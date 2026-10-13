"""Entrenamiento de la GCN que estima (xi, sigma) de la GPD por estación.

Este código usó como referencia 2 repositorios de GitHub, los cuales se tomaron de inspiración 
para adaptar este código a nuestro problema. :

- deepGPD (Wilson et al., 2022): https://github.com/TylerPWilson/deepGPD/tree/main
- pytorch-GCN (senadkurtisi): https://github.com/senadkurtisi/pytorch-GCN/tree/main


Cada "data" (train / val / test) es un dict de tensores con todo el conjunto (lote completo,
como en ambos repos):
    "vecs"  : (B, ndim_v, l, t)  covariables vectoriales         [opcional]
    "sets"  : (B, ndim_s, l, t)  excesos previos, NaN de relleno [opcional]
    "y"     : (B, l)             exceso observado y = Y - u (NaN si no hubo)
    "mask"  : (B, l)             True donde y es válido          [opcional]
    "maxis" : lo agrega add_maxis()
D es la matriz de distancias (haversine) que recibe el modelo.

El modelo debe cumplir:
    model(vecs, sets, D, maxis=maxis) -> (B, l, 2), con [..., 0] = xi y [..., 1] = sigma

Uso (equivalente al main.py del repo de la GCN):
    add_maxis(train_data, val_data, test_data)
    model = Combiner(ds_params, gcn_params)
    config = TrainConfig()
    history = training_loop(model, train_data, val_data, D, config)
    evaluate_on_test(model, test_data, D, config)
    pred_test = predict(model, test_data, D)          # (B, l, 2) = (xi, sigma)
"""


import copy
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

try:
    from utils.loss_function import gp_penalized
except ModuleNotFoundError:  # si se ejecuta como `python utils/train.py`
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.loss_function import gp_penalized


# Posición de cada parámetro en la última dimensión de la salida del modelo
# (es el orden en que lo entrega el Constrainer: primero xi, luego sigma).
XI_IDX, SIGMA_IDX = 0, 1


@dataclass
class TrainConfig:
    """Se reunen todos los hiperparámetros del entrenamiento para modificarlos más facilmente
 
    Para cambiar un valor solo se edita este mismo en la función o al llamando al entrenamiento con un 
    valor diferente como parametro.

    Se usa "dataclass" en lugar de "argparse" porque esta es más corta y se me es más facil de leer que argparse.
 
    Atributos:
        nepochs: Número de pasos de entrenamiento. Cada paso usa todo el conjunto de train.
        eval_every: Cada cuántos pasos se evalúa en validación.
        lr: Tasa de aprendizaje de Adam (qué tanto se mueven los pesos en cada paso).
        weight_decay: Regularización L2 de Adam. 0 = sin regularizar.
        grad_clip: Tope a la norma del gradiente para evitar saltos bruscos (None = sin tope)
        use_early_stopping: Si es True, el entrenamiento se detiene al dejar de mejorar.
        patience: Evaluaciones consecutivas sin mejora que se toleran antes de parar. Se
            cuenta en evaluaciones.
        max_zero_prob: Un modelo solo cuenta como "el mejor" si su zero_prob en validación
            es menor o igual a este valor (por defecto 1%).
        device: "cpu" o "cuda".
        seed: Semilla aleatoria para poder repetir los resultados.
        ckpt_path: Archivo donde se guarda el mejor modelo. None = no guardar.
        verbose: Si es True, imprime el progreso.
    """

    nepochs: int = 1500                
    eval_every: int = 50               
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    use_early_stopping: bool = True
    patience: int = 10                 
    max_zero_prob: float = 0.01        
    device: str = "cpu" #Cambiar por "cuda" en caso de tener GPU
    seed: int = 42
    ckpt_path: Optional[str] = "checkpoints/best.pt"
    verbose: bool = True


def set_seed(seed: int) -> None:
    # Fijamos las semmillas para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(data: dict, device: torch.device) -> dict:
    """Como mi equipo cuenta con una tarjeta gráfica puede correr el entrenamiento en ella
     En dado caso de no contar con una, unicamente asegurarse que el device sea "cpu" y no "cuda" en el TrainConfig
     """
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}



def add_maxis(train_data: dict, *other_data: dict) -> None:
    """Agrega "maxis" a train/val/test usando siempre el máximo de 'y' en train.

    Tiene la misma forma que y. Donde no hubo excesos en train se usa 1.

    'maxis' es el mayor exceso observado en cada nodo durante el entrenamiento. El
    Constrainer del modelo lo usa para que xi nunca vuelva imposible una observación
    (garantiza 1 + xi*y/sigma > 0 para los y de entrenamiento).

    Se usó de inspiración el repo "https://github.com/TylerPWilson/deepGPD/blob/main/training.py" 

    Args:
        train_data: Dict con "y" de entrenamiento. Se modifica en el lugar.
        *other_data: Dicts de val y/o test. Se modifican en el lugar.
    """
    y = train_data["y"]
    y_inf = torch.where(torch.isfinite(y), y, torch.full_like(y, float("-inf")))
    train_max = y_inf.amax(dim=(0, 1))
    train_max = torch.where(torch.isinf(train_max), torch.ones_like(train_max), train_max)
    for data in (train_data, *other_data):
        data["maxis"] = train_max.expand_as(data["y"]).clone()



def gpd_nll(xi, sigma, y, mask=None, xi_eps=1e-4, min_arg=1e-6):
    """Log-verosimilitud negativa (NLL) promedio de la GPD. Es la pérdida que se minimiza.

    Mide qué tan creíbles son (xi, sigma) dados los excesos observados: un valor bajo
    significa que la distribución propuesta explica bien los datos.

    log f(y) = -log(sigma) - (1 + 1/xi) * log(1 + xi*y/sigma)    si xi != 0
    log f(y) = -log(sigma) - y/sigma                              si xi == 0
    
    Notación:
        B: número de batches
        l: número de estaciones
        t: número de intervalos de tiempo
    
    Args:
        xi: Parámetro de forma, tensor (B, l).
        sigma: Parámetro de escala, tensor (B, l), debe ser > 0.
        y: Excesos observados (B, l, t), con NaN donde no hubo exceso.
        mask: Opcional (B, l, t). False = ignorar esa entrada.
        xi_eps: Umbral para tratar xi como 0.
        min_arg: Valor mínimo del argumento del log.
 
    Returns:
        Tensor escalar con la NLL promedio sobre las entradas válidas
    """
    y = y.squeeze()
    if y.dim() == 2:    # Si la forma es (B, l) pasar a (B, l, 1)
        y = y.unsqueeze(2)
        if mask is not None:
            mask = mask.unsqueeze(2)
          
    # Cambiar forma de parámetros para que sean congruentes con y  
    if xi.dim() == y.dim() - 1:
        xi = xi.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        
    valid = torch.isfinite(y) & (y >= 0)
    if mask is not None:
        valid = valid & mask.bool()

    y_safe = torch.where(valid, y, torch.zeros_like(y))
    sigma = sigma.clamp_min(1e-8)

    small = xi.abs() < xi_eps
    xi_safe = torch.where(small, torch.ones_like(xi), xi)  # evita dividir entre 0

    z = (1.0 + xi_safe * y_safe / sigma).clamp_min(min_arg)
    ll_xi = -torch.log(sigma) - (1.0 + 1.0 / xi_safe) * torch.log(z)
    ll_0 = -torch.log(sigma) - y_safe / sigma
    ll = torch.where(small, ll_0, ll_xi)

    n = valid.sum()
    if n == 0:
        return ll.sum() * 0.0
    return -(ll * valid).sum() / n



def forward(model: nn.Module, data: dict, D: torch.Tensor, with_metrics: bool = True):
    """Hace una pasada al modelo y calcula las 2 medidas de error.

    Pasos: 
    (1) el modelo predice (xi, sigma) para cada estación 
    (2) se calcula la pérdida derivable gpd_nll
    (3) opcionalmente se calculan las métricas de gp_penalized

    Las 2 medidas de error:
    `loss` tiene gradiente y sirve para entrnar, `llk_loss` y
    `zero_prob` vienen de gp_penalized (no tienen gradiente) y sirven para medir y elegir
    el mejor modelo.

    Args:
        model: Modelo con firma model(vecs, sets, D, maxis=...) -> (B, l, 2).
        data: Dict con "y" y "maxis", y opcionalmente "vecs", "sets" y "mask".
        D: Matriz de distancias (haversine) que recibe el modelo.
        with_metrics: Si es False se omite gp_penalized (más rápido durante el entrenamiento).
 
    Returns:
        loss: Tensor escalar con gradiente (gpd_nll).
        llk_loss: float, NLL penalizada de gp_penalized (None si with_metrics=False).
        zero_prob: float, fracción de y fuera del soporte (None si with_metrics=False).
        pred: Tensor (B, l, 2); [..., 0] = xi y [..., 1] = sigma.
    """
    pred = model(data.get("vecs"), data.get("sets"), D, maxis=data.get("maxis"))
    xi, sigma = pred[..., XI_IDX], pred[..., SIGMA_IDX]
    loss = gpd_nll(xi, sigma, data["y"], data.get("mask"))

    llk_loss = zero_prob = None
    if with_metrics:
        y = data["y"]
        valid = torch.isfinite(y) & (y >= 0)
        if data.get("mask") is not None:
            valid = valid & data["mask"].bool()
        if valid.any():
            y_nan = torch.where(valid, y, torch.full_like(y, float("nan")))  # gp_penalized ignora NaN
            llk_loss, zero_prob = (float(v) for v in gp_penalized(y_nan, xi, sigma))
        else:
            llk_loss, zero_prob = float("nan"), float("nan")
    return loss, llk_loss, zero_prob, pred



@torch.no_grad()
def evaluate(model: nn.Module, data: dict, D: torch.Tensor) -> dict:
    """
    Mide el desempeño del modelo en un conjunto de datos sin modificar sus pesos
    Returns:
        Dict con "nll" (NLL penalizada), "zero_prob" (fracción de 'y' fuera
        del soporte), "xi_mean" y "sigma_mean" (promedios de las predicciones).
    """
    model.eval()
    _, llk_loss, zero_prob, pred = forward(model, data, D)
    return {"nll": llk_loss, "zero_prob": zero_prob,
            "xi_mean": pred[..., XI_IDX].mean().item(),
            "sigma_mean": pred[..., SIGMA_IDX].mean().item()}


@torch.no_grad()
def predict(model: nn.Module, data: dict, D: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """
    Parámetros (xi, sigma) predichos, como arreglo numpy (B, l, 2).
    
    con xi y sigma se calcula la probabilidad de superar la norma.
    Pone el modelo en modo evaluación (sin gradiente)

    Args:
        model: Modelo ya entrenado.
        data: Dict de datos (con "maxis" ya agregado).
        D: Matriz de distancias.
        device: "cpu" o "cuda".
 
    Returns:
        np.ndarray (B, l, 2): [..., 0] = xi y [..., 1] = sigma.
    """
    dev = torch.device(device)
    model.to(dev).eval()
    data = _to_device(data, dev)
    _, _, _, pred = forward(model, data, D.to(dev), with_metrics=False)
    return pred.cpu().numpy()


def evaluate_on_test(model: nn.Module, test_data: dict, D: torch.Tensor,
                     config: TrainConfig) -> dict:
    """
    Evalua el modelo final en el conjunto de test e imprime los resultados
    Returns:
        Dict de evaluate(): nll, zero_prob, xi_mean y sigma_mean.
    """
    device = torch.device(config.device)
    model.to(device)
    res = evaluate(model, _to_device(test_data, device), D.to(device))
    if config.verbose:
        print(f"\nTest NLL: {res['nll']:.4f} | zero_prob: {res['zero_prob']:.4f} | "
              f"xi medio: {res['xi_mean']:+.3f} | sigma medio: {res['sigma_mean']:.3f}")
    return res



def training_loop(model: nn.Module, train_data: dict, val_data: Optional[dict],
                  D: torch.Tensor, config: TrainConfig) -> dict:
    """
    Entrena `model` con lote completo y devuelve el historial.

    Cada `eval_every` pasos evalúa en validación con gp_penalized. Un modelo es el
    "mejor hasta ahora" si baja la NLL de validación Y zero_prob <= max_zero_prob.
    Al terminar deja cargado el mejor modelo. val_data debe
    ser POSTERIOR en el tiempo a train_data. Si val_data es None se monitorea train.

    Pasos que sigue:
    1. Prepara semilla, dispositivo, datos y el optimizador (Adam).
    2. En cada paso: forward sobre todo train -> pérdida (gpd_nll) -> backward() calcula
         cuánto mover cada peso -> optimizer.step() los mueve.
    3. Cada `eval_every` pasos evalúa en validación con gp_penalized. El modelo es "el
         mejor hasta ahora" si baja la NLL de validación Y zero_prob <= max_zero_prob. Si lo
         es, guarda sus pesos (en memoria y en ckpt_path) y reinicia el contador de paciencia.
    4. Si pasan `patience` evaluaciones sin mejora, se detiene (early stopping).
    5. Al terminar carga en el modelo los mejores pesos.

    Args:
        model: Modelo (p. ej. Combiner). Se modifica en el lugar.
        train_data: Dict de tensores de entrenamiento, con "maxis" ya agregado.
        val_data: Dict de tensores de validación (o None).
        D: Matriz de distancias (haversine).
        config: TrainConfig con los hiperparámetros.
 
    Returns:
        Dict con listas por evaluación ("step", "train_nll", "val_nll", "val_zero_prob",
        "val_xi_mean", "val_sigma_mean") y "best_val_nll". Sirve para graficar las curvas
    """
    set_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    D = D.to(device)
    train_data = _to_device(train_data, device)
    monitor_data = _to_device(val_data, device) if val_data is not None else train_data

    y_tr = train_data["y"]
    if not (torch.isfinite(y_tr) & (y_tr >= 0)).any():
        raise ValueError("train_data no tiene observaciones válidas en y.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    history = {"step": [], "train_nll": [], "val_nll": [], "val_zero_prob": [],
               "val_xi_mean": [], "val_sigma_mean": []}
    best_nll, best_state, patience_counter, stopped_early = float("inf"), None, 0, False
    t_start = time.time()

    for epoch in tqdm(range(config.nepochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = forward(model, train_data, D, with_metrics=False)
        if not torch.isfinite(loss):
            print(f"  [aviso] paso {epoch}: pérdida no finita, se omite el paso")
            continue
        loss.backward()
        if config.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if epoch % config.eval_every == 0:
            ev = evaluate(model, monitor_data, D)
            history["step"].append(epoch)
            history["train_nll"].append(loss.item())
            history["val_nll"].append(ev["nll"])
            history["val_zero_prob"].append(ev["zero_prob"])
            history["val_xi_mean"].append(ev["xi_mean"])
            history["val_sigma_mean"].append(ev["sigma_mean"])

            improved = ev["nll"] < best_nll - 1e-6 and ev["zero_prob"] <= config.max_zero_prob
            if improved:
                best_nll, patience_counter = ev["nll"], 0
                best_state = copy.deepcopy(model.state_dict())
                if config.ckpt_path is not None:
                    Path(config.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": best_state, "step": epoch, "val_nll": best_nll}, config.ckpt_path)
            else:
                patience_counter += 1

            if config.verbose:
                tag = "  *mejor hasta ahora*" if improved else ""
                print(f"paso {epoch:5d} | train NLL {loss.item():.4f} | val NLL {ev['nll']:.4f} | "
                      f"zero_prob {ev['zero_prob']:.4f} | xi {ev['xi_mean']:+.3f} | "
                      f"sigma {ev['sigma_mean']:.3f}{tag}")

            if config.use_early_stopping and patience_counter >= config.patience:
                stopped_early = True
                break

    if config.verbose:
        if stopped_early:
            print(f"EARLY STOPPING: sin mejora en {config.patience} evaluaciones. Paso {epoch}.")
        print(f"Tiempo total de entrenamiento: {time.time() - t_start:.2f} s")

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[aviso] ningún punto cumplió zero_prob <= max_zero_prob; se deja el último modelo.")
    history["best_val_nll"] = best_nll
    return history



def multiple_runs(model_fn, train_data: dict, val_data: Optional[dict], test_data: dict,
                  D: torch.Tensor, config: TrainConfig, num_runs: int = 5) -> list[dict]:
    """
    Repite el entrenamiento varias veces con distintas inicializaciones y resume el test.
    Esta idea se toma del repo "https://github.com/senadkurtisi/pytorch-GCN/blob/main/src/training_evaluation.py",
    ya que una sola corrida puede salir bien o mal al azar como los pesos arrancan aleatorios, por eso repite el entrenamiento
    varias veces para ver la estaboilidad del modelo 

    en cada corrida crea un modelo nuevo con model_fn(), lo entrena con una
    semilla distinta (seed + número de corrida), lo evalúa en test y guarda su mejor modelo
    en un checkpoint propio (..._run0.pt, ..._run1.pt, ...). Al final imprime la media y la
    desviación estándar de la NLL de test.
 
    Args:
        model_fn: Función sin argumentos que devuelve un modelo nuevo.
        train_data, val_data, test_data: Dicts de datos, con "maxis" ya agregado.
        D: Matriz de distancias.
        config: TrainConfig base (cada corrida cambia seed y ckpt_path, y silencia el progreso).
        num_runs: Número de corridas.
 
    Returns:
        Lista con el dict de resultados de test (evaluate) de cada corrida.
    """
    results = []
    for r in range(num_runs):
        print(f"Corrida {r + 1}/{num_runs}")
        ckpt = None
        if config.ckpt_path is not None:
            p = Path(config.ckpt_path)
            ckpt = str(p.with_name(f"{p.stem}_run{r}{p.suffix}"))
        cfg = replace(config, seed=config.seed + r, ckpt_path=ckpt, verbose=False)
        model = model_fn()
        training_loop(model, train_data, val_data, D, cfg)
        results.append(evaluate_on_test(model, test_data, D, cfg))

    nll = [x["nll"] for x in results]
    print(f"TEST NLL: media {np.mean(nll):.4f} | desv. est. {np.std(nll):.4f}")
    return results