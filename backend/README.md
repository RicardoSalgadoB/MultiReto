# Backend · API de ozono (FastAPI)

API que alimenta el dashboard de React. Sirve el modelo final del reporte (notebook
`notebooks/04e_gam_gpd.ipynb`): un **GAM con respuesta binaria** para
ζ = P(máximo de mañana > 63 ppb) y una **GPD** del exceso para extrapolar a la norma de 90 ppb.

```
P(máx > c) = ζ · [1 + ξ (c − 63) / σ]^(−1/ξ),   log σ = b₀ + b₁ logit ζ + spline(logit ζ)
```

## Cómo correrlo

Desde la carpeta `backend/`:

```bash
pip install -r requirements.txt        # o: uv add fastapi "uvicorn[standard]" pytest httpx
uvicorn app.app:app --reload           # http://localhost:8000
```

- Documentación interactiva (para probar cada endpoint): <http://localhost:8000/docs>
- Pruebas: `python -m pytest -q`
- El frontend de Vite (`http://localhost:5173`) ya tiene permiso (CORS). Para otro origen:
  `CORS_ORIGINS="http://localhost:5173,https://mi-sitio.com" uvicorn app.app:app`

La API **no necesita los datos crudos**: usa los archivos ya exportados en `models/` y `data/`.

## Estructura

```
backend/
├── app/app.py           API (endpoints)
├── source/
│   ├── config.py        constantes: estaciones, variables, umbrales, rutas
│   ├── datos.py         limpieza y tabla diaria (igual que el notebook)
│   ├── entrenar.py      entrena y exporta el modelo y los datos de la API
│   └── predict.py       predicción con los parámetros en JSON (sin statsmodels)
├── models/
│   ├── gam.json         nudos y coeficientes de cada spline, efectos de estación y calendario
│   ├── gpd.json         b, ξ, nudos del spline de logit ζ y el corte de alerta τ
│   └── metricas.json    BSS, AUC, aciertos y falsas alarmas en 2025 y 2026
├── data/
│   ├── diario.csv.gz    estación-día: variables del modelo y predicción guardada
│   ├── historico_o3.csv.gz   O3 horario por estación
│   └── dependencia.json      matrices χ(0.95), χ̄ y Spearman
└── tests/test_api.py
```

## Volver a entrenar

Solo si cambian los datos o el modelo. Necesita `data/procesados/datos_no_imputados.csv` y
`data/pronosticos/pronostico_dia_anterior.csv` en la raíz del repositorio:

```bash
python -m source.entrenar
# o con otras rutas:
python -m source.entrenar --datos RUTA/datos_no_imputados.csv --pronostico RUTA/pronostico.csv
```

El script elige α con el BSS de 2025, ajusta la GPD, fija τ (detectar 70 % de los días con exceso
en 2025) y **verifica que `predict.py` dé lo mismo que statsmodels** (diferencia < 1e-8); si no, se detiene.

## Endpoints

| Tab | Método y ruta | Qué devuelve |
|---|---|---|
| — | `GET /estaciones` | Clave, nombre y coordenadas (formato del componente `Map`: `id`, `name`, `latitude`, `longitude`) |
| — | `GET /metricas` | Desempeño en 2025 y 2026 |
| 1 | `GET /historico/rango` | Primer y último día con datos |
| 1 | `GET /historico/hora?fecha=2026-05-15&hora=15` | O3 de esa hora por estación (`value`) y la predicción para ese día (`p90`, `alerta`, `exceso_real`) |
| 1 | `GET /historico/serie?estacion=SUR&fecha=2026-05-15&hora=15&dias=7` | Serie horaria de los 7 días anteriores |
| 2 | `GET /escenario/base?fecha=2026-05-14` | Valores reales de un día para llenar el sidebar, con límites de cada control |
| 2 | `POST /prediccion` | ζ = P(>63), P(>90), σ y alerta por estación para un escenario |
| 2 | `GET /gam/curvas?estacion=SUR` | Curvas del GAM (efecto en logit ζ y probabilidad para esa estación) |
| 3 | `GET /dependencia?tipo=chi` | Matriz y pares (`chi`, `chibar` o `spearman`) con distancia |

### Ejemplo de `POST /prediccion`

```json
{
  "fecha_base": "2026-05-14",
  "clima": {"T_max_pron": 36, "WSR_media_hoy": 6},
  "estaciones": {"SUR": {"o3max": 95}},
  "niveles": [90, 105]
}
```

- `fecha_base`: día "de hoy" del que se toman el ozono y el clima (por defecto, el último con datos).
- `clima`: cambia el clima de **todas** las estaciones; lo que no se mande se queda como el día base.
- `estaciones`: cambios por estación (cualquier variable de ozono o de clima).
- Respuesta por estación: `zeta` (GAM), `p90` (GAM + GPD), `probabilidades` por nivel, `alerta`
  (`p90 ≥ τ`), `entradas` usadas, `recortadas` (valores fuera del rango de entrenamiento, que se
  llevaron al límite) y `completadas` (faltantes que se llenaron con la mediana de entrenamiento).

## Cosas que el frontend debe saber

1. **El modelo es diario.** Predice el máximo de un día con los datos del día anterior; en el Tab 1 la
   predicción es la misma para todas las horas del día.
2. **Las predicciones de 2023–2024 son de entrenamiento** (`periodo = "entrenamiento"`): el modelo ya vio
   esos datos. Las honestas son las de 2025 (validación) y 2026 (prueba).
3. **El GAM no usa la dirección del viento, la presión ni la lluvia.** Un control de dirección del viento
   no cambiaría nada.
4. **El umbral auxiliar es 63 ppb**, no 60.
5. **Las curvas del GAM son las mismas para las 15 estaciones** (es un solo modelo); la estación solo
   sube o baja la curva (`efecto_estacion`).
6. En el Tab 3 hay **correlación de Spearman y dependencia de colas**, no covarianza.
