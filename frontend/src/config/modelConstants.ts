/**
 * Constantes del modelo que el frontend NO recibe de la API: se copiaron del backend o salen de una decisión
 * de interfaz. Si el backend cambia, hay que revisar este archivo (y la tabla del README).
 *
 * Todo lo que la API sí devuelve (u_ppb, norma_ppb, corte_alerta, rangos de entrenamiento en `controles`,
 * q y periodo de la dependencia...) se lee de la respuesta y NO se repite aquí.
 */
import type { ClimateKey, DependenceKind, ModelInputKey } from "../api/types";

// ---------------------------------------------------------------------------
// Copiadas del backend
// ---------------------------------------------------------------------------

/**
 * Límites de validación de los controles de clima.
 * Origen: backend/app/app.py, clase `Clima` (cada `Field(None, ge=..., le=...)`).
 * La API no los publica. Sirven de rango de los controles sin rango de entrenamiento (`min/max` null en
 * /escenario/base) y de tope del campo numérico.
 */
export const CLIMATE_LIMITS: Record<ClimateKey, [number, number]> = {
  SR_max_hoy: [0, 1.5], // kW/m²
  RH_media_hoy: [0, 100], // %
  WSR_media_hoy: [0, 150], // km/h
  T_max_hoy: [-10, 50], // °C
  T_max_pron: [-10, 50], // °C
  T_min_pron: [-10, 50], // °C
};

/**
 * Qué variables del GAM dependen de cada control. El backend informa las `recortadas` con los nombres del modelo
 * y las temperaturas crudas alimentan también `dT_pron` y `rango_pron`.
 * Origen: backend/source/predict.py (`completar_variables` y `acotar`) y backend/source/config.py
 * (`CLIMA_D`, `CLIMA_ENTRADA`).
 */
export const MODEL_VARS_OF_CONTROL: Record<ClimateKey, string[]> = {
  SR_max_hoy: ["SR_max_hoy"],
  RH_media_hoy: ["RH_media_hoy"],
  WSR_media_hoy: ["WSR_media_hoy"],
  T_max_hoy: ["dT_pron"],
  T_max_pron: ["T_max_pron", "dT_pron", "rango_pron"],
  T_min_pron: ["rango_pron"],
};

/**
 * Variables del GAM que no vienen en `entradas` y se derivan de ellas.
 * Origen: backend/source/predict.py, `ModeloOzono.completar_variables`:
 *   dT_pron    = T_max_pron − T_max_hoy
 *   rango_pron = T_max_pron − T_min_pron
 */
export const DERIVED_VARIABLES: Record<string, { minuend: ModelInputKey; subtrahend: ModelInputKey }> = {
  dT_pron: { minuend: "T_max_pron", subtrahend: "T_max_hoy" },
  rango_pron: { minuend: "T_max_pron", subtrahend: "T_min_pron" },
};

/**
 * Nombres de las variables derivadas (no están en `controles`).
 * Origen: backend/source/config.py, diccionario `DESCRIPCIONES`.
 */
export const DERIVED_VARIABLE_LABELS: Record<string, string> = {
  dT_pron: "Cambio: máxima pronosticada − máxima de hoy",
  rango_pron: "Rango pronosticado: máxima − mínima",
};

/**
 * Temperaturas pronosticadas: variables que faltan cuando el día no tiene pronóstico.
 * Origen: backend/source/config.py (`CLIMA_ENTRADA`) y la lista `valores_completados` de /escenario/base.
 */
export const FORECAST_VARIABLES: ClimateKey[] = ["T_max_pron", "T_min_pron"];

// ---------------------------------------------------------------------------
// Decisiones de la interfaz (no vienen del backend)
// ---------------------------------------------------------------------------

/** Niveles extra (ppb) de la tabla "probabilidad por nivel"; el nivel de la norma sale de la API. Deben ser ≥ u_ppb. */
export const LEVEL_BELOW_NORM = 80;
export const LEVEL_ABOVE_NORM = 105;

/** Dominio del color del Tab 1: ozono 0–120 ppb (los valores mayores se muestran con el color de 120). */
export const OZONE_DOMAIN: [number, number] = [0, 120];

/** Dominio del color del Tab 2: probabilidad 0–1 (fijo). */
export const PROBABILITY_DOMAIN: [number, number] = [0, 1];

/** Máximo de días que retrocede "Ir al último día con pronóstico" (una petición a /escenario/base por día). */
export const FORECAST_SEARCH_MAX_DAYS = 31;

// ---------------------------------------------------------------------------
// Tab 3: escalas de la dependencia (elegidas con los valores reales de backend/data/dependencia.json)
// ---------------------------------------------------------------------------

/**
 * Dominio fijo del color y del grosor de las líneas por medida. Valores observados en los 105 pares
 * (backend/data/dependencia.json, sin nulos ni negativos):
 *   chi      0.200 – 0.800  -> [0.2, 0.8]
 *   chibar   0.300 – 0.861  -> [0.3, 0.9]
 *   spearman 0.474 – 0.954  -> [0.4, 1.0]
 * Los valores fuera del dominio se muestran con el color y el grosor del extremo.
 */
export const DEPENDENCE_DOMAINS: Record<DependenceKind, [number, number]> = {
  chi: [0.2, 0.8],
  chibar: [0.3, 0.9],
  spearman: [0.4, 1.0],
};

/**
 * Un par tiene "pocos días conjuntos" si su `dias_conjuntos` queda por debajo de este cuantil de los pares de la
 * respuesta. En los datos actuales `dias_conjuntos` va de 9 a 38 sin huecos naturales (P10 = 13, P25 = 17,
 * mediana = 25), así que se usa el primer cuartil como criterio relativo en vez de inventar un número.
 */
export const LOW_JOINT_DAYS_QUANTILE = 0.25;
