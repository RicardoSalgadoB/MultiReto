/**
 * Contratos de la API (backend/app/app.py). Los nombres de los campos son los que
 * devuelve el backend (en español); los identificadores propios del frontend van en inglés.
 *
 * Los `number | null` vienen de `limpio()` del backend: NaN se serializa como null.
 */

// ---------------------------------------------------------------------------
// Generales
// ---------------------------------------------------------------------------
/** Clave de estación: CE, NE, NE2, NE3, NO, NO2, NO3, NTE, NTE2, SE, SE2, SE3, SO, SO2, SUR. */
export type StationCode = string;

/** GET /estaciones: el formato también sirve como punto del mapa. */
export interface Station {
  id: number;
  /** Clave de la estación (CE, NE, ...). */
  name: StationCode;
  /** Nombre largo ("Centro", "Noreste", ...). */
  nombre: string;
  latitude: number;
  longitude: number;
}

/** GET / */
export interface ApiInfo {
  mensaje: string;
  /** Umbral auxiliar del modelo (ppb). */
  u_ppb: number;
  /** Norma horaria de ozono (ppb). */
  norma_ppb: number;
  /** τ: se emite alerta si P(máx > norma) ≥ corte_alerta. */
  corte_alerta: number;
}

/** Periodo al que pertenece el día. Valores reales de `diario.csv.gz` (con tilde en "validación"). */
export type Period = "entrenamiento" | "validación" | "prueba";

// ---------------------------------------------------------------------------
// Variables del modelo
// ---------------------------------------------------------------------------
/** Variables de clima que el usuario puede mover (Clima en app.py). */
export const CLIMATE_KEYS = [
  "SR_max_hoy",
  "RH_media_hoy",
  "WSR_media_hoy",
  "T_max_hoy",
  "T_max_pron",
  "T_min_pron",
] as const;
export type ClimateKey = (typeof CLIMATE_KEYS)[number];

/** Variables de persistencia del ozono. */
export const OZONE_KEYS = ["o3max", "o3media", "horas63", "horas63_7d", "o3max_ciudad"] as const;
export type OzoneKey = (typeof OZONE_KEYS)[number];

/** Entradas crudas del escenario (ozono + clima). */
export type ModelInputKey = OzoneKey | ClimateKey;

// ---------------------------------------------------------------------------
// Tab 1: histórico
// ---------------------------------------------------------------------------
/** GET /historico/rango */
export interface HistoricRange {
  /** YYYY-MM-DD */
  inicio: string;
  /** YYYY-MM-DD */
  fin: string;
}

export interface HistoricPoint extends Station {
  /** O3 de esa hora (ppb); null si no se midió. */
  value: number | null;
  periodo: Period | null;
  /** P(máx del día > u) · GAM */
  zeta: number | null;
  /** P(máx del día > norma) · GAM + GPD */
  p90: number | null;
  alerta: boolean | null;
  /** Máximo observado del día (ppb). */
  max_dia: number | null;
  /** El máximo del día superó la norma. */
  exceso_real: boolean | null;
}

/** GET /historico/hora */
export interface HistoricHourResponse {
  fecha: string;
  hora: number;
  corte_alerta: number;
  puntos: HistoricPoint[];
}

export interface SeriesSample {
  /** ISO local sin zona horaria, p. ej. "2026-05-15T15:00:00". */
  fecha_hora: string;
  o3: number | null;
}

/** GET /historico/serie */
export interface SeriesResponse {
  estacion: StationCode;
  norma_ppb: number;
  u_ppb: number;
  serie: SeriesSample[];
}

// ---------------------------------------------------------------------------
// Tab 2: predicciones
// ---------------------------------------------------------------------------
export interface ControlSpec {
  descripcion: string;
  unidad: string;
  /** null en las temperaturas crudas (T_max_hoy, T_min_pron): usar los límites de Clima. */
  min: number | null;
  max: number | null;
}

/** GET /escenario/base */
export interface BaseScenarioResponse {
  fecha_base: string;
  fecha_objetivo: string;
  clima_ciudad: Record<ClimateKey, number | null>;
  estaciones: Record<StationCode, Record<ModelInputKey, number | null>>;
  valores_completados: Record<StationCode, string[]>;
  controles: Record<ModelInputKey, ControlSpec>;
  nota: string;
}

/** Cuerpo de POST /prediccion. Lo que no se manda se toma del día base. */
export interface PredictionRequest {
  fecha_base?: string;
  fecha_objetivo?: string;
  clima?: Partial<Record<ClimateKey, number>>;
  estaciones?: Record<StationCode, Partial<Record<ModelInputKey, number>>>;
  /** Niveles (ppb) para P(máx > nivel); cada uno debe ser ≥ u_ppb. */
  niveles?: number[];
}

export interface PredictionPoint extends Station {
  /** Igual a p90: es el valor con el que se colorea el mapa. */
  value: number | null;
  zeta: number | null;
  p90: number | null;
  /** P(máx > nivel) por nivel; las claves son el nivel como texto ("90", "105"). */
  probabilidades: Record<string, number | null>;
  sigma: number | null;
  alerta: boolean;
  entradas: Record<ModelInputKey, number | null>;
  /** Variables que se llevaron al límite del rango de entrenamiento. */
  recortadas: string[];
  /** Variables faltantes que se llenaron con la mediana de entrenamiento. */
  completadas: string[];
}

/** POST /prediccion */
export interface PredictionResponse {
  fecha_base: string;
  fecha_objetivo: string;
  u_ppb: number;
  norma_ppb: number;
  corte_alerta: number;
  /** Forma (ξ) de la GPD, común a todas las estaciones. */
  xi: number;
  puntos: PredictionPoint[];
}

export interface GamCurve {
  variable: string;
  x: number[];
  /** Efecto de la variable en logit ζ (centrado). Igual para todas las estaciones. */
  efecto_logit: number[];
  /** Solo si se manda estación: P(máx > u) de un día típico de esa estación. */
  probabilidad?: number[];
  descripcion: string;
  unidad: string;
}

/** GET /gam/curvas */
export interface GamCurvesResponse {
  estacion: StationCode | null;
  /** Desplazamiento α_s en logit ζ por estación. */
  efecto_estacion: Record<StationCode, number>;
  /** Estación de referencia (α = 0). */
  referencia: StationCode;
  curvas: Record<string, GamCurve>;
}

// ---------------------------------------------------------------------------
// Tab 3: dependencia
// ---------------------------------------------------------------------------
export type DependenceKind = "chi" | "chibar" | "spearman";

export interface DependencePair {
  a: StationCode;
  b: StationCode;
  /** null si el par no tiene suficientes datos: no se dibuja. */
  valor: number | null;
  distancia_km: number;
  dias_conjuntos: number;
}

/** GET /dependencia */
export interface DependenceResponse {
  tipo: DependenceKind;
  /** Cuantil con el que se calculó χ (0.95). */
  q: number;
  periodo: string;
  estaciones: Station[];
  matriz: (number | null)[][];
  /** 105 pares, cada uno una sola vez. */
  pares: DependencePair[];
}

// ---------------------------------------------------------------------------
// Errores
// ---------------------------------------------------------------------------
/** Un error de validación de FastAPI (422). */
export interface ValidationIssue {
  /** Ruta del campo, p. ej. ["body", "clima", "RH_media_hoy"]. */
  loc: (string | number)[];
  msg: string;
}
