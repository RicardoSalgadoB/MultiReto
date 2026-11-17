import { apiGet, apiPost } from "./client";
import type {
  BaseScenarioResponse,
  GamCurvesResponse,
  PredictionRequest,
  PredictionResponse,
  StationCode,
} from "./types";

/** GET /escenario/base : valores reales de un día (por defecto, el último con datos). */
export function getBaseScenario(date?: string, signal?: AbortSignal): Promise<BaseScenarioResponse> {
  return apiGet<BaseScenarioResponse>("/escenario/base", { fecha: date }, signal);
}

/** POST /prediccion */
export function postPrediction(body: PredictionRequest, signal?: AbortSignal): Promise<PredictionResponse> {
  return apiPost<PredictionResponse>("/prediccion", body, signal);
}

/** GET /gam/curvas : las 11 curvas del GAM (comunes a todas las estaciones). */
export function getGamCurves(station?: StationCode, points = 100, signal?: AbortSignal): Promise<GamCurvesResponse> {
  return apiGet<GamCurvesResponse>("/gam/curvas", { estacion: station, puntos: points }, signal);
}
