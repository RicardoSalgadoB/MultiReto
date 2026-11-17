import { getGamCurves } from "./prediction";
import type { GamCurvesResponse, StationCode } from "./types";

/**
 * Las curvas del GAM no cambian mientras el modelo no cambie, así que se piden una sola vez por estación
 * y se guardan mientras dure la sesión de la página.
 */
const cache = new Map<StationCode, GamCurvesResponse>();

/** Respuesta ya guardada (sin pedir nada), o null. */
export function peekGamCurves(station: StationCode): GamCurvesResponse | null {
  return cache.get(station) ?? null;
}

export async function getGamCurvesCached(station: StationCode, signal?: AbortSignal): Promise<GamCurvesResponse> {
  const hit = cache.get(station);
  if (hit) return hit;
  const response = await getGamCurves(station, 100, signal);
  cache.set(station, response);
  return response;
}
