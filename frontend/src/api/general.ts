import { apiGet } from "./client";
import type { ApiInfo, Station } from "./types";

/** GET / : umbrales y corte de alerta vigentes en el backend. */
export function getInfo(signal?: AbortSignal): Promise<ApiInfo> {
  return apiGet<ApiInfo>("/", undefined, signal);
}

/** GET /estaciones : las 15 estaciones con coordenadas. */
export function getStations(signal?: AbortSignal): Promise<Station[]> {
  return apiGet<Station[]>("/estaciones", undefined, signal);
}
