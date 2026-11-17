import { apiGet } from "./client";
import type { DependenceKind, DependenceResponse } from "./types";

/** GET /dependencia : matriz y pares entre estaciones (chi, chibar o spearman). */
export function getDependence(kind: DependenceKind, signal?: AbortSignal): Promise<DependenceResponse> {
  return apiGet<DependenceResponse>("/dependencia", { tipo: kind }, signal);
}
