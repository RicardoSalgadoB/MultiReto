import { getDependence } from "./dependence";
import type { DependenceKind, DependenceResponse } from "./types";

/**
 * /dependencia solo se pide al cambiar de medida y se guarda por medida durante la sesión: volver a una medida ya
 * vista no genera petición, y seleccionar una estación nunca pide nada.
 */
const cache = new Map<DependenceKind, DependenceResponse>();

/** Respuesta ya guardada (sin pedir nada), o null. */
export function peekDependence(kind: DependenceKind): DependenceResponse | null {
  return cache.get(kind) ?? null;
}

export async function getDependenceCached(kind: DependenceKind, signal?: AbortSignal): Promise<DependenceResponse> {
  const hit = cache.get(kind);
  if (hit) return hit;
  const response = await getDependence(kind, signal);
  cache.set(kind, response);
  return response;
}
