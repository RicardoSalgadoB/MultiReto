import { getHistoricRange } from "./historic";
import type { HistoricRange } from "./types";

/** El rango de fechas disponibles no cambia durante la sesión: se pide una sola vez aunque se cambie de pestaña. */
let cached: HistoricRange | null = null;

/** Rango ya guardado (sin pedir nada), o null. */
export function peekHistoricRange(): HistoricRange | null {
  return cached;
}

export async function getHistoricRangeCached(signal?: AbortSignal): Promise<HistoricRange> {
  if (cached) return cached;
  cached = await getHistoricRange(signal);
  return cached;
}
