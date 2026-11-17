import { apiGet } from "./client";
import type { HistoricHourResponse, HistoricRange, SeriesResponse, StationCode } from "./types";

/** GET /historico/rango */
export function getHistoricRange(signal?: AbortSignal): Promise<HistoricRange> {
  return apiGet<HistoricRange>("/historico/rango", undefined, signal);
}

/** GET /historico/hora : fecha en YYYY-MM-DD y hora 0..23. */
export function getHistoricHour(date: string, hour: number, signal?: AbortSignal): Promise<HistoricHourResponse> {
  return apiGet<HistoricHourResponse>("/historico/hora", { fecha: date, hora: hour }, signal);
}

/** GET /historico/serie : serie horaria de los `days` días anteriores al momento indicado. */
export function getHistoricSeries(
  station: StationCode,
  date: string,
  hour: number,
  days = 7,
  signal?: AbortSignal,
): Promise<SeriesResponse> {
  return apiGet<SeriesResponse>("/historico/serie", { estacion: station, fecha: date, hora: hour, dias: days }, signal);
}
