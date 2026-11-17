import type { HistoricRange } from "../api/types";

/**
 * Fechas y horas del histórico. Todo se calcula en UTC sobre la "hora de pared" que entrega la API
 * (sin zona horaria), así el resultado no depende de la zona del navegador.
 */
export interface Moment {
  /** YYYY-MM-DD */
  fecha: string;
  /** 0..23 */
  hora: number;
}

/** Hora inicial por defecto: hora típica del pico de ozono. */
const DEFAULT_HOUR = 15;

const HOUR_MS = 3_600_000;
const DAY_MS = 24 * HOUR_MS;
const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;
const ISO_DATETIME = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/;

export function isIsoDate(value: string): boolean {
  const match = ISO_DATE.exec(value);
  if (!match) return false;
  const [year, month, day] = [Number(match[1]), Number(match[2]), Number(match[3])];
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day;
}

export function toUtcMs(moment: Moment): number {
  const [year, month, day] = moment.fecha.split("-").map(Number);
  return Date.UTC(year, month - 1, day, moment.hora);
}

function fromUtcMs(ms: number): Moment {
  const date = new Date(ms);
  return { fecha: date.toISOString().slice(0, 10), hora: date.getUTCHours() };
}

/** "2026-05-15T15:00:00" -> milisegundos UTC; null si no tiene ese formato. */
export function parseSampleTime(value: string): number | null {
  const match = ISO_DATETIME.exec(value);
  if (!match) return null;
  return Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]), Number(match[4]), Number(match[5]));
}

export function rangeBounds(range: HistoricRange): { first: number; last: number } {
  return {
    first: toUtcMs({ fecha: range.inicio, hora: 0 }),
    last: toUtcMs({ fecha: range.fin, hora: 23 }),
  };
}

export function clampMoment(moment: Moment, range: HistoricRange): Moment {
  const { first, last } = rangeBounds(range);
  return fromUtcMs(Math.min(Math.max(toUtcMs(moment), first), last));
}

export function shiftMoment(moment: Moment, hours: number, range: HistoricRange): Moment {
  return clampMoment(fromUtcMs(toUtcMs(moment) + hours * HOUR_MS), range);
}

/**
 * Momento inicial a partir de los parámetros de la URL. Cada parámetro inválido o fuera de rango se
 * reemplaza por su valor por defecto (último día disponible, 15:00) sin lanzar errores.
 */
export function parseMomentParams(fecha: string | null, hora: string | null, range: HistoricRange): Moment {
  const validDate = fecha !== null && isIsoDate(fecha) && fecha >= range.inicio && fecha <= range.fin;
  const validHour = hora !== null && /^\d{1,2}$/.test(hora) && Number(hora) <= 23;
  return clampMoment({ fecha: validDate ? fecha : range.fin, hora: validHour ? Number(hora) : DEFAULT_HOUR }, range);
}

/** "2026-05-15" + n días (n puede ser negativo). */
export function addDays(fecha: string, days: number): string {
  return fromUtcMs(toUtcMs({ fecha, hora: 0 }) + days * DAY_MS).fecha;
}

/** Fecha base del Tab 2 a partir de ?base=: válida y dentro del rango, o null (= la última con datos). */
export function parseBaseParam(value: string | null, range: HistoricRange): string | null {
  return value !== null && isIsoDate(value) && value >= range.inicio && value <= range.fin ? value : null;
}

// ---------------------------------------------------------------------------
// Formato (español de México, siempre en UTC)
// ---------------------------------------------------------------------------
const longDate = new Intl.DateTimeFormat("es-MX", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
const pointFormat = new Intl.DateTimeFormat("es-MX", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" });
const dayTickFormat = new Intl.DateTimeFormat("es-MX", { weekday: "short", day: "numeric", timeZone: "UTC" });

export function formatHour(hour: number): string {
  return `${String(hour).padStart(2, "0")}:00`;
}

/** "viernes, 15 de mayo de 2026" */
export function formatDay(fecha: string): string {
  return longDate.format(toUtcMs({ fecha, hora: 0 }));
}

/** "viernes, 15 de mayo de 2026, 15:00" */
export function formatMoment(moment: Moment): string {
  return `${longDate.format(toUtcMs({ ...moment, hora: 0 }))}, ${formatHour(moment.hora)}`;
}

/** "vie, 15 may, 14:00" */
export function formatPoint(ms: number): string {
  return pointFormat.format(ms);
}

/** Marcas de medianoche entre dos instantes, con etiqueta "vie 15". */
export function dayTicks(firstMs: number, lastMs: number): { value: number; label: string }[] {
  const ticks: { value: number; label: string }[] = [];
  for (let t = Math.ceil(firstMs / DAY_MS) * DAY_MS; t <= lastMs; t += DAY_MS) {
    ticks.push({ value: t, label: dayTickFormat.format(t) });
  }
  return ticks;
}
