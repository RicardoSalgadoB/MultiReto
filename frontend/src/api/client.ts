import { API_URL } from "../config/env";
import type { ValidationIssue } from "./types";

export type ApiErrorKind =
  /** No hubo respuesta (servidor apagado, sin red, bloqueo de CORS). */
  | "network"
  /** 5xx sin cuerpo JSON: el proxy o el rewrite no alcanzaron el backend. */
  | "unreachable"
  /** Respuesta 2xx que no es JSON (p. ej. el SPA devolviendo index.html porque /api no está configurado). */
  | "bad-format"
  /** 4xx/5xx con `detail` de FastAPI. */
  | "http";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  /** Errores por campo (solo 422). */
  readonly issues: ValidationIssue[];
  /** Petición que falló, para mostrar en detalles técnicos. */
  readonly request: string;

  constructor(
    kind: ApiErrorKind,
    message: string,
    request: string,
    status: number | null = null,
    issues: ValidationIssue[] = [],
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.issues = issues;
    this.request = request;
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

// ---------------------------------------------------------------------------
// Servidor lento: aviso a los 4 s y tope de 60 s (un servidor gratuito puede tardar hasta un minuto en despertar)
// ---------------------------------------------------------------------------
const SLOW_AFTER_MS = 4000;
const REQUEST_TIMEOUT_MS = 60_000;
let slowCount = 0;
const slowListeners = new Set<() => void>();

/** Para useSyncExternalStore: ¿hay alguna petición pendiente desde hace más de 4 s? */
export function subscribeSlow(listener: () => void): () => void {
  slowListeners.add(listener);
  return () => slowListeners.delete(listener);
}
export function getSlowSnapshot(): boolean {
  return slowCount > 0;
}

/** Marca una petición como "lenta" si pasan 4 s; devuelve la función que la da por terminada (también al abortar). */
function trackRequest(): () => void {
  let slow = false;
  const timer = setTimeout(() => {
    slow = true;
    slowCount += 1;
    slowListeners.forEach((listener) => listener());
  }, SLOW_AFTER_MS);
  return () => {
    clearTimeout(timer);
    if (slow) {
      slowCount -= 1;
      slowListeners.forEach((listener) => listener());
    }
  };
}

/**
 * Mensajes cuando la API no responde como API. En desarrollo se explica cómo encender el backend; en producción el texto
 * está pensado para quien administra el sitio y no expone detalles internos.
 */
function unreachableMessage(status: number): string {
  if (import.meta.env.DEV) {
    return `El servicio de predicciones no respondió (HTTP ${status}). Revisa que el backend esté encendido en el puerto 8000.`;
  }
  return (
    "El servicio de datos no está disponible o todavía no está conectado a este sitio. Quien administra el sitio debe indicar la " +
    "dirección (HTTPS) del backend en la configuración de despliegue (ver «Checklist de Vercel» en el README). " +
    "Si solo estás consultando, vuelve a intentarlo en unos minutos."
  );
}

function badFormatMessage(): string {
  if (import.meta.env.DEV) {
    return "La respuesta no tiene el formato esperado (se esperaba JSON). Revisa que /api esté redirigido al backend (proxy de Vite) o que VITE_API_URL sea correcta.";
  }
  return (
    "El sitio no está conectado al servicio de datos: la dirección de datos devolvió una página web en lugar de información. " +
    "Quien administra el sitio debe revisar la configuración del backend (ver «Checklist de Vercel» en el README)."
  );
}

/** Convierte cualquier cosa lanzada en un ApiError (los abortos se manejan antes, con isAbortError). */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  const message = error instanceof Error ? error.message : "Error desconocido.";
  return new ApiError("network", message, "");
}

type Query = Record<string, string | number | boolean | undefined | null>;

interface RequestOptions {
  params?: Query;
  body?: unknown;
  signal?: AbortSignal;
}

function buildUrl(path: string, params?: Query): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null) search.set(key, String(value));
  }
  const query = search.toString();
  return `${API_URL}${path}${query ? `?${query}` : ""}`;
}

/** Extrae `detail` de una respuesta de error de FastAPI (texto o lista de errores de validación). */
function parseDetail(payload: unknown): { message: string | null; issues: ValidationIssue[] } {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) {
    return { message: null, issues: [] };
  }
  const detail = (payload as { detail: unknown }).detail;
  if (typeof detail === "string") return { message: detail, issues: [] };
  if (Array.isArray(detail)) {
    const issues = detail.filter(
      (item): item is ValidationIssue =>
        typeof item === "object" &&
        item !== null &&
        Array.isArray((item as ValidationIssue).loc) &&
        typeof (item as ValidationIssue).msg === "string",
    );
    return { message: null, issues };
  }
  return { message: null, issues: [] };
}

async function request<T>(method: "GET" | "POST", path: string, options: RequestOptions): Promise<T> {
  const url = buildUrl(path, options.params);
  const label = `${method} ${url}`;
  const hasBody = options.body !== undefined;

  let response: Response;
  let text: string;
  const finishTracking = trackRequest();
  // AbortSignal.timeout aborta a los 60 s (no antes); AbortSignal.any conserva la cancelación del llamador.
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
  try {
    response = await fetch(url, {
      method,
      signal,
      headers: hasBody
        ? { Accept: "application/json", "Content-Type": "application/json" }
        : { Accept: "application/json" },
      body: hasBody ? JSON.stringify(options.body) : undefined,
    });
    text = await response.text();
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new ApiError("network", "El servidor tardó más de un minuto en responder. Vuelve a intentarlo en unos minutos.", label);
    }
    if (isAbortError(error)) throw error;
    throw new ApiError(
      "network",
      "No se pudo conectar con el servicio de predicciones. Verifica que el backend esté en marcha y vuelve a intentar.",
      label,
    );
  } finally {
    finishTracking();
  }

  const isJson = (response.headers.get("content-type") ?? "").includes("json");
  let payload: unknown = null;
  if (isJson && text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ApiError("bad-format", "La respuesta del servidor no es un JSON válido.", label, response.status);
    }
  }

  if (!response.ok) {
    const { message, issues } = parseDetail(payload);
    if (message !== null || issues.length > 0) {
      const fallback =
        response.status === 422 ? "Hay valores no válidos en la solicitud." : "La solicitud no se pudo completar.";
      throw new ApiError("http", message ?? fallback, label, response.status, issues);
    }
    // Una respuesta de error sin `detail` de FastAPI no viene del backend: es el proxy o el rewrite que no lo alcanzó.
    throw new ApiError("unreachable", unreachableMessage(response.status), label, response.status);
  }

  if (!isJson) {
    throw new ApiError(
      "bad-format",
      badFormatMessage(),
      label,
      response.status,
    );
  }
  // El contrato de cada endpoint se asume (no hay validación en ejecución); ver api/types.ts.
  return payload as T;
}

export function apiGet<T>(path: string, params?: Query, signal?: AbortSignal): Promise<T> {
  return request<T>("GET", path, { params, signal });
}

export function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>("POST", path, { body, signal });
}
