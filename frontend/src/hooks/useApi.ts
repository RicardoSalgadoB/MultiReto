import { useCallback, useEffect, useMemo, useState } from "react";
import { type ApiError, isAbortError, toApiError } from "../api/client";

export interface ApiState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  /** Vuelve a pedir los datos (sirve para el botón "Reintentar"). */
  reload: () => void;
}

interface UseApiOptions {
  /** Si es false no se pide nada (p. ej. mientras falta un parámetro). Por defecto, true. */
  enabled?: boolean;
  /** Mantiene el dato anterior visible mientras llega el nuevo (evita parpadeos). Por defecto, false. */
  keepPreviousData?: boolean;
}

interface Settled<T> {
  /** Petición a la que corresponde este resultado. */
  request: object | null;
  data: T | null;
  error: ApiError | null;
}

/**
 * Pide datos a la API con cancelación de peticiones obsoletas.
 *
 * `fetcher` debe ser estable (función de módulo o `useCallback`): cuando cambia, se cancela la
 * petición en curso y se lanza otra. Los parámetros de la petición van dentro del `useCallback`.
 *
 * `loading` y `error` se derivan en el render comparando la petición vigente con la última resuelta,
 * así el efecto solo actualiza estado cuando llega una respuesta.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  { enabled = true, keepPreviousData = false }: UseApiOptions = {},
): ApiState<T> {
  const [attempt, setAttempt] = useState(0);
  const [settled, setSettled] = useState<Settled<T>>({ request: null, data: null, error: null });
  const request = useMemo(() => ({ fetcher, attempt }), [fetcher, attempt]);

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();

    fetcher(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setSettled({ request, data, error: null });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || isAbortError(error)) return;
        setSettled((prev) => ({ request, data: prev.data, error: toApiError(error) }));
      });

    return () => controller.abort();
  }, [request, fetcher, enabled]);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);

  const isCurrent = settled.request === request;
  return {
    data: isCurrent || keepPreviousData ? settled.data : null,
    error: isCurrent ? settled.error : null,
    loading: enabled && !isCurrent,
    reload,
  };
}
