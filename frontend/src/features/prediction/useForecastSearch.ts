import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, isAbortError, toApiError } from "../../api/client";
import { getBaseScenario } from "../../api/prediction";
import { FORECAST_SEARCH_MAX_DAYS } from "../../config/modelConstants";
import { addDays } from "../../lib/dates";
import { forecastCompletedAnywhere } from "./forecast";

export type ForecastSearchState =
  | { status: "idle" }
  | { status: "searching"; tried: number }
  | { status: "not-found"; from: string; tried: number }
  | { status: "error"; error: ApiError };

/**
 * Busca, solo cuando se pide, el último día anterior a `from` con pronóstico de temperatura: retrocede día por día
 * con /escenario/base (máximo FORECAST_SEARCH_MAX_DAYS intentos, uno tras otro) hasta un día en que T_max_pron no
 * figure en `valores_completados`. Se puede cancelar y se cancela sola al desmontar.
 */
export function useForecastSearch(onFound: (date: string) => void) {
  const [state, setState] = useState<ForecastSearchState>({ status: "idle" });
  const controllerRef = useRef<AbortController | null>(null);
  const onFoundRef = useRef(onFound);
  useEffect(() => {
    onFoundRef.current = onFound;
  });
  useEffect(() => () => controllerRef.current?.abort(), []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setState({ status: "idle" });
  }, []);

  const start = useCallback(async (from: string, earliest: string) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ status: "searching", tried: 0 });

    let date = from;
    let tried = 0;
    while (tried < FORECAST_SEARCH_MAX_DAYS) {
      date = addDays(date, -1);
      if (date < earliest) break;
      tried += 1;
      setState({ status: "searching", tried });
      try {
        const scenario = await getBaseScenario(date, controller.signal);
        if (!forecastCompletedAnywhere(scenario)) {
          setState({ status: "idle" });
          onFoundRef.current(date);
          return;
        }
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) return;
        // Un día sin datos (404) cuenta como intento y se sigue con el anterior; cualquier otro error detiene la búsqueda.
        if (error instanceof ApiError && error.status === 404) continue;
        setState({ status: "error", error: toApiError(error) });
        return;
      }
    }
    if (!controller.signal.aborted) setState({ status: "not-found", from, tried });
  }, []);

  return { state, start, cancel };
}
