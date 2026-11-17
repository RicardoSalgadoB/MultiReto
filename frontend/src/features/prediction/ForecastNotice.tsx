import type { HistoricRange } from "../../api/types";
import { FORECAST_SEARCH_MAX_DAYS } from "../../config/modelConstants";
import { formatDay } from "../../lib/dates";
import { useForecastSearch } from "./useForecastSearch";

interface ForecastNoticeProps {
  /** Fecha base actual (la búsqueda empieza el día anterior). */
  baseDate: string;
  range: HistoricRange;
  onGoToDate: (date: string) => void;
}

/** Aviso del día sin pronóstico de temperatura y botón (manual) para ir al último día con pronóstico. */
export default function ForecastNotice({ baseDate, range, onGoToDate }: ForecastNoticeProps) {
  const { state, start, cancel } = useForecastSearch(onGoToDate);

  return (
    <div className="notice notice--warn" role="status">
      <span>Este día no tiene pronóstico de temperatura; se usó la mediana de entrenamiento, por eso el resultado no es representativo.</span>
      {state.status === "searching" ? (
        <>
          <span aria-live="polite">
            Buscando el último día con pronóstico… ({state.tried} de {FORECAST_SEARCH_MAX_DAYS})
          </span>
          <button type="button" className="button" onClick={cancel}>
            Cancelar
          </button>
        </>
      ) : (
        <button type="button" className="button" onClick={() => void start(baseDate, range.inicio)}>
          Ir al último día con pronóstico
        </button>
      )}
      {state.status === "not-found" && (
        <span role="alert">
          No se encontró un día con pronóstico en los {state.tried} días anteriores al {formatDay(state.from)}.
        </span>
      )}
      {state.status === "error" && <span role="alert">No se pudo completar la búsqueda. {state.error.message}</span>}
    </div>
  );
}
