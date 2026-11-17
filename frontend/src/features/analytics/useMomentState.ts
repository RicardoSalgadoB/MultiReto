import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import type { HistoricRange } from "../../api/types";
import { useDebounced } from "../../hooks/useDebounced";
import { parseMomentParams, type Moment } from "../../lib/dates";

/** Espera tras el último cambio de fecha u hora antes de pedir datos y actualizar la URL. */
const COMMIT_DELAY_MS = 250;

/**
 * Momento (fecha y hora) del Tab 1.
 * - `draft`: lo que muestran los controles, cambia al instante.
 * - `committed`: `draft` con retraso; es lo que pide datos y lo que se escribe en la URL (?fecha=&hora=).
 *
 * El valor inicial sale de la URL si es válido y está en rango; si no, el último día disponible a las 15:00.
 */
export function useMomentState(range: HistoricRange) {
  const [params, setParams] = useSearchParams();
  const [draft, setDraft] = useState<Moment>(() => parseMomentParams(params.get("fecha"), params.get("hora"), range));
  const committed = useDebounced(draft, COMMIT_DELAY_MS);

  // setSearchParams cambia de identidad en cada render: se guarda la última para no re-lanzar el efecto.
  const setParamsRef = useRef(setParams);
  useEffect(() => {
    setParamsRef.current = setParams;
  });

  // La URL siempre refleja el estado. Se reescribe también si algo la cambia por fuera (p. ej. pulsar "Mapa de análisis"
  // en la barra de navegación la deja sin parámetros). replace: arrastrar el slider no llena el historial.
  const urlFecha = params.get("fecha");
  const urlHora = params.get("hora");
  useEffect(() => {
    if (urlFecha !== committed.fecha || urlHora !== String(committed.hora)) {
      setParamsRef.current({ fecha: committed.fecha, hora: String(committed.hora) }, { replace: true });
    }
  }, [committed.fecha, committed.hora, urlFecha, urlHora]);

  return { draft, setDraft, committed };
}
