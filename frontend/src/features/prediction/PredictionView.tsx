import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { getBaseScenario } from "../../api/prediction";
import type { HistoricRange } from "../../api/types";
import PageShell from "../../components/layout/PageShell";
import { ErrorView, LoadingView } from "../../components/layout/StatusViews";
import { useApi } from "../../hooks/useApi";
import { parseBaseParam } from "../../lib/dates";
import ScenarioWorkspace from "./ScenarioWorkspace";

const TITLE = "Mapa de predicciones";

/**
 * Fecha base del Tab 2 (?base=YYYY-MM-DD) y carga del escenario de ese día.
 * Sin parámetro válido se usa la última fecha con datos (la que devuelve la API sin parámetros).
 */
export default function PredictionView({ range }: { range: HistoricRange }) {
  const [params, setParams] = useSearchParams();
  // null = "la última con datos", que decide el backend.
  const [baseDate, setBaseDate] = useState<string | null>(() => parseBaseParam(params.get("base"), range));

  const fetcher = useCallback((signal: AbortSignal) => getBaseScenario(baseDate ?? undefined, signal), [baseDate]);
  const scenario = useApi(fetcher, { keepPreviousData: true });

  // La URL siempre refleja la fecha base realmente cargada (los valores de los controles no van en la URL).
  const setParamsRef = useRef(setParams);
  useEffect(() => {
    setParamsRef.current = setParams;
  });
  const loadedBase = scenario.data?.fecha_base;
  const urlBase = params.get("base");
  useEffect(() => {
    if (loadedBase && urlBase !== loadedBase) setParamsRef.current({ base: loadedBase }, { replace: true });
  }, [loadedBase, urlBase]);

  if (!scenario.data) {
    if (scenario.error) {
      return (
        <PageShell title={TITLE}>
          <ErrorView error={scenario.error} onRetry={scenario.reload} title="No se pudo cargar el escenario" />
          {baseDate !== null && (
            <p className="status-view">
              <button type="button" className="button" onClick={() => setBaseDate(null)}>
                Usar el último día disponible
              </button>
            </p>
          )}
        </PageShell>
      );
    }
    return (
      <PageShell title={TITLE}>
        <LoadingView label="Cargando el escenario del día real…" />
      </PageShell>
    );
  }

  return (
    <ScenarioWorkspace
      scenario={scenario.data}
      range={range}
      baseDate={baseDate ?? scenario.data.fecha_base}
      baseLoading={scenario.loading}
      baseError={scenario.error}
      onBaseDateChange={setBaseDate}
    />
  );
}
