import { getHistoricRangeCached, peekHistoricRange } from "../api/rangeCache";
import { usePageTitle } from "../hooks/usePageTitle";
import PageShell from "../components/layout/PageShell";
import { ErrorView, LoadingView } from "../components/layout/StatusViews";
import AnalyticsView from "../features/analytics/AnalyticsView";
import { useApi } from "../hooks/useApi";

const TITLE = "Mapa de análisis";

/** Tab 1. El rango de fechas disponible viene de la API; la vista solo se monta cuando ya se conoce. */
export default function AnalyticsMap() {
  usePageTitle("Mapa de análisis");
  // Si el rango ya se pidió en otra pestaña, se usa al instante y no hay petición.
  const cached = peekHistoricRange();
  const api = useApi(getHistoricRangeCached, { enabled: cached === null });
  const range = { data: cached ?? api.data, error: cached === null ? api.error : null, reload: api.reload };

  if (range.error) {
    return (
      <PageShell title={TITLE}>
        <ErrorView error={range.error} onRetry={range.reload} title="No se pudo cargar el rango de fechas" />
      </PageShell>
    );
  }
  if (!range.data) {
    return (
      <PageShell title={TITLE}>
        <LoadingView label="Cargando el rango de fechas…" />
      </PageShell>
    );
  }
  return <AnalyticsView range={range.data} />;
}
