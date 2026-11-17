import { useMemo, type Ref } from "react";
import type { ApiInfo, HistoricPoint, SeriesResponse } from "../../api/types";
import LineChart, { type ChartPoint, type ReferenceLine } from "../../components/charts/LineChart";
import { EmptyView, ErrorView, LoadingView } from "../../components/layout/StatusViews";
import type { ApiState } from "../../hooks/useApi";
import { dayTicks, formatDay, formatHour, formatMoment, formatPoint, parseSampleTime, toUtcMs, type Moment } from "../../lib/dates";
import { formatNumber, formatPercent } from "../../lib/format";
import Verdict from "../../components/ui/Verdict";
import "../../components/ui/Panel.css";
import "./StationSeriesPanel.css";

/** Serie recibida junto con el momento para el que se pidió (para colocar bien la marca). */
export interface SeriesResult {
  moment: Moment;
  response: SeriesResponse;
}

interface StationSeriesPanelProps {
  point: HistoricPoint;
  info: ApiInfo;
  /** Momento al que corresponden los datos del punto. */
  pointMoment: Moment;
  series: ApiState<SeriesResult>;
  onClose: () => void;
  ref?: Ref<HTMLElement>;
}

function yesNo(value: boolean | null): string {
  return value === null ? "sin dato" : value ? "Sí" : "No";
}

export default function StationSeriesPanel({ point, info, pointMoment, series, onClose, ref }: StationSeriesPanelProps) {
  // Mientras llega la serie de una estación nueva, no se muestra la de la anterior.
  const current = series.data && series.data.response.estacion === point.name ? series.data : null;

  const chartData = useMemo<ChartPoint[]>(
    () =>
      (current?.response.serie ?? []).flatMap((sample) => {
        const x = parseSampleTime(sample.fecha_hora);
        return x === null ? [] : [{ x, y: sample.o3 }];
      }),
    [current],
  );
  const xTicks = useMemo(() => (chartData.length > 1 ? dayTicks(chartData[0].x, chartData[chartData.length - 1].x) : []), [chartData]);
  const referenceLines = useMemo<ReferenceLine[]>(
    () =>
      current
        ? [
            { value: current.response.u_ppb, label: "Umbral del modelo", shortLabel: `Umbral ${formatNumber(current.response.u_ppb)} ppb`, kind: "threshold" },
            { value: current.response.norma_ppb, label: "Norma", shortLabel: `Norma ${formatNumber(current.response.norma_ppb)} ppb`, kind: "limit" },
          ]
        : [],
    [current],
  );
  const marker = useMemo(() => (current ? { x: toUtcMs(current.moment), label: "Momento seleccionado" } : undefined), [current]);

  const hasMeasurements = chartData.some((p) => p.y !== null);

  return (
    <section className="panel series-panel" ref={ref} tabIndex={-1} aria-labelledby="series-panel-title">
      <header className="panel-head">
        <div className="panel-title">
          <h2 id="series-panel-title">
            {point.nombre} <span className="panel-code">({point.name})</span>
          </h2>
          <p className="panel-note">Cifras del {formatDay(pointMoment.fecha)}. La predicción es diaria: es la misma para las 24 horas.</p>
        </div>
        <button type="button" className="button" onClick={onClose}>
          Cerrar
        </button>
      </header>

      <dl className="stat-grid">
        <div>
          <dt>Ozono a las {formatHour(pointMoment.hora)}</dt>
          <dd data-muted={point.value === null ? "" : undefined}>{point.value === null ? "sin medición" : `${formatNumber(point.value)} ppb`}</dd>
        </div>
        <div>
          <dt>Máximo observado ese día</dt>
          <dd data-muted={point.max_dia === null ? "" : undefined}>{point.max_dia === null ? "sin dato" : `${formatNumber(point.max_dia)} ppb`}</dd>
        </div>
        <div>
          <dt>Etapa 1 (GAM): P(máximo &gt; {formatNumber(info.u_ppb)} ppb)</dt>
          <dd data-muted={point.zeta === null ? "" : undefined}>{point.zeta === null ? "sin predicción" : formatPercent(point.zeta)}</dd>
        </div>
        <div>
          <dt>Etapa 2 (GPD): P(máximo &gt; {formatNumber(info.norma_ppb)} ppb)</dt>
          <dd data-muted={point.p90 === null ? "" : undefined}>{point.p90 === null ? "sin predicción" : formatPercent(point.p90)}</dd>
        </div>
        <div>
          <dt>Alerta (corte: P ≥ {formatPercent(info.corte_alerta)})</dt>
          <dd data-muted={point.alerta === null ? "" : undefined}>{point.alerta === null ? "sin dato" : <Verdict alert={point.alerta} size="md" />}</dd>
        </div>
        <div>
          <dt>Exceso real (máximo &gt; {formatNumber(info.norma_ppb)} ppb)</dt>
          <dd data-muted={point.exceso_real === null ? "" : undefined}>{yesNo(point.exceso_real)}</dd>
        </div>
      </dl>

      <h3 className="series-title">Ozono de los 7 días anteriores</h3>
      {series.error ? (
        <ErrorView error={series.error} onRetry={series.reload} title="No se pudo cargar la serie" />
      ) : !current ? (
        <LoadingView label="Cargando la serie…" />
      ) : !hasMeasurements ? (
        <EmptyView title="Sin mediciones">Esta estación no registró ozono en los 7 días previos a ese momento.</EmptyView>
      ) : (
        <div className="series-chart" aria-busy={series.loading} data-loading={series.loading ? "" : undefined}>
          <LineChart
            data={chartData}
            seriesName="Ozono"
            unit="ppb"
            ariaLabel={`Ozono horario de ${point.name} en los 7 días previos a ${formatMoment(current.moment)}`}
            formatX={formatPoint}
            xTicks={xTicks}
            referenceLines={referenceLines}
            marker={marker}
          />
        </div>
      )}
    </section>
  );
}
