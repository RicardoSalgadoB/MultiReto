import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getHistoricHour, getHistoricSeries } from "../../api/historic";
import type { HistoricPoint, HistoricRange } from "../../api/types";
import PageShell from "../../components/layout/PageShell";
import { ErrorView } from "../../components/layout/StatusViews";
import MapLegend from "../../components/map/MapLegend";
import StationMap from "../../components/map/StationMap";
import { useAppData } from "../../context/app-data-context";
import { useApi } from "../../hooks/useApi";
import { scrollBehavior, useReducedMotion } from "../../hooks/useReducedMotion";
import { formatMoment, rangeBounds, shiftMoment, toUtcMs, type Moment } from "../../lib/dates";
import { formatNumber } from "../../lib/format";
import PeriodNotice from "./PeriodNotice";
import StationSeriesPanel, { type SeriesResult } from "./StationSeriesPanel";
import TimeControls from "./TimeControls";
import { OZONE_DOMAIN } from "../../config/modelConstants";
import { OZONE_SCALE, getLabel, getMarkerStyle, legendShapes, renderTooltip } from "./ozoneMarkers";
import { useMomentState } from "./useMomentState";

const SERIES_DAYS = 7;
const PLAY_INTERVAL_MS = 1200;
/** Con "reducir movimiento", la reproducción avanza más despacio. */
const PLAY_INTERVAL_REDUCED_MS = 2500;

export default function AnalyticsView({ range }: { range: HistoricRange }) {
  const { stations, info } = useAppData();
  const { draft, setDraft, committed } = useMomentState(range);
  const [playing, setPlaying] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const panelRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();

  // --- Reproducción: avanza una hora cada PLAY_INTERVAL_MS hasta el final del rango -------------
  const atEnd = toUtcMs(draft) >= rangeBounds(range).last;
  const isPlaying = playing && !atEnd;
  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => setDraft((prev) => shiftMoment(prev, 1, range)), reducedMotion ? PLAY_INTERVAL_REDUCED_MS : PLAY_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [isPlaying, range, setDraft, reducedMotion]);

  const handleChange = useCallback(
    (moment: Moment) => {
      setPlaying(false);
      setDraft(moment);
    },
    [setDraft],
  );
  const togglePlay = useCallback(() => setPlaying((value) => !value), []);

  // --- Datos de la hora: se cancela la petición anterior si cambia el momento ---------------------
  const hourFetcher = useCallback(
    (signal: AbortSignal) => getHistoricHour(committed.fecha, committed.hora, signal),
    [committed.fecha, committed.hora],
  );
  const hourData = useApi(hourFetcher, { keepPreviousData: true });

  // Mientras llega el primer dato, las estaciones salen de la API sin valores ni predicción.
  const placeholders = useMemo<HistoricPoint[]>(
    () => stations.map((s) => ({ ...s, value: null, periodo: null, zeta: null, p90: null, alerta: null, max_dia: null, exceso_real: null })),
    [stations],
  );
  const points = hourData.data?.puntos ?? placeholders;
  const selected = points.find((p) => p.id === selectedId);
  const period = hourData.data?.puntos.find((p) => p.periodo !== null)?.periodo ?? null;

  // --- Serie de la estación seleccionada ---------------------------------------------------------
  const selectedName = selected?.name;
  const seriesFetcher = useCallback(
    async (signal: AbortSignal): Promise<SeriesResult> => {
      if (selectedName === undefined) throw new Error("No hay estación seleccionada.");
      const response = await getHistoricSeries(selectedName, committed.fecha, committed.hora, SERIES_DAYS, signal);
      return { moment: { fecha: committed.fecha, hora: committed.hora }, response };
    },
    [selectedName, committed.fecha, committed.hora],
  );
  const series = useApi(seriesFetcher, { enabled: selectedName !== undefined, keepPreviousData: true });

  // --- Callbacks y datos estables para el mapa y la leyenda ---------------------------------------
  const handleSelect = useCallback((id: number) => setSelectedId(id), []);
  const handleBackgroundClick = useCallback(() => setSelectedId(null), []);
  const handleClose = useCallback(() => setSelectedId(null), []);
  const goToPanel = useCallback(() => {
    panelRef.current?.scrollIntoView({ behavior: scrollBehavior(reducedMotion), block: "start" });
    panelRef.current?.focus({ preventScroll: true });
  }, [reducedMotion]);

  const ticks = useMemo(
    () => [
      { value: info.u_ppb, label: "umbral del modelo" },
      { value: info.norma_ppb, label: "norma" },
    ],
    [info],
  );
  const shapes = useMemo(() => legendShapes(info), [info]);

  const shownMoment: Moment | null = hourData.data ? { fecha: hourData.data.fecha, hora: hourData.data.hora } : null;
  const notFound = hourData.error?.status === 404;

  return (
    <PageShell
      title="Mapa de análisis"
      description="Ozono observado por estación en cualquier hora de 2023 a 2026 y alerta del modelo para ese día."
      toolbar={<TimeControls range={range} moment={draft} playing={isPlaying} onChange={handleChange} onTogglePlay={togglePlay} />}
    >
      <PeriodNotice period={period} />

      {hourData.error &&
        (notFound ? (
          <p className="notice notice--warn" role="alert">
            No hay datos para este momento. {hourData.error.message}
          </p>
        ) : (
          <ErrorView error={hourData.error} onRetry={hourData.reload} />
        ))}

      <p className="map-caption" aria-live="polite">
        {shownMoment ? `Mostrando: ${formatMoment(shownMoment)}` : "Cargando datos…"}
        {hourData.loading && shownMoment && " · actualizando…"}
        {selected && (
          <>
            {" · "}
            <button type="button" className="link-button" onClick={goToPanel}>
              {selected.name} seleccionada: ver serie de {SERIES_DAYS} días
            </button>
          </>
        )}
      </p>

      <StationMap
        points={points}
        colorDomain={OZONE_DOMAIN}
        colorScale={OZONE_SCALE}
        getMarkerStyle={getMarkerStyle}
        getLabel={getLabel}
        renderTooltip={renderTooltip}
        selectedId={selectedId}
        onSelect={handleSelect}
        onBackgroundClick={handleBackgroundClick}
        ariaLabel="Mapa del ozono por estación a la hora elegida"
      />

      <MapLegend
        title="Ozono a esa hora"
        colorDomain={OZONE_DOMAIN}
        colorScale={OZONE_SCALE}
        unit="ppb"
        ticks={ticks}
        shapes={shapes}
        noValueLabel="Sin medición de ozono a esa hora"
      >
        <p>
          La forma indica si el modelo predice exceso de la norma ({formatNumber(info.norma_ppb)} ppb) para ese día. Los valores mayores que{" "}
          {formatNumber(OZONE_DOMAIN[1])} ppb se muestran con el color de {formatNumber(OZONE_DOMAIN[1])} ppb. En una estación con un rombo de contorno gris grueso y centro
          translúcido (sin color de relleno) no hay predicción y el color no se aplica; el valor sigue en el tooltip.
        </p>
      </MapLegend>

      {selected ? (
        <StationSeriesPanel
          ref={panelRef}
          point={selected}
          info={info}
          pointMoment={shownMoment ?? committed}
          series={series}
          onClose={handleClose}
        />
      ) : (
        <p className="empty-hint">Elige una estación en el mapa (con clic, o con Tab y Enter) para ver su serie de ozono de los {SERIES_DAYS} días anteriores.</p>
      )}
    </PageShell>
  );
}
