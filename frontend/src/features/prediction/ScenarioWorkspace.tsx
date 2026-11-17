import { useCallback, useMemo, useRef, useState } from "react";
import { postPrediction } from "../../api/prediction";
import type { ApiError } from "../../api/client";
import {
  CLIMATE_KEYS,
  OZONE_KEYS,
  type BaseScenarioResponse,
  type ClimateKey,
  type HistoricRange,
  type PredictionPoint,
  type PredictionRequest,
} from "../../api/types";
import PageShell from "../../components/layout/PageShell";
import { ErrorView } from "../../components/layout/StatusViews";
import MapLegend from "../../components/map/MapLegend";
import StationMap from "../../components/map/StationMap";
import { useAppData } from "../../context/app-data-context";
import { useApi } from "../../hooks/useApi";
import { useDebounced } from "../../hooks/useDebounced";
import { scrollBehavior, useReducedMotion } from "../../hooks/useReducedMotion";
import { formatPercent } from "../../lib/format";
import {
  CLIMATE_LIMITS,
  LEVEL_ABOVE_NORM,
  LEVEL_BELOW_NORM,
  MODEL_VARS_OF_CONTROL,
  PROBABILITY_DOMAIN,
} from "../../config/modelConstants";
import { CLIMATE_ORDER, SAME_VALUE_EPSILON } from "./climateSpecs";
import { PROBABILITY_SCALE, getLabel, getMarkerStyle, legendShapes, renderTooltip } from "./predictionMarkers";
import ForecastNotice from "./ForecastNotice";
import { forecastMissingEverywhere } from "./forecast";
import ScenarioSidebar from "./ScenarioSidebar";
import StationPredictionPanel from "./StationPredictionPanel";

/** Espera tras el último cambio de un control antes de pedir una nueva predicción. */
const PREDICTION_DELAY_MS = 300;
/** Color de las formas de la leyenda: un valor bajo de la escala, solo para mostrar la forma. */
const LEGEND_GLYPH_VALUE = 0.3;

type Edits = Partial<Record<ClimateKey, number>>;
/** Las ediciones pertenecen a una fecha base: al cambiar de día dejan de valer y los controles vuelven al día nuevo. */
interface EditState {
  key: string;
  values: Edits;
}
const NO_EDITS: Edits = {};

function emptyInputs(): PredictionPoint["entradas"] {
  const entries = [...OZONE_KEYS, ...CLIMATE_KEYS].map((key) => [key, null] as const);
  return Object.fromEntries(entries) as PredictionPoint["entradas"];
}

interface ScenarioWorkspaceProps {
  scenario: BaseScenarioResponse;
  range: HistoricRange;
  /** Fecha base pedida (puede ir por delante de `scenario` mientras carga). */
  baseDate: string;
  baseLoading: boolean;
  /** Error al cargar la fecha base pedida, si lo hubo (el escenario anterior se conserva). */
  baseError: ApiError | null;
  onBaseDateChange: (date: string) => void;
}

/** Tab 2 una vez cargado el escenario del día base: controles, predicción, mapa y panel de la estación. */
export default function ScenarioWorkspace({ scenario, range, baseDate, baseLoading, baseError, onBaseDateChange }: ScenarioWorkspaceProps) {
  const { stations, info } = useAppData();
  const baseKey = scenario.fecha_base;
  const panelRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // --- Valores de los controles: día real + ediciones del usuario ----------------------------------
  const baseline = useMemo(() => {
    const result = {} as Record<ClimateKey, number>;
    for (const key of CLIMATE_ORDER) {
      const [lo, hi] = CLIMATE_LIMITS[key];
      result[key] = scenario.clima_ciudad[key] ?? (lo + hi) / 2;
    }
    return result;
  }, [scenario]);

  const [edits, setEdits] = useState<EditState>({ key: baseKey, values: NO_EDITS });
  const activeEdits = edits.key === baseKey ? edits.values : NO_EDITS;
  const values = useMemo(() => ({ ...baseline, ...activeEdits }), [baseline, activeEdits]);
  const changedKeys = useMemo(() => new Set(Object.keys(activeEdits) as ClimateKey[]), [activeEdits]);

  const handleChange = useCallback(
    (key: ClimateKey, value: number) => {
      setEdits((prev) => {
        const next = { ...(prev.key === baseKey ? prev.values : NO_EDITS) };
        if (Math.abs(value - baseline[key]) < SAME_VALUE_EPSILON) delete next[key];
        else next[key] = value;
        return { key: baseKey, values: next };
      });
    },
    [baseKey, baseline],
  );
  const handleResetKey = useCallback(
    (key: ClimateKey) => {
      setEdits((prev) => {
        const next = { ...(prev.key === baseKey ? prev.values : NO_EDITS) };
        delete next[key];
        return { key: baseKey, values: next };
      });
    },
    [baseKey],
  );
  const handleResetAll = useCallback(() => setEdits({ key: baseKey, values: NO_EDITS }), [baseKey]);

  // --- Predicción: debounce, solo los campos cambiados, cancelación de las obsoletas ---------------
  const debouncedEdits = useDebounced(edits, PREDICTION_DELAY_MS);
  const sentEdits = debouncedEdits.key === baseKey ? debouncedEdits.values : NO_EDITS;
  const pending = edits !== debouncedEdits;

  const bodyJson = useMemo(() => {
    const body: PredictionRequest = {
      fecha_base: baseKey,
      niveles: [LEVEL_BELOW_NORM, info.norma_ppb, LEVEL_ABOVE_NORM],
      ...(Object.keys(sentEdits).length > 0 ? { clima: sentEdits } : {}),
    };
    return JSON.stringify(body);
  }, [baseKey, info.norma_ppb, sentEdits]);
  const fetcher = useCallback((signal: AbortSignal) => postPrediction(JSON.parse(bodyJson) as PredictionRequest, signal), [bodyJson]);
  const prediction = useApi(fetcher, { keepPreviousData: true });

  const placeholders = useMemo<PredictionPoint[]>(
    () =>
      stations.map((s) => ({
        ...s,
        value: null,
        zeta: null,
        p90: null,
        probabilidades: {},
        sigma: null,
        alerta: false,
        entradas: emptyInputs(),
        recortadas: [],
        completadas: [],
      })),
    [stations],
  );
  const points = prediction.data?.puntos ?? placeholders;
  const selected = points.find((p) => p.id === selectedId);
  const alertCount = points.filter((p) => p.p90 !== null && p.alerta).length;
  const recalculating = prediction.loading || pending;

  // --- Insignias por control y errores 422 junto al control que los causó --------------------------
  const { clippedBy, completedBy } = useMemo(() => {
    const clipped = {} as Record<ClimateKey, string[]>;
    const completed = {} as Record<ClimateKey, string[]>;
    for (const key of CLIMATE_ORDER) {
      clipped[key] = points.filter((p) => MODEL_VARS_OF_CONTROL[key].some((v) => p.recortadas.includes(v))).map((p) => p.name);
      completed[key] = points.filter((p) => p.completadas.includes(key)).map((p) => p.name);
    }
    return { clippedBy: clipped, completedBy: completed };
  }, [points]);

  const { issuesByKey, generalError } = useMemo(() => {
    const error = prediction.error;
    const byKey: Partial<Record<ClimateKey, string[]>> = {};
    if (!error) return { issuesByKey: byKey, generalError: null };
    let unmatched = error.status !== 422 || error.issues.length === 0;
    for (const issue of error.issues) {
      const field = String(issue.loc[issue.loc.length - 1]);
      if ((CLIMATE_ORDER as string[]).includes(field)) (byKey[field as ClimateKey] ??= []).push(issue.msg);
      else unmatched = true;
    }
    return { issuesByKey: byKey, generalError: unmatched ? error : null };
  }, [prediction.error]);

  // --- Callbacks y datos estables para el mapa y la leyenda -----------------------------------------
  const handleSelect = useCallback((id: number) => setSelectedId(id), []);
  const handleBackgroundClick = useCallback(() => setSelectedId(null), []);
  const handleClose = useCallback(() => setSelectedId(null), []);
  const goToPanel = useCallback(() => {
    panelRef.current?.scrollIntoView({ behavior: scrollBehavior(reducedMotion), block: "start" });
    panelRef.current?.focus({ preventScroll: true });
  }, [reducedMotion]);

  const noForecast = useMemo(() => forecastMissingEverywhere(scenario, stations.map((station) => station.name)), [scenario, stations]);

  const ticks = useMemo(() => [{ value: info.corte_alerta, label: "corte de alerta" }], [info.corte_alerta]);
  const shapes = useMemo(() => legendShapes(info), [info]);

  return (
    <PageShell
      title="Mapa de predicciones"
      description="Probabilidad de que el máximo de ozono de mañana supere la norma, según el clima que elijas."
      sidebar={
        <ScenarioSidebar
          scenario={scenario}
          range={range}
          baseDate={baseDate}
          baseLoading={baseLoading}
          onBaseDateChange={onBaseDateChange}
          values={values}
          baseline={baseline}
          changedKeys={changedKeys}
          clippedBy={clippedBy}
          completedBy={completedBy}
          issuesByKey={issuesByKey}
          totalStations={stations.length}
          onChangeValue={handleChange}
          onResetKey={handleResetKey}
          onResetAll={handleResetAll}
        />
      }
    >
      {noForecast && <ForecastNotice baseDate={baseKey} range={range} onGoToDate={onBaseDateChange} />}

      {baseError && (
        <p className="notice notice--warn" role="alert">
          No se pudo cargar esa fecha base. {baseError.message} Se conserva el escenario anterior.
        </p>
      )}
      {generalError && <ErrorView error={generalError} onRetry={prediction.reload} title="No se pudo calcular la predicción" />}

      <p className="map-caption" aria-live="polite">
        {prediction.data ? (
          <strong>
            {alertCount} de {points.length} estaciones en alerta
          </strong>
        ) : (
          "Calculando la predicción…"
        )}
        {recalculating && prediction.data && <span className="recalculating"> · recalculando…</span>}
        {selected && (
          <>
            {" · "}
            <button type="button" className="link-button" onClick={goToPanel}>
              {selected.name} seleccionada: ver detalle
            </button>
          </>
        )}
      </p>

      <StationMap
        points={points}
        colorDomain={PROBABILITY_DOMAIN}
        colorScale={PROBABILITY_SCALE}
        getMarkerStyle={getMarkerStyle}
        getLabel={getLabel}
        renderTooltip={renderTooltip}
        selectedId={selectedId}
        onSelect={handleSelect}
        onBackgroundClick={handleBackgroundClick}
        ariaLabel="Mapa de la probabilidad de superar la norma por estación"
      />

      <MapLegend
        title="P(máximo de mañana > norma)"
        colorDomain={PROBABILITY_DOMAIN}
        colorScale={PROBABILITY_SCALE}
        formatValue={formatPercent}
        ticks={ticks}
        shapes={shapes}
        glyphValue={LEGEND_GLYPH_VALUE}
      >
        <p>El color es la probabilidad (escala fija de 0 a 100 %); la forma indica si la probabilidad alcanza el corte de alerta.</p>
      </MapLegend>

      {selected && prediction.data ? (
        <StationPredictionPanel
          ref={panelRef}
          point={selected}
          prediction={prediction.data}
          controles={scenario.controles}
          recalculating={recalculating}
          onClose={handleClose}
        />
      ) : (
        <p className="empty-hint">Elige una estación en el mapa (con clic, o con Tab y Enter) para ver sus dos etapas del modelo y las curvas del GAM.</p>
      )}
    </PageShell>
  );
}
