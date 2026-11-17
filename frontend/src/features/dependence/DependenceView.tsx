import { useCallback, useMemo, useState } from "react";
import { getDependenceCached, peekDependence } from "../../api/dependenceCache";
import type { DependenceKind } from "../../api/types";
import PageShell from "../../components/layout/PageShell";
import { ErrorView } from "../../components/layout/StatusViews";
import StationMap from "../../components/map/StationMap";
import type { StationMapPoint } from "../../components/map/types";
import { DEPENDENCE_DOMAINS } from "../../config/modelConstants";
import { useAppData } from "../../context/app-data-context";
import { useApi } from "../../hooks/useApi";
import { intensityScale, type ColorScale } from "../../lib/colors";
import { formatFixed } from "../../lib/format";
import DependenceLegend from "./DependenceLegend";
import DependenceLines from "./DependenceLines";
import { buildDependenceModel, otherStation } from "./dependenceModel";
import DependenceTable from "./DependenceTable";
import { describeKind } from "./measureInfo";
import ModeToggle from "./ModeToggle";
import { useDependenceState } from "./useDependenceState";

/** Una escala fija por medida (referencias estables). */
const SCALES: Record<DependenceKind, ColorScale> = {
  chi: intensityScale(DEPENDENCE_DOMAINS.chi),
  chibar: intensityScale(DEPENDENCE_DOMAINS.chibar),
  spearman: intensityScale(DEPENDENCE_DOMAINS.spearman),
};

interface StationPoint extends StationMapPoint {
  nombre: string;
}

const formatValue = (value: number) => formatFixed(value, 2);
/** Etiqueta permanente junto a cada marcador: el código de la estación. */
const renderStationCode = (point: StationPoint) => point.name;

/** Tab 3: dependencia entre estaciones (Spearman o dependencia de colas χ / χ̄). */
export default function DependenceView() {
  const { stations } = useAppData();
  const stationNames = useMemo(() => stations.map((s) => s.name), [stations]);
  const { mode, measure, station, kind, setMode, setMeasure, toggleStation, clearStation } = useDependenceState(stationNames);
  const [activeKey, setActiveKey] = useState<string | null>(null);

  // Una petición por medida, guardada: alternar medidas ya vistas o seleccionar una estación no pide nada.
  const cached = peekDependence(kind);
  const fetcher = useCallback((signal: AbortSignal) => getDependenceCached(kind, signal), [kind]);
  const api = useApi(fetcher, { enabled: cached === null, keepPreviousData: true });
  const data = cached ?? (api.data?.tipo === kind ? api.data : null);
  const loading = cached === null && api.loading;

  const model = useMemo(() => (data ? buildDependenceModel(data) : null), [data]);
  const domain = DEPENDENCE_DOMAINS[kind];
  const scale = SCALES[kind];
  const { label, help } = describeKind(kind, data?.q ?? null);

  // --- Marcadores (estables: no se recrean al seleccionar) ---------------------------------------------
  const points = useMemo<StationPoint[]>(
    () => stations.map((s) => ({ id: s.id, name: s.name, nombre: s.nombre, latitude: s.latitude, longitude: s.longitude, value: null })),
    [stations],
  );
  const idToName = useMemo(() => new Map(stations.map((s) => [s.id, s.name])), [stations]);
  const selectedId = useMemo(() => stations.find((s) => s.name === station)?.id ?? null, [stations, station]);

  const handleSelect = useCallback(
    (id: number) => {
      const code = idToName.get(id);
      if (code === undefined) return;
      toggleStation(code);
      setActiveKey(null);
    },
    [idToName, toggleStation],
  );
  const handleBackgroundClick = useCallback(() => {
    clearStation();
    setActiveKey(null);
  }, [clearStation]);
  const handlePairClick = useCallback((key: string) => setActiveKey((prev) => (prev === key ? null : key)), []);

  // Valor de cada estación respecto a la seleccionada, para el texto accesible de los marcadores.
  const relatedValues = useMemo(
    () => (model && station ? new Map((model.byStation.get(station) ?? []).map((pair) => [otherStation(pair, station), pair.valor])) : null),
    [model, station],
  );
  const getLabel = useCallback(
    (point: StationPoint) => {
      const base = `${point.name}, ${point.nombre}`;
      if (station === null) return base;
      if (point.name === station) return `${base}, seleccionada`;
      const value = relatedValues?.get(point.name);
      return value === undefined ? base : `${base}; ${label} con ${station}: ${value === null ? "sin valor" : formatValue(value)}`;
    },
    [station, relatedValues, label],
  );

  const activePair = model && activeKey ? model.byKey.get(activeKey) : undefined;
  const drawnCount = model ? model.pairs.length - model.nullCount : 0;

  return (
    <PageShell
      title="Dependencia entre estaciones"
      description="Qué tan relacionados están los máximos diarios de ozono de cada par de estaciones. Selecciona una estación para ver sus líneas con las demás."
      toolbar={<ModeToggle mode={mode} measure={measure} onMode={setMode} onMeasure={setMeasure} />}
    >
      <div className="notice notice--info">
        <span>{help}</span>
        {data && <span>Periodo: {data.periodo}.</span>}
      </div>

      {api.error && !data && <ErrorView error={api.error} onRetry={api.reload} />}

      <p className="map-caption" aria-live="polite">
        {loading || !model ? (
          api.error ? "No se pudieron cargar los datos." : "Cargando…"
        ) : station === null ? (
          <>
            <strong>{label}</strong> · sin estación seleccionada: {drawnCount} líneas muy tenues
          </>
        ) : (
          <>
            <strong>{label}</strong> · {station} seleccionada: sus {model.byStation.get(station)?.filter((p) => p.valor !== null).length ?? 0} líneas resaltadas
          </>
        )}
      </p>

      <StationMap
        points={points}
        getLabel={getLabel}
        renderTooltip={renderStationCode}
        tooltipPermanent
        selectedId={selectedId}
        onSelect={handleSelect}
        onBackgroundClick={handleBackgroundClick}
        ariaLabel="Mapa de las estaciones y la dependencia entre ellas"
      >
        {model && (
          <DependenceLines
            model={model}
            domain={domain}
            scale={scale}
            selected={station}
            activeKey={activeKey}
            measureLabel={label}
            formatValue={formatValue}
            onPairClick={handlePairClick}
          />
        )}
      </StationMap>

      <div className="pair-readout-region" aria-live="polite">
        {activePair ? (
          <div className="pair-readout">
            <span className="pair-id">
              {activePair.a} – {activePair.b}
            </span>
            <span className="pair-value">
              <span className="pair-measure">{label} = </span>
              {activePair.valor === null ? "sin valor" : formatValue(activePair.valor)}
            </span>
            <span className="pair-meta">{formatFixed(activePair.distanciaKm, 1)} km</span>
            <span className="pair-meta">
              {activePair.diasConjuntos} días conjuntos
              {activePair.low ? ` (${model?.lowLabel.charAt(0).toLowerCase()}${model?.lowLabel.slice(1)})` : ""}
            </span>
          </div>
        ) : (
          <p className="map-caption">Pasa el cursor sobre una línea, tócala o elige una fila de la lista para ver su valor, distancia y días conjuntos.</p>
        )}
      </div>

      <DependenceLegend measureLabel={label} domain={domain} scale={scale} model={model} />

      {model && <DependenceTable model={model} selected={station} activeKey={activeKey} measureLabel={label} onActivate={handlePairClick} />}
    </PageShell>
  );
}

