import { memo, useEffect, useMemo, useRef } from "react";
import type { LatLngTuple, PathOptions, Polyline as LeafletPolyline } from "leaflet";
import { Pane, Polyline, Tooltip } from "react-leaflet";
import type { StationCode } from "../../api/types";
import type { ColorScale } from "../../lib/colors";
import { formatNumber } from "../../lib/format";
import type { DependenceModel } from "./dependenceModel";
import { CASING_COLOR, CASING_EXTRA_WEIGHT, CASING_OPACITY, LOW_DAYS_DASH, lineOpacity, lineWeight, normalized, type Emphasis } from "./dependenceStyle";

/** Los marcadores están en markerPane (z-index 600): las líneas quedan siempre por debajo y nunca interceptan su clic. */
const PANE_NAME = "dependence-lines";
const PANE_Z_INDEX = 450;

/**
 * Línea invisible y ancha: facilita pasar el cursor o tocar (≈ 14 px). `bubblingMouseEvents: false` evita que el
 * clic en una línea llegue al mapa y lo trate como un clic en el fondo (que deselecciona).
 */
const HIT_OPTIONS: PathOptions = { color: "#000", weight: 14, opacity: 0, lineCap: "round", bubblingMouseEvents: false };

interface PairLineProps {
  pairKey: string;
  positions: LatLngTuple[];
  color: string;
  weight: number;
  opacity: number;
  dashed: boolean;
  emphasis: Emphasis;
  /** Si la línea responde al cursor (las de fondo no responden cuando hay una estación seleccionada). */
  interactive: boolean;
  title: string;
  details: string[];
  onPairClick: (key: string) => void;
}

const PairLine = memo(function PairLine({ pairKey, positions, color, weight, opacity, dashed, emphasis, interactive, title, details, onPairClick }: PairLineProps) {
  const visibleRef = useRef<LeafletPolyline>(null);
  const visible = useMemo<PathOptions>(
    () => ({ color, weight, opacity, dashArray: dashed ? LOW_DAYS_DASH : undefined, lineCap: "round", lineJoin: "round" }),
    [color, weight, opacity, dashed],
  );
  const casing = useMemo<PathOptions>(
    () => ({ color: CASING_COLOR, weight: weight + CASING_EXTRA_WEIGHT, opacity: CASING_OPACITY, lineCap: "round", lineJoin: "round" }),
    [weight],
  );
  const handlers = useMemo(() => ({ click: () => onPairClick(pairKey) }), [onPairClick, pairKey]);

  // Las líneas resaltadas se dibujan encima de las tenues.
  useEffect(() => {
    if (emphasis === "related") visibleRef.current?.bringToFront();
  }, [emphasis, weight]);

  return (
    <>
      {emphasis === "related" && <Polyline positions={positions} pathOptions={casing} interactive={false} />}
      {/* `interactive` va como propiedad (no en pathOptions): Leaflet la lee al crear la capa, y pathOptions se aplica después. */}
      <Polyline ref={visibleRef} positions={positions} pathOptions={visible} interactive={false} />
      {interactive && (
        <Polyline positions={positions} pathOptions={HIT_OPTIONS} bubblingMouseEvents={false} eventHandlers={handlers}>
          <Tooltip sticky direction="top">
            <strong>{title}</strong>
            {details.map((line) => (
              <span key={line} className="pair-tooltip-line">
                <br />
                {line}
              </span>
            ))}
          </Tooltip>
        </Polyline>
      )}
    </>
  );
});

interface DependenceLinesProps {
  model: DependenceModel;
  domain: [number, number];
  scale: ColorScale;
  selected: StationCode | null;
  activeKey: string | null;
  /** Etiqueta de la medida para el tooltip, p. ej. "χ(0.95)". */
  measureLabel: string;
  formatValue: (value: number) => string;
  /** Debe ser estable (useCallback). */
  onPairClick: (key: string) => void;
}

/**
 * Las líneas entre estaciones. Sin selección: todas muy tenues. Con una estación: sus líneas resaltadas (color y
 * grosor por valor) y las demás aún más tenues. Los pares sin valor no se dibujan. Los del cuartil con menos días conjuntos van
 * con trazo discontinuo.
 */
export default function DependenceLines({ model, domain, scale, selected, activeKey, measureLabel, formatValue, onPairClick }: DependenceLinesProps) {
  const lines = useMemo(
    () =>
      model.pairs.flatMap((pair) => {
        if (pair.valor === null) return [];
        const emphasis: Emphasis = selected === null ? "overview" : pair.a === selected || pair.b === selected ? "related" : "unrelated";
        const active = pair.key === activeKey;
        const t = normalized(pair.valor, domain);
        const details = [
          `${measureLabel}: ${formatValue(pair.valor)}`,
          `Distancia: ${formatNumber(pair.distanciaKm)} km`,
          `Días conjuntos: ${pair.diasConjuntos}`,
        ];
        if (pair.low) details.push(model.lowLabel);
        return [
          {
            pair,
            emphasis,
            color: scale(pair.valor),
            weight: lineWeight(t, emphasis) + (active ? 3 : 0),
            opacity: active ? 1 : lineOpacity(emphasis),
            interactive: emphasis !== "unrelated",
            title: `${pair.a} – ${pair.b}`,
            details,
          },
        ];
      }),
    [model, domain, scale, selected, activeKey, measureLabel, formatValue],
  );

  return (
    <Pane name={PANE_NAME} style={{ zIndex: PANE_Z_INDEX }}>
      {lines.map((line) => (
        <PairLine
          key={line.pair.key}
          pairKey={line.pair.key}
          positions={line.pair.positions}
          color={line.color}
          weight={line.weight}
          opacity={line.opacity}
          dashed={line.pair.low}
          emphasis={line.emphasis}
          interactive={line.interactive}
          title={line.title}
          details={line.details}
          onPairClick={onPairClick}
        />
      ))}
    </Pane>
  );
}
