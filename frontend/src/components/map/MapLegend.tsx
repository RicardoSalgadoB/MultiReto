import { useMemo, type ReactNode } from "react";
import { NO_VALUE_COLOR, sampleScale, viridisScale, type ColorScale } from "../../lib/colors";
import { markerSvg } from "./markerSvg";
import type { LegendShape } from "./types";
import "./MapLegend.css";

export interface LegendTick {
  value: number;
  /** Texto opcional bajo el valor, p. ej. "norma". */
  label?: string;
}

interface MapLegendProps {
  /** Qué representa el color, p. ej. "Ozono a esa hora". */
  title: string;
  colorDomain: [number, number];
  /** La misma escala que se le pasó al mapa; por defecto, viridis sobre `colorDomain`. */
  colorScale?: ColorScale;
  unit?: string;
  /** Marcas sobre la barra (p. ej. la norma). Se incluyen siempre los extremos del dominio. */
  ticks?: LegendTick[];
  formatValue?: (value: number) => string;
  /** Entradas de la leyenda de formas. */
  shapes?: LegendShape[];
  /** Valor de la escala con el que se colorean las formas de la leyenda (por defecto, el punto medio). */
  glyphValue?: number;
  /** Si se da, se agrega la entrada "sin dato" con este texto. */
  noValueLabel?: string;
  /** Notas o explicaciones adicionales. */
  children?: ReactNode;
}

function ShapeGlyph({ shape, hollow, ring, fill }: Omit<LegendShape, "label"> & { fill: string }) {
  // El SVG lo genera markerSvg con valores propios (sin texto del usuario).
  return <span className="legend-glyph" aria-hidden="true" dangerouslySetInnerHTML={{ __html: markerSvg({ shape, hollow, ring, fill, size: 24 }) }} />;
}

/** Leyenda en banda compacta: escala de color con marcas a la izquierda y formas a la derecha. Va bajo el mapa, no sobre él. */
export default function MapLegend({
  title,
  colorDomain,
  colorScale,
  unit,
  ticks = [],
  formatValue = String,
  shapes = [],
  glyphValue,
  noValueLabel,
  children,
}: MapLegendProps) {
  const [lo, hi] = colorDomain;
  const scale = useMemo(() => colorScale ?? viridisScale([lo, hi]), [colorScale, lo, hi]);
  const gradient = useMemo(() => `linear-gradient(to right, ${sampleScale(scale, [lo, hi]).join(", ")})`, [scale, lo, hi]);
  const unitText = unit ? ` ${unit}` : "";

  const allTicks: LegendTick[] = [{ value: lo }, ...ticks.filter((t) => t.value > lo && t.value < hi), { value: hi }];
  const sample = (value: number) => scale(Math.min(Math.max(value, lo), hi));
  // Las marcas con texto se escalonan en dos filas para que no se encimen.
  const labeledCount = allTicks.filter((tick) => tick.label).length;
  let labeledSoFar = 0;

  return (
    <section className="map-legend" aria-label="Leyenda del mapa">
      <div className="legend-scale">
        <div className="legend-head">
          <h3>{title}</h3>
          {unit && <span className="legend-unit">{unit}</span>}
        </div>
        <div className="legend-bar" role="img" aria-label={`Escala de color de ${formatValue(lo)} a ${formatValue(hi)}${unitText}`} style={{ background: gradient }} />
        <div className="legend-ticks" data-labeled={labeledCount > 0 ? "" : undefined}>
          {allTicks.map((tick) => {
            const labelRow = tick.label ? labeledSoFar++ % 2 : 0;
            return (
              <span key={tick.value} className="legend-tick" style={{ left: `${((tick.value - lo) / (hi - lo)) * 100}%` }}>
                <span aria-hidden={tick.label ? undefined : true}>{formatValue(tick.value)}</span>
                {tick.label && (
                  <span className="legend-tick-label" data-row={labelRow}>
                    {tick.label}
                  </span>
                )}
              </span>
            );
          })}
        </div>
      </div>

      {(shapes.length > 0 || noValueLabel) && (
        <ul className="legend-items">
          {shapes.map((entry) => (
            <li key={entry.label}>
              <ShapeGlyph shape={entry.shape} hollow={entry.hollow} ring={entry.ring} fill={sample(glyphValue ?? hi * 0.55 + lo * 0.45)} />
              <span>{entry.label}</span>
            </li>
          ))}
          {noValueLabel && (
            <li>
              <ShapeGlyph fill={NO_VALUE_COLOR} />
              <span>{noValueLabel}</span>
            </li>
          )}
        </ul>
      )}

      {children && <div className="legend-notes">{children}</div>}
    </section>
  );
}
