import { useMemo } from "react";
import "../../components/map/MapLegend.css";
import { sampleScale, type ColorScale } from "../../lib/colors";
import { formatFixed } from "../../lib/format";
import type { DependenceModel } from "./dependenceModel";
import { CASING_COLOR, CASING_EXTRA_WEIGHT, LOW_DAYS_DASH, lineOpacity, lineWeight, normalized } from "./dependenceStyle";

/** Opacidad de la muestra «muy tenue» de la leyenda: mayor que la del mapa, solo para que se vea. */
const SAMPLE_FAINT_OPACITY = 0.35;
import "./dependence.css";

interface DependenceLegendProps {
  /** Nombre de la medida, p. ej. "χ(0.95)". */
  measureLabel: string;
  domain: [number, number];
  scale: ColorScale;
  model: DependenceModel | null;
}

/**
 * Muestra de una línea (color y grosor). Lleva un contorno claro fino, igual que las líneas resaltadas del mapa: así se lee
 * igual en la superficie clara y en la oscura, sin un parche de fondo. `soft`: contorno tenue para la muestra de líneas de contexto.
 */
function LineSample({ color, weight, opacity = 1, dashed = false, soft = false }: { color: string; weight: number; opacity?: number; dashed?: boolean; soft?: boolean }) {
  return (
    <svg className="line-sample" width="64" height="16" viewBox="0 0 64 16" aria-hidden="true" focusable="false">
      <line x1="4" y1="8" x2="60" y2="8" stroke={CASING_COLOR} strokeWidth={weight + (soft ? 1.5 : CASING_EXTRA_WEIGHT)} strokeLinecap="round" strokeOpacity={soft ? 0.4 : 0.9} />
      <line x1="4" y1="8" x2="60" y2="8" stroke={color} strokeWidth={weight} strokeOpacity={opacity} strokeLinecap="round" strokeDasharray={dashed ? LOW_DAYS_DASH : undefined} />
    </svg>
  );
}

/** Leyenda del Tab 3 en banda compacta: el valor se ve en el color Y en el grosor (el color no es el único canal). */
export default function DependenceLegend({ measureLabel, domain, scale, model }: DependenceLegendProps) {
  const [lo, hi] = domain;
  const mid = (lo + hi) / 2;
  const gradient = useMemo(() => `linear-gradient(to right, ${sampleScale(scale, domain).join(", ")})`, [scale, domain]);
  const fmt = (value: number) => formatFixed(value, 2);

  return (
    <section className="map-legend" aria-label="Leyenda del mapa">
      <div className="legend-scale">
        <div className="legend-head">
          <h3>{measureLabel}: color y grosor</h3>
        </div>
        <div
          className="legend-bar"
          role="img"
          aria-label={`Escala de color de ${fmt(lo)} a ${fmt(hi)}; los valores fuera de ese rango se muestran con el color del extremo`}
          style={{ background: gradient }}
        />
        <div className="legend-ticks" aria-hidden="true">
          <span className="legend-tick" style={{ left: "0%" }}>
            ≤ {fmt(lo)}
          </span>
          <span className="legend-tick" style={{ left: "50%" }}>
            {fmt(mid)}
          </span>
          <span className="legend-tick" style={{ left: "100%" }}>
            ≥ {fmt(hi)}
          </span>
        </div>
        <ul className="width-samples" aria-label="Grosor de las líneas con una estación seleccionada">
          {[lo, mid, hi].map((value) => (
            <li key={value}>
              <LineSample color={scale(value)} weight={lineWeight(normalized(value, domain), "related")} />
              <span>{fmt(value)}</span>
            </li>
          ))}
        </ul>
        <p className="legend-hint">Más oscuro y más grueso = dependencia más fuerte (la escala es fija para esta medida).</p>
      </div>

      <ul className="legend-items legend-items--stack">
        <li>
          <LineSample color={scale(mid)} weight={lineWeight(normalized(mid, domain), "overview")} opacity={SAMPLE_FAINT_OPACITY} soft />
          <span>
            Líneas muy tenues: sin estación seleccionada, las líneas se dibujan así de tenues (y también los pares que no incluyen a la seleccionada).{" "}
            <em>Muestra ampliada: en el mapa son más tenues.</em>
          </span>
        </li>
        <li>
          <LineSample color={scale(mid)} weight={lineWeight(normalized(mid, domain), "related")} opacity={lineOpacity("related")} />
          <span>Líneas resaltadas: pares que incluyen a la estación seleccionada</span>
        </li>
        <li>
          <LineSample color={scale(mid)} weight={3} dashed />
          <span>Trazo discontinuo: {model ? `${model.lowLabel}; ${model.lowCount} pares` : "pares con menos días conjuntos"}</span>
        </li>
      </ul>

      {model && (
        <div className="legend-notes">
          <p>
            {model.nullCount > 0
              ? `${model.nullCount} ${model.nullCount === 1 ? "par sin valor no se dibuja" : "pares sin valor no se dibujan"} en esta medida (aparecen como «sin valor» en la lista).`
              : "En esta medida todos los pares tienen valor, así que se dibujan todos."}
          </p>
          <p>
            Todos los pares se basan en pocos días conjuntos (de {model.minDays} a {model.maxDays}); el trazo discontinuo solo señala los que tienen menos.
          </p>
        </div>
      )}
    </section>
  );
}
