import { useId, useMemo, useState, type KeyboardEvent, type PointerEvent } from "react";
import { scaleLinear } from "d3-scale";
import { useElementWidth } from "../../hooks/useElementWidth";
import { formatNumber } from "../../lib/format";
import "./LineChart.css";

const MARGIN_FULL = { top: 28, right: 14, bottom: 26, left: 44 };
const MARGIN_COMPACT = { top: 10, right: 10, bottom: 22, left: 42 };
const MIN_WIDTH = 220;
const DEFAULT_WIDTH = 640;
const MIN_TICK_GAP_PX = 64;
const NO_LINES: ReferenceLine[] = [];

export interface ChartPoint {
  /** Eje x numérico (milisegundos, ppb, °C...). */
  x: number;
  /** null = sin dato: se deja un hueco, no se interpola. */
  y: number | null;
}

export interface ReferenceLine {
  value: number;
  label: string;
  /** Texto corto que se dibuja dentro de la gráfica (p. ej. "Norma 90 ppb"); la leyenda conserva `label` completo. */
  shortLabel?: string;
  /** "threshold" = punteada; "limit" = discontinua y más marcada. */
  kind: "threshold" | "limit";
}

export interface ChartTick {
  value: number;
  label: string;
}

interface LineChartProps {
  /** Puntos ordenados por x. */
  data: ChartPoint[];
  /** Nombre de la serie, p. ej. "Ozono". */
  seriesName: string;
  /** Unidad de y para tooltip, resumen y leyenda; "" si no aplica (p. ej. porcentajes ya formateados). */
  unit: string;
  /** Descripción corta de lo que muestra la gráfica (aria-label). */
  ariaLabel: string;
  /** Texto de un punto en el tooltip y en el resumen. */
  formatX: (x: number) => string;
  /** Marcas del eje x. Si se omiten, se calculan numéricamente. */
  xTicks?: ChartTick[];
  formatXTick?: (x: number) => string;
  formatY?: (y: number) => string;
  /** Texto de las marcas del eje y (por defecto, `formatY`). */
  formatYTick?: (y: number) => string;
  /** "from-zero": el eje y arranca en `yMin` (por defecto 0). "fit": se ajusta a los datos (efectos con valores negativos). */
  yRange?: "from-zero" | "fit";
  yMin?: number;
  referenceLines?: ReferenceLine[];
  /**
   * Marca vertical, p. ej. el momento seleccionado o el valor actual de la variable. Si `x` cae fuera del dominio
   * de la gráfica, la marca se lleva al borde y `clampedNote` se agrega al tooltip de ese punto y al resumen.
   */
  marker?: { x: number; label: string; clampedNote?: string };
  /** Título visible sobre la gráfica. */
  title?: string;
  /** Versión pequeña para miniaturas: menos márgenes, sin leyenda y con el resumen solo para lectores de pantalla. */
  compact?: boolean;
  /** Salto de Re Pág / Av Pág en puntos (por defecto 24). */
  pageStep?: number;
}

/** Índice del punto cuya x está más cerca de `x` (data ordenada por x). */
function nearestIndex(data: ChartPoint[], x: number): number {
  let lo = 0;
  let hi = data.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (data[mid].x < x) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(data[lo - 1].x - x) <= Math.abs(data[lo].x - x)) return lo - 1;
  return lo;
}

/** Una sola ruta SVG para toda la serie; cada hueco abre un subtrazo nuevo (M). */
function buildPath(data: ChartPoint[], toX: (x: number) => number, toY: (y: number) => number): string {
  let path = "";
  let inRun = false;
  for (let i = 0; i < data.length; i++) {
    const { x, y } = data[i];
    if (y === null || !Number.isFinite(y)) {
      inRun = false;
      continue;
    }
    const px = toX(x).toFixed(1);
    const py = toY(y).toFixed(1);
    if (inRun) {
      path += `L${px} ${py}`;
    } else {
      path += `M${px} ${py}`;
      // Punto aislado: segmento de largo cero que, con trazo redondeado, se dibuja como punto.
      const next = data[i + 1];
      if (!next || next.y === null || !Number.isFinite(next.y)) path += `L${px} ${py}`;
    }
    inRun = true;
  }
  return path;
}

/** Deja solo las marcas que caben sin encimarse. */
function thinTicks(ticks: ChartTick[], toX: (x: number) => number): ChartTick[] {
  const kept: ChartTick[] = [];
  let lastX = -Infinity;
  for (const tick of ticks) {
    const px = toX(tick.value);
    if (px - lastX >= MIN_TICK_GAP_PX) {
      kept.push(tick);
      lastX = px;
    }
  }
  return kept;
}

/**
 * Gráfica de línea en SVG propio (sin librerías), reutilizable para series de tiempo y curvas numéricas.
 * - Ancho responsivo: el viewBox mide lo mismo que el contenedor, así el texto conserva su tamaño.
 * - Teclado: Tab para enfocar; ← → mueven un punto, Re Pág / Av Pág saltan `pageStep`, Inicio / Fin van a los extremos.
 * - Pasar el cursor (o deslizar un dedo en horizontal) muestra línea guía y tooltip.
 */
export default function LineChart({
  data,
  seriesName,
  unit,
  ariaLabel,
  formatX,
  xTicks,
  formatXTick = formatNumber,
  formatY = formatNumber,
  formatYTick,
  yRange = "from-zero",
  yMin = 0,
  referenceLines = NO_LINES,
  marker,
  title,
  compact = false,
  pageStep = 24,
}: LineChartProps) {
  const [frameRef, measured] = useElementWidth<HTMLDivElement>();
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const summaryId = useId();

  const margin = compact ? MARGIN_COMPACT : MARGIN_FULL;
  const width = Math.max(measured || DEFAULT_WIDTH, MIN_WIDTH);
  const height = compact ? Math.min(Math.max(Math.round(width * 0.5), 150), 190) : Math.min(Math.max(Math.round(width * 0.42), 210), 290);
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const withUnit = (text: string) => (unit ? `${text} ${unit}` : text);
  const tickFormatY = formatYTick ?? formatY;

  const { toX, toY, yTicks, hasData } = useMemo(() => {
    const valid = data.filter((p) => p.y !== null && Number.isFinite(p.y)) as { x: number; y: number }[];
    const values = [...valid.map((p) => p.y), ...referenceLines.map((r) => r.value)];
    let domain: [number, number];
    if (yRange === "fit" && values.length > 0) {
      const lo = Math.min(...values);
      const hi = Math.max(...values);
      const pad = (hi - lo || 1) * 0.08;
      domain = [lo - pad, hi + pad];
    } else {
      domain = [yMin, Math.max(...values, yMin + 1) * 1.06];
    }
    const y = scaleLinear().domain(domain).nice().range([margin.top + plotH, margin.top]);
    const x = scaleLinear()
      .domain(data.length > 1 ? [data[0].x, data[data.length - 1].x] : [0, 1])
      .range([margin.left, margin.left + plotW]);
    return { toX: x, toY: y, yTicks: y.ticks(compact ? 3 : 5), hasData: valid.length > 0 };
  }, [data, referenceLines, yRange, yMin, margin.top, margin.left, plotW, plotH, compact]);

  const path = useMemo(() => buildPath(data, toX, toY), [data, toX, toY]);
  const visibleXTicks = useMemo(() => {
    const candidates = xTicks ?? toX.ticks(compact ? 4 : 6).map((value) => ({ value, label: formatXTick(value) }));
    return thinTicks(candidates, toX);
  }, [xTicks, toX, compact, formatXTick]);

  const summary = useMemo(() => {
    const valid = data.filter((p): p is { x: number; y: number } => p.y !== null && Number.isFinite(p.y));
    if (valid.length === 0) return `Sin valores de ${seriesName.toLowerCase()} en este periodo.`;
    const lowest = valid.reduce((a, b) => (b.y < a.y ? b : a));
    const highest = valid.reduce((a, b) => (b.y > a.y ? b : a));
    const latest = valid[valid.length - 1];
    const missing = data.length - valid.length;
    const fmt = (v: number) => (unit ? `${formatY(v)} ${unit}` : formatY(v));
    return (
      `${seriesName}, de ${formatX(data[0].x)} a ${formatX(data[data.length - 1].x)}: ` +
      `mínimo ${fmt(lowest.y)}, máximo ${fmt(highest.y)}, ` +
      `último valor ${fmt(latest.y)} (${formatX(latest.x)}). ` +
      (missing > 0 ? `${missing} de ${data.length} puntos sin dato.` : "Sin huecos.")
    );
  }, [data, seriesName, unit, formatX, formatY]);

  // La marca se lleva al borde si el valor queda fuera del dominio de la gráfica.
  const [domainLo, domainHi] = toX.domain();
  const markerX = marker ? Math.min(Math.max(marker.x, domainLo), domainHi) : null;
  const markerClamped = marker !== undefined && markerX !== marker.x;
  const markerIndex = markerX !== null && data.length > 0 ? nearestIndex(data, markerX) : null;
  const active = activeIndex !== null && activeIndex < data.length ? data[activeIndex] : null;
  // El punto del borde donde se dibuja una marca acotada lleva la nota de que el valor real quedó fuera.
  const clampedNote = markerClamped && marker && activeIndex !== null && activeIndex === markerIndex ? marker.clampedNote : undefined;
  const activeText = active ? `${formatX(active.x)}: ${active.y === null ? "sin dato" : withUnit(formatY(active.y))}${clampedNote ? `. ${clampedNote}` : ""}` : "";
  const hasGaps = data.some((p) => p.y === null);

  function handlePointer(event: PointerEvent<SVGSVGElement>) {
    if (data.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (width / rect.width);
    setActiveIndex(nearestIndex(data, toX.invert(Math.min(Math.max(px, margin.left), margin.left + plotW))));
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (data.length === 0) return;
    const last = data.length - 1;
    const base = activeIndex ?? markerIndex ?? last;
    let next: number;
    switch (event.key) {
      case "ArrowRight":
        next = base + 1;
        break;
      case "ArrowLeft":
        next = base - 1;
        break;
      case "PageUp":
        next = base - pageStep;
        break;
      case "PageDown":
        next = base + pageStep;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = last;
        break;
      case "Escape":
        setActiveIndex(null);
        return;
      default:
        return;
    }
    event.preventDefault();
    setActiveIndex(Math.min(Math.max(next, 0), last));
  }

  const tooltipX = active ? toX(active.x) : 0;
  const tooltipShift = tooltipX < 96 ? "0%" : tooltipX > width - 96 ? "-100%" : "-50%";

  return (
    <figure className={`line-chart${compact ? " line-chart--compact" : ""}`}>
      {title && <figcaption className="chart-title">{title}</figcaption>}
      <div
        ref={frameRef}
        className="chart-frame"
        tabIndex={0}
        role="group"
        aria-label={`${ariaLabel}. Gráfica interactiva: usa las flechas izquierda y derecha para recorrer los puntos.`}
        aria-describedby={summaryId}
        onKeyDown={handleKeyDown}
        onFocus={() => setActiveIndex((prev) => prev ?? markerIndex ?? (data.length > 0 ? data.length - 1 : null))}
        onBlur={() => setActiveIndex(null)}
      >
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height={height}
          role="img"
          aria-label={ariaLabel}
          aria-describedby={summaryId}
          onPointerMove={handlePointer}
          onPointerDown={handlePointer}
          onPointerLeave={() => setActiveIndex(null)}
        >
          {/* Rejilla y eje y */}
          {yTicks.map((tick) => (
            <g key={tick}>
              <line className="chart-grid" x1={margin.left} x2={margin.left + plotW} y1={toY(tick)} y2={toY(tick)} />
              <text className="chart-axis-text" x={margin.left - 6} y={toY(tick)} textAnchor="end" dominantBaseline="middle">
                {tickFormatY(tick)}
              </text>
            </g>
          ))}
          {/* Eje x */}
          <line className="chart-baseline" x1={margin.left} x2={margin.left + plotW} y1={margin.top + plotH} y2={margin.top + plotH} />
          {visibleXTicks.map((tick) => (
            <g key={tick.value}>
              <line className="chart-tick" x1={toX(tick.value)} x2={toX(tick.value)} y1={margin.top + plotH} y2={margin.top + plotH + 4} />
              <text className="chart-axis-text" x={toX(tick.value)} y={margin.top + plotH + (compact ? 14 : 16)} textAnchor="middle">
                {tick.label}
              </text>
            </g>
          ))}
          {unit && !compact && (
            <text className="chart-axis-text chart-unit" x={margin.left - 6} y={margin.top - 16} textAnchor="end">
              {unit}
            </text>
          )}

          {/* Líneas de referencia etiquetadas */}
          {referenceLines.map((ref) => (
            <g key={ref.label}>
              <line className={`chart-ref chart-ref--${ref.kind}`} x1={margin.left} x2={margin.left + plotW} y1={toY(ref.value)} y2={toY(ref.value)} />
              <text className={`chart-ref-text chart-ref-text--${ref.kind}`} x={margin.left + plotW - 6} y={toY(ref.value) - 5} textAnchor="end">
                {ref.shortLabel ?? `${ref.label} (${withUnit(formatY(ref.value))})`}
              </text>
            </g>
          ))}

          {/* Marca vertical (momento seleccionado o valor actual de la variable) */}
          {marker && markerX !== null && (
            <g>
              <line className="chart-marker" x1={toX(markerX)} x2={toX(markerX)} y1={margin.top} y2={margin.top + plotH} />
              <polygon
                className="chart-marker-head"
                points={`${toX(markerX) - 5},${margin.top - 8} ${toX(markerX) + 5},${margin.top - 8} ${toX(markerX)},${margin.top}`}
              />
            </g>
          )}

          {/* Serie: un solo path */}
          <path className="chart-line" d={path} />

          {/* Línea guía y punto activo */}
          {active && (
            <g pointerEvents="none">
              <line className="chart-guide" x1={toX(active.x)} x2={toX(active.x)} y1={margin.top} y2={margin.top + plotH} />
              {active.y !== null && <circle className="chart-dot" cx={toX(active.x)} cy={toY(active.y)} r={4.5} />}
            </g>
          )}
        </svg>

        {active && (
          <div className="chart-tooltip" style={{ left: tooltipX, transform: `translateX(${tooltipShift})` }} aria-hidden="true">
            <span className="chart-tooltip-time">{formatX(active.x)}</span>
            <strong>{active.y === null ? "sin dato" : withUnit(formatY(active.y))}</strong>
            {clampedNote && <span className="chart-tooltip-note">{clampedNote}</span>}
          </div>
        )}
        <p className="sr-only" aria-live="polite">
          {activeText}
        </p>
      </div>

      <p id={summaryId} className={compact ? "sr-only" : "chart-summary"}>
        {summary}
        {markerClamped && marker ? ` ${marker.clampedNote ?? `La marca «${marker.label}» queda fuera del rango y se muestra en el borde.`}` : ""}
      </p>

      {!compact && (
        <ul className="chart-legend" aria-label="Leyenda de la gráfica">
          <li>
            <span className="swatch swatch--line" aria-hidden="true" />
            {seriesName}
            {unit ? ` (${unit})` : ""}
          </li>
          {referenceLines.map((ref) => (
            <li key={ref.label}>
              <span className={`swatch swatch--${ref.kind}`} aria-hidden="true" />
              {ref.label} ({withUnit(formatY(ref.value))})
            </li>
          ))}
          {marker && (
            <li>
              <span className="swatch swatch--marker" aria-hidden="true" />
              {marker.label}
            </li>
          )}
          {hasGaps && hasData && <li>Los huecos en la línea son horas sin dato (no se interpolan).</li>}
        </ul>
      )}
    </figure>
  );
}
