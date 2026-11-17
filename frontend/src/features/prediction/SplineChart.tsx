import { useMemo } from "react";
import type { GamCurve, StationCode } from "../../api/types";
import LineChart, { type ChartPoint } from "../../components/charts/LineChart";
import { formatNumber, formatNumber2, formatPercent, formatPercentAxis } from "../../lib/format";

export type CurveMode = "logit" | "probabilidad";

interface SplineChartProps {
  curve: GamCurve;
  mode: CurveMode;
  station: StationCode;
  /** Valor actual de la variable en la estación; null si no se puede determinar. */
  currentValue: number | null;
  /** Texto breve si el valor se calculó a partir de otras entradas (variables derivadas). */
  derivedNote?: string;
}

/** Una curva del GAM: efecto de la variable en logit ζ, o probabilidad de un día típico de la estación. */
export default function SplineChart({ curve, mode, station, currentValue, derivedNote }: SplineChartProps) {
  const values = mode === "logit" ? curve.efecto_logit : curve.probabilidad;

  const data = useMemo<ChartPoint[]>(() => curve.x.map((x, i) => ({ x, y: values?.[i] ?? null })), [curve.x, values]);
  const unit = curve.unidad;
  const formatX = useMemo(() => (x: number) => `${formatNumber(x)} ${unit}`, [unit]);
  const lo = curve.x[0];
  const hi = curve.x[curve.x.length - 1];
  const outside = currentValue !== null && (currentValue < lo || currentValue > hi);
  const clampedNote = currentValue === null ? undefined : `Valor actual: ${formatX(currentValue)}, fuera del rango de la curva; la marca se muestra en el borde.`;
  const marker = useMemo(
    () => (currentValue === null ? undefined : { x: currentValue, label: `Valor actual en ${station}`, clampedNote }),
    [currentValue, station, clampedNote],
  );

  const isLogit = mode === "logit";
  return (
    <div className="spline-chart">
      <LineChart
        title={`${curve.descripcion} (${unit})`}
        data={data}
        seriesName={isLogit ? "Efecto en logit ζ" : `Probabilidad en ${station}`}
        unit=""
        ariaLabel={
          isLogit
            ? `Efecto de «${curve.descripcion}» en el logit de ζ (curva común a todas las estaciones)`
            : `Probabilidad de que el máximo de mañana supere el umbral del modelo en ${station} según «${curve.descripcion}»`
        }
        formatX={formatX}
        formatXTick={formatNumber}
        formatY={isLogit ? formatNumber2 : formatPercent}
        formatYTick={isLogit ? formatNumber2 : formatPercentAxis}
        yRange="fit"
        marker={marker}
        compact
        pageStep={10}
      />
      {outside && clampedNote && <p className="spline-note spline-note--warn">{clampedNote}</p>}
      {derivedNote && <p className="spline-note">{derivedNote}</p>}
      {currentValue === null && <p className="spline-note">No se pudo determinar el valor actual de esta variable: sin marca.</p>}
    </div>
  );
}
