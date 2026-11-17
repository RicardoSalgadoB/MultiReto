import type { ReactNode } from "react";
import type { ApiInfo, PredictionPoint } from "../../api/types";
import type { LegendShape, MarkerStyle } from "../../components/map/types";
import { PROBABILITY_DOMAIN } from "../../config/modelConstants";
import { viridisScale } from "../../lib/colors";
import { formatPercent } from "../../lib/format";

/**
 * Cómo se dibuja cada estación en el Tab 2. Todo es de módulo (referencias estables), igual que en el Tab 1.
 * El color es P(máximo de mañana > norma), con dominio fijo (PROBABILITY_DOMAIN); la forma indica si supera el corte de alerta.
 */
export const PROBABILITY_SCALE = viridisScale(PROBABILITY_DOMAIN);

const ALERT: MarkerStyle = { shape: "triangle" };
const NO_ALERT: MarkerStyle = { shape: "circle" };
const NO_PREDICTION: MarkerStyle = { shape: "diamond", hollow: true };

export function getMarkerStyle(point: PredictionPoint): MarkerStyle {
  if (point.p90 === null) return NO_PREDICTION;
  return point.alerta ? ALERT : NO_ALERT;
}

function probabilityText(point: PredictionPoint): string {
  return point.p90 === null ? "sin predicción" : formatPercent(point.p90);
}

function alertText(point: PredictionPoint): string {
  if (point.p90 === null) return "sin predicción";
  return point.alerta ? "supera el corte de alerta" : "no supera el corte de alerta";
}

/** Nombre accesible de cada marcador. */
export function getLabel(point: PredictionPoint): string {
  return `${point.name}, probabilidad de superar la norma: ${probabilityText(point)}; ${alertText(point)}`;
}

export function renderTooltip(point: PredictionPoint): ReactNode {
  return (
    <>
      <strong>{point.name}</strong> · {point.nombre}
      <br />
      P(máx &gt; norma): {probabilityText(point)}
      <br />
      {alertText(point)}
    </>
  );
}

/** Leyenda de formas; el corte sale de la API. */
export function legendShapes(info: ApiInfo): LegendShape[] {
  return [
    { shape: "triangle", label: `Supera el corte de alerta (P ≥ ${formatPercent(info.corte_alerta)})` },
    { shape: "circle", label: "No supera el corte de alerta" },
  ];
}
