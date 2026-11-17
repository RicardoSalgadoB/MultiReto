import type { ReactNode } from "react";
import type { ApiInfo, HistoricPoint } from "../../api/types";
import type { LegendShape, MarkerStyle } from "../../components/map/types";
import { OZONE_DOMAIN } from "../../config/modelConstants";
import { viridisScale } from "../../lib/colors";
import { formatNumber } from "../../lib/format";

/**
 * Cómo se dibuja cada estación en el Tab 1. Todo está a nivel de módulo (referencias estables) para que
 * los marcadores no se recreen al cambiar la hora: solo se actualizan los que cambian de aspecto.
 */

export const OZONE_SCALE = viridisScale(OZONE_DOMAIN);

const TRIANGLE: MarkerStyle = { shape: "triangle" };
const CIRCLE: MarkerStyle = { shape: "circle" };
const NO_PREDICTION: MarkerStyle = { shape: "diamond", hollow: true };
const TRIANGLE_RING: MarkerStyle = { shape: "triangle", ring: true };
const CIRCLE_RING: MarkerStyle = { shape: "circle", ring: true };
const NO_PREDICTION_RING: MarkerStyle = { shape: "diamond", hollow: true, ring: true };

/** Forma = ¿el modelo predice exceso (alerta)? Anillo doble = el máximo observado del día superó la norma. */
export function getMarkerStyle(point: HistoricPoint): MarkerStyle {
  const ring = point.exceso_real === true;
  if (point.alerta === null) return ring ? NO_PREDICTION_RING : NO_PREDICTION;
  if (point.alerta) return ring ? TRIANGLE_RING : TRIANGLE;
  return ring ? CIRCLE_RING : CIRCLE;
}

function alertText(point: HistoricPoint): string {
  if (point.alerta === null) return "sin predicción para ese día";
  return point.alerta ? "predice exceso (alerta)" : "no predice exceso";
}

function ozoneText(point: HistoricPoint): string {
  return point.value === null ? "sin medición" : `${formatNumber(point.value)} ppb`;
}

/** Nombre accesible de cada marcador (aria-label). */
export function getLabel(point: HistoricPoint): string {
  const observed = point.exceso_real === true ? "; el máximo observado del día superó la norma" : "";
  return `${point.name}, ozono: ${ozoneText(point)}; modelo: ${alertText(point)}${observed}`;
}

export function renderTooltip(point: HistoricPoint): ReactNode {
  return (
    <>
      <strong>{point.name}</strong> · {point.nombre}
      <br />
      Ozono: {ozoneText(point)}
      <br />
      Modelo: {alertText(point)}
      {point.exceso_real === true && (
        <>
          <br />
          Máximo del día sobre la norma
        </>
      )}
    </>
  );
}

/** Entradas de la leyenda de formas; la norma sale de la API. */
export function legendShapes(info: ApiInfo): LegendShape[] {
  return [
    { shape: "triangle", label: "El modelo predice exceso (alerta)" },
    { shape: "circle", label: "El modelo no predice exceso" },
    { shape: "diamond", hollow: true, label: "Sin predicción para ese día" },
    { shape: "circle", ring: true, label: `Anillo doble: el máximo observado del día superó la norma (${formatNumber(info.norma_ppb)} ppb)` },
  ];
}
