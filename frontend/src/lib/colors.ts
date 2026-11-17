import { scaleSequential } from "d3-scale";
import { interpolateViridis, interpolateYlOrRd } from "d3-scale-chromatic";

/** Gris neutro para "sin dato". */
export const NO_VALUE_COLOR = "#9ca3af";

export type ColorScale = (value: number) => string;

/** Escala viridis sobre un dominio fijo (los valores fuera del dominio se llevan al extremo). */
export function viridisScale(domain: [number, number]): ColorScale {
  const scale = scaleSequential(interpolateViridis).domain(domain);
  return (value) => scale(Math.min(Math.max(value, domain[0]), domain[1]));
}

/** Colores equiespaciados del dominio completo, para dibujar un degradado en la leyenda. */
export function sampleScale(scale: ColorScale, domain: [number, number], steps = 12): string[] {
  const [lo, hi] = domain;
  return Array.from({ length: steps }, (_, i) => scale(lo + ((hi - lo) * i) / (steps - 1)));
}

/**
 * Escala de intensidad para líneas sobre un mapa claro: naranja-rojo (débil) a rojo oscuro (fuerte). Empieza en el punto de
 * la rampa YlOrRd (0.63) donde el color ya contrasta ≥ 3:1 con el fondo del mapa (#f2efe9); antes era casi invisible.
 * Los valores fuera del dominio se llevan al extremo.
 */
const INTENSITY_START = 0.63;
export function intensityScale(domain: [number, number]): ColorScale {
  const [lo, hi] = domain;
  return (value) => interpolateYlOrRd(INTENSITY_START + (1 - INTENSITY_START) * Math.min(Math.max((value - lo) / (hi - lo), 0), 1));
}
