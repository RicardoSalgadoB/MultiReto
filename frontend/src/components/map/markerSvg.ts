import { NO_VALUE_COLOR } from "../../lib/colors";
import type { MarkerShape, MarkerStyle } from "./types";

export interface MarkerSvgOptions extends MarkerStyle {
  /** Color de relleno; null = sin dato. */
  fill: string | null;
  selected?: boolean;
  /** Lado del SVG en píxeles. */
  size?: number;
}

const SHAPES: readonly MarkerShape[] = ["circle", "triangle", "square", "diamond"];
// Colores que producen d3 (hex o rgb()/rgba() numéricos) y las constantes del proyecto.
const SAFE_COLOR = /^(#[0-9a-f]{3,8}|rgba?\(\s*[0-9.]+\s*(,\s*[0-9.]+\s*){2,3}\))$/i;
const DEFAULT_SIZE = 40;
/** Contorno oscuro de las formas: se distingue del mapa claro con cualquier color de relleno (≥ 12:1). */
const OUTLINE = "#1f2937";

/*
 * Este módulo construye una cadena SVG que se inserta con innerHTML (divIcon de Leaflet y leyenda).
 * Por eso solo interpola valores validados aquí mismo: formas de una unión literal, colores con
 * formato numérico y tamaños finitos. Si algo no es válido se usa el estilo neutro por defecto.
 * Nunca recibe nombres de estación ni texto de la API.
 */
function safeShape(value: unknown): MarkerShape {
  return SHAPES.includes(value as MarkerShape) ? (value as MarkerShape) : "circle";
}

function safeColor(value: unknown): string {
  return typeof value === "string" && SAFE_COLOR.test(value) ? value : NO_VALUE_COLOR;
}

function safeSize(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.min(Math.max(Math.round(value), 8), 128) : DEFAULT_SIZE;
}

/** Forma centrada en el origen (media anchura ≈ 9 unidades del viewBox). */
function shapeElement(shape: MarkerShape, attrs: string): string {
  switch (shape) {
    case "square":
      return `<rect x="-8" y="-8" width="16" height="16" rx="1.5" ${attrs}/>`;
    case "triangle":
      return `<polygon points="0,-10 10.5,8 -10.5,8" ${attrs}/>`;
    case "diamond":
      return `<polygon points="0,-11 11,0 0,11 -11,0" ${attrs}/>`;
    case "circle":
      return `<circle r="9" ${attrs}/>`;
  }
}

/** SVG de un marcador como texto. Lo usan el `divIcon` de Leaflet y la leyenda, así el dibujo es idéntico. */
export function markerSvg(options: MarkerSvgOptions): string {
  const shape = safeShape(options.shape);
  const fill = options.fill === null ? NO_VALUE_COLOR : safeColor(options.fill);
  const size = safeSize(options.size);
  const hollow = options.hollow === true;
  const ring = options.ring === true;
  const selected = options.selected === true;

  // vector-effect mantiene el grosor del trazo al escalar los anillos.
  const ringAttrs = 'fill="none" vector-effect="non-scaling-stroke"';
  const parts: string[] = [];

  if (selected) {
    parts.push('<circle r="17.5" style="fill:var(--map-halo);fill-opacity:.18;stroke:var(--map-halo)" stroke-width="2.5"/>');
  }
  if (ring) {
    parts.push(`<g transform="scale(1.4)">${shapeElement(shape, `${ringAttrs} stroke="#000" stroke-width="4"`)}</g>`);
    parts.push(`<g transform="scale(1.2)">${shapeElement(shape, `${ringAttrs} stroke="#fff" stroke-width="2.2"`)}</g>`);
  }
  const body = hollow
    ? 'fill="rgba(255,255,255,0.35)" stroke="#4b5563" stroke-width="2.5"'
    : `fill="${fill}" stroke="${OUTLINE}" stroke-width="1.5"`;
  parts.push(shapeElement(shape, body));

  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-20 -20 40 40" width="${size}" height="${size}" ` +
    `aria-hidden="true" focusable="false">${parts.join("")}</svg>`
  );
}
