/**
 * Estilo de las líneas. El valor se codifica con dos canales: color (escala de intensidad) y grosor.
 * "overview": sin estación seleccionada (muy tenues); "related": líneas de la estación seleccionada;
 * "unrelated": las demás cuando hay una estación seleccionada (aún más tenues).
 */
export type Emphasis = "overview" | "related" | "unrelated";

/** Posición del valor dentro del dominio fijo de la medida: 0 a 1. */
export function normalized(value: number, domain: [number, number]): number {
  return Math.min(Math.max((value - domain[0]) / (domain[1] - domain[0]), 0), 1);
}

export function lineWeight(t: number, emphasis: Emphasis): number {
  if (emphasis === "related") return 2 + 6 * t; // 2 a 8 px
  if (emphasis === "overview") return 0.75 + 1.25 * t; // 0.75 a 2 px: contexto, no ruido
  return 1;
}

export function lineOpacity(emphasis: Emphasis): number {
  if (emphasis === "related") return 0.95;
  if (emphasis === "overview") return 0.16;
  return 0.05;
}

/** Trazo discontinuo para los pares del cuartil con menos días conjuntos. */
export const LOW_DAYS_DASH = "7 6";

/** Contorno claro bajo las líneas resaltadas: las separa del dibujo del mapa base (que es claro en ambos temas). */
export const CASING_COLOR = "#ffffff";
export const CASING_EXTRA_WEIGHT = 3;
export const CASING_OPACITY = 0.85;
