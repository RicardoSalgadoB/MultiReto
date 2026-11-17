/** Formas disponibles para los marcadores (no depender solo del color). */
export type MarkerShape = "circle" | "triangle" | "square" | "diamond";

/** Aspecto de un marcador, además del color de relleno (que sale del valor). */
export interface MarkerStyle {
  /** Por defecto, "circle". */
  shape?: MarkerShape;
  /** Sin relleno de color, trazo gris (p. ej. "sin predicción"). */
  hollow?: boolean;
  /** Anillo doble (negro por fuera, blanco por dentro). */
  ring?: boolean;
}

/** Mínimo que necesita el mapa de cada estación. Las estaciones de la API ya lo cumplen. */
export interface StationMapPoint {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  /** Valor con el que se colorea; null = sin dato (se dibuja en gris neutro). */
  value: number | null;
}

/** Una entrada de la leyenda de formas. */
export interface LegendShape extends MarkerStyle {
  label: string;
}
