/**
 * Variables de entorno (se incrustan al compilar). Nunca poner secretos aquí.
 * Un valor vacío cuenta como "no definido".
 */

/** Base de la API: mismo origen por defecto (proxy de Vite en desarrollo, rewrite de Vercel en producción). */
export const API_URL: string = (import.meta.env.VITE_API_URL || "/api").replace(/\/+$/, "");

export const TILE_URL: string =
  import.meta.env.VITE_TILE_URL || "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
