/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base de la API. Por defecto "/api" (mismo origen). */
  readonly VITE_API_URL?: string;
  /** Plantilla de teselas del mapa base. Por defecto, OpenStreetMap. */
  readonly VITE_TILE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
