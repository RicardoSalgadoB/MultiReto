import { useEffect } from "react";

const SITE = "Ozono en Monterrey";

/** Título del documento por ruta ("Mapa de análisis · Ozono en Monterrey"): lo anuncian los lectores de pantalla al navegar. */
export function usePageTitle(title: string): void {
  useEffect(() => {
    document.title = `${title} · ${SITE}`;
  }, [title]);
}
