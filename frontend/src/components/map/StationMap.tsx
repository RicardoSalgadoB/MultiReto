import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { MapContainer, TileLayer, useMapEvents } from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./StationMap.css";

import { MAP_CENTER, MAP_MAX_ZOOM, MAP_ZOOM, TILE_ATTRIBUTION, TILE_URL } from "../../config/map";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { viridisScale, type ColorScale } from "../../lib/colors";
import StationMarker from "./StationMarker";
import type { MarkerStyle, StationMapPoint } from "./types";

const HINT_MS = 1600;

export interface StationMapProps<P extends StationMapPoint> {
  points: P[];
  /** Dominio de la escala de color por defecto (viridis). */
  colorDomain?: [number, number];
  /** Escala inyectable; si se da, `colorDomain` solo sirve a la leyenda. Debe ser estable (useMemo o constante). */
  colorScale?: ColorScale;
  /** Forma, relleno hueco y anillo de cada punto. Debe ser estable (useCallback o función de módulo). */
  getMarkerStyle?: (point: P) => MarkerStyle;
  /** Texto accesible (aria-label y tooltip). Por defecto, "{name}: {value}" o "{name}: sin dato". */
  getLabel?: (point: P) => string;
  /** Contenido del tooltip al pasar el cursor; por defecto, el texto de `getLabel`. */
  renderTooltip?: (point: P) => ReactNode;
  /** El tooltip se muestra siempre junto al marcador (etiquetas con el código de la estación). */
  tooltipPermanent?: boolean;
  /** Si se da, hacer clic abre un popup con este contenido. */
  renderPopup?: (point: P) => ReactNode;
  selectedId?: number | null;
  /** Debe ser estable (useCallback). */
  onSelect?: (id: number) => void;
  /** Clic en el fondo del mapa (no en un marcador): sirve para deseleccionar. Debe ser estable. */
  onBackgroundClick?: () => void;
  center?: [number, number];
  zoom?: number;
  /** Texto accesible de la región del mapa. */
  ariaLabel?: string;
  /**
   * Evita que el mapa secuestre el scroll de la página (activado por defecto): el zoom con la rueda
   * exige Ctrl (o ⌘) y, en pantallas táctiles, un dedo desplaza la página y dos dedos mueven el mapa.
   * Un aviso discreto lo explica cuando el usuario lo intenta. Con `false` se usa el comportamiento normal de Leaflet.
   */
  gestureHandling?: boolean;
  /** Capas de react-leaflet dibujadas encima (p. ej. líneas del Tab 3). Los marcadores quedan por encima. */
  children?: ReactNode;
}

function BackgroundClick({ onClick }: { onClick?: () => void }) {
  useMapEvents({ click: () => onClick?.() });
  return null;
}

function defaultLabel(point: StationMapPoint): string {
  return `${point.name}: ${point.value === null ? "sin dato" : point.value}`;
}

/** Pantalla táctil principal (se evalúa una vez, al montar). */
function usesCoarsePointer(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(pointer: coarse)").matches;
}

/**
 * Mapa de estaciones con marcadores de forma, color y borde configurables.
 * El `MapContainer` se monta una sola vez: cambiar props o datos no recarga las teselas.
 */
export default function StationMap<P extends StationMapPoint>({
  points,
  colorDomain = [0, 120],
  colorScale,
  getMarkerStyle,
  getLabel = defaultLabel,
  renderTooltip,
  tooltipPermanent = false,
  renderPopup,
  selectedId = null,
  onSelect,
  onBackgroundClick,
  center = MAP_CENTER,
  zoom = MAP_ZOOM,
  ariaLabel = "Mapa de las estaciones de monitoreo",
  gestureHandling = true,
  children,
}: StationMapProps<P>) {
  const reducedMotion = useReducedMotion();
  const [domainLo, domainHi] = colorDomain;
  const scale = useMemo<ColorScale>(() => colorScale ?? viridisScale([domainLo, domainHi]), [colorScale, domainLo, domainHi]);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const hintTimer = useRef<number | undefined>(undefined);
  const [hint, setHint] = useState<string | null>(null);
  // Con gestos táctiles controlados, un dedo no arrastra el mapa (así la página sigue desplazándose).
  const [coarse] = useState(usesCoarsePointer);

  const showHint = useCallback((text: string) => {
    setHint(text);
    window.clearTimeout(hintTimer.current);
    hintTimer.current = window.setTimeout(() => setHint(null), HINT_MS);
  }, []);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!gestureHandling || !wrapper) return;

    // En fase de captura: sin Ctrl/⌘ el evento no llega a Leaflet, así que no hace zoom y el navegador
    // desplaza la página (el listener es pasivo y no cancela el scroll).
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) return;
      event.stopPropagation();
      showHint("Usa Ctrl + rueda para hacer zoom");
    };
    const onTouchMove = (event: TouchEvent) => {
      if (coarse && event.touches.length === 1) showHint("Usa dos dedos para mover el mapa");
    };
    wrapper.addEventListener("wheel", onWheel, { capture: true, passive: true });
    wrapper.addEventListener("touchmove", onTouchMove, { passive: true });
    return () => {
      wrapper.removeEventListener("wheel", onWheel, { capture: true });
      wrapper.removeEventListener("touchmove", onTouchMove);
      window.clearTimeout(hintTimer.current);
    };
  }, [gestureHandling, coarse, showHint]);

  return (
    <div className="map-wrapper" role="region" aria-label={ariaLabel} ref={wrapperRef}>
      <MapContainer
        center={center}
        zoom={zoom}
        maxZoom={MAP_MAX_ZOOM}
        scrollWheelZoom
        dragging={!(gestureHandling && coarse)}
        // Opciones de Leaflet que se leen al montar: sin animaciones de zoom ni fundidos si se pidió reducir el movimiento.
        zoomAnimation={!reducedMotion}
        fadeAnimation={!reducedMotion}
        markerZoomAnimation={!reducedMotion}
        className="map-container"
      >
        <TileLayer attribution={TILE_ATTRIBUTION} url={TILE_URL} maxZoom={MAP_MAX_ZOOM} />
        <BackgroundClick onClick={onBackgroundClick} />
        {children}
        {points.map((point) => (
          <StationMarker
            key={point.id}
            point={point}
            scale={scale}
            selected={point.id === selectedId}
            getMarkerStyle={getMarkerStyle}
            getLabel={getLabel}
            renderTooltip={renderTooltip}
            renderPopup={renderPopup}
            permanentLabel={tooltipPermanent}
            onSelect={onSelect}
          />
        ))}
      </MapContainer>
      <div className="map-hint-region" role="status" aria-live="polite">
        {hint && <p className="map-hint">{hint}</p>}
      </div>
    </div>
  );
}
