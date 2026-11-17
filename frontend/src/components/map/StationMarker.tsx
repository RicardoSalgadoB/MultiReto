import { memo, useEffect, useMemo, useRef, type ReactNode } from "react";
import { divIcon, type LeafletKeyboardEvent, type Marker as LeafletMarker } from "leaflet";
import { Marker, Popup, Tooltip } from "react-leaflet";
import type { ColorScale } from "../../lib/colors";
import { markerSvg } from "./markerSvg";
import type { MarkerStyle, StationMapPoint } from "./types";

const DEFAULT_STYLE: MarkerStyle = {};
const ICON_SIZE = 40;

interface StationMarkerProps<P extends StationMapPoint> {
  /** Debe conservar su referencia mientras los datos de la estación no cambien: así `memo` evita re-renders. */
  point: P;
  scale: ColorScale;
  selected: boolean;
  getMarkerStyle?: (point: P) => MarkerStyle;
  /** Nombre accesible (aria-label) y texto por defecto del tooltip. */
  getLabel: (point: P) => string;
  renderTooltip?: (point: P) => ReactNode;
  renderPopup?: (point: P) => ReactNode;
  /** Muestra el tooltip siempre (como etiqueta junto al marcador) en vez de al pasar el cursor. No cambia después de montar. */
  permanentLabel?: boolean;
  /** Debe ser estable (useCallback) para que el marcador no se redibuje de más. */
  onSelect?: (id: number) => void;
}

/**
 * Marcador de una estación. Recibe el punto y las funciones (en vez de estilo, etiqueta y tooltip ya calculados)
 * para que `memo` compare solo referencias estables: al mover un control cuyo efecto aún no llega a los datos,
 * ningún marcador se vuelve a renderizar.
 */
function StationMarker<P extends StationMapPoint>({
  point,
  scale,
  selected,
  getMarkerStyle,
  getLabel,
  renderTooltip,
  renderPopup,
  permanentLabel = false,
  onSelect,
}: StationMarkerProps<P>) {
  const markerRef = useRef<LeafletMarker>(null);
  const { id, latitude, longitude, value } = point;

  const { shape, hollow, ring } = getMarkerStyle?.(point) ?? DEFAULT_STYLE;
  const fill = value === null ? null : scale(value);
  const label = getLabel(point);
  const html = useMemo(() => markerSvg({ shape, hollow, ring, fill, selected, size: ICON_SIZE }), [shape, hollow, ring, fill, selected]);
  const icon = useMemo(
    () => divIcon({ html, className: "station-marker", iconSize: [ICON_SIZE, ICON_SIZE], iconAnchor: [ICON_SIZE / 2, ICON_SIZE / 2] }),
    [html],
  );
  const position = useMemo<[number, number]>(() => [latitude, longitude], [latitude, longitude]);

  // Leaflet ya pone tabindex=0 y role="button"; aquí se agrega el nombre accesible y el estado.
  useEffect(() => {
    const element = markerRef.current?.getElement();
    element?.setAttribute("aria-label", label);
    element?.setAttribute("aria-pressed", String(selected));
  }, [label, selected, icon]);

  const eventHandlers = useMemo(
    () => ({
      click: () => onSelect?.(id),
      // Leaflet solo abre popups con Enter; aquí Enter y Espacio seleccionan la estación.
      keydown: (event: LeafletKeyboardEvent) => {
        const key = event.originalEvent.key;
        if (key === "Enter" || key === " ") {
          event.originalEvent.preventDefault();
          onSelect?.(id);
        }
      },
    }),
    [id, onSelect],
  );

  const popup = renderPopup?.(point);
  return (
    <Marker ref={markerRef} position={position} icon={icon} eventHandlers={eventHandlers} keyboard zIndexOffset={selected ? 1000 : 0}>
      <Tooltip
        direction={permanentLabel ? "right" : "top"}
        offset={permanentLabel ? [14, 0] : [0, -12]}
        permanent={permanentLabel}
        className={permanentLabel ? "station-label" : undefined}
      >
        {renderTooltip?.(point) ?? label}
      </Tooltip>
      {popup && <Popup>{popup}</Popup>}
    </Marker>
  );
}

// memo conserva el tipo genérico del componente.
export default memo(StationMarker) as typeof StationMarker;
