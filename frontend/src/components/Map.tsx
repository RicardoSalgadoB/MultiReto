
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./Map.css";

import { scaleSequential } from "d3-scale";
import { interpolateViridis } from "d3-scale-chromatic";

export interface MapPoint {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  value: number;
}

interface MapProps {
  center?: [number, number];
  zoom?: number;
  points?: MapPoint[];
}

const threshold = 90;

export default function Map({
  center = [25.6866, -100.3161],
  zoom = 8,
  points = [],
}: MapProps) {
  const values = points.map((p) => p.value);

  const colorScale = scaleSequential(interpolateViridis)
    .domain([
      Math.min(0),
      Math.max(120),
    ]);

  return (
    <div className="map-wrapper">
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom={true}
        className="map-container"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        
        {points.map((point) => {
          const aboveThreshold = point.value > threshold;

          return (
            <CircleMarker
              key={point.id}
              center={[point.latitude, point.longitude]}
              radius={8}
              pathOptions={{
                fillColor: colorScale(point.value),
                fillOpacity: 0.85,

                // Conditional outline
                color: aboveThreshold ? "#000000" : "#ffffff",

                // Optional: thicker outline above threshold
                weight: aboveThreshold ? 3 : 1,
              }}
            >
              <Popup>
                <strong>{point.name}</strong>
                <br />
                Ozone: {point.value} ppb
              </Popup>
            </CircleMarker>
          );
        })}

      </MapContainer>
    </div>
  );
}
