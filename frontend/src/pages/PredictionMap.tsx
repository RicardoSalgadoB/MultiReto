import Map, { type MapPoint } from "../components/Map";

const points: MapPoint[] = [
  { id: 1, name: "CE", latitude: 25.675833, longitude: -100.338333, value: 30 },
  { id: 2, name: "NE", latitude: 25.744722, longitude: -100.253056, value: 30 },
  { id: 3, name: "NE2", latitude: 25.777222, longitude: -100.188056, value: 30 },
  { id: 4, name: "NE3", latitude: 25.790556, longitude: -100.078333, value: 30 },
  { id: 5, name: "NO", latitude: 25.762778, longitude: -100.369167, value: 30 },
  { id: 6, name: "NO2", latitude: 25.800278, longitude: -100.584444, value: 30 },
  { id: 7, name: "NO3", latitude: 25.768333, longitude: -100.463611, value: 30 },
  { id: 8, name: "NTE", latitude: 25.798611, longitude: -100.327222, value: 30 },
  { id: 9, name: "NTE2", latitude: 25.729722, longitude: -100.310000, value: 30 },
  { id: 10, name: "SE", latitude: 25.665278, longitude: -100.243611, value: 70 },
  { id: 11, name: "SE2", latitude: 25.645833, longitude: -100.095278, value: 30 },
  { id: 12, name: "SE3", latitude: 25.601111, longitude: -99.999167, value: 30 },
  { id: 13, name: "SO", latitude: 25.675556, longitude: -100.458056, value: 80 },
  { id: 14, name: "SO2", latitude: 25.665000, longitude: -100.412778, value: 30 },
  { id: 15, name: "SUR", latitude: 25.616944, longitude: -100.273889, value: 100 },
];



export default function PredictionsMap() {
  return (
    <div>
      <h1>Mapa de Predicciones</h1>

      <Map
        center={[25.70, -100.30]}
        zoom={11}
        points={points}
      />
    </div>
  );
}