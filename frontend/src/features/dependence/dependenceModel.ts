import type { LatLngTuple } from "leaflet";
import type { DependencePair, DependenceResponse, StationCode } from "../../api/types";
import { LOW_JOINT_DAYS_QUANTILE } from "../../config/modelConstants";
import { formatNumber } from "../../lib/format";
import { lowerQuantile } from "../../lib/stats";

/** Un par de estaciones listo para dibujar y listar. */
export interface PairView {
  /** Clave estable del par: "NO|NTE". */
  key: string;
  a: StationCode;
  b: StationCode;
  /** null = sin valor: el par no se dibuja. */
  valor: number | null;
  distanciaKm: number;
  diasConjuntos: number;
  /** Está entre el 25 % de pares con menos días conjuntos (criterio relativo, no de confiabilidad). */
  low: boolean;
  /** Extremos de la línea [lat, lon]; la misma referencia mientras no cambie la respuesta (react-leaflet la compara). */
  positions: LatLngTuple[];
}

export interface DependenceModel {
  pairs: PairView[];
  byKey: Map<string, PairView>;
  /** Pares de cada estación, del más fuerte al más débil (sin valor al final). */
  byStation: Map<StationCode, PairView[]>;
  /** Todos los pares, del más fuerte al más débil. */
  ranked: PairView[];
  /** Un par tiene "pocos días conjuntos" si su `dias_conjuntos` es menor que esto. */
  lowThreshold: number;
  /** "Entre el 25 % de pares con menos días conjuntos (< 17)": el criterio es relativo, no una medida de confiabilidad. */
  lowLabel: string;
  lowCount: number;
  /** Menor y mayor `dias_conjuntos` de la respuesta. */
  minDays: number;
  maxDays: number;
  /** Cuantil `q` de la respuesta (días conjuntos = días en que ambas estaciones superan ese cuantil). */
  q: number;
  nullCount: number;
  /** Nombre largo de cada estación. */
  names: Map<StationCode, string>;
}

export function otherStation(pair: PairView, code: StationCode): StationCode {
  return pair.a === code ? pair.b : pair.a;
}

function compare(x: PairView, y: PairView): number {
  if (x.valor === null && y.valor === null) return x.key.localeCompare(y.key);
  if (x.valor === null) return 1;
  if (y.valor === null) return -1;
  return y.valor - x.valor || x.key.localeCompare(y.key);
}

/** Construye los pares una sola vez por respuesta: las coordenadas salen de `estaciones` de la misma respuesta. */
export function buildDependenceModel(data: DependenceResponse): DependenceModel {
  const coords = new Map<StationCode, LatLngTuple>(data.estaciones.map((s) => [s.name, [s.latitude, s.longitude]]));
  const names = new Map<StationCode, string>(data.estaciones.map((s) => [s.name, s.nombre]));
  const sortedDays = data.pares.map((p) => p.dias_conjuntos).sort((x, y) => x - y);
  const lowThreshold = sortedDays.length > 0 ? lowerQuantile(sortedDays, LOW_JOINT_DAYS_QUANTILE) : 0;

  const pairs: PairView[] = data.pares.flatMap((p: DependencePair) => {
    const from = coords.get(p.a);
    const to = coords.get(p.b);
    if (!from || !to) return [];
    return [
      {
        key: `${p.a}|${p.b}`,
        a: p.a,
        b: p.b,
        valor: p.valor,
        distanciaKm: p.distancia_km,
        diasConjuntos: p.dias_conjuntos,
        low: p.dias_conjuntos < lowThreshold,
        positions: [from, to],
      },
    ];
  });

  const byStation = new Map<StationCode, PairView[]>();
  for (const pair of pairs) {
    for (const code of [pair.a, pair.b]) byStation.set(code, [...(byStation.get(code) ?? []), pair]);
  }
  for (const list of byStation.values()) list.sort(compare);

  return {
    pairs,
    lowLabel: `Entre el ${formatNumber(LOW_JOINT_DAYS_QUANTILE * 100)} % de pares con menos días conjuntos (< ${lowThreshold})`,
    minDays: sortedDays[0] ?? 0,
    maxDays: sortedDays[sortedDays.length - 1] ?? 0,
    q: data.q,
    byKey: new Map(pairs.map((p) => [p.key, p])),
    byStation,
    ranked: [...pairs].sort(compare),
    lowThreshold,
    lowCount: pairs.filter((p) => p.low).length,
    nullCount: pairs.filter((p) => p.valor === null).length,
    names,
  };
}
