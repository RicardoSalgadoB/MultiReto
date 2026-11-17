import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import type { DependenceKind, StationCode } from "../../api/types";

export type DependenceMode = "spearman" | "colas";
export type TailMeasure = "chi" | "chibar";

function parseMode(value: string | null): DependenceMode {
  return value === "spearman" || value === "colas" ? value : "colas";
}

function parseMeasure(value: string | null): TailMeasure {
  return value === "chi" || value === "chibar" ? value : "chi";
}

function parseStation(value: string | null, stationNames: StationCode[]): StationCode | null {
  return value !== null && stationNames.includes(value) ? value : null;
}

/**
 * Estado del Tab 3 en la URL: ?modo=spearman|colas, ?medida=chi|chibar (solo con modo colas) y ?estacion=CLAVE.
 * Cada parámetro inválido cae a su valor por defecto (colas, chi, sin estación) sin error.
 */
export function useDependenceState(stationNames: StationCode[]) {
  const [params, setParams] = useSearchParams();
  const [mode, setMode] = useState<DependenceMode>(() => parseMode(params.get("modo")));
  const [measure, setMeasure] = useState<TailMeasure>(() => parseMeasure(params.get("medida")));
  const [station, setStation] = useState<StationCode | null>(() => parseStation(params.get("estacion"), stationNames));

  const setParamsRef = useRef(setParams);
  useEffect(() => {
    setParamsRef.current = setParams;
  });
  const urlMode = params.get("modo");
  const urlMeasure = params.get("medida");
  const urlStation = params.get("estacion");
  useEffect(() => {
    const expectedMeasure = mode === "colas" ? measure : null;
    if (urlMode === mode && urlMeasure === expectedMeasure && urlStation === station) return;
    const next: Record<string, string> = { modo: mode };
    if (mode === "colas") next.medida = measure;
    if (station !== null) next.estacion = station;
    setParamsRef.current(next, { replace: true });
  }, [mode, measure, station, urlMode, urlMeasure, urlStation]);

  /** Seleccionar la estación ya seleccionada la deselecciona. */
  const toggleStation = useCallback((code: StationCode) => setStation((prev) => (prev === code ? null : code)), []);
  const clearStation = useCallback(() => setStation(null), []);

  const kind: DependenceKind = mode === "spearman" ? "spearman" : measure;
  return { mode, measure, station, kind, setMode, setMeasure, toggleStation, clearStation };
}
