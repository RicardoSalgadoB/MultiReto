import { createContext, useContext } from "react";
import type { ApiInfo, Station } from "../api/types";

/** Datos que se cargan una sola vez y que usan las tres pestañas. */
export interface AppData {
  stations: Station[];
  /** Umbrales y corte de alerta vigentes (siempre los de la API, nunca constantes copiadas). */
  info: ApiInfo;
}

export const AppDataContext = createContext<AppData | null>(null);

export function useAppData(): AppData {
  const value = useContext(AppDataContext);
  if (value === null) throw new Error("useAppData debe usarse dentro de <AppDataProvider>.");
  return value;
}
