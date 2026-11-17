import { useMemo, type ReactNode } from "react";
import { getInfo, getStations } from "../api/general";
import { ErrorView, LoadingView } from "../components/layout/StatusViews";
import { useApi } from "../hooks/useApi";
import { AppDataContext } from "./app-data-context";

/** Carga estaciones y umbrales una sola vez; mientras tanto (o si falla) muestra el estado en lugar de las páginas. */
export default function AppDataProvider({ children }: { children: ReactNode }) {
  const stations = useApi(getStations);
  const info = useApi(getInfo);

  const value = useMemo(
    () => (stations.data && info.data ? { stations: stations.data, info: info.data } : null),
    [stations.data, info.data],
  );

  const error = stations.error ?? info.error;
  if (error) {
    return (
      <ErrorView
        error={error}
        onRetry={() => {
          stations.reload();
          info.reload();
        }}
      />
    );
  }
  if (value === null) return <LoadingView label="Conectando con el servicio de predicciones…" />;

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}
