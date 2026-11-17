import { Outlet } from "react-router";
import AppDataProvider from "../context/AppDataProvider";

/** Las tres pestañas con datos cargan estaciones y umbrales una sola vez; la página 404 queda fuera (no depende de la API). */
export default function DataLayout() {
  return (
    <AppDataProvider>
      <Outlet />
    </AppDataProvider>
  );
}
