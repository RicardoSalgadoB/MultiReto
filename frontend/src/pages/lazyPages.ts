import { lazy } from "react";

// Carga diferida por ruta: cada pestaña va en su propio chunk (y Leaflet, que solo ellas usan, en el suyo).
export const AnalyticsMap = lazy(() => import("./AnalyticsMap"));
export const PredictionMap = lazy(() => import("./PredictionMap"));
export const DependenceMap = lazy(() => import("./DependenceMap"));
