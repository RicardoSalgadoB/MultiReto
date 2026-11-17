import { createBrowserRouter } from "react-router";

import App from "./layouts/App";
import DataLayout from "./layouts/DataLayout";
import NotFound from "./pages/NotFound";
import { AnalyticsMap, DependenceMap, PredictionMap } from "./pages/lazyPages";
import RouteError from "./pages/RouteError";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    errorElement: <RouteError />,
    children: [
      {
        element: <DataLayout />,
        children: [
          { index: true, element: <AnalyticsMap /> },
          { path: "prediction", element: <PredictionMap /> },
          { path: "dependence", element: <DependenceMap /> },
        ],
      },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

export default router;
