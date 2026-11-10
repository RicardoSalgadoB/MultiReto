
import { createBrowserRouter } from "react-router";

import AnalyticsMap from "./pages/AnalyticsMap";
import PredictionMap from "./pages/PredictionMap";
import DependenceMap from "./pages/DependenceMap";

import App from "./layouts/App";


const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        index: true,
        element: <AnalyticsMap />,
      },
      {
        path: "prediction",
        element: <PredictionMap />,
      },
      {
        path: "dependence",
        element: <DependenceMap />,
      },
    ],
  },
]);

export default router;
