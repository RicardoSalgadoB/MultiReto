import { useCallback, useState } from "react";
import { getGamCurvesCached, peekGamCurves } from "../../api/gamCurvesCache";
import type { ModelInputKey, PredictionPoint } from "../../api/types";
import { ErrorView, LoadingView } from "../../components/layout/StatusViews";
import { useApi } from "../../hooks/useApi";
import { DERIVED_VARIABLES } from "../../config/modelConstants";
import { formatNumber2 } from "../../lib/format";
import SplineChart, { type CurveMode } from "./SplineChart";
import "./GamCurvesBlock.css";

function inputOf(point: PredictionPoint, key: ModelInputKey): number | null {
  if (!Object.prototype.hasOwnProperty.call(point.entradas, key)) return null;
  const value = point.entradas[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Valor actual de la variable en la estación. Las variables que están en `entradas` se leen tal cual; `dT_pron` y
 * `rango_pron` no vienen, así que se calculan con las fórmulas del backend (backend/source/predict.py,
 * `ModeloOzono.completar_variables`): dT_pron = T_max_pron − T_max_hoy y rango_pron = T_max_pron − T_min_pron.
 */
function currentValueOf(point: PredictionPoint, variable: string): number | null {
  const derived = DERIVED_VARIABLES[variable];
  if (derived) {
    const minuend = inputOf(point, derived.minuend);
    const subtrahend = inputOf(point, derived.subtrahend);
    return minuend === null || subtrahend === null ? null : minuend - subtrahend;
  }
  return inputOf(point, variable as ModelInputKey);
}

/** Las 11 curvas del GAM de una estación. Se piden una vez por estación y se guardan. */
export default function GamCurvesBlock({ point }: { point: PredictionPoint }) {
  const station = point.name;
  const [mode, setMode] = useState<CurveMode>("logit");

  // Si ya se pidieron, se usan al instante y no se hace ninguna petición.
  const cached = peekGamCurves(station);
  const fetcher = useCallback((signal: AbortSignal) => getGamCurvesCached(station, signal), [station]);
  const api = useApi(fetcher, { enabled: cached === null, keepPreviousData: true });

  const curves = cached ?? api.data;
  // En logit la forma es la misma para todas las estaciones; en probabilidad depende de la estación.
  const stale = curves !== null && curves.estacion !== station;
  const loading = cached === null && api.loading;

  let body;
  if (cached === null && api.error) {
    body = <ErrorView error={api.error} onRetry={api.reload} title="No se pudieron cargar las curvas" />;
  } else if (curves === null || (stale && mode === "probabilidad")) {
    body = <LoadingView label="Cargando las curvas del GAM…" />;
  } else {
    const alpha = curves.efecto_estacion[station];
    body = (
      <>
        <p className="curves-note">
          La forma de cada curva es <strong>común a las 15 estaciones</strong> (es un solo GAM): la estación solo la sube o baja con su efecto fijo.
          {alpha !== undefined && (
            <>
              {" "}
              Efecto de {station}: <strong>{alpha > 0 ? "+" : ""}{formatNumber2(alpha)}</strong> en logit ζ respecto a {curves.referencia} (referencia = 0).
            </>
          )}{" "}
          La línea vertical marca el valor actual de la variable en {station}; con un valor fuera del rango de la curva, el modelo lo lleva al límite.
        </p>
        <div className="curves-grid" aria-busy={loading} data-loading={loading ? "" : undefined}>
          {Object.entries(curves.curvas).map(([variable, curve]) => (
            <SplineChart
              key={variable}
              curve={curve}
              mode={mode}
              station={station}
              currentValue={currentValueOf(point, variable)}
              derivedNote={variable in DERIVED_VARIABLES ? "Valor actual calculado con las entradas de la estación." : undefined}
            />
          ))}
        </div>
      </>
    );
  }

  return (
    <section className="curves-block" aria-labelledby={`curves-title-${station}`}>
      <div className="curves-head">
        <h3 id={`curves-title-${station}`}>Curvas del GAM (etapa 1)</h3>
        <div className="mode-toggle" role="group" aria-label="Qué mostrar en las curvas">
          <button type="button" className="button" aria-pressed={mode === "logit"} onClick={() => setMode("logit")}>
            Efecto en logit ζ
          </button>
          <button type="button" className="button" aria-pressed={mode === "probabilidad"} onClick={() => setMode("probabilidad")}>
            Probabilidad en {station}
          </button>
        </div>
      </div>
      {body}
    </section>
  );
}
