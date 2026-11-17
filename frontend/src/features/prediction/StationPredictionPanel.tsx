import type { Ref } from "react";
import type { BaseScenarioResponse, PredictionPoint, PredictionResponse } from "../../api/types";
import Verdict from "../../components/ui/Verdict";
import "../../components/ui/Panel.css";
import { DERIVED_VARIABLE_LABELS } from "../../config/modelConstants";
import { formatDay } from "../../lib/dates";
import { formatFixed, formatNumber, formatPercent } from "../../lib/format";
import GamCurvesBlock from "./GamCurvesBlock";
import "./StationPredictionPanel.css";

interface StationPredictionPanelProps {
  point: PredictionPoint;
  prediction: PredictionResponse;
  controles: BaseScenarioResponse["controles"];
  /** El panel muestra lo que se calculó para este momento; true si hay un recálculo en curso. */
  recalculating: boolean;
  onClose: () => void;
  ref?: Ref<HTMLElement>;
}

function describe(variable: string, controles: BaseScenarioResponse["controles"]): string {
  const known = (controles as Record<string, { descripcion: string } | undefined>)[variable];
  return known?.descripcion ?? DERIVED_VARIABLE_LABELS[variable] ?? variable;
}

export default function StationPredictionPanel({ point, prediction, controles, recalculating, onClose, ref }: StationPredictionPanelProps) {
  const { u_ppb: u, norma_ppb: norma, corte_alerta: corte, xi } = prediction;
  const levels = Object.entries(point.probabilidades)
    .map(([level, probability]) => ({ level: Number(level), probability }))
    .sort((a, b) => a.level - b.level);

  const pct = (value: number) => `${Math.min(Math.max(value, 0), 1) * 100}%`;

  return (
    <section className="panel prediction-panel" ref={ref} tabIndex={-1} aria-labelledby="prediction-panel-title" aria-busy={recalculating}>
      <header className="panel-head">
        <div className="panel-title">
          <h2 id="prediction-panel-title">
            {point.nombre} <span className="panel-code">({point.name})</span>
          </h2>
          <p className="panel-note" aria-live="polite">
            Predicción del máximo del {formatDay(prediction.fecha_objetivo)}, con datos del {formatDay(prediction.fecha_base)}.
            {recalculating ? " Recalculando…" : ""}
          </p>
        </div>
        <div className="panel-actions">
          <Verdict alert={point.alerta} />
          <button type="button" className="button" onClick={onClose}>
            Cerrar
          </button>
        </div>
      </header>

      <div className="readouts">
        <section className="readout" aria-labelledby="stage1-title">
          <h3 id="stage1-title">Etapa 1 · GAM</h3>
          <p className="readout-label">ζ = P(máximo &gt; {formatNumber(u)} ppb)</p>
          <p className="readout-value">
            <span className="sr-only">= </span>
            {point.zeta === null ? "sin dato" : formatPercent(point.zeta)}
          </p>
          {point.zeta !== null && (
            <div className="meter" aria-hidden="true">
              <div className="meter-fill" style={{ width: pct(point.zeta) }} />
            </div>
          )}
          <p className="readout-help">Probabilidad de que mañana haya un episodio (el máximo supere el umbral auxiliar del modelo).</p>
        </section>

        <section className="readout" aria-labelledby="stage2-title">
          <h3 id="stage2-title">Etapa 2 · GPD</h3>
          <p className="readout-label">P(máximo &gt; {formatNumber(norma)} ppb)</p>
          <p className="readout-value">
            <span className="sr-only">= </span>
            {point.p90 === null ? "sin dato" : formatPercent(point.p90)}
          </p>
          {point.p90 !== null && (
            <div className="meter" aria-hidden="true">
              <div className="meter-fill" style={{ width: pct(point.p90) }} />
              <div className="meter-mark" style={{ left: pct(corte) }} />
            </div>
          )}
          <p className="readout-help">
            Frente al corte de alerta: <strong>{formatPercent(corte)}</strong> (marca magenta en la barra).
          </p>
        </section>
      </div>

      <section className="formula-card" aria-labelledby="formula-title">
        <h3 id="formula-title" className="panel-section-title">
          Cómo se combinan las dos etapas
        </h3>
        <p className="formula" aria-label="Fórmula">
          <span className="formula-lhs">P</span> = <span className="formula-term">ζ</span> · [1 + <span className="formula-term">ξ</span> · ({formatNumber(norma)} −{" "}
          {formatNumber(u)}) / <span className="formula-term">σ</span>]
          <sup>−1/ξ</sup>
        </p>
        <dl className="formula-values">
          <div>
            <dt>ζ</dt>
            <dd>{point.zeta === null ? "—" : formatPercent(point.zeta)}</dd>
          </div>
          <div>
            <dt>σ</dt>
            <dd>{point.sigma === null ? "—" : `${formatFixed(point.sigma, 1)} ppb`}</dd>
          </div>
          <div>
            <dt>ξ</dt>
            <dd>{formatFixed(xi, 3)}</dd>
          </div>
        </dl>
        <p className="readout-help">ξ es común a todas las estaciones; σ depende de ζ.</p>
      </section>

      <div className="readouts">
        <section className="readout" aria-labelledby="levels-title">
          <h3 id="levels-title">Probabilidad por nivel</h3>
          <ul className="levels">
            {levels.map(({ level, probability }) => (
              <li key={level} data-norm={level === norma ? "" : undefined}>
                <span className="level-name">
                  P(máximo &gt; {formatNumber(level)} ppb){level === norma ? " · norma" : ""}
                </span>
                <strong className="level-value">{probability === null ? "sin dato" : formatPercent(probability)}</strong>
              </li>
            ))}
          </ul>
        </section>

        <section className="readout" aria-labelledby="alert-title">
          <h3 id="alert-title">Alerta frente al corte</h3>
          <p className="readout-label">{point.p90 === null ? "Sin predicción" : <>P(máximo &gt; {formatNumber(norma)} ppb) = {formatPercent(point.p90)}</>}</p>
          {point.p90 !== null && (
            <p className="readout-value" data-size="md">
              {formatPercent(point.p90)} {point.alerta ? "≥" : "<"} {formatPercent(corte)}
            </p>
          )}
          <p className="readout-help readout-help--strong">
            Corte de alerta: P ≥ {formatPercent(corte)}.{" "}
            <strong>{point.alerta ? "Supera el corte: se emite alerta." : "No supera el corte: sin alerta."}</strong>
          </p>
        </section>
      </div>

      {point.recortadas.length > 0 && (
        <div className="callout callout--warn">
          <p className="callout-title">Recortadas</p>
          <p>
            Valor fuera del rango de entrenamiento: el modelo lo llevó al límite. {point.recortadas.map((v) => describe(v, controles)).join("; ")}.
          </p>
        </div>
      )}
      {point.completadas.length > 0 && (
        <div className="callout callout--info">
          <p className="callout-title">Completadas</p>
          <p>Dato faltante: se usó la mediana de entrenamiento. {point.completadas.map((v) => describe(v, controles)).join("; ")}.</p>
        </div>
      )}

      <GamCurvesBlock point={point} />
    </section>
  );
}
