import { useId, useState } from "react";
import type { BaseScenarioResponse, ClimateKey, HistoricRange } from "../../api/types";
import { formatDay, isIsoDate } from "../../lib/dates";
import ClimateControl from "./ClimateControl";
import { CLIMATE_ORDER } from "./climateSpecs";
import WindDial from "./WindDial";
import "./ScenarioSidebar.css";

interface ScenarioSidebarProps {
  scenario: BaseScenarioResponse;
  range: HistoricRange;
  /** Fecha base pedida (puede ir por delante de `scenario` mientras carga). */
  baseDate: string;
  baseLoading: boolean;
  onBaseDateChange: (date: string) => void;
  values: Record<ClimateKey, number>;
  baseline: Record<ClimateKey, number>;
  changedKeys: ReadonlySet<ClimateKey>;
  clippedBy: Record<ClimateKey, string[]>;
  completedBy: Record<ClimateKey, string[]>;
  issuesByKey: Partial<Record<ClimateKey, string[]>>;
  totalStations: number;
  /** Estas funciones deben ser estables (useCallback). */
  onChangeValue: (key: ClimateKey, value: number) => void;
  onResetKey: (key: ClimateKey) => void;
  onResetAll: () => void;
}

/** Sidebar del Tab 2: fecha base, clima de todas las estaciones y dial de viento (colapsable en pantallas angostas). */
export default function ScenarioSidebar({
  scenario,
  range,
  baseDate,
  baseLoading,
  onBaseDateChange,
  values,
  baseline,
  changedKeys,
  clippedBy,
  completedBy,
  issuesByKey,
  totalStations,
  onChangeValue,
  onResetKey,
  onResetAll,
}: ScenarioSidebarProps) {
  const contentId = useId();
  const [open, setOpen] = useState(false);
  // La dirección del viento es solo informativa: estado local que nunca se envía a la API.
  const [windDirection, setWindDirection] = useState(0);

  const changed = changedKeys.size;
  const summary = changed === 0 ? "sin cambios" : `${changed} ${changed === 1 ? "modificado" : "modificados"}`;

  return (
    <div className="scenario-sidebar">
      <button type="button" className="button sidebar-toggle" aria-expanded={open} aria-controls={contentId} onClick={() => setOpen((value) => !value)}>
        <span className="toggle-text">
          Controles del escenario
          <span className="toggle-count">{summary}</span>
        </span>
        <svg className="toggle-chevron" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
          <path d="m3 6 5 5 5-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      <h2 className="sidebar-title">Controles del escenario</h2>
      <p className="sidebar-count" aria-live="polite">
        {changed === 0 ? "Sin cambios respecto al día real" : `${changed} ${changed === 1 ? "control modificado" : "controles modificados"}`}
      </p>

      <div id={contentId} className="sidebar-content" data-open={open ? "" : undefined}>
        <label className="base-field">
          <span>Fecha base (el «hoy» del escenario)</span>
          <input
            type="date"
            value={baseDate}
            min={range.inicio}
            max={range.fin}
            onChange={(event) => {
              const next = event.target.value;
              if (isIsoDate(next) && next >= range.inicio && next <= range.fin) onBaseDateChange(next);
            }}
          />
        </label>
        <p className="control-help" aria-live="polite">
          Se predice el máximo de {formatDay(scenario.fecha_objetivo)} con los datos de ese día.{baseLoading ? " Cargando fecha…" : ""}
        </p>

        <button type="button" className="button reset-button" onClick={onResetAll} disabled={changed === 0}>
          Restablecer al día real
        </button>

        <p className="control-help sidebar-intro">
          Los valores iniciales son el promedio de las 15 estaciones del día real. Un control aplica su valor a todas las estaciones; lo demás (ozono de
          hoy, etc.) se queda como en el día real.
        </p>

        <div className="controls-list">
          {CLIMATE_ORDER.map((key) => (
            <ClimateControl
              key={key}
              controlKey={key}
              spec={scenario.controles[key]}
              value={values[key]}
              baseline={baseline[key]}
              changed={changedKeys.has(key)}
              clippedStations={clippedBy[key]}
              completedStations={completedBy[key]}
              totalStations={totalStations}
              issues={issuesByKey[key] ?? NO_ISSUES}
              onChange={onChangeValue}
              onReset={onResetKey}
            />
          ))}
        </div>

        <WindDial value={windDirection} onChange={setWindDirection} />
      </div>
    </div>
  );
}

const NO_ISSUES: string[] = [];
