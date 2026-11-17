import { memo, useId, useState, type CSSProperties } from "react";
import type { ClimateKey, ControlSpec } from "../../api/types";
import { formatFixed } from "../../lib/format";
import { CLIMATE_LIMITS } from "../../config/modelConstants";
import { CLIMATE_DECIMALS } from "./climateSpecs";
import "./ClimateControl.css";

interface ClimateControlProps {
  controlKey: ClimateKey;
  spec: ControlSpec;
  /** Valor actual del control. */
  value: number;
  /** Valor del día real (promedio de las 15 estaciones). */
  baseline: number;
  changed: boolean;
  /** Estaciones donde la API recortó una variable que depende de este control. */
  clippedStations: string[];
  /** Estaciones donde falta el dato y la API usó la mediana de entrenamiento. */
  completedStations: string[];
  totalStations: number;
  /** Errores de validación (422) de este campo. */
  issues: string[];
  onChange: (key: ClimateKey, value: number) => void;
  onReset: (key: ClimateKey) => void;
}

function ClimateControl({
  controlKey,
  spec,
  value,
  baseline,
  changed,
  clippedStations,
  completedStations,
  totalStations,
  issues,
  onChange,
  onReset,
}: ClimateControlProps) {
  const id = useId();
  const [limitLo, limitHi] = CLIMATE_LIMITS[controlKey];
  const decimals = CLIMATE_DECIMALS[controlKey];
  const hasTrainingRange = spec.min !== null && spec.max !== null;
  // El deslizador se acota al rango de entrenamiento si existe; el campo numérico llega hasta el límite de validación.
  const sliderLo = hasTrainingRange ? (spec.min as number) : limitLo;
  const sliderHi = hasTrainingRange ? (spec.max as number) : limitHi;
  const unit = spec.unidad;

  // Texto del campo numérico: se edita libremente y se confirma al terminar (así se puede escribir "-" o "3.").
  // Mientras no se edita, se muestra el valor actual (por ejemplo, el del deslizador).
  const plain = (v: number) => Number(v.toFixed(decimals)).toString();
  const [text, setText] = useState("");
  const [editing, setEditing] = useState(false);

  function commitText() {
    setEditing(false);
    const parsed = Number(text);
    if (text.trim() === "" || !Number.isFinite(parsed)) return;
    const clamped = Math.min(Math.max(parsed, limitLo), limitHi);
    // Salir del campo sin editar (el texto es el valor que ya se muestra) no confirma nada: no marca el control como modificado.
    if (clamped === Number(plain(value))) return;
    onChange(controlKey, clamped);
  }

  const fmt = (v: number) => formatFixed(v, decimals);

  return (
    <div className="climate-control" data-changed={changed ? "" : undefined}>
      <div className="control-head">
        <label htmlFor={`${id}-num`}>{spec.descripcion}</label>
        {changed && (
          <button type="button" className="link-button" onClick={() => onReset(controlKey)}>
            Volver a {fmt(baseline)} {unit}
          </button>
        )}
      </div>

      <div className="control-inputs">
        <input
          type="range"
          min={sliderLo}
          max={sliderHi}
          step="any"
          value={Math.min(Math.max(value, sliderLo), sliderHi)}
          style={{ "--fill": `${((Math.min(Math.max(value, sliderLo), sliderHi) - sliderLo) / (sliderHi - sliderLo)) * 100}%` } as CSSProperties}
          aria-label={`${spec.descripcion}, deslizador`}
          aria-valuetext={`${fmt(value)} ${unit}`}
          onChange={(event) => onChange(controlKey, Number(event.target.value))}
        />
        <span className="control-number">
          <input
            id={`${id}-num`}
            type="number"
            inputMode="decimal"
            min={limitLo}
            max={limitHi}
            step="any"
            value={editing ? text : plain(value)}
            aria-invalid={issues.length > 0 ? true : undefined}
            aria-describedby={`${id}-help`}
            onFocus={() => {
              setText(plain(value));
              setEditing(true);
            }}
            onChange={(event) => {
              setText(event.target.value);
              const parsed = Number(event.target.value);
              if (event.target.value.trim() !== "" && Number.isFinite(parsed) && parsed >= limitLo && parsed <= limitHi) onChange(controlKey, parsed);
            }}
            onBlur={commitText}
            onKeyDown={(event) => {
              if (event.key === "Enter") commitText();
            }}
          />
          <span className="control-unit">{unit}</span>
        </span>
      </div>

      <p id={`${id}-help`} className="control-help">
        {hasTrainingRange
          ? `Rango de entrenamiento: ${fmt(sliderLo)} a ${fmt(sliderHi)} ${unit}. `
          : `Sin rango de entrenamiento; límite permitido: ${fmt(limitLo)} a ${fmt(limitHi)} ${unit}. `}
        {hasTrainingRange && `Se acepta hasta ${fmt(limitLo)}–${fmt(limitHi)} ${unit}. `}
        Día real: {fmt(baseline)} {unit}.
      </p>

      {clippedStations.length > 0 && (
        <div className="callout callout--warn" title={`Estaciones: ${clippedStations.join(", ")}`}>
          <p className="callout-title">
            Recortada en {clippedStations.length} de {totalStations} estaciones
          </p>
          <p>Valor fuera del rango de entrenamiento: el modelo lo llevó al límite (las curvas no extrapolan).</p>
        </div>
      )}
      {completedStations.length > 0 && (
        <div className="callout callout--info" title={`Estaciones: ${completedStations.join(", ")}`}>
          <p className="callout-title">
            Dato faltante en {completedStations.length} de {totalStations}: mediana de entrenamiento
          </p>
        </div>
      )}

      {issues.length > 0 && (
        <p className="control-error" role="alert">
          {issues.join(" ")}
        </p>
      )}
    </div>
  );
}

function sameList(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((item, index) => item === b[index]);
}

/** Compara las listas por contenido: se recrean en cada predicción aunque no cambien. */
function areEqual(prev: ClimateControlProps, next: ClimateControlProps): boolean {
  return (
    prev.controlKey === next.controlKey &&
    prev.spec === next.spec &&
    prev.value === next.value &&
    prev.baseline === next.baseline &&
    prev.changed === next.changed &&
    prev.totalStations === next.totalStations &&
    prev.onChange === next.onChange &&
    prev.onReset === next.onReset &&
    sameList(prev.clippedStations, next.clippedStations) &&
    sameList(prev.completedStations, next.completedStations) &&
    sameList(prev.issues, next.issues)
  );
}

export default memo(ClimateControl, areEqual);
