import type { CSSProperties } from "react";
import type { HistoricRange } from "../../api/types";
import { clampMoment, formatHour, formatMoment, isIsoDate, rangeBounds, shiftMoment, toUtcMs, type Moment } from "../../lib/dates";
import "./TimeControls.css";

interface TimeControlsProps {
  range: HistoricRange;
  moment: Moment;
  playing: boolean;
  /** Cambio hecho por el usuario (detiene la reproducción). */
  onChange: (moment: Moment) => void;
  onTogglePlay: () => void;
}

/** Fecha, hora (0–23), saltos de ±1 h y ±1 día, y reproducción automática. */
export default function TimeControls({ range, moment, playing, onChange, onTogglePlay }: TimeControlsProps) {
  const { first, last } = rangeBounds(range);
  const current = toUtcMs(moment);
  const atStart = current <= first;
  const atEnd = current >= last;

  const jump = (hours: number) => onChange(shiftMoment(moment, hours, range));

  return (
    <div className="time-controls">
      <p className="time-current">{formatMoment(moment)}</p>

      <div className="time-row">
        <label className="time-field">
          <span>Fecha</span>
          <input
            type="date"
            value={moment.fecha}
            min={range.inicio}
            max={range.fin}
            onChange={(event) => {
              // El campo se vacía o queda a medias mientras se escribe: solo se aceptan fechas completas.
              if (isIsoDate(event.target.value)) onChange(clampMoment({ ...moment, fecha: event.target.value }, range));
            }}
          />
        </label>

        <label className="time-field time-field--slider">
          <span>
            Hora: <strong className="time-hour">{formatHour(moment.hora)}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={23}
            step={1}
            value={moment.hora}
            style={{ "--fill": `${(moment.hora / 23) * 100}%` } as CSSProperties}
            aria-valuetext={formatHour(moment.hora)}
            onChange={(event) => onChange(clampMoment({ ...moment, hora: Number(event.target.value) }, range))}
          />
        </label>

        <div className="time-buttons">
          <div className="time-jumps" role="group" aria-label="Moverse en el tiempo">
            <button type="button" className="button" onClick={() => jump(-24)} disabled={atStart}>
              −1 día
            </button>
            <button type="button" className="button" onClick={() => jump(-1)} disabled={atStart}>
              −1 h
            </button>
            <button type="button" className="button" onClick={() => jump(1)} disabled={atEnd}>
              +1 h
            </button>
            <button type="button" className="button" onClick={() => jump(24)} disabled={atEnd}>
              +1 día
            </button>
          </div>
          <button type="button" className="button" onClick={onTogglePlay} disabled={atEnd && !playing} aria-pressed={playing}>
            <svg className="button-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
              {playing ? <path d="M3.5 2h3.2v12H3.5V2Zm5.8 0h3.2v12H9.3V2Z" /> : <path d="M4 2.2 13.4 8 4 13.8V2.2Z" />}
            </svg>
            {playing ? "Pausar" : "Reproducir"}
          </button>
        </div>
      </div>
    </div>
  );
}
