import { useId, useRef, type Dispatch, type KeyboardEvent, type PointerEvent, type SetStateAction } from "react";
import "./WindDial.css";

const ROSE = ["Norte", "Noreste", "Este", "Sureste", "Sur", "Suroeste", "Oeste", "Noroeste"];
const SIZE = 148;
const CENTER = SIZE / 2;
const RADIUS = 52;

/** Normaliza a [0, 360). */
function wrap(degrees: number): number {
  return ((Math.round(degrees) % 360) + 360) % 360;
}

/** "Noreste, 45 grados" */
function windDirectionText(degrees: number): string {
  const value = wrap(degrees);
  return `${ROSE[Math.round(value / 45) % 8]}, ${value} grados`;
}

interface WindDialProps {
  /** Grados desde el norte, en sentido horario (0–359). */
  value: number;
  /** Acepta un valor o una función del valor anterior (así varias pulsaciones seguidas no se pisan). */
  onChange: Dispatch<SetStateAction<number>>;
}

/**
 * Dial de dirección del viento: se arrastra con el puntero (mouse, lápiz o dedo) o se mueve con el teclado.
 * Es solo informativo: el modelo no usa la dirección del viento y este valor nunca se envía a la API.
 */
export default function WindDial({ value, onChange }: WindDialProps) {
  const labelId = useId();
  const dragging = useRef(false);
  const angle = wrap(value);

  function angleFromPointer(event: PointerEvent<SVGSVGElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const dx = event.clientX - (rect.left + rect.width / 2);
    const dy = event.clientY - (rect.top + rect.height / 2);
    // 0° = arriba (norte); crece en sentido horario.
    return wrap((Math.atan2(dx, -dy) * 180) / Math.PI);
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.parentElement?.focus();
    onChange(angleFromPointer(event));
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (dragging.current) onChange(angleFromPointer(event));
  }

  function handlePointerUp(event: PointerEvent<SVGSVGElement>) {
    dragging.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    let step: (previous: number) => number;
    switch (event.key) {
      case "ArrowRight":
      case "ArrowUp":
        step = (previous) => wrap(previous + 1);
        break;
      case "ArrowLeft":
      case "ArrowDown":
        step = (previous) => wrap(previous - 1);
        break;
      case "PageUp":
        step = (previous) => wrap(previous + 15);
        break;
      case "PageDown":
        step = (previous) => wrap(previous - 15);
        break;
      case "Home":
      case "End":
        step = () => 0; // norte
        break;
      default:
        return;
    }
    event.preventDefault();
    onChange(step);
  }

  const ticks = Array.from({ length: 16 }, (_, i) => i * 22.5);
  const rad = (angle * Math.PI) / 180;
  const tipX = CENTER + Math.sin(rad) * (RADIUS - 6);
  const tipY = CENTER - Math.cos(rad) * (RADIUS - 6);

  return (
    <div className="wind-dial">
      <p id={labelId} className="wind-dial-label">
        Dirección del viento
      </p>
      <div
        className="wind-dial-control"
        role="slider"
        tabIndex={0}
        aria-labelledby={labelId}
        aria-valuemin={0}
        aria-valuemax={359}
        aria-valuenow={angle}
        aria-valuetext={windDirectionText(angle)}
        aria-orientation="horizontal"
        onKeyDown={handleKeyDown}
      >
        <svg
          className="wind-dial-svg"
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          width={SIZE}
          height={SIZE}
          aria-hidden="true"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        >
          <circle className="dial-face" cx={CENTER} cy={CENTER} r={RADIUS + 12} />
          {ticks.map((deg) => {
            const r = (deg * Math.PI) / 180;
            const major = deg % 90 === 0;
            const inner = RADIUS + (major ? 1 : 5);
            return (
              <line
                key={deg}
                className={major ? "dial-tick dial-tick--major" : "dial-tick"}
                x1={CENTER + Math.sin(r) * inner}
                y1={CENTER - Math.cos(r) * inner}
                x2={CENTER + Math.sin(r) * (RADIUS + 10)}
                y2={CENTER - Math.cos(r) * (RADIUS + 10)}
              />
            );
          })}
          <text className="dial-letter" x={CENTER} y={15} textAnchor="middle">
            N
          </text>
          <text className="dial-letter" x={SIZE - 8} y={CENTER + 4} textAnchor="middle">
            E
          </text>
          <text className="dial-letter" x={CENTER} y={SIZE - 6} textAnchor="middle">
            S
          </text>
          <text className="dial-letter" x={8} y={CENTER + 4} textAnchor="middle">
            O
          </text>
          <line className="dial-needle" x1={CENTER} y1={CENTER} x2={tipX} y2={tipY} />
          <circle className="dial-tip" cx={tipX} cy={tipY} r={6} />
          <circle className="dial-hub" cx={CENTER} cy={CENTER} r={4} />
        </svg>
      </div>
      <p className="wind-dial-value" aria-hidden="true">
        {windDirectionText(angle)}
      </p>
      <div className="callout callout--warn wind-dial-note">
        <p>
          <strong>No afecta la predicción:</strong> el modelo no usa dirección del viento. Este valor no se envía a la API.
        </p>
      </div>
    </div>
  );
}
