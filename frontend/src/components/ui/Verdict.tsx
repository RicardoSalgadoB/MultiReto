import "./Verdict.css";

interface VerdictProps {
  /** true = alerta; false = sin alerta. */
  alert: boolean;
  /** "lg" para el encabezado de un panel; "md" dentro de una celda. */
  size?: "md" | "lg";
}

/**
 * Veredicto de alerta. Nunca depende del color: "Alerta" lleva relleno de tinta invertida y un triángulo; "Sin alerta",
 * solo contorno y un círculo. El texto siempre dice el estado.
 */
export default function Verdict({ alert, size = "lg" }: VerdictProps) {
  return (
    <span className="verdict" data-state={alert ? "alert" : "clear"} data-size={size}>
      <svg className="verdict-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        {alert ? (
          <path d="M8 1.6 15.2 14.4H0.8L8 1.6Z" fill="currentColor" />
        ) : (
          <circle cx="8" cy="8" r="5.6" fill="none" stroke="currentColor" strokeWidth="1.6" />
        )}
      </svg>
      <span>{alert ? "Alerta" : "Sin alerta"}</span>
    </span>
  );
}
