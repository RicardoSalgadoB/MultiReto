import type { ReactNode } from "react";
import type { ApiError } from "../../api/client";
import "./StatusViews.css";

/** Iconos propios (SVG en línea, decorativos): el estado siempre se dice también con texto. */
function ErrorIcon() {
  return (
    <svg className="status-icon" viewBox="0 0 24 24" width="28" height="28" aria-hidden="true" focusable="false">
      <path fill="currentColor" d="M12 2 1 21h22L12 2Zm-1 7h2v6h-2V9Zm0 8h2v2h-2v-2Z" />
    </svg>
  );
}

function EmptyIcon() {
  return (
    <svg className="status-icon" viewBox="0 0 24 24" width="28" height="28" aria-hidden="true" focusable="false">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M8 12h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function LoadingView({ label = "Cargando…" }: { label?: string }) {
  return (
    <div className="status-view status-view--loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <p>{label}</p>
    </div>
  );
}

interface ErrorViewProps {
  error: ApiError;
  onRetry?: () => void;
  title?: string;
}

export function ErrorView({ error, onRetry, title = "No se pudieron cargar los datos" }: ErrorViewProps) {
  return (
    <div className="status-view status-view--error" role="alert">
      <ErrorIcon />
      <h2>{title}</h2>
      <p>{error.message}</p>
      {error.issues.length > 0 && (
        <ul className="status-issues">
          {error.issues.map((issue, index) => (
            <li key={index}>
              <code>{issue.loc.filter((part) => part !== "body").join(".")}</code>: {issue.msg}
            </li>
          ))}
        </ul>
      )}
      {onRetry && (
        <button type="button" className="button" onClick={onRetry}>
          Reintentar
        </button>
      )}
      {error.request && (
        <details className="status-details">
          <summary>Detalles técnicos</summary>
          <code>{error.request}</code>
          {error.status !== null && <span> → HTTP {error.status}</span>}
        </details>
      )}
    </div>
  );
}

interface EmptyViewProps {
  title: string;
  children?: ReactNode;
}

export function EmptyView({ title, children }: EmptyViewProps) {
  return (
    <div className="status-view">
      <EmptyIcon />
      <h2>{title}</h2>
      {children && <p>{children}</p>}
    </div>
  );
}
