import type { ReactNode } from "react";
import "./PageShell.css";

interface PageShellProps {
  title: string;
  description?: ReactNode;
  /** Barra de controles bajo el encabezado (p. ej. fecha y hora). */
  toolbar?: ReactNode;
  /** Panel lateral izquierdo (controles del escenario). */
  sidebar?: ReactNode;
  /** Panel de detalle a la derecha (estación seleccionada). */
  panel?: ReactNode;
  /** Zona del mapa. */
  children: ReactNode;
}

/** Estructura común de página: encabezado, (toolbar), sidebar opcional, mapa y panel opcional. */
export default function PageShell({ title, description, toolbar, sidebar, panel, children }: PageShellProps) {
  return (
    <section className="page">
      <div className="page-header">
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {toolbar && <div className="page-toolbar">{toolbar}</div>}
      <div className="page-body" data-sidebar={sidebar ? "" : undefined} data-panel={panel ? "" : undefined}>
        {sidebar && (
          <aside className="page-sidebar" aria-label="Controles">
            {sidebar}
          </aside>
        )}
        <div className="page-map">{children}</div>
        {panel && (
          <aside className="page-panel" aria-label="Detalle">
            {panel}
          </aside>
        )}
      </div>
    </section>
  );
}
