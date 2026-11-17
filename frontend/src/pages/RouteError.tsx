import { isRouteErrorResponse, Link, useRouteError } from "react-router";
import Navbar from "../components/layout/Navbar";
import PageShell from "../components/layout/PageShell";
import { usePageTitle } from "../hooks/usePageTitle";

/** `errorElement` global: errores al cargar una sección (p. ej. tras una actualización del sitio) o fallos inesperados. */
export default function RouteError() {
  usePageTitle("Algo salió mal");
  const error = useRouteError();
  const detail = isRouteErrorResponse(error) ? `${error.status} ${error.statusText}` : null;
  return (
    <>
      <Navbar />
      <main id="contenido">
        <PageShell title="Algo salió mal" description="No se pudo mostrar esta sección.">
          <p>
            Recarga la página para intentarlo de nuevo. Si el problema continúa, vuelve al inicio.
            {detail ? ` (${detail})` : ""}
          </p>
          <p>
            <button type="button" className="button" onClick={() => window.location.reload()}>
              Recargar la página
            </button>{" "}
            <Link to="/">Volver al mapa de análisis</Link>
          </p>
        </PageShell>
      </main>
    </>
  );
}
