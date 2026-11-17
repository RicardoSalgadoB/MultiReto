import { Link } from "react-router";
import PageShell from "../components/layout/PageShell";
import { usePageTitle } from "../hooks/usePageTitle";

/** Ruta inexistente. No depende de la API, así que funciona aunque el backend esté apagado. */
export default function NotFound() {
  usePageTitle("Página no encontrada");
  return (
    <PageShell title="Página no encontrada" description="La dirección que abriste no existe en este sitio.">
      <section className="empty-links" aria-label="Secciones del sitio">
        <p>Elige una sección:</p>
        <ul>
          <li>
            <Link to="/">Mapa de análisis</Link>
          </li>
          <li>
            <Link to="/prediction">Mapa de predicciones</Link>
          </li>
          <li>
            <Link to="/dependence">Dependencia entre estaciones</Link>
          </li>
        </ul>
      </section>
    </PageShell>
  );
}
