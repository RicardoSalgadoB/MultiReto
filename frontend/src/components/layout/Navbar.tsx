import { NavLink } from "react-router";
import "./Navbar.css";

const LINKS = [
  { to: "/", label: "Mapa de análisis", end: true },
  { to: "/prediction", label: "Mapa de predicciones", end: false },
  { to: "/dependence", label: "Dependencia entre estaciones", end: false },
];

/** Marca: el mismo icono «O₃» del favicon (public/favicon.svg), dibujado en línea para heredar los colores del tema. */
function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 64 64" width="28" height="28" aria-hidden="true" focusable="false">
      <rect width="64" height="64" rx="14" className="brand-mark-bg" />
      <text x="32" y="43" textAnchor="middle" fontSize="30" fontWeight="700" className="brand-mark-text">
        O<tspan fontSize="18" dy="7">3</tspan>
      </text>
    </svg>
  );
}

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        <span className="navbar-brand">
          <BrandMark />
          Ozono Monterrey
        </span>
        <nav aria-label="Secciones">
          <ul>
            {LINKS.map((link) => (
              <li key={link.to}>
                <NavLink to={link.to} end={link.end}>
                  {link.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
