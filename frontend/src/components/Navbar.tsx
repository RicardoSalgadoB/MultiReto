import { Link } from "react-router";

function Navbar() {
  return (
    <nav>
      <Link to="/">Mapa de Análisis</Link>
      <> | </>
      <Link to="/prediction">Mapa de Predicciones</Link>
      <> | </>
      <Link to="/dependence">Mapa de Dependencias y Covarianzas</Link>
    </nav>
  );
}

export default Navbar;