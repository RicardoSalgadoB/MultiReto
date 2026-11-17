import { Suspense, useRef, type MouseEvent } from "react";
import { Outlet } from "react-router";
import { LoadingView } from "../components/layout/StatusViews";
import Navbar from "../components/layout/Navbar";
import SlowServerNotice from "../components/layout/SlowServerNotice";

/** Estructura común: enlace para saltar al contenido, barra de navegación, aviso de servidor lento y la página activa. */
export default function App() {
  const mainRef = useRef<HTMLElement>(null);

  // El enlace mueve el foco a <main> sin tocar la URL (un #hash entraría al historial del router).
  const skip = (event: MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    mainRef.current?.focus();
    mainRef.current?.scrollIntoView();
  };

  return (
    <>
      <a href="#contenido" className="skip-link" onClick={skip}>
        Saltar al contenido
      </a>
      <Navbar />
      <SlowServerNotice />
      <main id="contenido" tabIndex={-1} ref={mainRef}>
        <Suspense fallback={<LoadingView label="Cargando la página…" />}>
          <Outlet />
        </Suspense>
      </main>
    </>
  );
}
