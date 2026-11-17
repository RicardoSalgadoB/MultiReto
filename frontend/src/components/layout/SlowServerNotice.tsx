import { useSyncExternalStore } from "react";
import { getSlowSnapshot, subscribeSlow } from "../../api/client";

/** Aviso discreto cuando alguna petición lleva más de unos segundos (p. ej. un servidor gratuito que está despertando). */
export default function SlowServerNotice() {
  const slow = useSyncExternalStore(subscribeSlow, getSlowSnapshot, () => false);
  return (
    <div role="status" aria-live="polite" className="slow-region">
      {slow && <p className="slow-notice">El servidor está iniciando o responde lento; puede tardar hasta un minuto.</p>}
    </div>
  );
}
