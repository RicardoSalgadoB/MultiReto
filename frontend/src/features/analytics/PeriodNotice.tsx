import type { Period } from "../../api/types";
import Badge from "../../components/ui/Badge";

const PERIOD_INFO: Record<Period, { tone: "warn" | "info"; badge: string; text: string }> = {
  entrenamiento: { tone: "warn", badge: "Periodo de entrenamiento", text: "El modelo ya vio estos datos." },
  validación: {
    tone: "info",
    badge: "Periodo de validación",
    text: "Se usó para elegir el corte de alerta, así que no es una evaluación completamente independiente.",
  },
  prueba: { tone: "info", badge: "Periodo de prueba", text: "El modelo no usó estos datos para ajustarse." },
};

/** Avisos fijos del Tab 1: la predicción es diaria y a qué periodo pertenece el día mostrado. */
export default function PeriodNotice({ period }: { period: Period | null }) {
  const info = period ? PERIOD_INFO[period] : null;
  return (
    <div className={`notice ${info?.tone === "warn" ? "notice--warn" : "notice--info"}`} data-period={period ?? undefined}>
      {info && <Badge tone={info.tone}>{info.badge}</Badge>}
      {info && <span>{info.text}</span>}
      <span>La predicción es diaria: es la misma para las 24 horas del día.</span>
    </div>
  );
}
