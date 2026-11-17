import type { ReactNode } from "react";
import "./Badge.css";

type Tone = "neutral" | "info" | "warn";

/** Etiqueta corta de estado. El tono nunca es lo único que lleva el significado: el texto siempre lo dice. */
export default function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`badge badge--${tone}`}>{children}</span>;
}
