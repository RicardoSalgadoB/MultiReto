import type { DependenceMode, TailMeasure } from "./useDependenceState";
import "./dependence.css";

interface ModeToggleProps {
  mode: DependenceMode;
  measure: TailMeasure;
  onMode: (mode: DependenceMode) => void;
  onMeasure: (measure: TailMeasure) => void;
}

/** Cambia entre correlación de Spearman y dependencia de colas (χ o χ̄). */
export default function ModeToggle({ mode, measure, onMode, onMeasure }: ModeToggleProps) {
  return (
    <div className="mode-controls">
      <div className="seg-group" role="group" aria-label="Tipo de medida">
        <button type="button" className="button" aria-pressed={mode === "spearman"} onClick={() => onMode("spearman")}>
          Correlación (Spearman)
        </button>
        <button type="button" className="button" aria-pressed={mode === "colas"} onClick={() => onMode("colas")}>
          Dependencia de colas
        </button>
      </div>
      {mode === "colas" && (
        <div className="seg-group" role="group" aria-label="Medida de dependencia de colas">
          <button type="button" className="button" aria-pressed={measure === "chi"} onClick={() => onMeasure("chi")}>
            χ
          </button>
          <button type="button" className="button" aria-pressed={measure === "chibar"} onClick={() => onMeasure("chibar")}>
            χ̄
          </button>
        </div>
      )}
    </div>
  );
}
