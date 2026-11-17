import type { DependenceKind } from "../../api/types";

/**
 * Origen de los datos (backend/source/entrenar.py, función `dependencia(E, q=0.95)`): χ y χ̄ se calculan con el MISMO
 * `q` (la respuesta trae uno solo) sobre los máximos diarios de ozono con al menos 18 h de datos, y `dias_conjuntos`
 * es el número de días en que AMBAS estaciones superan el cuantil `q` (`ambos.sum()` en `chi_y_chibar`).
 *
 * Nombre y explicación breve de cada medida. `q` sale de la respuesta de la API (nunca se escribe a mano);
 * mientras no se conoce, el nombre y el texto no lo mencionan.
 */
export function describeKind(kind: DependenceKind, q: number | null): { label: string; help: string } {
  const withQ = q === null ? "" : `(${q})`;
  switch (kind) {
    case "spearman":
      return {
        label: "Correlación de Spearman",
        help: "Correlación de rangos de los máximos diarios de ozono entre dos estaciones: qué tanto suben y bajan juntos en todo el rango de valores, no solo en los extremos.",
      };
    case "chi":
      return {
        label: `χ${withQ}`,
        help: `χ${withQ}: probabilidad de que una estación tenga un episodio extremo de ozono${q === null ? "" : ` (por encima del cuantil ${q})`} dado que la otra también lo tiene.`,
      };
    case "chibar":
      return {
        label: `χ̄${withQ}`,
        help: `χ̄${withQ}: indica si la dependencia entre los extremos de dos estaciones se mantiene en los valores más altos; cuanto más cerca de 1, más se mantiene.`,
      };
  }
}
