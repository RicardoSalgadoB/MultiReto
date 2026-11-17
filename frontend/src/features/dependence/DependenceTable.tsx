import { useState } from "react";
import type { StationCode } from "../../api/types";
import Badge from "../../components/ui/Badge";
import { formatFixed, formatNumber } from "../../lib/format";
import { otherStation, type DependenceModel, type PairView } from "./dependenceModel";
import "./dependence.css";

/** Cuántos pares se listan sin estación seleccionada (hasta que se pidan todos). */
const NETWORK_TOP = 10;

interface DependenceTableProps {
  model: DependenceModel;
  selected: StationCode | null;
  activeKey: string | null;
  measureLabel: string;
  /** Debe ser estable (useCallback). */
  onActivate: (key: string) => void;
}

/**
 * Alternativa accesible al hover: lista ordenada de las estaciones más dependientes de la seleccionada (o, sin
 * selección, de los pares más fuertes de la red). Cada fila tiene un botón (navegable con Tab) que resalta su línea.
 */
export default function DependenceTable({ model, selected, activeKey, measureLabel, onActivate }: DependenceTableProps) {
  const [showAll, setShowAll] = useState(false);
  const all: PairView[] = selected === null ? model.ranked : (model.byStation.get(selected) ?? []);
  const rows = selected === null && !showAll ? all.slice(0, NETWORK_TOP) : all;
  const title = selected === null ? `Pares más fuertes de la red (${measureLabel})` : `Estaciones más dependientes de ${selected} (${measureLabel})`;
  const hasLow = rows.some((row) => row.low);

  return (
    <section className="dependence-table-section" aria-labelledby="dependence-table-title">
      <h2 id="dependence-table-title">{title}</h2>
      <p className="table-help">
        {selected === null
          ? "Selecciona una estación en el mapa para ver solo las suyas. El botón de cada fila resalta su línea en el mapa."
          : `Ordenadas de mayor a menor. El botón de cada fila resalta en el mapa la línea entre ${selected} y esa estación.`}
      </p>
      <div className="table-scroll" role="region" aria-label={title} tabIndex={0}>
        <table className="dependence-table">
          <caption className="sr-only">{title}</caption>
          <thead>
            <tr>
              <th scope="col">{selected === null ? "Par" : "Estación"}</th>
              <th scope="col">{measureLabel}</th>
              <th scope="col">Distancia</th>
              <th scope="col">Días conjuntos</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const other = selected === null ? null : otherStation(row, selected);
              const label = other === null ? `${row.a} – ${row.b}` : other;
              const longName = other === null ? `${model.names.get(row.a) ?? row.a} – ${model.names.get(row.b) ?? row.b}` : (model.names.get(other) ?? "");
              const active = row.key === activeKey;
              return (
                <tr key={row.key} data-active={active ? "" : undefined}>
                  <th scope="row">
                    <button type="button" className="row-button" aria-pressed={active} onClick={() => onActivate(row.key)} title={longName}>
                      {label}
                    </button>
                    <span className="row-name">{longName}</span>
                  </th>
                  <td className="cell-value" data-muted={row.valor === null ? "" : undefined}>
                    {row.valor === null ? "sin valor" : formatFixed(row.valor, 2)}
                  </td>
                  <td>{formatNumber(row.distanciaKm)} km</td>
                  <td>
                    {row.diasConjuntos} {row.low && <Badge tone="warn">&lt; {model.lowThreshold}</Badge>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="table-help">
        Días conjuntos: días en que ambas estaciones superaron el cuantil {model.q}; se cuentan igual en las tres medidas.
      </p>
      <div className="callout callout--warn">
        <p className="callout-title">Pocos días conjuntos</p>
        <p>
          Todos los pares se basan en pocos (de {model.minDays} a {model.maxDays}).
          {hasLow && ` «< ${model.lowThreshold}»: ${model.lowLabel.charAt(0).toLowerCase()}${model.lowLabel.slice(1)}.`}
        </p>
      </div>
      {selected === null && all.length > NETWORK_TOP && (
        <button type="button" className="button" aria-expanded={showAll} onClick={() => setShowAll((value) => !value)}>
          {showAll ? `Mostrar solo los ${NETWORK_TOP} más fuertes` : `Mostrar los ${all.length} pares`}
        </button>
      )}
    </section>
  );
}
