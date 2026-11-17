import type { BaseScenarioResponse } from "../../api/types";
import { FORECAST_VARIABLES } from "../../config/modelConstants";

/**
 * ¿Faltan las temperaturas pronosticadas (T_max_pron y T_min_pron) en TODAS las estaciones?
 * En ese caso el backend las llenó con la mediana de entrenamiento y el resultado no es representativo.
 */
export function forecastMissingEverywhere(scenario: BaseScenarioResponse, stationNames: string[]): boolean {
  if (stationNames.length === 0) return false;
  return stationNames.every((name) => {
    const completed = scenario.valores_completados[name] ?? [];
    return FORECAST_VARIABLES.every((variable) => completed.includes(variable));
  });
}

/** ¿T_max_pron figura en `valores_completados` de alguna estación? (el día "con pronóstico" es el que no). */
export function forecastCompletedAnywhere(scenario: BaseScenarioResponse): boolean {
  return Object.values(scenario.valores_completados).some((variables) => variables.includes("T_max_pron"));
}
