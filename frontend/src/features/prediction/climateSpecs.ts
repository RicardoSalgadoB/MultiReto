import type { ClimateKey } from "../../api/types";

/** Controles del sidebar, en el orden en que se muestran. */
export const CLIMATE_ORDER: ClimateKey[] = ["T_max_hoy", "T_max_pron", "T_min_pron", "RH_media_hoy", "WSR_media_hoy", "SR_max_hoy"];

/** Decimales con que se muestra cada control. */
export const CLIMATE_DECIMALS: Record<ClimateKey, number> = {
  SR_max_hoy: 2,
  RH_media_hoy: 1,
  WSR_media_hoy: 1,
  T_max_hoy: 1,
  T_max_pron: 1,
  T_min_pron: 1,
};

/** Dos valores más cercanos que esto se consideran iguales (el control vuelve a "sin cambios"). */
export const SAME_VALUE_EPSILON = 1e-6;
