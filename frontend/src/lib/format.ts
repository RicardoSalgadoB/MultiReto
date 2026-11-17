const numberFormat = new Intl.NumberFormat("es-MX", { maximumFractionDigits: 1 });
const percentValue = new Intl.NumberFormat("es-MX", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

/** 83.5 -> "83.5" (un decimal como máximo). */
export function formatNumber(value: number): string {
  return numberFormat.format(value);
}

/** 0.1673 -> "16.7 %". */
export function formatPercent(value: number): string {
  return `${percentValue.format(value * 100)} %`;
}

/** Número con exactamente `digits` decimales: formatFixed(-0.16770, 3) -> "-0.168". */
export function formatFixed(value: number, digits: number): string {
  return new Intl.NumberFormat("es-MX", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

/** Hasta dos decimales (efectos en logit). */
export function formatNumber2(value: number): string {
  return new Intl.NumberFormat("es-MX", { maximumFractionDigits: 2 }).format(value);
}

/** Porcentaje entero para ejes: 0.256 -> "26 %". */
export function formatPercentAxis(value: number): string {
  return `${Math.round(value * 100)} %`;
}
