/** Cuantil "inferior" de una lista ya ordenada de menor a mayor: sorted[floor(p·(n−1))]. */
export function lowerQuantile(sorted: number[], p: number): number {
  return sorted[Math.floor(p * (sorted.length - 1))];
}
