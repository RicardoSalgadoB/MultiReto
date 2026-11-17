import { usePageTitle } from "../hooks/usePageTitle";
import DependenceView from "../features/dependence/DependenceView";

/** Tab 3. Los datos de dependencia se piden por medida (ver DependenceView). */
export default function DependenceMap() {
  usePageTitle("Dependencia entre estaciones");
  return <DependenceView />;
}
