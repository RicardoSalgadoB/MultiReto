import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)
from scipy import stats


def by_day(df: pd.DataFrame, df_probs: pd.DataFrame):
    o3_by_day = (
        df.assign(Fecha=pd.to_datetime(df["Fecha y hora"]).dt.normalize())
        .groupby("Fecha")["O3"]
        .apply(lambda values: values.gt(90).any())
        .rename("O3_over_90")
    )
    probs_by_day = (
        df_probs.assign(Fecha=pd.to_datetime(df_probs["Fecha"]).dt.normalize())
        .groupby("Fecha")["prob_over_threshold"]
        .max()
    )

    return o3_by_day.to_frame().join(probs_by_day, how="inner").reset_index()


def by_station(df: pd.DataFrame, df_probs: pd.DataFrame):
    o3_by_station = (
        df.assign(Fecha=pd.to_datetime(df["Fecha y hora"]).dt.normalize())
        .groupby(["Fecha", "Estacion"])["O3"]
        .apply(lambda values: values.gt(90).any())
        .rename("O3_over_90")
    )
    probs_by_station = (
        df_probs.assign(Fecha=pd.to_datetime(df_probs["Fecha"]).dt.normalize())
        .groupby(["Fecha", "Estacion"])["prob_over_threshold"]
        .max()
    )

    return o3_by_station.to_frame().join(probs_by_station, how="inner").reset_index()


def plot_confusion_matrix(
    df_by_day: pd.DataFrame, probability_threshold: float = 0.5
) -> ConfusionMatrixDisplay:
    actual = df_by_day["O3_over_90"].astype(bool)
    predicted = df_by_day["prob_over_threshold"].ge(probability_threshold)

    display = ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=[False, True],
        display_labels=["Sin Excedencia", "Con Excedencia"],
        values_format="d",
        cmap="Blues",
    )
    display.ax_.set_title(
        f"Excedencias diarias de O3 (exactitud = {accuracy_score(actual, predicted):.1%}, "
        f"umbral de probabilidad = {probability_threshold:.2f})"
    )
    display.figure_.tight_layout()
    return display


def plot_roc_curve(df: pd.DataFrame) -> float:
    actual = df["O3_over_90"].astype(bool)
    probabilities = df["prob_over_threshold"]
    valid_mask = pd.notna(df["O3_over_90"]) & pd.notna(df["prob_over_threshold"])

    actual_clean = df.loc[valid_mask, "O3_over_90"]
    probabilities_clean = df.loc[valid_mask, "prob_over_threshold"]

    # Compute False Positive Rate, True Positive Rate, and AUC
    fpr, tpr, _ = roc_curve(actual_clean, probabilities_clean)
    roc_auc = roc_auc_score(actual_clean, probabilities_clean)

    fig, ax = plt.subplots()
    # Plot ROC curve
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

    # Diagonal random guess line
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Sin habilidad (0.5)")

    ax.set(
        xlabel="Tasa de Falsos Positivos (FPR)",
        ylabel="Tasa de Verdaderos Positivos (TPR)",
        title="Curva ROC para Excedencias diarias de O3",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    ax.legend()
    fig.tight_layout()

    return float(roc_auc)

def analysis(df):
    exceedance_probabilities = df.loc[
        df["O3_over_90"], "prob_over_threshold"
    ]
    non_exceedance_probabilities = df.loc[
        ~df["O3_over_90"], "prob_over_threshold"
    ]
    print(f"Mean probability on exceedance days: {exceedance_probabilities.mean()}")
    print(f"Mean probability on non-exceedance days: {non_exceedance_probabilities.mean()}")
    
    a = np.asarray(exceedance_probabilities, dtype=float)
    b = np.asarray(non_exceedance_probabilities, dtype=float)
    
    t_val, p_val = stats.ttest_ind(a, b)

    alpha = 0.05
    deg_f = len(exceedance_probabilities)+len(non_exceedance_probabilities)-2

    crit_t = stats.t.ppf(1 - alpha/2, deg_f)

    print("\nT-value:", t_val)
    print("P-Value:", p_val)
    print("Critical t-value:", crit_t)

    print('\nT-test Result:')
    if np.abs(t_val) > crit_t:
        print('Significant difference found.\n')
    else:
        print('No significant difference.\n')

    print('P-test Result:')
    if p_val < alpha:
        print('Reject H0. Significant difference found.\n')
    else:
        print('Fail to reject H0. No strong evidence of difference.\n')

    probability_threshold = 0.20
    actual = df["O3_over_90"].astype(bool)
    predicted = df["prob_over_threshold"].ge(probability_threshold)
    print(
        "Classification report "
        f"(probability threshold = {probability_threshold:.2f}):"
    )
    print(
        classification_report(
            actual,
            predicted,
            labels=[False, True],
            target_names=["Sin Excedentes", "Con Excedentes"],
            zero_division=0,
        )
    )

    plot_confusion_matrix(df, probability_threshold)
    plot_roc_curve(df)
    plt.show()


def main():
    df = pd.read_csv('data/datos_corregidos.csv')
    df_probs = pd.read_csv('data/predictions_O3_probs.csv')
    df_by_day = by_day(df, df_probs)
    df_by_station = by_station(df, df_probs)
    
    # df_by_day = df_by_day[df_by_day["Fecha"] != "2023-01-02 00:00:00"]
    # df_by_day["prob_over_threshold"] = df_by_day["O3_over_90"].shift(1)
    
    analysis(df_by_day)
    


if __name__ == '__main__':
    main()