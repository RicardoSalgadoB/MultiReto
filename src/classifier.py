import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
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
        display_labels=["No exceedance", "Exceedance"],
        values_format="d",
        cmap="Blues",
    )
    display.ax_.set_title(
        f"Daily O3 exceedance (accuracy = {accuracy_score(actual, predicted):.1%}, "
        f"probability threshold = {probability_threshold:.2f})"
    )
    display.figure_.tight_layout()
    return display


def plot_precision_recall_curve(df_by_day: pd.DataFrame) -> float:
    actual = df_by_day["O3_over_90"].astype(bool)
    probabilities = df_by_day["prob_over_threshold"]
    precision, recall, _ = precision_recall_curve(actual, probabilities)
    average_precision = average_precision_score(actual, probabilities)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"Average precision = {average_precision:.3f}")
    ax.axhline(
        actual.mean(),
        color="gray",
        linestyle="--",
        label=f"Positive prevalence = {actual.mean():.3f}",
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Daily O3 exceedance precision-recall curve",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    ax.legend()
    fig.tight_layout()
    return float(average_precision)


def analysis(df):
    exceedance_probabilities = df.loc[
        df["O3_over_90"], "prob_over_threshold"
    ]
    non_exceedance_probabilities = df.loc[
        ~df["O3_over_90"], "prob_over_threshold"
    ]
    print(f"Mean probability on exceedance days: {exceedance_probabilities.mean()}")
    print(f"Mean probability on non-exceedance days: {non_exceedance_probabilities.mean()}")
    
    t_val, p_val = stats.ttest_ind(exceedance_probabilities, non_exceedance_probabilities)

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

    probability_threshold = 0.25
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
            target_names=["No exceedance", "Exceedance"],
            zero_division=0,
        )
    )

    plot_confusion_matrix(df, probability_threshold)
    plt.show()


def main():
    df = pd.read_csv('data/datos_no_imputados.csv')
    df_probs = pd.read_csv('data/predictions_O3_probs.csv')
    df_by_day = by_day(df, df_probs)
    df_by_station = by_station(df, df_probs)
    
    analysis(df_by_day)
    


if __name__ == '__main__':
    main()