
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)   
from sklearn.preprocessing import LabelEncoder










import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)   
from sklearn.preprocessing import LabelEncoder

def evaluate_pseudo_labels(
    csv_path,
    use_names=True,
    normalize_cm=None,
    figsize=(8, 6),
    cmap="Blues",
    plot_=True,
):

    # ---- Load CSV ----
    df = pd.read_csv(csv_path)

    if use_names:
        y_true_raw = df["gt_category_name"].astype(str).values
        y_pred_raw = df["pseudo_category_name"].astype(str).values
        label_type = "names"

    else:
        y_true_raw = df["gt_category_id"].astype(int).values
        y_pred_raw = df["pseudo_category_id"].astype(int).values
        label_type = "ids"




    le = LabelEncoder()
    all_labels_raw = np.concatenate([y_true_raw, y_pred_raw], axis=0)
    le.fit(all_labels_raw)
    print(f"y_true_raw unique values: {np.unique(y_true_raw)}")
    print(f"y_pred_raw unique values: {np.unique(y_pred_raw)}")
    y_true = le.transform(y_true_raw)
    y_pred = le.transform(y_pred_raw)
    print(f"y_true unique values after encoding: {np.unique(y_true)}")
    print(f"y_pred unique values after encoding: {np.unique(y_pred)}")
    # breakpoint()
    labels = le.classes_  # label names/ids in cm order
    print("\033[95m" + "Predicted class counts: " + str(np.bincount(y_pred))+ " Total: " + str(len(y_pred)) + "\033[0m") # ---- Basic metrics ----
    print("\033[95m" + "True class counts: " + str(np.bincount(y_true))+ " Total: " + str(len(y_true)) + "\033[0m") # ---- Basic metrics ----
    # ---- Basic metrics ----
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro",zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted",zero_division=0)
    #MRR calculation
    reciprocal_ranks = (y_true == y_pred).astype(float)
    mrr = np.mean(reciprocal_ranks)
    print(f"Evaluation using category {label_type} (excluding unknown/-1)")
    print(f"Accuracy:        {acc:.4f}")
    print(f"F1 (macro):      {f1_macro:.4f}")
    print(f"F1 (micro):      {f1_micro:.4f}")
    print(f"F1 (weighted):   {f1_weighted:.4f}")
    print(f"MRR:             {mrr:.4f}")
    print()

    # ---- Classification report ----
    report = classification_report(
        y_true,
        y_pred,
        target_names=[str(l) for l in labels],
        digits=4,
        zero_division=0
    )
    print("Classification report:")
    print(report)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, normalize=normalize_cm)
    if plot_:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        # Tick labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Annotate cells (optional; can get crowded with many classes)
        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                text_color = "white" if val > thresh else "black"
                ax.text(
                    j, i, f"{val:.2f}" if normalize_cm else int(val),
                    ha="center", va="center", color=text_color, fontsize=8
                )

        plt.tight_layout()
        plt.show()

    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels.tolist(),
    }
    return results
