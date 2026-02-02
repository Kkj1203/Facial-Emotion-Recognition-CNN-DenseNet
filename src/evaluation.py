# ===============================
# Evaluation
# ===============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from config import *

def evaluate_model(model, test_gen):

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.labels

    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    print("ROC-AUC Score:", roc_auc_score(y_true_bin, preds))
