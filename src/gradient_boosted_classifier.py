from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import config 

def select_gbm_parameters(
        X_train, 
        X_val, 
        y_train, 
        y_val, 
        label_col=config.LABEL, 
        output_csv=config.GBM_RESULTS_LOCATION, 
        use_sample_weights=False
    ):

    output_csv = output_csv

    keys, values = zip(*config.GBM_PARAM_GRID.items())
    grid_combos = [dict(zip(keys, v)) for v in product(*values)]
    custom_weights = {0: 1.0, 1: 2.0} # this could be tuned given more time
    sample_weights = compute_sample_weight(class_weight=custom_weights, y=y_train)

    results = []

    for params in grid_combos:
        print(f"Training with params: {params}")
        model = GradientBoostingClassifier(**params, random_state=42)

        # Hypothesis: weighting will help recall
        if use_sample_weights:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Metrics
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)

        row = params.copy()
        row.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc
        })

        results.append(row)

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="f1", ascending=False, inplace=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return results_df

def create_gbm_with_set_hyperparameters(
        X_train, 
        X_val, 
        y_train, 
        y_val, 
        params, use_sample_weights=False
    ):
    print(f"Training GBM with params: {params}")
    model = GradientBoostingClassifier(**params, random_state=42)

    # Hypothesis: weighting will help recall
    if use_sample_weights:
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    row = params.copy()
    row.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    return model

def evaluate_gbm(model, X_val, y_val, plot_path=None):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)

    # Print metrics
    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4)
    }
    print(metrics)

    # Plot PR curve
    precs, recalls, _ = precision_recall_curve(y_val, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precs, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if plot_path:
        plt.savefig(plot_path)
        print(f"Saved PR curve to {plot_path}")
    else:
        plt.show()

    plt.close()
    return metrics