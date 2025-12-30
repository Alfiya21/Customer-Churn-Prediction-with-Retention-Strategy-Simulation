from sklearn.metrics import classification_report, roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    Print classification metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc:.4f}")

    return y_prob
