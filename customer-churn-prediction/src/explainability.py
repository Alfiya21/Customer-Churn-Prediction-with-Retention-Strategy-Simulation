import shap
import matplotlib.pyplot as plt


def generate_shap_summary(model, X_test):
    """
    Generate SHAP summary plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(
    shap_values[1],
    X_test.values,
    feature_names=X_test.columns,
    show=False
)

    plt.tight_layout()
    plt.show()
