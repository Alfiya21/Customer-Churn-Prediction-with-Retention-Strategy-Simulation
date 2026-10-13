import numpy as np
import shap
import matplotlib.pyplot as plt


def generate_shap_summary(model, X_test, save_path=None):
    """Generate SHAP summary plot for the churn (positive) class.

    Works with old SHAP (list of arrays) and new SHAP (>=0.45, array of shape [n, features, classes]).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]

    shap.summary_plot(sv, X_test.values, feature_names=X_test.columns, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    plt.show()
