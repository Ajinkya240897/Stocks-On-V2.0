"""
SHAP explainability wrapper for tree models.
Returns a matplotlib figure, shap_values array, and explainer object.
"""
import shap, matplotlib.pyplot as plt, pandas as pd, numpy as np

def explain_tree_model(bst, X: pd.DataFrame, max_display: int = 10, plot_type: str = "bar"):
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X)
    fig = plt.figure(figsize=(6,4))
    if plot_type=="bar":
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    else:
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    return fig, shap_values, explainer
