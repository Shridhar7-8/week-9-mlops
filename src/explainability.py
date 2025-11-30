import shap
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate, count
from sklearn.metrics import accuracy_score

def run_fairness_analysis(y_test, y_pred, sensitive_features):
    """
    Uses Fairlearn to check metrics based on the sensitive 'location' attribute.
    """
    print("\n--- ⚖️ Fairlearn Analysis (Sensitive Attribute: Location) ---")
    
    # Define metrics we want to compare
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,  # How often model predicts positive
        "count": count # How many samples in each group
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    print(metric_frame.by_group)
    return metric_frame

def run_shap_analysis(model, X_train, X_test):
    """
    Generates SHAP plots for the Virginica class.
    """
    print("\n--- 🔍 Generating SHAP Explanations ---")
    
    # Initialize Explainer
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values (using X_test for speed, or X_train for full depth)
    shap_values = explainer(X_test)
    
    # Class 2 is Virginica in the Iris dataset
    print("   Plotting Summary Plot for Class: Virginica...")
    
    plt.figure()
    plt.title("SHAP Summary Plot (Virginica)")
    
    # Plotting for Class index 2 (Virginica)
    shap.summary_plot(shap_values[..., 2], X_test, show=False)
    
    # Save the plot
    plt.savefig("shap_virginica_summary.png")
    print("✅ SHAP plot saved as 'shap_virginica_summary.png'")
