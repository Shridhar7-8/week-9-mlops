from src.data_loader import load_and_process_data
from src.train import train_model, evaluate_accuracy
from src.explainability import run_fairness_analysis, run_shap_analysis

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test, feature_names = load_and_process_data()
    
    # 2. Train Model
    model = train_model(X_train, y_train)
    
    # 3. Basic Evaluation
    y_pred = evaluate_accuracy(model, X_test, y_test)
    
    # 4. Fairlearn Analysis
    # We pass the 'location' column from X_test as the sensitive feature
    run_fairness_analysis(y_test, y_pred, sensitive_features=X_test['location'])
    
    # 5. SHAP Analysis
    run_shap_analysis(model, X_train, X_test)

if __name__ == "__main__":
    main()
