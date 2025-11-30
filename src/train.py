from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    """
    Trains a Random Forest Classifier.
    """
    print("🔄 Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    return model

def evaluate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Overall Accuracy: {acc:.2f}")
    return y_pred
