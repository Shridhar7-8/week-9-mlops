# 🧪 Week 8: MLOps - Data Poisoning Experiments

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*Exploring the impact of data poisoning on machine learning model performance*

</div>

---

## ⚖️ Week 9: MLOps - Fairness & Explainability

<div align="center">

**Incorporating sensitive attribute analysis and model explainability into the IRIS dataset classification pipeline**

</div>

---

## 📌 Assignment Objective

The goal of this week is to enhance a standard Machine Learning pipeline by introducing **Fairness checks** and **Model Explainability**.

### Key Components:

1. **Data Augmentation**: Introduce a synthetic `"location"` attribute to the IRIS dataset (values 0 and 1 assigned randomly)
2. **Fairness Analysis**: Incorporate Fairlearn to analyze if the model treats different locations equally
3. **Explainability**: Use SHAP (SHapley Additive exPlanations) to interpret model predictions, specifically focusing on the Virginica class

---

## 📂 Project Structure

```
week-9-mlops/
├── .github/
│   └── workflows/
│       └── main.yml         # GitHub Actions (CI/CD) pipeline
├── data/
│   └── iris.csv             # The dataset
├── src/
│   ├── data_loader.py       # Loads data & adds 'location' attribute
│   ├── train.py             # Trains Random Forest model
│   └── explainability.py    # Generates Fairlearn metrics & SHAP plots
├── main.py                  # Entry point for the pipeline
├── pyproject.toml           # Dependencies configuration
└── README.md                # Project documentation
```

---

## 🚀 Installation & Usage

This project uses **`uv`** for fast dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/Shridhar7-8/week-9-mlops.git
cd week-9-mlops
```

### 2. Install Dependencies

```bash
pip install uv
uv pip install --system .
```

### 3. Run the Pipeline

```bash
python main.py
```

This will train the model, print fairness metrics to the console, and generate the SHAP summary plot.

---

## 📊 Results & Analysis

### 1. Fairness Analysis (Fairlearn)

We treated `location` as a sensitive attribute. Since `location` was assigned randomly, we expect the model to exhibit **Demographic Parity** (similar selection rates for both locations).

**Sample Output:**

```
          accuracy  selection_rate  count
location                                 
0         1.000000        0.312500   16.0
1         1.000000        0.357143   14.0
```

**Interpretation:**
- Both locations show perfect accuracy (1.0)
- Selection rates are similar, indicating no systematic bias
- The random assignment of location is correctly treated as non-discriminatory by the model

---

### 2. Explainability (SHAP)

We generated a SHAP summary plot to understand how features contribute to classifying a flower as **Virginica**.

**Interpretation of SHAP for Virginica:**

- **Petal Length & Petal Width**: These are the top features. High values (Red dots) typically push the prediction towards Virginica (positive SHAP value)
- **Sepal Features**: Secondary importance in classification
- **Location**: As expected, the `location` feature has zero or negligible impact (dots clustered at 0), confirming the model did not learn bias from this random attribute

---

## 🤖 CI/CD Automation

This project includes a **GitHub Actions workflow** that:

1. ✅ Installs dependencies via `uv`
2. ✅ Runs the full ML pipeline
3. ✅ Generates a Report (Accuracy + Fairness metrics) and publishes it to the Job Summary
4. ✅ Auto-commits the generated SHAP plot back to the repository for easy viewing

**Workflow triggers on:**
- Push to `main` branch
- Pull requests
- Manual dispatch

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **scikit-learn** | Machine learning models |
| **Fairlearn** | Fairness assessment |
| **SHAP** | Model explainability |
| **uv** | Fast dependency management |
| **GitHub Actions** | CI/CD automation |

---

## 📈 Key Takeaways

- ✨ **Fairness matters**: Even with synthetic attributes, checking for bias is crucial
- 🔍 **Explainability builds trust**: SHAP helps understand what drives predictions
- 🚀 **Automation accelerates**: CI/CD ensures reproducibility and consistency
- 🎯 **MLOps best practices**: Combining fairness and explainability creates more responsible AI

---

## 👤 Author

**Shridhar**

- GitHub: [@Shridhar7-8](https://github.com/Shridhar7-8)

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

</div>
