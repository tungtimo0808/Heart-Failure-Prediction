# Heart Failure / Heart Disease Prediction (ML + Flask)

A machine learning project that predicts whether a patient is likely to have **heart disease** from clinical + demographic features.  
The project includes:
- Data cleaning + preprocessing
- Exploratory Data Analysis (EDA)
- Training & evaluating multiple ML models (Logistic Regression, SVM, Random Forest, XGBoost)
- Model interpretability (SHAP + permutation importance)
- A simple **Flask web app** for real-time prediction

> Dataset: *Heart Failure Prediction Dataset* (Kaggle, fedesoriano, 2021).

---

## Demo (Flask Web App)
The web app lets users input health indicators (age, sex, BP, cholesterol, ECG results, etc.) and returns:
- Predicted class (Positive / Negative)
- Probability score

---

## Project Structure
Typical structure in this repository:


---

## Dataset
The dataset contains **918 records** and **12 features**, including:
- Age, Sex
- ChestPainType
- RestingBP, Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope  
Target: `HeartDisease` (1 = disease, 0 = normal)

### Data Cleaning (important)
- `RestingBP = 0` (invalid): replaced with **median**
- `Cholesterol = 0` (many cases): treated as missing and imputed using **KNN Imputer**
- Categorical features:
  - Binary encoding: `Sex`, `ExerciseAngina`
  - One-hot encoding: `ChestPainType`, `RestingECG`, `ST_Slope`

---

## Models & Results (Test Set)
Models trained and tuned using stratified CV (ROC-AUC as main metric):

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy |
|------|------------|--------------|----------|
| Logistic Regression | ~0.922 | ~0.932 | ~0.88 |
| SVC (RBF) | ~0.920 | ~0.937 | ~0.86 |
| Random Forest | ~0.929 | ~0.923 | ~0.86 |
| XGBoost | ~0.933 | ~0.932 | **~0.89** |

**Conclusion:** XGBoost provides the most balanced performance and stability overall.

---

## Interpretability (XAI)
Two methods are used to understand feature impact:
- **SHAP** (global importance + direction of influence)
- **Permutation Importance** (performance drop when shuffling features)

Top influential signals include ECG-related / symptom-related features such as:
- ST slope (especially `Up` / `Flat`)
- Chest pain type (notably `ASY`)
- Exercise angina, Oldpeak, MaxHR  
Traditional indicators like cholesterol may contribute less in this dataset.

---

## How to Run

### 1) Create environment & install dependencies
```bash
# (recommended) create venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
