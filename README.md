# Placement Outcome Prediction using Machine Learning (SVM & Random Forest)

## 📌 Project Overview
This project implements a complete **machine learning classification pipeline** using Python to analyze structured student data and predict a future outcome based on academic performance and aptitude-related features.

The goal of the project is **not just prediction**, but to demonstrate a **disciplined ML workflow** including data preparation, feature engineering, model selection, evaluation, and validation.

Support Vector Machine (SVM) is used as the **primary model**, while Random Forest is implemented as a **comparison model**.

---

## 🎯 Objectives
- Build a classification model using real-world structured data
- Apply proper data cleaning and preprocessing techniques
- Perform feature engineering to improve model learning
- Train and evaluate SVM and Random Forest models
- Compare models using meaningful metrics beyond accuracy
- Validate model stability using cross-validation

---

## 🧠 Machine Learning Approach

### Problem Type
- Supervised Learning
- Binary Classification

### Models Used
- **Primary Model:** Support Vector Machine (SVM)
- **Secondary Model:** Random Forest Classifier

SVM was selected as the primary model due to its strong performance on structured, scaled data and its stable decision boundary after tuning.

---

## 🗂️ Project Structure
    -data/
      -students_data.csv
    -notebooks/
      -analysis.ipynp
    -models/
      -svm_model.pkl
      -random_forest_model.pkl
    -Preview/
    -README.md
---


## 📊 Dataset Description
The dataset consists of **1000+ student records** with the following attributes:

| Feature | Description |
|------|------------|
| gender | Student gender |
| ug_department | Undergraduate department |
| internships | Number of internships |
| academic_strength | Engineered feature (avg of academic scores) |
| aptitude_strength | Engineered feature (avg of aptitude scores) |
| outcome | Target variable (binary) |

> Raw academic and aptitude scores were consolidated through feature engineering to reduce redundancy and improve learning efficiency.

---

## ⚙️ Key Project Phases

### Phase 1: Data Understanding & Cleaning
- Handled missing values
- Removed duplicate records
- Corrected data types
- Encoded categorical variables

### Phase 2: Feature Engineering
- Created `academic_strength`
- Created `aptitude_strength`
- Dropped redundant raw features

### Phase 3: Data Preparation
- Train-test split (80–20 with stratification)
- Feature scaling using StandardScaler (for SVM)

### Phase 4: Model Training
- SVM with RBF kernel
- Hyperparameter tuning using GridSearchCV
- Random Forest training without scaling

### Phase 5: Model Evaluation
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### Phase 6: Validation
- Cross-validation (5-fold)
- Stability analysis using mean and standard deviation

---

## 📈 Evaluation Metrics
Models were evaluated using:
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross-validation scores

Accuracy was **not used as the sole metric**, as it can be misleading in classification problems.

---

## 🏆 Model Selection
Although both models performed competitively, **SVM was selected as the final model** due to:
- Higher or comparable F1-score
- Better stability across cross-validation folds
- Suitability for structured, scaled feature space

---

## 💾 Model Persistence
Trained models were saved using `joblib` for reuse:
- `svm_model.pkl`
- `random_forest_model.pkl`

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 🚀 How to Run the Project

1. Clone the repository
   ```bash
   git clone <https://github.com/PeruguHari/Placement-prediction>
2.Install dependencies

  - pip install pandas numpy scikit-learn matplotlib seaborn'
3.Open the notebook

  - jupyter notebook notebooks/analysis.ipynb

4.Run all cells in order

---
## 📌 Key Learnings

- Importance of feature engineering

- Why accuracy alone is insufficient

- Differences between margin-based and tree-based models

- Model stability through cross-validation

- End-to-end ML project structuring

## 📄 Author

- **Perugu Hari**
- **Bachelor of Computer Science**
