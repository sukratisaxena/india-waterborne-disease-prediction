# Problem Statement
Build and evaluate multiple machine learning classification models to predict **waterborne disease category** (target: `disease`) using the **India Waterborne Disease Dataset**. Compare model performance using standard evaluation metrics and summarize results.

---

## Dataset Description
- **Dataset Name:** India Waterborne Disease Dataset  
- **Source:** Kaggle (via `kagglehub`)  
- **File Used:** `waterborne_disease_dataset.csv`  
- **Target Column:** `disease`  
- **Task Type:** Multiclass classification  
- **Classes Present:** `Cholera`, `Dysentery`, `Giardiasis`, `Hepatitis_A`, `Hepatitis_E`, `Leptospirosis`, `No_Disease`, `Typhoid`

---

## Models Used + Metrics

### Comparison Table (Evaluation Metrics)

---

## Observations on Model Performance 

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high AUC (0.9907) and stable overall performance, but lower accuracy than tree-based ensembles, indicating limited ability to capture complex feature interactions. |
| Decision Tree | Higher accuracy than linear and distance-based models, showing good non-linear learning; slightly behind ensembles, likely due to overfitting limits from using a single tree. |
| kNN | Comparable to Logistic Regression in accuracy, but lower AUC than tree-based methods; performance suggests sensitivity to feature space sparsity and scaling even with preprocessing. |
| Naive Bayes (Multinomial) | Lowest scores across metrics; assumptions of feature independence and multinomial distribution reduce effectiveness on mixed numeric + categorical structured data. |
| Random Forest | Strong performance with high AUC and improved accuracy over single Decision Tree; ensemble averaging reduces overfitting and improves generalization. |
| XGBoost | Best overall performance across all metrics (highest Accuracy, AUC, and MCC), indicating superior handling of complex patterns via boosted trees and regularization. |
