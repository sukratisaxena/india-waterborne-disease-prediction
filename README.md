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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8878 | 0.9907 | 0.8877 | 0.8878 | 0.8874 | 0.8574 |
| Decision Tree | 0.9263 | 0.9935 | 0.9278 | 0.9263 | 0.9267 | 0.9063 |
| kNN | 0.8858 | 0.9834 | 0.8863 | 0.8858 | 0.8848 | 0.8547 |
| Naive Bayes (Multinomial) | 0.8214 | 0.9769 | 0.8239 | 0.8214 | 0.8204 | 0.7729 |
| Random Forest (Ensemble) | 0.9159 | 0.9938 | 0.9171 | 0.9159 | 0.9152 | 0.8934 |
| XGBoost (Ensemble) | 0.9517 | 0.9981 | 0.9515 | 0.9517 | 0.9515 | 0.9385 |

---

## Observations on Model Performance 

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high AUC (0.9907) and stable overall performance, but lower accuracy than tree-based ensembles, indicating limited ability to capture complex feature interactions. |
| Decision Tree | Higher accuracy than linear and distance-based models, showing good non-linear learning; slightly behind ensembles, likely due to overfitting limits from using a single tree. |
| kNN | Comparable to Logistic Regression in accuracy, but lower AUC than tree-based methods; performance suggests sensitivity to feature space sparsity and scaling even with preprocessing. |
| Naive Bayes (Multinomial) | Lowest scores across metrics; assumptions of feature independence and multinomial distribution reduce effectiveness on mixed numeric + categorical structured data. |
| Random Forest (Ensemble) | Strong performance with high AUC and improved accuracy over single Decision Tree; ensemble averaging reduces overfitting and improves generalization. |
| XGBoost (Ensemble) | Best overall performance across all metrics (highest Accuracy, AUC, and MCC), indicating superior handling of complex patterns via boosted trees and regularization. |
