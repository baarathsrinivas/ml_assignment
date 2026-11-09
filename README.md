# Mushroom Edibility Classification

## Problem Statement

This project compares supervised classifiers for predicting whether a mushroom is edible or poisonous from its categorical characteristics. The target column is `class`: `e` means edible and `p` means poisonous. The source combines mushrooms of unknown edibility with the poisonous class, following the dataset description.

## Dataset Description

The local dataset is `data/mushrooms.csv`, based on the UCI Mushroom dataset. It has 8,124 instances, 22 predictor attributes, and two target classes. The class counts are 4,208 edible and 3,916 poisonous. The `?` markers in `stalk-root` are treated as missing and imputed with the most frequent category using training data only.

| Item | Description |
| --- | --- |
| Target | `class` (`e` = edible, `p` = poisonous) |
| Predictors | `cap-shape`, `cap-surface`, `cap-color`, `bruises`, `odor`, `gill-attachment`, `gill-spacing`, `gill-size`, `gill-color`, `stalk-shape`, `stalk-root`, `stalk-surface-above-ring`, `stalk-surface-below-ring`, `stalk-color-above-ring`, `stalk-color-below-ring`, `veil-type`, `veil-color`, `ring-number`, `ring-type`, `spore-print-color`, `population`, `habitat` |
| Instances | 8,124 |
| Predictor count | 22 |
| Source | [UCI Machine Learning Repository: Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom) |

## GitHub Repository

[https://github.com/baarathsrinivas/ml_assignment](https://github.com/baarathsrinivas/ml_assignment)

## Models and Results

The models were trained using an 80/20 stratified split. The test set has 800 labeled instances. Numeric features are median-imputed and standardized; categorical features, if present, are imputed with the most frequent value and one-hot encoded. Preprocessing is fitted on the training partition only.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9988 | 1.0000 | 0.9988 | 0.9988 | 0.9988 | 0.9975 |
| Decision Tree | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| K-Nearest Neighbors | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Gaussian Naive Bayes | 0.9569 | 0.9963 | 0.9601 | 0.9569 | 0.9569 | 0.9170 |
| Support Vector Machine | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### Model Observations and Winner

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.9988, MCC 0.9975, AUC 1.0000. The near-perfect result shows the classes are almost linearly separable in the encoded feature space on this split. |
| Decision Tree | Scored 1.0000 on all six metrics. It represents categorical feature interactions and gives a comparatively interpretable decision structure. |
| KNN | Also 1.0000 on all six metrics. Neighbor voting separates this test set perfectly after one-hot encoding; inference cost and sensitivity to feature representation matter for larger or shifted data. |
| Naive Bayes | Accuracy 0.9569, MCC 0.9170, below the other classifiers. The conditional-independence assumption is a less suitable fit for dependent mushroom attributes. |
| Support Vector Machine | Tied at 1.0000 on every metric. Its margin-based boundary separates the encoded test examples perfectly, but it does not outperform the simpler tied models on this split. |
| Overall, Winner for your dataset? | **Decision Tree**, tied with KNN and SVM at 1.0000 on all six metrics. It is the tie-breaker because it offers the most inspectable decision structure without sacrificing measured performance. One random split does not prove generalization; cross-validation and an independent external test set are recommended. |

## Run the Project

Install the dependencies and train the models:

```bash
pip install -r requirements.txt
python train_models.py
```

Training writes `test_data.csv` and the fitted preprocessing/model artifacts under `model/`. Start the evaluation app with:

```bash
streamlit run app.py
```

The app accepts a labeled CSV upload or uses the generated `test_data.csv`. It loads the selected pre-trained model and preprocessor for inference; it does not retrain models.

Run the smoke tests with:

```bash
pytest -q
```