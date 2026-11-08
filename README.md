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

[Insert repository URL](https://github.com/<username>/<repository>)

## Models and Results

The models were trained using an 80/20 stratified split. The test set has 800 labeled instances. Numeric features are median-imputed and standardized; categorical features, if present, are imputed with the most frequent value and one-hot encoded. Preprocessing is fitted on the training partition only.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9988 | 1.0000 | 0.9988 | 0.9988 | 0.9988 | 0.9975 |
| Decision Tree | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| K-Nearest Neighbors | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Gaussian Naive Bayes | 0.9569 | 0.9963 | 0.9601 | 0.9569 | 0.9569 | 0.9170 |
| Support Vector Machine | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### Model Observations

- **Logistic Regression:** Reached 0.9988 accuracy and 0.9975 MCC, with an AUC of 1.0000. Its near-perfect result shows the classes are almost linearly separable in the encoded feature space on this split.
- **Decision Tree:** Achieved 1.0000 on all six metrics. The tree can represent categorical feature interactions and provides a comparatively interpretable decision structure.
- **K-Nearest Neighbors:** Also achieved 1.0000 on all six metrics. Neighbor voting separates this test set perfectly after one-hot encoding, though inference cost and sensitivity to feature representation should be considered for larger or shifted data.
- **Gaussian Naive Bayes:** Scored 0.9569 accuracy and 0.9170 MCC, below the other classifiers. The gap is consistent with the model's conditional-independence assumption being a less suitable fit for dependent mushroom attributes.
- **Support Vector Machine:** Tied for 1.0000 on every metric. Its margin-based boundary separates the encoded test examples perfectly, but it does not outperform the simpler tied models on this split.

### Winner

**Overall test-set winner: Decision Tree, tied with K-Nearest Neighbors and Support Vector Machine.** All three attained 1.0000 across all six metrics, so the holdout results do not distinguish them statistically. The Decision Tree is the practical tie-breaker here because it offers a more inspectable decision structure without sacrificing measured performance. Perfect results on one random split are not proof of generalization; cross-validation and an independent external test set are recommended.

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