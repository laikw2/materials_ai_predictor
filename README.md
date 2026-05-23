# Polymer Cross-Presentation AI Prediction

Reproducible ML project for predicting `cross_presentation_pct` from polymer design descriptors.

The main task is regression. A secondary median-split classifier is included only for ROC-AUC, PR-AUC, confusion matrix, precision, recall, accuracy, and F1 reporting.

## Quick Start

```powershell
cd polymer_cross_presentation_ai
python src\train_regression.py --quick --n-iter 4
python src\train_binary_classifier.py --n-iter 4
python src\predict.py --example
streamlit run app\streamlit_app.py
```

For a heavier repeated-CV run, use:

```powershell
python src\train_regression.py --repeats 20 --n-iter 8
python src\train_binary_classifier.py --n-iter 8
```

## Outputs

- `models/best_regression_model.joblib`
- `models/best_classifier_model.joblib`
- `reports/metrics/*.csv`
- `reports/metrics/*.json`
- `reports/figures/*.png`

## Important Scientific Note

This model is intended for ML-assisted polymer design prioritization, not for replacing wetlab validation. Predictions outside the training descriptor ranges should be treated as extrapolative hypotheses.
