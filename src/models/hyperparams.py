"""
Default hyperparameter configurations for model training.

The churn dataset has a class imbalance (~26.5% positive class),
so we use scale_pos_weight to compensate. This is calculated as:
  (count of negatives) / (count of positives) ≈ 73.5 / 26.5 ≈ 2.77 → rounded to 3.0
"""

# XGBoost default configuration
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "scale_pos_weight": 3.0,  # Compensate for 26.5% churn imbalance
    "eval_metric": "logloss",
    "random_state": 42,
    "use_label_encoder": False,
}

# Hyperparameter search space (for future Optuna integration)
XGBOOST_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "scale_pos_weight": [2.0, 3.0, 4.0],
}

# Model registry settings
MODEL_NAME = "churn-model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
EXPERIMENT_NAME = "telco-churn-experiment"
