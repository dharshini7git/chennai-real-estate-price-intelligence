import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "datasets/chennai_housing_combined.csv"

MODEL_FOLDER = "models"
RESULT_FOLDER = "final_results"
INTELLIGENCE_FOLDER = "price_intelligence"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(INTELLIGENCE_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Original dataset:", df.shape)


# ============================================================
# 3. FINAL FEATURE SET
# ============================================================
#
# IMPORTANT:
# price_per_sqft is NOT used because it is derived from price.
#
# source_dataset is also NOT used as an input feature.
# A new property does not inherently belong to one of the
# historical source datasets.
#
# property_id is an identifier, not a useful property feature.
#
# reg_fee and commission are target leakage and are absent
# from the combined dataset anyway.
# ============================================================

features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "total_rooms",
    "building_age",

    "location",
    "property_type",
    "parking",
    "construction_status",

    "swimming_pool",
    "gymnasium",
    "landscaped_gardens",
    "club_house",
    "power_backup",
    "intercom",
    "security_24x7",
    "childrens_play_area"
]

target = "price"


# Keep only available columns
features = [
    col for col in features
    if col in df.columns
]

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 4. REMOVE INVALID TARGETS
# ============================================================

valid = y > 0

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

print("Final ML dataset:", X.shape)


# ============================================================
# 5. COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 6. PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 8. MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
}


# ============================================================
# 9. TRAIN + TEST EVALUATION
# ============================================================

results = {}

trained_pipelines = {}


print("\n================================")
print("FINAL MODEL EVALUATION")
print("================================")


for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    trained_pipelines[name] = pipeline

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))


# ============================================================
# 10. SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="RMSE"
)

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(results_df)


results_df.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "final_model_comparison.csv"
    )
)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.index[0]

print("\nSelected model based on test RMSE:")
print(best_model_name)

best_test_pipeline = trained_pipelines[
    best_model_name
]


# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = best_test_pipeline.predict(
    X_test
)

test_results = X_test.copy()

test_results["actual_price"] = y_test.values
test_results["predicted_price"] = test_predictions

test_results["error"] = (
    test_results["actual_price"]
    - test_results["predicted_price"]
)

test_results["absolute_error"] = (
    abs(test_results["error"])
)

test_results.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "final_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 13. 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n================================")
print("5-FOLD CROSS-VALIDATION")
print("================================")

rf_pipeline = trained_pipelines["Random Forest"]

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

print("\nR² scores:")
print(cv_scores)

print(
    "\nMean CV R²:",
    round(cv_scores.mean(), 4)
)

print(
    "Std CV R²:",
    round(cv_scores.std(), 4)
)


cv_results = pd.DataFrame({
    "fold": range(1, 6),
    "R2": cv_scores
})

cv_results.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "random_forest_cv_results.csv"
    ),
    index=False
)


# ============================================================
# 14. OUT-OF-FOLD PREDICTIONS
# ============================================================
#
# This is the important part for price intelligence.
#
# Every property gets a prediction from a model that did NOT
# train on that property.
# ============================================================

print("\n================================")
print("GENERATING OUT-OF-FOLD PREDICTIONS")
print("================================")


oof_predictions = np.zeros(
    len(X)
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for fold, (train_index, valid_index) in enumerate(
    kf.split(X),
    start=1
):

    print("Processing fold", fold)

    X_fold_train = X.iloc[train_index]
    X_fold_valid = X.iloc[valid_index]

    y_fold_train = y.iloc[train_index]

    fold_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    fold_model.fit(
        X_fold_train,
        y_fold_train
    )

    oof_predictions[valid_index] = (
        fold_model.predict(
            X_fold_valid
        )
    )


# ============================================================
# 15. OOF EVALUATION
# ============================================================

oof_mae = mean_absolute_error(
    y,
    oof_predictions
)

oof_rmse = np.sqrt(
    mean_squared_error(
        y,
        oof_predictions
    )
)

oof_r2 = r2_score(
    y,
    oof_predictions
)


print("\nOOF MAE :", round(oof_mae, 2))
print("OOF RMSE:", round(oof_rmse, 2))
print("OOF R²  :", round(oof_r2, 4))


# ============================================================
# 16. PRICE INTELLIGENCE
# ============================================================

intelligence = X.copy()

intelligence["actual_price"] = y.values

intelligence["fair_price"] = oof_predictions

intelligence["price_difference"] = (
    intelligence["actual_price"]
    - intelligence["fair_price"]
)

intelligence["price_gap_percent"] = (
    intelligence["price_difference"]
    / intelligence["fair_price"]
) * 100


def classify_price(gap):

    if gap <= -10:
        return "Potentially Underpriced"

    elif gap >= 10:
        return "Potentially Overpriced"

    else:
        return "Fairly Priced"


intelligence["price_category"] = (
    intelligence["price_gap_percent"]
    .apply(classify_price)
)


# ============================================================
# 17. SAVE PRICE INTELLIGENCE
# ============================================================

intelligence.to_csv(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "price_intelligence_oof.csv"
    ),
    index=False
)


# Separate files

underpriced = intelligence[
    intelligence["price_category"]
    == "Potentially Underpriced"
]

overpriced = intelligence[
    intelligence["price_category"]
    == "Potentially Overpriced"
]

fairly_priced = intelligence[
    intelligence["price_category"]
    == "Fairly Priced"
]


underpriced.to_csv(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "potentially_underpriced_oof.csv"
    ),
    index=False
)

overpriced.to_csv(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "potentially_overpriced_oof.csv"
    ),
    index=False
)

fairly_priced.to_csv(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "fairly_priced_oof.csv"
    ),
    index=False
)


# ============================================================
# 18. PRICE INTELLIGENCE SUMMARY
# ============================================================

category_counts = (
    intelligence["price_category"]
    .value_counts()
)

print("\n================================")
print("FINAL PRICE INTELLIGENCE")
print("================================")

print(category_counts)

print(
    "\nAverage price gap:",
    round(
        intelligence[
            "price_gap_percent"
        ].mean(),
        2
    ),
    "%"
)


# ============================================================
# 19. TRAIN FINAL MODEL ON ALL DATA
# ============================================================
#
# This model is used by the future dashboard.
#
# It is trained on all available historical data AFTER
# evaluation has already been completed.
# ============================================================

print("\n================================")
print("TRAINING FINAL DASHBOARD MODEL")
print("================================")


final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


final_model.fit(
    X,
    y
)


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_FOLDER,
    "final_price_model.pkl"
)

joblib.dump(
    final_model,
    final_model_path
)


print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n================================")
print("FINAL PIPELINE COMPLETED")
print("================================")

print("Best evaluated model:", best_model_name)

print(
    "Test R²:",
    round(
        results_df.loc[
            best_model_name,
            "R2"
        ],
        4
    )
)

print(
    "5-Fold CV Mean R²:",
    round(
        cv_scores.mean(),
        4
    )
)

print(
    "OOF R²:",
    round(
        oof_r2,
        4
    )
)

print("\nFinal dashboard model:")
print(final_model_path)