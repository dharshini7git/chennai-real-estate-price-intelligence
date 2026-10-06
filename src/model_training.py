import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "datasets/chennai_housing_combined.csv"

MODEL_FOLDER = "models"
RESULT_FOLDER = "model_results"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES
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
    "source_dataset",

    # Selected amenities
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


# Keep only columns that actually exist
features = [
    col for col in features
    if col in df.columns
]

print("\nFeatures used:")
for feature in features:
    print("-", feature)


# ============================================================
# 4. CREATE X AND y
# ============================================================

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 5. REMOVE INVALID TARGET VALUES
# ============================================================

valid_rows = y > 0

X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("\nFinal ML dataset:", X.shape)


# ============================================================
# 6. IDENTIFY COLUMN TYPES
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
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 9. MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
}


# ============================================================
# 10. TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n--------------------------------")
    print("Training:", name)
    print("--------------------------------")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
)

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(results_df)


# Save results
results_df.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_model_name
]

print("\n================================")
print("BEST MODEL")
print("================================")

print(best_model_name)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_FOLDER,
    "best_price_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved to:")
print(model_path)


# ============================================================
# 14. SAVE TEST PREDICTIONS
# ============================================================

best_predictions = best_model.predict(X_test)

prediction_results = X_test.copy()

prediction_results["actual_price"] = y_test.values

prediction_results["predicted_price"] = best_predictions

prediction_results["price_difference"] = (
    prediction_results["actual_price"]
    - prediction_results["predicted_price"]
)

prediction_results.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "test_predictions.csv"
    ),
    index=False
)


print("\n================================")
print("MODEL TRAINING COMPLETED")
print("================================")
# ============================================================
# STEP 4 - MODEL EXPLAINABILITY
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


EXPLAIN_FOLDER = "explainability"
os.makedirs(EXPLAIN_FOLDER, exist_ok=True)


# ============================================================
# 1. Get Random Forest pipeline
# ============================================================

rf_pipeline = trained_models["Random Forest"]

rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_model = rf_pipeline.named_steps["model"]


# ============================================================
# 2. Get transformed feature names
# ============================================================

feature_names = rf_preprocessor.get_feature_names_out()

feature_importances = rf_model.feature_importances_


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances
})


# Remove preprocessing prefixes
importance_df["feature"] = (
    importance_df["feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)


# Sort by importance
importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print("\n================================")
print("TOP FEATURE IMPORTANCE")
print("================================")

print(
    importance_df.head(20).to_string(index=False)
)


# Save feature importance
importance_df.to_csv(
    os.path.join(
        EXPLAIN_FOLDER,
        "feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 3. Feature importance plot
# ============================================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.title("Top Features Influencing Random Forest Predictions")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        EXPLAIN_FOLDER,
        "01_feature_importance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. Actual vs Predicted
# ============================================================

actual = y_test.values
predicted = rf_pipeline.predict(X_test)


plt.figure(figsize=(8, 8))

plt.scatter(
    actual,
    predicted,
    alpha=0.3
)

# Perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Property Prices")
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        EXPLAIN_FOLDER,
        "02_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. Residual Analysis
# ============================================================

residuals = actual - predicted


plt.figure(figsize=(10, 6))

plt.scatter(
    predicted,
    residuals,
    alpha=0.3
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Prediction Residuals")
plt.xlabel("Predicted Price (INR)")
plt.ylabel("Residual (Actual - Predicted)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        EXPLAIN_FOLDER,
        "03_prediction_residuals.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 6. Save Random Forest evaluation
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    predicted
)

rf_rmse = mean_squared_error(
    y_test,
    predicted
) ** 0.5

rf_r2 = r2_score(
    y_test,
    predicted
)


evaluation = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "value": [
        rf_mae,
        rf_rmse,
        rf_r2
    ]
})


evaluation.to_csv(
    os.path.join(
        EXPLAIN_FOLDER,
        "random_forest_evaluation.csv"
    ),
    index=False
)


print("\n================================")
print("EXPLAINABILITY COMPLETED")
print("================================")

print("\nFiles created:")

for file in sorted(os.listdir(EXPLAIN_FOLDER)):
    print("-", file)
# ============================================================
# STEP 5 - FAIR PRICE AND PRICE INTELLIGENCE
# ============================================================

INTELLIGENCE_FOLDER = "price_intelligence"

os.makedirs(
    INTELLIGENCE_FOLDER,
    exist_ok=True
)


# ------------------------------------------------------------
# 1. Predict prices for the complete dataset
# ------------------------------------------------------------

all_predictions = rf_pipeline.predict(X)


# ------------------------------------------------------------
# 2. Create price intelligence dataframe
# ------------------------------------------------------------

price_intelligence = X.copy()

price_intelligence["actual_price"] = y.values

price_intelligence["fair_price"] = all_predictions


# ------------------------------------------------------------
# 3. Calculate price difference
# ------------------------------------------------------------

price_intelligence["price_difference"] = (
    price_intelligence["actual_price"]
    - price_intelligence["fair_price"]
)


# ------------------------------------------------------------
# 4. Calculate percentage difference
# ------------------------------------------------------------

price_intelligence["price_gap_percent"] = (
    price_intelligence["price_difference"]
    / price_intelligence["fair_price"]
) * 100


# ------------------------------------------------------------
# 5. Classify properties
# ------------------------------------------------------------

def classify_price(row):

    gap = row["price_gap_percent"]

    if gap <= -10:
        return "Potentially Underpriced"

    elif gap >= 10:
        return "Potentially Overpriced"

    else:
        return "Fairly Priced"


price_intelligence["price_category"] = (
    price_intelligence.apply(
        classify_price,
        axis=1
    )
)


# ------------------------------------------------------------
# 6. Save complete price intelligence dataset
# ------------------------------------------------------------

price_intelligence.to_csv(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "price_intelligence.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 7. Print summary
# ------------------------------------------------------------

print("\n================================")
print("PRICE INTELLIGENCE SUMMARY")
print("================================")

print(
    price_intelligence["price_category"]
    .value_counts()
)


# ------------------------------------------------------------
# 8. Average price gap
# ------------------------------------------------------------

print("\nAverage price gap:")

print(
    round(
        price_intelligence["price_gap_percent"].mean(),
        2
    ),
    "%"
)


# ------------------------------------------------------------
# 9. Potentially underpriced properties
# ------------------------------------------------------------

underpriced = (
    price_intelligence[
        price_intelligence["price_category"]
        == "Potentially Underpriced"
    ]
    .sort_values(
        "price_gap_percent"
    )
)


underpriced.to_csv(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "potentially_underpriced.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 10. Potentially overpriced properties
# ------------------------------------------------------------

overpriced = (
    price_intelligence[
        price_intelligence["price_category"]
        == "Potentially Overpriced"
    ]
    .sort_values(
        "price_gap_percent",
        ascending=False
    )
)


overpriced.to_csv(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "potentially_overpriced.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 11. Price category visualization
# ------------------------------------------------------------

category_counts = (
    price_intelligence["price_category"]
    .value_counts()
)


plt.figure(figsize=(9, 6))

plt.bar(
    category_counts.index,
    category_counts.values
)

plt.title(
    "Property Price Intelligence Classification"
)

plt.xlabel("Price Category")
plt.ylabel("Number of Properties")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "01_price_categories.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 12. Price gap distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    price_intelligence["price_gap_percent"],
    bins=50
)

plt.axvline(
    x=-10,
    linestyle="--"
)

plt.axvline(
    x=10,
    linestyle="--"
)

plt.title(
    "Distribution of Price Gap from Model Fair Price"
)

plt.xlabel(
    "Price Gap (%)"
)

plt.ylabel(
    "Number of Properties"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        INTELLIGENCE_FOLDER,
        "02_price_gap_distribution.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 13. Final output
# ------------------------------------------------------------

print("\n================================")
print("PRICE INTELLIGENCE COMPLETED")
print("================================")

print("\nFiles created:")

for file in sorted(
    os.listdir(INTELLIGENCE_FOLDER)
):
    print("-", file)