import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. File paths
# ==============================

DATA_FOLDER = "datasets"
OUTPUT_FOLDER = "eda_plots"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==============================
# 2. Load datasets
# ==============================

metropolitan = pd.read_csv(
    os.path.join(DATA_FOLDER, "cleaned_metropolitan_chennai.csv")
)

house_price = pd.read_csv(
    os.path.join(DATA_FOLDER, "cleaned_chennai_house_price.csv")
)

housing_sales = pd.read_csv(
    os.path.join(DATA_FOLDER, "cleaned_chennai_housing_sales.csv")
)

combined = pd.read_csv(
    os.path.join(DATA_FOLDER, "chennai_housing_combined.csv")
)


# ==============================
# 3. Basic information
# ==============================

print("\n===== DATASET SHAPES =====")

print("Metropolitan:", metropolitan.shape)
print("House Price:", house_price.shape)
print("Housing Sales:", housing_sales.shape)
print("Combined:", combined.shape)


print("\n===== COMBINED DATASET INFO =====")
print(combined.info())


print("\n===== BASIC STATISTICS =====")
print(combined[["price", "area_sqft", "bedrooms"]].describe())


# ==============================
# 4. Price Distribution
# ==============================

plt.figure(figsize=(10, 6))

plt.hist(combined["price"].dropna(), bins=50)

plt.title("Distribution of Property Prices")
plt.xlabel("Price (INR)")
plt.ylabel("Number of Properties")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "01_price_distribution.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 5. Area vs Price
# ==============================

plt.figure(figsize=(10, 6))

plt.scatter(
    combined["area_sqft"],
    combined["price"],
    alpha=0.4
)

plt.title("Area vs Property Price")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "02_area_vs_price.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 6. Price vs Bedrooms
# ==============================

bedroom_price = (
    combined
    .groupby("bedrooms")["price"]
    .mean()
    .dropna()
)

plt.figure(figsize=(10, 6))

plt.bar(
    bedroom_price.index.astype(str),
    bedroom_price.values
)

plt.title("Average Property Price by Number of Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Price (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "03_price_by_bedrooms.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 7. Top Locations by Average Price
# ==============================

location_price = (
    combined
    .groupby("location")["price"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

# Only locations having at least 20 properties
location_price = location_price[
    location_price["count"] >= 20
].head(15)

plt.figure(figsize=(12, 7))

plt.barh(
    location_price.index[::-1],
    location_price["mean"][::-1]
)

plt.title("Top Locations by Average Property Price")
plt.xlabel("Average Price (INR)")
plt.ylabel("Location")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "04_average_price_by_location.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 8. Price per sqft by Location
# ==============================

location_pps = (
    combined
    .groupby("location")["price_per_sqft"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

# Only locations with at least 20 properties
location_pps = location_pps[
    location_pps["count"] >= 20
].head(15)

plt.figure(figsize=(12, 7))

plt.barh(
    location_pps.index[::-1],
    location_pps["mean"][::-1]
)

plt.title("Top Locations by Average Price per Sqft")
plt.xlabel("Average Price per Sqft (INR)")
plt.ylabel("Location")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "05_price_per_sqft_by_location.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 9. Correlation Heatmap
# ==============================

numeric_columns = combined.select_dtypes(
    include="number"
).columns

correlation = combined[numeric_columns].corr()

plt.figure(figsize=(12, 9))

plt.imshow(correlation, cmap="coolwarm", aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "06_correlation_heatmap.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 10. Dataset Source Comparison
# ==============================

source_counts = combined["source_dataset"].value_counts()

plt.figure(figsize=(10, 6))

plt.bar(
    source_counts.index,
    source_counts.values
)

plt.title("Number of Properties by Dataset Source")
plt.xlabel("Dataset Source")
plt.ylabel("Number of Properties")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "07_dataset_source_comparison.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 11. Average Price by Dataset
# ==============================

source_price = (
    combined
    .groupby("source_dataset")["price"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.bar(
    source_price.index,
    source_price.values
)

plt.title("Average Property Price by Dataset Source")
plt.xlabel("Dataset Source")
plt.ylabel("Average Price (INR)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "08_average_price_by_source.png"),
    dpi=300
)

plt.show()
plt.close()


# ==============================
# 12. Final message
# ==============================

print("\n================================")
print("EDA COMPLETED SUCCESSFULLY")
print("================================")

print(f"\nPlots saved in: {OUTPUT_FOLDER}/")

print("\nGenerated files:")

for file in sorted(os.listdir(OUTPUT_FOLDER)):
    print("-", file)
# ============================================================
# STEP 2 - FEATURE RELATIONSHIP ANALYSIS
# ============================================================

print("\n================================")
print("STEP 2: FEATURE RELATIONSHIP ANALYSIS")
print("================================")


# ------------------------------------------------------------
# 1. Correlation of numerical features with PRICE
# ------------------------------------------------------------

numeric_data = combined.select_dtypes(include="number")

price_correlation = (
    numeric_data.corr()["price"]
    .sort_values(ascending=False)
)

# Remove price itself
price_correlation = price_correlation.drop("price")

print("\n===== CORRELATION WITH PRICE =====")
print(price_correlation)


# Save correlation values
price_correlation.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "price_feature_correlations.csv"
    )
)


# ------------------------------------------------------------
# 2. Plot top positive and negative correlations
# ------------------------------------------------------------

# Remove derived/leakage variables from this visualization
correlation_plot = price_correlation.drop(
    labels=["price_per_sqft"],
    errors="ignore"
)

# Select 15 strongest relationships
top_correlations = (
    correlation_plot.abs()
    .sort_values(ascending=False)
    .head(15)
)

top_features = top_correlations.index

top_values = correlation_plot[top_features].sort_values()


plt.figure(figsize=(10, 7))

plt.barh(
    top_values.index,
    top_values.values
)

plt.title("Top Features Correlated with Property Price")
plt.xlabel("Correlation with Price")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "09_top_price_correlations.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 3. Area vs Price - improved version
# ------------------------------------------------------------

area_data = combined[
    ["area_sqft", "price"]
].dropna()

plt.figure(figsize=(10, 6))

plt.scatter(
    area_data["area_sqft"],
    area_data["price"],
    alpha=0.3
)

plt.title("Relationship Between Area and Property Price")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "10_area_price_relationship.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 4. Bedrooms vs Price
# ------------------------------------------------------------

bedroom_data = (
    combined
    .groupby("bedrooms")["price"]
    .agg(["mean", "median", "count"])
    .dropna()
)

print("\n===== PRICE BY BEDROOMS =====")
print(bedroom_data)

bedroom_data.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "price_by_bedrooms.csv"
    )
)


plt.figure(figsize=(9, 6))

plt.bar(
    bedroom_data.index.astype(str),
    bedroom_data["mean"]
)

plt.title("Average Property Price by Number of Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Average Price (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "11_bedrooms_price_relationship.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 5. Bathrooms vs Price
# ------------------------------------------------------------

bathroom_data = (
    combined
    .groupby("bathrooms")["price"]
    .agg(["mean", "median", "count"])
    .dropna()
)

print("\n===== PRICE BY BATHROOMS =====")
print(bathroom_data)


plt.figure(figsize=(9, 6))

plt.bar(
    bathroom_data.index.astype(str),
    bathroom_data["mean"]
)

plt.title("Average Property Price by Number of Bathrooms")
plt.xlabel("Bathrooms")
plt.ylabel("Average Price (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "12_bathrooms_price_relationship.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 6. Building Age vs Price
# ------------------------------------------------------------

age_data = combined[
    ["building_age", "price"]
].dropna()

# Limit visualization to reasonable age range
age_data = age_data[
    age_data["building_age"] <= 50
]

plt.figure(figsize=(10, 6))

plt.scatter(
    age_data["building_age"],
    age_data["price"],
    alpha=0.3
)

plt.title("Building Age vs Property Price")
plt.xlabel("Building Age (years)")
plt.ylabel("Price (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "13_building_age_vs_price.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 7. Price per sqft distribution
# ------------------------------------------------------------

pps_data = combined[
    "price_per_sqft"
].dropna()

# Use 1st to 99th percentile only for visualization
lower = pps_data.quantile(0.01)
upper = pps_data.quantile(0.99)

pps_filtered = pps_data[
    (pps_data >= lower) &
    (pps_data <= upper)
]

plt.figure(figsize=(10, 6))

plt.hist(
    pps_filtered,
    bins=50
)

plt.title("Distribution of Price per Square Foot")
plt.xlabel("Price per Sqft (INR)")
plt.ylabel("Number of Properties")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "14_price_per_sqft_distribution.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 8. Dataset-wise price comparison
# ------------------------------------------------------------

dataset_summary = (
    combined
    .groupby("source_dataset")["price"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\n===== DATASET PRICE SUMMARY =====")
print(dataset_summary)

dataset_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "dataset_price_summary.csv"
    )
)


# ------------------------------------------------------------
# 9. Dataset-wise area comparison
# ------------------------------------------------------------

area_summary = (
    combined
    .groupby("source_dataset")["area_sqft"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\n===== DATASET AREA SUMMARY =====")
print(area_summary)

area_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "dataset_area_summary.csv"
    )
)


# ------------------------------------------------------------
# 10. Final message
# ------------------------------------------------------------

print("\n================================")
print("STEP 2 COMPLETED")
print("================================")

print("\nNew files created:")

for file in sorted(os.listdir(OUTPUT_FOLDER)):
    print("-", file)