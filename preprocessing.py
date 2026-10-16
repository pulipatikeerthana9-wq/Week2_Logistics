import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# LOGISTICS DATA PREPROCESSING - WEEK 2
# ============================================================

FILE_PATH = "DataCoSupplyChainDataset.csv"

print("=" * 70)
print("LOGISTICS DATA COLLECTION, CLEANING AND PREPROCESSING")
print("=" * 70)


# ============================================================
# 1. DATA COLLECTION / LOADING
# ============================================================

df = pd.read_csv(FILE_PATH, encoding="latin1")

original_rows, original_columns = df.shape

print("\n--- DATASET OVERVIEW ---")
print("Rows:", original_rows)
print("Columns:", original_columns)


# ============================================================
# 2. INITIAL DATA QUALITY ASSESSMENT
# ============================================================

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES BEFORE CLEANING ---")

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0].sort_values(
    ascending=False
)

print(missing_before)

print("\n--- DUPLICATE RECORDS ---")
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)


# ============================================================
# 3. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

empty_columns = df.columns[df.isnull().all()].tolist()

print("\n--- COMPLETELY EMPTY COLUMNS ---")
print(empty_columns)

if empty_columns:
    df = df.drop(columns=empty_columns)

print("Columns after removing empty columns:", len(df.columns))


# ============================================================
# 4. REMOVE DUPLICATE RECORDS
# ============================================================

if duplicate_count > 0:
    df = df.drop_duplicates()

print("\n--- DUPLICATE HANDLING ---")
print("Duplicates removed:", duplicate_count)
print("Rows after duplicate removal:", len(df))


# ============================================================
# 5. HANDLE CUSTOMER LAST NAME
# ============================================================

print("\n--- CUSTOMER LAST NAME ---")

if "Customer Lname" in df.columns:

    missing_lname = df["Customer Lname"].isnull().sum()

    df["Customer Lname"] = df["Customer Lname"].fillna("Unknown")

    print("Missing values:", missing_lname)
    print("Replacement: 'Unknown'")


# ============================================================
# 6. HANDLE POSTAL / ZIP CODE FIELDS
# ============================================================

print("\n--- ZIP CODE HANDLING ---")

# ZIP codes are identifiers, not continuous numerical variables.
# Therefore, median imputation is avoided.

if "Customer Zipcode" in df.columns:

    missing_customer_zip = df["Customer Zipcode"].isnull().sum()

    df["Customer Zipcode"] = (
        df["Customer Zipcode"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    df["Customer Zipcode"] = df["Customer Zipcode"].replace(
        "-1", "Unknown"
    )

    print(
        "Customer Zipcode missing values handled:",
        missing_customer_zip
    )


if "Order Zipcode" in df.columns:

    missing_order_zip = df["Order Zipcode"].isnull().sum()

    df["Order Zipcode"] = (
        df["Order Zipcode"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    df["Order Zipcode"] = df["Order Zipcode"].replace(
        "-1", "Unknown"
    )

    print(
        "Order Zipcode missing values handled:",
        missing_order_zip
    )


# ============================================================
# 7. DATE CONVERSION
# ============================================================

print("\n--- DATE CONVERSION ---")

date_columns = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for column in date_columns:

    if column in df.columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        print(f"{column}: converted to datetime")


# ============================================================
# 8. CHECK INVALID DATE VALUES
# ============================================================

print("\n--- INVALID DATE VALUES ---")

for column in date_columns:

    if column in df.columns:

        invalid_dates = df[column].isnull().sum()

        print(
            f"{column}: {invalid_dates} invalid/missing dates"
        )


# ============================================================
# 9. OUTLIER DETECTION USING IQR
# ============================================================

print("\n--- OUTLIER DETECTION USING IQR ---")

outlier_columns = [
    "Sales",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Profit Ratio",
    "Days for shipping (real)"
]

outlier_summary = []

for column in outlier_columns:

    if column in df.columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        )

        outlier_count = outlier_mask.sum()

        outlier_summary.append({
            "Column": column,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers": outlier_count
        })

        print(f"\n{column}")
        print("Q1:", Q1)
        print("Q3:", Q3)
        print("IQR:", IQR)
        print("Lower Bound:", lower_bound)
        print("Upper Bound:", upper_bound)
        print("Outliers:", outlier_count)


# ============================================================
# 10. CREATE OUTLIER SUMMARY FILE
# ============================================================

outlier_df = pd.DataFrame(outlier_summary)

outlier_df.to_csv(
    "outlier_summary.csv",
    index=False
)

print("\nOutlier summary saved as: outlier_summary.csv")


# ============================================================
# 11. CREATE ANALYTICAL DATASET
# ============================================================

print("\n--- ANALYTICAL DATASET ---")

# Select numerical variables suitable for normalization.
# Identifier fields and ZIP codes are intentionally excluded.

scale_columns = [
    "Sales",
    "Order Item Quantity",
    "Days for shipping (real)"
]

available_scale_columns = [
    column
    for column in scale_columns
    if column in df.columns
]

analytical_df = df.copy()


# ============================================================
# 12. MIN-MAX NORMALIZATION
# ============================================================

print("\n--- MIN-MAX NORMALIZATION ---")

scaler = MinMaxScaler()

analytical_df[available_scale_columns] = scaler.fit_transform(
    analytical_df[available_scale_columns]
)

for column in available_scale_columns:

    print(
        f"{column}: normalized to range "
        f"{analytical_df[column].min():.2f} - "
        f"{analytical_df[column].max():.2f}"
    )


# ============================================================
# 13. BEFORE / AFTER MISSING VALUE CHECK
# ============================================================

print("\n--- MISSING VALUES AFTER CLEANING ---")

missing_after = analytical_df.isnull().sum().sum()

print(
    "Total remaining missing values:",
    missing_after
)


# ============================================================
# 14. SAVE CLEANED DATASET
# ============================================================

cleaned_file = "DataCoSupplyChain_Cleaned.csv"

df.to_csv(
    cleaned_file,
    index=False
)

print("\nCleaned dataset saved as:")
print(cleaned_file)


# ============================================================
# 15. SAVE NORMALIZED ANALYTICAL DATASET
# ============================================================

normalized_file = "DataCoSupplyChain_Normalized.csv"

analytical_df.to_csv(
    normalized_file,
    index=False
)

print("\nNormalized analytical dataset saved as:")
print(normalized_file)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 70)

print("Original rows:", original_rows)
print("Final rows:", len(df))

print("Original columns:", original_columns)
print("Final columns:", len(df.columns))

print("Duplicate rows removed:", duplicate_count)
print("Empty columns removed:", len(empty_columns))

print(
    "Remaining missing values:",
    missing_after
)

print("\nFiles generated:")
print("1. DataCoSupplyChain_Cleaned.csv")
print("2. DataCoSupplyChain_Normalized.csv")
print("3. outlier_summary.csv")

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)