import pandas as pd

# ============================================================
# LOGISTICS DATA QUALITY AND ANALYSIS SUMMARY
# ============================================================

FILE_PATH = "DataCoSupplyChain_Cleaned.csv"

df = pd.read_csv(
    FILE_PATH,
    encoding="latin1",
    low_memory=False
)

print("=" * 70)
print("LOGISTICS DATA QUALITY AND ANALYSIS SUMMARY")
print("=" * 70)

# SALES SUMMARY
print("\n--- SALES SUMMARY ---")

print("Minimum Sales:", round(df["Sales"].min(), 2))
print("Maximum Sales:", round(df["Sales"].max(), 2))
print("Average Sales:", round(df["Sales"].mean(), 2))
print("Median Sales:", round(df["Sales"].median(), 2))

# SHIPPING PERFORMANCE
print("\n--- SHIPPING PERFORMANCE ---")

average_actual = df["Days for shipping (real)"].mean()
average_scheduled = df["Days for shipment (scheduled)"].mean()

print(
    "Average actual shipping days:",
    round(average_actual, 2)
)

print(
    "Average scheduled shipping days:",
    round(average_scheduled, 2)
)

print(
    "Average shipping delay:",
    round(
        average_actual - average_scheduled,
        2
    ),
    "days"
)

# LATE DELIVERY RISK
print("\n--- LATE DELIVERY RISK ---")

late_risk_count = df["Late_delivery_risk"].sum()

late_risk_percentage = (
    late_risk_count / len(df)
) * 100

print(
    "Orders with late-delivery risk:",
    late_risk_count
)

print(
    "Late-delivery risk percentage:",
    round(late_risk_percentage, 2),
    "%"
)

# ORDER QUANTITY
print("\n--- ORDER QUANTITY ---")

print(
    "Average order item quantity:",
    round(df["Order Item Quantity"].mean(), 2)
)

print(
    "Maximum order item quantity:",
    df["Order Item Quantity"].max()
)

# DISCOUNT
print("\n--- DISCOUNT SUMMARY ---")

print(
    "Average discount:",
    round(df["Order Item Discount"].mean(), 2)
)

print(
    "Maximum discount:",
    round(df["Order Item Discount"].max(), 2)
)

# PROFIT
print("\n--- PROFIT SUMMARY ---")

print(
    "Average profit per order:",
    round(df["Order Profit Per Order"].mean(), 2)
)

print(
    "Minimum profit per order:",
    round(df["Order Profit Per Order"].min(), 2)
)

print(
    "Maximum profit per order:",
    round(df["Order Profit Per Order"].max(), 2)
)

# DELIVERY STATUS
print("\n--- DELIVERY STATUS ---")

print(
    df["Delivery Status"].value_counts()
)

# SHIPPING MODE
print("\n--- SHIPPING MODE ---")

print(
    df["Shipping Mode"].value_counts()
)

# MARKET
print("\n--- ORDERS BY MARKET ---")

print(
    df["Market"].value_counts()
)

# FINAL DATA QUALITY CHECK
print("\n--- FINAL DATA QUALITY CHECK ---")

print(
    "Total missing values:",
    df.isnull().sum().sum()
)

print(
    "Total duplicate rows:",
    df.duplicated().sum()
)

print(
    "Final dataset shape:",
    df.shape
)

print("\n" + "=" * 70)
print("ANALYSIS SUMMARY COMPLETED")
print("=" * 70)