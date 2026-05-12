import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 2000
dates = pd.date_range("2023-01-01", "2024-12-31", periods=n)
regions = np.random.choice(["North", "South", "East", "West"], n)
categories = np.random.choice(["Electronics", "Clothing", "Books", "Home", "Sports"], n)
products_map = {
    "Electronics": ["Laptop", "Phone", "Tablet", "Headphones"],
    "Clothing": ["T-Shirt", "Jeans", "Dress", "Jacket"],
    "Books": ["Fiction", "Non-Fiction", "Textbook", "Comic"],
    "Home": ["Lamp", "Chair", "Table", "Rug"],
    "Sports": ["Shoes", "Ball", "Racket", "Gloves"],
}
product_names = [np.random.choice(products_map[c]) for c in categories]
base_rev = {"Electronics": 500, "Clothing": 80, "Books": 25, "Home": 150, "Sports": 60}
revenue = [max(5.0, round(base_rev[c] * np.random.uniform(0.5, 2.5), 2)) for c in categories]
units = [max(1, int(r / base_rev[c] * np.random.randint(1, 10))) for r, c in zip(revenue, categories)]
for i in np.random.choice(n, 30, replace=False):
    revenue[i] = round(revenue[i] * np.random.choice([0.05, 8.0]), 2)

df = pd.DataFrame({
    "order_date": dates,
    "region": regions,
    "category": categories,
    "product": product_names,
    "revenue": revenue,
    "units": units,
    "discount": np.round(np.random.uniform(0, 0.4, n), 2),
    "profit": [round(r * np.random.uniform(0.05, 0.35), 2) for r in revenue],
})
Path("data/demo").mkdir(parents=True, exist_ok=True)
df.to_csv("data/demo/ecommerce_sales.csv", index=False)
print(f"ecommerce_sales.csv: {len(df)} rows")

# global_superstore
n2 = 1500
df2 = pd.DataFrame({
    "order_date": pd.date_range("2021-01-01", periods=n2, freq="8h"),
    "segment": np.random.choice(["Consumer", "Corporate", "Home Office"], n2),
    "country": np.random.choice(["United States", "Canada", "Mexico"], n2),
    "region": np.random.choice(["East", "West", "Central", "South"], n2),
    "category": np.random.choice(["Furniture", "Office Supplies", "Technology"], n2),
    "sub_category": np.random.choice(["Chairs", "Phones", "Binders", "Tables", "Storage", "Copiers"], n2),
    "product_name": [f"Product_{i % 200}" for i in range(n2)],
    "sales": np.round(np.random.exponential(250, n2), 2),
    "quantity": np.random.randint(1, 15, n2),
    "discount": np.round(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], n2), 1),
    "profit": np.round(np.random.normal(50, 120, n2), 2),
    "shipping_cost": np.round(np.random.exponential(15, n2), 2),
})
df2.to_csv("data/demo/global_superstore.csv", index=False)
print(f"global_superstore.csv: {len(df2)} rows")
