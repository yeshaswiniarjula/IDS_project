
import pandas as pd

df = pd.read_parquet("KDDTest.parquet")
df.to_csv("KDDTest.csv", index=False)

print("✅ Converted successfully!")
