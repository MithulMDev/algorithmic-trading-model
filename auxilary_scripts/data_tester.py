import pandas as pd

# Replace 'file.parquet' with the path to your Parquet file
df = pd.read_parquet("E:\\trading_algo_model\\parquet_files\\ACC.parquet")

# Show the first few rows of the DataFrame
print(df.head())