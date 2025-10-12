import pandas as pd
import os


# Replace 'file.parquet' with the path to your Parquet file
df = pd.read_parquet("E:\\trading_algo_model\\parquet_files\\ADANIPORTS.parquet")
df1 = pd.read_csv("E:\\trading_algo_model\\organized_files_data\\ACC_F1.txt\\nifty-banknifty-intraday-data-train_2017_AUG_01AUG_01AUG_ACC_F1.csv")

# Show the first few rows of the DataFrame
print(df.head())
print(len(df))

print(" ")
print(df1.head())
print(len(df1))



# Path to the folder containing CSV files
folder_path = r"E:\trading_algo_model\file_processors\aggregated_csv_files"  # change this to your folder

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)
            print(f"{filename}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
