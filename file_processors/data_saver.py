import os
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from pathlib import Path

def process_csvs_to_mongodb(folder_path, mongo_uri, db_name, collection_name):
    """
    Read CSV files from a folder and insert combined rows into MongoDB.
    
    Args:
        folder_path: Path to folder containing CSV files
        mongo_uri: MongoDB connection string
        db_name: Database name
        collection_name: Collection name
    """
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Get all CSV files in the folder
    csv_files = list(Path(folder_path).glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read all CSVs into memory (first 52000 rows each)
    dataframes = {}
    for csv_file in csv_files:
        filename = csv_file.name
        print(f"Reading {filename}...")
        df = pd.read_csv(csv_file, nrows=52000)
        dataframes[filename] = df
        print(f"  Loaded {len(df)} rows from {filename}")
    
    # Get the starting timestamp
    current_timestamp = datetime.now()
    
    # Process and insert row by row
    num_rows = 52000
    print(f"\nInserting {num_rows} documents into MongoDB...")
    
    for i in range(num_rows):
        # Create document with timestamp
        document = {
            "timestamp": current_timestamp + timedelta(seconds=i)
        }
        
        # Add data from each CSV file
        for filename, df in dataframes.items():
            if i < len(df):
                # Convert row to dict and add to document
                row_dict = df.iloc[i].to_dict()
                document[filename] = row_dict
            else:
                # If this CSV has fewer rows, add null
                document[filename] = None
        
        # Insert into MongoDB
        collection.insert_one(document)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Inserted {i + 1}/{num_rows} documents")
    
    print(f"\nCompleted! Inserted {num_rows} documents into {db_name}.{collection_name}")
    client.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "E:\\trading_algo_model\\file_processors\\streamable_csv_files"  # Change to your folder path
    MONGO_URI = "mongodb://localhost:27017/"  # Change to your MongoDB URI
    DB_NAME = "ohlcv_data"
    COLLECTION_NAME = "combined_rows"
    
    process_csvs_to_mongodb(FOLDER_PATH, MONGO_URI, DB_NAME, COLLECTION_NAME)