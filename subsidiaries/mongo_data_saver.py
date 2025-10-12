import os
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pathlib import Path
from dotenv import load_dotenv

def test_mongodb_connection(mongo_uri):
    """Test MongoDB connection before processing."""
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB!")
        client.close()
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"✗ Failed to connect to MongoDB: {e}")
        print("\nPlease ensure:")
        print("1. Docker container is running: docker-compose ps")
        print("2. MongoDB is healthy: docker-compose logs mongodb")
        return False

def process_csvs_to_mongodb(folder_path, mongo_uri, db_name, collection_name):
    """
    Read CSV files from a folder and insert combined rows into MongoDB.
    
    Args:
        folder_path: Path to folder containing CSV files
        mongo_uri: MongoDB connection string
        db_name: Database name
        collection_name: Collection name
    """
    # Test connection first
    if not test_mongodb_connection(mongo_uri):
        return
    
    # Connect to MongoDB
    print(f"\nConnecting to MongoDB...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Get all CSV files in the folder
    csv_files = list(Path(folder_path).glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in the folder.")
        client.close()
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read all CSVs into memory (first 52000 rows each)
    dataframes = {}
    for csv_file in csv_files:
        filename = csv_file.name
        print(f"Reading {filename}...")
        try:
            df = pd.read_csv(csv_file, nrows=52000)
            dataframes[filename] = df
            print(f"  ✓ Loaded {len(df)} rows from {filename}")
        except Exception as e:
            print(f"  ✗ Error reading {filename}: {e}")
            continue
    
    if not dataframes:
        print("No data loaded from CSV files.")
        client.close()
        return
    
    # Get the starting timestamp
    current_timestamp = datetime.now()
    
    # Process and insert row by row
    num_rows = 52000
    print(f"\nInserting {num_rows} documents into MongoDB...")
    print(f"Database: {db_name}, Collection: {collection_name}\n")
    
    documents_to_insert = []
    batch_size = 1000
    
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
                # Convert any NaN values to None for MongoDB
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                document[filename] = row_dict
            else:
                # If this CSV has fewer rows, add null
                document[filename] = None
        
        documents_to_insert.append(document)
        
        # Insert in batches for better performance
        if len(documents_to_insert) >= batch_size:
            try:
                collection.insert_many(documents_to_insert)
                print(f"  ✓ Inserted {i + 1}/{num_rows} documents")
                documents_to_insert = []
            except Exception as e:
                print(f"  ✗ Error inserting batch at row {i}: {e}")
                documents_to_insert = []
    
    # Insert remaining documents
    if documents_to_insert:
        try:
            collection.insert_many(documents_to_insert)
            print(f"  ✓ Inserted {num_rows}/{num_rows} documents")
        except Exception as e:
            print(f"  ✗ Error inserting final batch: {e}")
    
    # Verify insertion
    doc_count = collection.count_documents({})
    print(f"\n{'='*60}")
    print(f"✓ Completed Successfully!")
    print(f"{'='*60}")
    print(f"Database: {db_name}")
    print(f"Collection: {collection_name}")
    print(f"Total documents in collection: {doc_count}")
    print(f"{'='*60}\n")
    
    client.close()


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration from environment variables
    FOLDER_PATH = os.getenv("CSV_FOLDER_PATH", "E:\\trading_algo_model\\file_processors\\streamable_csv_files")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:your_secure_password_here@localhost:27017/")
    DB_NAME = os.getenv("MONGO_DB_NAME", "ohlcv_data")
    COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "combined_rows")
    
    print("="*60)
    print("MongoDB Data Loader")
    print("="*60)
    print(f"CSV Folder: {FOLDER_PATH}")
    print(f"MongoDB URI: {MONGO_URI.split('@')[0]}@***")  # Hide password in logs
    print(f"Database: {DB_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print("="*60 + "\n")
    
    process_csvs_to_mongodb(FOLDER_PATH, MONGO_URI, DB_NAME, COLLECTION_NAME)