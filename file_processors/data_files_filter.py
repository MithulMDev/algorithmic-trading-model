# this takes the csv files of selected tickers and aggregates them and then saves in the streamable_csv_files dir

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import gc

def concatenate_csv_files(root_folder, subfolder_list, output_folder="aggregated_files", 
                          output_format="csv", chunk_size=100000):
    """
    Aggregate CSV files from subfolders into single files.
    Handles CSVs with or without headers correctly.
    
    Args:
        root_folder: Root folder containing the subfolders
        subfolder_list: List of subfolder names to process
        output_folder: Folder where output files will be saved
        output_format: 'csv' or 'parquet'
        chunk_size: Number of rows to read at a time
    """
    
    root_path = Path(root_folder).resolve()
    output_path = Path(output_folder).resolve()
    
    if not root_path.exists():
        print(f"❌ Error: Root folder '{root_path}' does not exist!")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output folder: {output_path}")
    print(f"📄 Output format: {output_format.upper()}\n")
    
    # Clean subfolder names
    clean_subfolder_names = []
    for name in subfolder_list:
        clean_name = name.replace('.txt', '') if name.endswith('.txt') else name
        clean_subfolder_names.append(clean_name)
    
    print(f"📊 Processing {len(clean_subfolder_names)} subfolders\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, subfolder_name in enumerate(clean_subfolder_names, 1):
        print(f"{'='*60}")
        print(f"[{idx}/{len(clean_subfolder_names)}] Processing: {subfolder_name}")
        print(f"{'='*60}")
        
        subfolder_path = find_subfolder(root_path, subfolder_name)
        
        if subfolder_path is None:
            print(f"⚠️  Subfolder not found: {subfolder_name}")
            skipped += 1
            print()
            continue
        
        print(f"✓ Found subfolder: {subfolder_path.relative_to(root_path)}")
        
        csv_files = sorted(list(subfolder_path.rglob("*.csv")))
        
        if not csv_files:
            print(f"⚠️  No .csv files found in {subfolder_name}")
            skipped += 1
            print()
            continue
        
        print(f"📄 Found {len(csv_files)} CSV file(s)")
        
        # Auto-detect headers and columns
        print("\n🔍 Detecting CSV structure...")
        has_header, column_names = detect_csv_structure(csv_files[0])
        
        if has_header:
            print(f"✓ CSV has headers: {column_names}")
        else:
            print(f"✓ CSV has NO headers. Using: {column_names}")
        
        # Determine output file
        if output_format == "csv":
            output_file = output_path / f"{subfolder_name}_combined.csv"
        else:
            output_file = output_path / f"{subfolder_name}.parquet"
        
        try:
            if output_format == "csv":
                success = aggregate_to_csv(csv_files, output_file, has_header, column_names, chunk_size)
            else:
                success = aggregate_to_parquet(csv_files, output_file, has_header, column_names, chunk_size)
            
            if success:
                file_size = output_file.stat().st_size
                print(f"\n✅ Saved: {output_file.name} ({format_file_size(file_size)})")
                successful += 1
            else:
                print(f"\n❌ Failed to process {subfolder_name}")
                failed += 1
                
        except Exception as e:
            print(f"\n❌ Error processing {subfolder_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        gc.collect()
        print()
    
    print(f"{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped (not found): {skipped}")
    print(f"📁 Output location: {output_path}")
    print(f"{'='*60}")


def detect_csv_structure(csv_file):
    """
    Auto-detect if CSV has headers and what the columns are.
    Returns: (has_header: bool, column_names: list)
    """
    # Read first few rows without assuming headers
    df_no_header = pd.read_csv(csv_file, nrows=5, header=None)
    df_with_header = pd.read_csv(csv_file, nrows=5)
    
    # Check if first row looks like headers (contains strings, not all numeric)
    first_row = df_no_header.iloc[0]
    
    # If first row is all strings and doesn't match data pattern, it's likely a header
    if df_with_header.shape[1] == df_no_header.shape[1]:
        # Check if header names make sense
        header_names = list(df_with_header.columns)
        
        # If column names are like "Type", "Date", etc., it has headers
        # If column names are like "0", "1", "2", it doesn't have headers
        if all(isinstance(col, str) and not col.isdigit() for col in header_names):
            return True, header_names
    
    # No header - create generic names or use common trading columns
    num_cols = df_no_header.shape[1]
    
    # Common trading data columns
    if num_cols == 8:
        column_names = ['Type', 'Date', 'Time', 'open', 'high', 'low', 'close', 'volume']
    else:
        column_names = [f'col_{i}' for i in range(num_cols)]
    
    return False, column_names


def aggregate_to_csv(csv_files, output_file, has_header, column_names, chunk_size):
    """
    Aggregate multiple CSV files into one CSV file.
    """
    print(f"\n💾 Aggregating to CSV...")
    
    total_rows = 0
    file_count = 0
    first_file = True
    
    for csv_file in csv_files:
        try:
            print(f"  📖 Reading: {csv_file.name}...", end=" ", flush=True)
            
            file_rows = 0
            
            # Read with correct header setting
            if has_header:
                reader = pd.read_csv(csv_file, chunksize=chunk_size)
            else:
                reader = pd.read_csv(csv_file, names=column_names, header=None, chunksize=chunk_size)
            
            for chunk in reader:
                # Write to CSV
                chunk.to_csv(
                    output_file, 
                    mode='a' if not first_file or file_rows > 0 else 'w',
                    header=first_file and file_rows == 0,
                    index=False
                )
                
                file_rows += len(chunk)
                total_rows += len(chunk)
                
                del chunk
                gc.collect()
            
            file_count += 1
            print(f"✓ ({file_rows:,} rows)")
            first_file = False
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if file_count > 0:
        print(f"\n📊 Combined: {total_rows:,} rows from {file_count} file(s)")
        return True
    
    return False


def aggregate_to_parquet(csv_files, output_file, has_header, column_names, chunk_size):
    """
    Aggregate multiple CSV files into one Parquet file with correct schema.
    """
    print(f"\n💾 Aggregating to Parquet...")
    
    writer = None
    total_rows = 0
    file_count = 0
    chunks_processed = 0
    
    for csv_file in csv_files:
        try:
            print(f"  📖 Reading: {csv_file.name}...", end=" ", flush=True)
            
            file_rows = 0
            
            # Read with correct header setting
            if has_header:
                reader = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
            else:
                reader = pd.read_csv(csv_file, names=column_names, header=None, 
                                   chunksize=chunk_size, low_memory=False)
            
            for chunk in reader:
                # Ensure column order is consistent
                chunk = chunk[column_names]
                
                # Convert to PyArrow Table
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                
                # Create writer on first chunk
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_file, 
                        table.schema, 
                        compression='snappy'
                    )
                
                writer.write_table(table)
                
                file_rows += len(chunk)
                total_rows += len(chunk)
                chunks_processed += 1
                
                del chunk, table
                
                if chunks_processed % 10 == 0:
                    gc.collect()
            
            file_count += 1
            print(f"✓ ({file_rows:,} rows)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if writer:
        writer.close()
        print(f"\n📊 Combined: {total_rows:,} rows from {file_count} file(s)")
        return True
    
    return False


def find_subfolder(root_path, subfolder_name):
    """Find subfolder with or without .txt extension."""
    if (root_path / subfolder_name).exists() and (root_path / subfolder_name).is_dir():
        return root_path / subfolder_name
    
    if (root_path / f"{subfolder_name}.txt").exists() and (root_path / f"{subfolder_name}.txt").is_dir():
        return root_path / f"{subfolder_name}.txt"
    
    for item in root_path.rglob(subfolder_name):
        if item.is_dir():
            return item
    
    for item in root_path.rglob(f"{subfolder_name}.txt"):
        if item.is_dir():
            return item
    
    return None


def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


if __name__ == "__main__":
    root_folder = "E:\\trading_algo_model\\organized_files_data\\trainable_files"
    
    subfolder_list = [
        "ACC", 
        "ADANIPORTS", 
        "BHARTIARTL", 
        "BOSCHLTD",
        "DLF", 
        "HCLTECH", 
        "SUNPHARMA", 
        "HINDUNILVR", 
        "SIEMENS",
        "TATASTEEL"
    ]
    
    # Choose output format: "csv" or "parquet"
    output_format = "csv"  # Change to "parquet" if you prefer
    output_folder = f"trainable_{output_format}_files"
    
    chunk_size = 100000
    
    concatenate_csv_files(root_folder, subfolder_list, output_folder, output_format, chunk_size)