import os
import pandas as pd
from pathlib import Path
import gc

def concatenate_csv_to_parquet(root_folder, subfolder_list, output_folder="parquet_files", chunk_size=50000):
    """
    Efficiently traverse subfolders, read text files containing CSV data in chunks,
    and save as parquet files without overloading memory.
    
    Args:
        root_folder: Root folder containing the subfolders
        subfolder_list: List of subfolder names to process
        output_folder: Folder where parquet files will be saved
        chunk_size: Number of rows to read at a time (adjust based on your RAM)
    """
    
    root_path = Path(root_folder).resolve()
    output_path = Path(output_folder).resolve()
    
    if not root_path.exists():
        print(f"❌ Error: Root folder '{root_path}' does not exist!")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output folder: {output_path}\n")
    
    # Clean subfolder names
    clean_subfolder_names = []
    for name in subfolder_list:
        clean_name = name.replace('.txt', '') if name.endswith('.txt') else name
        clean_subfolder_names.append(clean_name)
    
    print(f"📊 Processing {len(clean_subfolder_names)} subfolders...\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, subfolder_name in enumerate(clean_subfolder_names, 1):
        print(f"{'='*60}")
        print(f"[{idx}/{len(clean_subfolder_names)}] Processing: {subfolder_name}")
        print(f"{'='*60}")
        
        # Find the subfolder
        subfolder_path = find_subfolder(root_path, subfolder_name)
        
        if subfolder_path is None:
            print(f"⚠️  Subfolder not found: {subfolder_name}")
            skipped += 1
            print()
            continue
        
        print(f"✓ Found subfolder: {subfolder_path.relative_to(root_path)}")
        
        # Find all .txt files
        txt_files = list(subfolder_path.rglob("*.txt"))
        
        if not txt_files:
            print(f"⚠️  No .txt files found in {subfolder_name}")
            skipped += 1
            print()
            continue
        
        print(f"📄 Found {len(txt_files)} text file(s)")
        
        # Process files efficiently using chunked reading
        output_file_path = output_path / f"{subfolder_name}.parquet"
        
        try:
            success = process_files_to_parquet(txt_files, output_file_path, chunk_size)
            
            if success:
                file_size = output_file_path.stat().st_size
                print(f"\n✅ Saved: {subfolder_name}.parquet ({format_file_size(file_size)})")
                successful += 1
            else:
                print(f"\n❌ Failed to process {subfolder_name}")
                failed += 1
                
        except Exception as e:
            print(f"\n❌ Error processing {subfolder_name}: {e}")
            failed += 1
        
        # Force garbage collection to free memory
        gc.collect()
        print()
    
    # Final summary
    print(f"{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped (not found): {skipped}")
    print(f"📁 Output location: {output_path}")
    print(f"{'='*60}")


def find_subfolder(root_path, subfolder_name):
    """Find subfolder with or without .txt extension."""
    # Try exact match
    if (root_path / subfolder_name).exists() and (root_path / subfolder_name).is_dir():
        return root_path / subfolder_name
    
    # Try with .txt
    if (root_path / f"{subfolder_name}.txt").exists() and (root_path / f"{subfolder_name}.txt").is_dir():
        return root_path / f"{subfolder_name}.txt"
    
    # Search recursively (first match only)
    for item in root_path.rglob(subfolder_name):
        if item.is_dir():
            return item
    
    for item in root_path.rglob(f"{subfolder_name}.txt"):
        if item.is_dir():
            return item
    
    return None


def process_files_to_parquet(txt_files, output_path, chunk_size):
    """
    Process text files efficiently using chunked reading and writing.
    This avoids loading all data into memory at once.
    """
    first_file = True
    total_rows = 0
    file_count = 0
    
    for txt_file in txt_files:
        try:
            print(f"  📖 Reading: {txt_file.name}...", end=" ", flush=True)
            
            # Read file in chunks to avoid memory issues
            chunk_iter = pd.read_csv(txt_file, chunksize=chunk_size, low_memory=False)
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                total_rows += len(chunk)
                
                if first_file and chunk_idx == 0:
                    # First chunk: create new parquet file
                    chunk.to_parquet(output_path, index=False, engine='pyarrow')
                    first_file = False
                else:
                    # Append to existing parquet file
                    # Read existing data
                    existing_df = pd.read_parquet(output_path)
                    # Concatenate
                    combined = pd.concat([existing_df, chunk], ignore_index=True)
                    # Write back
                    combined.to_parquet(output_path, index=False, engine='pyarrow')
                    
                    # Free memory
                    del existing_df
                    del combined
                    gc.collect()
            
            file_count += 1
            print(f"✓ ({total_rows} rows so far)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if file_count == 0:
        print(f"❌ No valid CSV files could be read")
        return False
    
    print(f"\n📊 Total: {total_rows} rows from {file_count} file(s)")
    return True


def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


if __name__ == "__main__":
    # Configuration
    root_folder = "E:\\trading_algo_model\\organized_files_data"  # Change this to your root folder
    
    # List of subfolders to process
    subfolder_list = [
        "ACC.txt", 
        "ADANIPORTS.txt", 
        "BHARTIARTL.txt", 
        "BOSCHLTD.txt",
        "DLF.txt", 
        "HCLTECH.txt", 
        "SUNPHARMA.txt", 
        "HINDUNILVR.txt", 
        "SIEMENS.txt",
        "TATASTEEL.txt"
    ]
    
    output_folder = "parquet_files"
    
    # Adjust chunk_size based on your available RAM
    # Larger = faster but uses more memory
    # Smaller = slower but safer for limited RAM
    # Default: 50,000 rows per chunk (adjust as needed)
    chunk_size = 50000
    
    concatenate_csv_to_parquet(root_folder, subfolder_list, output_folder, chunk_size)