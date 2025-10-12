import csv
from pathlib import Path
import sys

def convert_txt_to_csv(root_folder, dry_run=True, delete_originals=True):
    """
    Recursively convert all .txt files to .csv files and optionally delete originals.
    
    Args:
        root_folder: Root folder to search for .txt files
        dry_run: If True, shows what would be done without actually doing it
        delete_originals: If True, deletes .txt files after successful conversion
    """
    
    root_path = Path(root_folder).resolve()
    
    if not root_path.exists():
        print(f"❌ Error: Folder '{root_path}' does not exist!")
        return
    
    print(f"📁 Scanning folder: {root_path}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'CONVERSION MODE'}")
    print(f"Delete originals: {delete_originals}\n")
    
    # Find all .txt files
    txt_files = list(root_path.rglob("*.txt"))
    
    if not txt_files:
        print("✅ No .txt files found.")
        return
    
    print(f"Found {len(txt_files)} text file(s)\n")
    
    # Statistics
    successful = 0
    failed = 0
    skipped = 0
    total_valid_rows = 0
    total_invalid_rows = 0
    
    # Header for CSV files
    header = ["Type", "Date", "Time", "open", "high", "low", "close", "volume"]
    
    for idx, txt_file in enumerate(txt_files, 1):
        print(f"[{idx}/{len(txt_files)}] Processing: {txt_file.name}")
        
        # Create output CSV file path (same location, .csv extension)
        csv_file = txt_file.with_suffix('.csv')
        
        try:
            # Read the text file
            with open(txt_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
            
            if not lines:
                print(f"  ⚠️  Empty file, skipping")
                skipped += 1
                print()
                continue
            
            valid_rows = 0
            invalid_rows = 0
            
            if dry_run:
                # Just validate in dry run mode
                for line in lines:
                    row = line.strip().split(',')
                    if len(row) == 8:
                        valid_rows += 1
                    else:
                        invalid_rows += 1
                
                print(f"  ✓ Would convert: {valid_rows} valid rows, {invalid_rows} invalid rows skipped")
                print(f"  → Output: {csv_file.relative_to(root_path)}")
                if delete_originals:
                    print(f"  → Would delete: {txt_file.name}")
                successful += 1
                total_valid_rows += valid_rows
                total_invalid_rows += invalid_rows
                
            else:
                # Actually convert the file
                with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
                    csv_writer = csv.writer(outfile)
                    
                    # Write header
                    csv_writer.writerow(header)
                    
                    # Process each line
                    for line in lines:
                        row = line.strip().split(',')
                        
                        if len(row) == 8:
                            csv_writer.writerow(row)
                            valid_rows += 1
                        else:
                            invalid_rows += 1
                
                print(f"  ✓ Converted: {valid_rows} valid rows, {invalid_rows} invalid rows skipped")
                print(f"  → Created: {csv_file.name}")
                
                # Delete original .txt file if requested
                if delete_originals:
                    txt_file.unlink()
                    print(f"  → Deleted: {txt_file.name}")
                
                successful += 1
                total_valid_rows += valid_rows
                total_invalid_rows += invalid_rows
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
        
        print()
    
    # Final summary
    print(f"{'='*60}")
    print(f"📊 CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    if dry_run:
        print(f"🔍 DRY RUN COMPLETE - No files were modified")
        print(f"   Files that would be converted: {successful}")
    else:
        print(f"✅ Successfully converted: {successful}")
        if delete_originals:
            print(f"🗑️  Original .txt files deleted: {successful}")
    
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped (empty): {skipped}")
    print(f"📊 Total valid rows: {total_valid_rows:,}")
    print(f"⚠️  Total invalid rows skipped: {total_invalid_rows:,}")
    print(f"{'='*60}")
    
    if dry_run:
        print(f"\n💡 To actually convert files, run with dry_run=False")


def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


if __name__ == "__main__":

    # Save all console prints to a Markdown file
    log_file = open("conversion_log.md", "w", encoding="utf-8")
    sys.stdout = log_file  # Redirect all print() output to the log file

    # Configuration
    root_folder = "E:\\trading_algo_model\\organized_files_data\\trainable_files"  # Change this to your root folder
    
    # STEP 1: DRY RUN - Preview what will happen
    print("=" * 60)
    print("STEP 1: DRY RUN - Preview conversions")
    print("=" * 60)
    convert_txt_to_csv(root_folder, dry_run=True, delete_originals=True)
    
    print("\n" + "=" * 60)
    print("actually converting files:")
    print("=" * 60)
    
    # to actually convert the files
    convert_txt_to_csv(root_folder, dry_run=False, delete_originals=True)
    
    # If you want to keep the original .txt files, use delete_originals=False:
    # convert_txt_to_csv(root_folder, dry_run=False, delete_originals=False)

    # Restore console output and close the file
    sys.stdout = sys.__stdout__
    log_file.close()
    print("✅ Logs saved to conversion_log.md")