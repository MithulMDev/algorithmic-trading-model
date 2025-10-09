import os
from pathlib import Path

def find_and_delete_nifty_files(folder_path, dry_run=True, case_sensitive=False):
    """
    Recursively find and delete all files containing "nifty" in their filename.
    
    Args:
        folder_path: Root folder to search
        dry_run: If True, only shows what would be deleted without actually deleting
        case_sensitive: If True, matches "nifty" exactly; if False, matches case-insensitively
    """
    
    folder = Path(folder_path).resolve()
    
    if not folder.exists():
        print(f"❌ Error: Folder '{folder}' does not exist!")
        return
    
    if not folder.is_dir():
        print(f"❌ Error: '{folder}' is not a directory!")
        return
    
    print(f"Scanning folder: {folder}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Mode: {'DRY RUN (no files will be deleted)' if dry_run else 'DELETE MODE'}\n")
    
    # Analytics variables
    total_files = 0
    total_folders = 0
    files_to_delete = []
    extension_counts = {}
    
    # Find all files containing "nifty"
    for item_path in folder.rglob("*"):
        if item_path.is_file():
            total_files += 1
            filename = item_path.name
            
            # Track file extensions
            ext = item_path.suffix.lower() if item_path.suffix else "(no extension)"
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # Check if "nifty" is in the filename
            if case_sensitive:
                contains_nifty = "_nifty50_" in filename
            else:
                contains_nifty = "_nifty50_" in filename.lower()
            
            if contains_nifty:
                files_to_delete.append(item_path)
        elif item_path.is_dir():
            total_folders += 1
    
    # Display analytics
    print("=" * 60)
    print("📊 ANALYTICS")
    print("=" * 60)
    print(f"Total files scanned:           {total_files:,}")
    print(f"Total folders scanned:         {total_folders:,}")
    print(f"Files containing 'nifty':      {len(files_to_delete):,}")
    
    if total_files > 0:
        percentage = (len(files_to_delete) / total_files) * 100
        print(f"Percentage of files w/ 'nifty': {percentage:.2f}%")
    
    # Calculate total size of files to delete
    total_size_to_delete = sum(f.stat().st_size for f in files_to_delete)
    print(f"Total size to delete:          {format_file_size(total_size_to_delete)}")
    
    # Show top 5 file extensions
    if extension_counts:
        print(f"\nTop file extensions found:")
        sorted_exts = sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_exts[:5]:
            print(f"  {ext}: {count:,} files")
    
    print("=" * 60)
    print()
    
    # Display findings
    if not files_to_delete:
        print("✅ No files containing 'nifty' found.")
        return
    
    print(f"Files to be deleted:\n")
    
    # Group files by extension for better overview
    nifty_extensions = {}
    for file_path in files_to_delete:
        ext = file_path.suffix.lower() if file_path.suffix else "(no extension)"
        if ext not in nifty_extensions:
            nifty_extensions[ext] = []
        nifty_extensions[ext].append(file_path)
    
    print(f"Breakdown by file type:")
    for ext, files in sorted(nifty_extensions.items()):
        ext_size = sum(f.stat().st_size for f in files)
        print(f"  {ext}: {len(files)} file(s) - {format_file_size(ext_size)}")
    print()
    
    for i, file_path in enumerate(files_to_delete, 1):
        rel_path = file_path.relative_to(folder)
        file_size = file_path.stat().st_size
        size_str = format_file_size(file_size)
        print(f"{i}. {rel_path} ({size_str})")
    
    print()
    
    # Delete files if not in dry run mode
    if dry_run:
        print("🔍 DRY RUN - No files were deleted.")
        print("   To actually delete these files, run with dry_run=False")
    else:
        # Ask for confirmation
        response = input(f"⚠️  Are you sure you want to delete {len(files_to_delete)} file(s)? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            deleted_count = 0
            failed_count = 0
            total_deleted_size = 0
            
            for file_path in files_to_delete:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    total_deleted_size += file_size
                    print(f"✓ Deleted: {file_path.relative_to(folder)}")
                except Exception as e:
                    failed_count += 1
                    print(f"✗ Failed to delete {file_path.relative_to(folder)}: {e}")
            
            print("\n" + "=" * 60)
            print("📊 DELETION SUMMARY")
            print("=" * 60)
            print(f"✅ Successfully deleted: {deleted_count} file(s)")
            print(f"   Total space freed:    {format_file_size(total_deleted_size)}")
            if failed_count > 0:
                print(f"❌ Failed to delete:     {failed_count} file(s)")
            print("=" * 60)
        else:
            print("❌ Deletion cancelled.")


def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


if __name__ == "__main__":
    folder_path = "E:\\trading_algo_model\\organized_files_data"  # Change this to your folder path
    
    # First, do a dry run to see what would be deleted
    print("=" * 60)
    print("STEP 1: DRY RUN - Preview files to be deleted")
    print("=" * 60)
    find_and_delete_nifty_files(folder_path, dry_run=True, case_sensitive=False)

    # to actually delete the files
    
    print("=" * 60)
    print("STEP 2: DELETE RUN - Preview files to be deleted")
    print("=" * 60)
    
    find_and_delete_nifty_files(folder_path, dry_run=False, case_sensitive=False)