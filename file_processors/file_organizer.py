import os
import shutil
from pathlib import Path
from collections import defaultdict

def organize_text_files(source_folder, output_folder):
    """
    Recursively find all .txt files, group them by filename,
    and organize them with path-prefixed names.
    
    Args:
        source_folder: Root folder to search for .txt files
        output_folder: Destination folder for organized files
    """
    
    # Convert to Path objects for easier manipulation
    source_path = Path(source_folder).resolve()
    output_path = Path(output_folder).resolve()
    
    # Get the root folder name
    root_folder_name = source_path.name
    
    # Dictionary to group files by their original filename
    # Key: original filename, Value: list of (full_path, prefixed_name)
    file_groups = defaultdict(list)
    
    print(f"Scanning folder: {source_path}")
    print(f"Root folder name: {root_folder_name}\n")
    
    # Recursively find all .txt files
    for txt_file in source_path.rglob("*.txt"):
        # Get relative path from source folder
        rel_path = txt_file.relative_to(source_path)
        
        # Get all parent folders in the path
        path_parts = list(rel_path.parts[:-1])  # Exclude filename
        
        # Create prefix: root_folder_part1_part2_..._filename
        prefix_parts = [root_folder_name] + path_parts
        prefix = "_".join(prefix_parts)
        
        # Create new filename with prefix
        original_filename = txt_file.name
        if prefix:
            new_filename = f"{prefix}_{original_filename}"
        else:
            new_filename = f"{root_folder_name}_{original_filename}"
        
        # Group by original filename
        file_groups[original_filename].append((txt_file, new_filename))
    
    print(f"Found {sum(len(files) for files in file_groups.values())} .txt files")
    print(f"Grouped into {len(file_groups)} unique filenames\n")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each group
    for original_filename, file_list in file_groups.items():
        # Create folder for this filename group
        group_folder = output_path / original_filename
        group_folder.mkdir(exist_ok=True)
        
        print(f"Processing group: {original_filename} ({len(file_list)} files)")
        
        # Track used names to handle duplicates
        used_names = {}
        
        for source_file, new_filename in file_list:
            # Handle duplicate names by adding number suffix
            final_filename = new_filename
            if final_filename in used_names:
                # Extract name and extension
                name_part = final_filename.rsplit('.', 1)[0]
                ext_part = final_filename.rsplit('.', 1)[1] if '.' in final_filename else ''
                
                # Add number suffix
                counter = 1
                while True:
                    if ext_part:
                        final_filename = f"{name_part}_{counter}.{ext_part}"
                    else:
                        final_filename = f"{name_part}_{counter}"
                    
                    if final_filename not in used_names:
                        break
                    counter += 1
            
            used_names[final_filename] = True
            
            # Copy file to new location
            destination = group_folder / final_filename
            shutil.copy2(source_file, destination)
            print(f"  ✓ {source_file.relative_to(source_path)} -> {original_filename}/{final_filename}")
    
    print(f"\n Done! All files organized in: {output_path}")


if __name__ == "__main__":
    
    source_folder = "C:\\Users\\dev\\Downloads\\nifty-banknifty-intraday-data-train"  # source folder
    output_folder = "E:\\trading_algo_model\\organized_files_data"       # output folder
    
    organize_text_files(source_folder, output_folder)