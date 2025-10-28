#!/usr/bin/env python3
"""
Script to copy files that contain the line '(set-info :status unsat)' from source to destination folder.

Usage:
    python copy_unsat_files.py <source_folder> <destination_folder>

The script will:
1. Recursively scan the source folder for files
2. Check each file for the specific line '(set-info :status unsat)'
3. Copy matching files to the destination folder (flat structure)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def contains_unsat_line(file_path):
    """
    Check if a file contains the line '(set-info :status unsat)'
    
    Args:
        file_path (Path): Path to the file to check
        
    Returns:
        bool: True if the file contains the target line, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip() == '(set-info :status unsat)':
                    return True
    except (IOError, OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        return False
    return False


def copy_unsat_files(source_folder, dest_folder):
    """
    Copy files containing '(set-info :status unsat)' from source to destination folder.
    
    Args:
        source_folder (Path): Source directory to scan
        dest_folder (Path): Destination directory to copy files to
    """
    source_path = Path(source_folder).resolve()
    dest_path = Path(dest_folder).resolve()
    
    # Validate source folder exists
    if not source_path.exists():
        print(f"Error: Source folder '{source_path}' does not exist.")
        sys.exit(1)
    
    if not source_path.is_dir():
        print(f"Error: Source path '{source_path}' is not a directory.")
        sys.exit(1)
    
    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = 0
    total_files = 0
    filename_counter = {}
    
    print(f"Scanning files in: {source_path}")
    print(f"Copying matching files to: {dest_path}")
    print("-" * 50)
    
    # Walk through all files in source directory
    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            total_files += 1
            
            # Check if file contains the target line
            if contains_unsat_line(file_path):
                # Copy file directly to destination (flat structure)
                dest_file_path = dest_path / file_path.name
                
                # Handle filename conflicts by adding a counter
                if dest_file_path.exists():
                    if file_path.name in filename_counter:
                        filename_counter[file_path.name] += 1
                    else:
                        filename_counter[file_path.name] = 1
                    
                    stem = dest_file_path.stem
                    suffix = dest_file_path.suffix
                    dest_file_path = dest_path / f"{stem}_{filename_counter[file_path.name]}{suffix}"
                
                # Copy the file
                try:
                    shutil.copy2(file_path, dest_file_path)
                    print(f"Copied: {dest_file_path.name}")
                    copied_files += 1
                except (IOError, OSError) as e:
                    print(f"Error copying {file_path.name}: {e}")
    
    print("-" * 50)
    print(f"Scan complete!")
    print(f"Total files scanned: {total_files}")
    print(f"Files copied: {copied_files}")


def main():
    """Main function to parse arguments and execute the copy operation."""
    parser = argparse.ArgumentParser(
        description="Copy files containing '(set-info :status unsat)' from source to destination folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python copy_unsat_files.py ./inputs ./unsat_files
    python copy_unsat_files.py "C:\\source\\folder" "C:\\dest\\folder"
        """
    )
    
    parser.add_argument(
        'source_folder',
        help='Source folder to scan for files'
    )
    
    parser.add_argument(
        'destination_folder',
        help='Destination folder to copy matching files to'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Source folder: {args.source_folder}")
        print(f"Destination folder: {args.destination_folder}")
        print(f"Looking for files containing: (set-info :status unsat)")
        print()
    
    copy_unsat_files(args.source_folder, args.destination_folder)


if __name__ == "__main__":
    main()
