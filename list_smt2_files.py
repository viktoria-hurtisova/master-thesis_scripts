#!/usr/bin/env python3
"""
Script to list all SMT2 files from a given directory.
Output format: theory,filename
  - theory: derived from the directory name
  - filename: name of the SMT2 file

Usage:
    python list_smt2_files.py <directory> [--output <output_file>]

"""

import argparse
from pathlib import Path


def list_smt2_files(directory: Path, output_file: Path):
    """
    List all SMT2 files from the given directory.
    The directory name is used as the theory name.
    
    Args:
        directory: Directory to scan for SMT2 files
        output_file: Path to the output file
    """
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        return
    
    # Use the directory name as the theory
    theory = directory.name
    
    entries = []
    
    # Find all .smt2 files in this directory (including subdirectories)
    for smt2_file in sorted(directory.rglob("*.smt2")):
        filename = smt2_file.name
        entries.append(f"{theory},{filename}")
    
    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("theory,filename\n")  # Header
        for entry in entries:
            f.write(entry + "\n")
    
    print(f"Written {len(entries)} entries to {output_file}")
    print(f"Theory: {theory}")
    print(f"Files found: {len(entries)}")


def main():
    parser = argparse.ArgumentParser(
        description="List all SMT2 files from a directory. "
                    "The directory name is used as the theory name."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan for SMT2 files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: smt2_file_list.txt in parent of scripts folder)"
    )
    
    args = parser.parse_args()
    
    # Resolve the directory path
    directory = args.directory.resolve()
    
    # Set default output file if not provided
    if args.output is None:
        script_dir = Path(__file__).parent
        output_file = script_dir.parent / "smt2_file_list.txt"
    else:
        output_file = args.output.resolve()
    
    print(f"Scanning directory: {directory}")
    print(f"Output file: {output_file}")
    print()
    
    list_smt2_files(directory, output_file)


if __name__ == "__main__":
    main()
