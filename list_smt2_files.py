#!/usr/bin/env python3
"""
Script to list all SMT2 files from QF_LRA_unsat and QF_UFLRA_unsat directories.
Output format: theory,mode,filename
  - theory: QF_LRA or QF_UFLRA
  - mode: incremental or non-incremental
  - filename: name of the SMT2 file
"""

import os
from pathlib import Path


def list_smt2_files(base_dir: Path, output_file: Path):
    """
    List all SMT2 files from QF_LRA_unsat and QF_UFLRA_unsat directories.
    
    Args:
        base_dir: Base directory containing the *_unsat folders
        output_file: Path to the output file
    """
    # Define the directories to scan
    directories = {
        "QF_LRA": base_dir / "QF_LRA_unsat",
        "QF_UFLRA": base_dir / "QF_UFLRA_unsat"
    }
    
    modes = ["incremental", "non-incremental"]
    
    entries = []
    
    for theory, theory_dir in directories.items():
        if not theory_dir.exists():
            print(f"Warning: Directory {theory_dir} does not exist, skipping...")
            continue
            
        for mode in modes:
            mode_dir = theory_dir / mode
            if not mode_dir.exists():
                print(f"Warning: Directory {mode_dir} does not exist, skipping...")
                continue
            
            # Find all .smt2 files in this directory (including subdirectories)
            for smt2_file in sorted(mode_dir.rglob("*.smt2")):
                filename = smt2_file.name
                entries.append(f"{theory},{mode},{filename}")
    
    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("theory,mode,filename\n")  # Header
        for entry in entries:
            f.write(entry + "\n")
    
    print(f"Written {len(entries)} entries to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for theory in directories.keys():
        for mode in modes:
            count = sum(1 for e in entries if e.startswith(f"{theory},{mode},"))
            print(f"  {theory} / {mode}: {count} files")


def main():
    # Get the script's directory and navigate to inputs folder
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent / "inputs"
    output_file = script_dir.parent / "smt2_file_list.txt"
    
    print(f"Scanning directories in: {base_dir}")
    print(f"Output file: {output_file}")
    print()
    
    list_smt2_files(base_dir, output_file)


if __name__ == "__main__":
    main()

