#!/usr/bin/env python3
"""cleanup_smt.py

A utility script to clean up SMT-LIB files in a directory by keeping only
the bipartitioned (.bi.smt2) files and removing all other .smt2 files.

This is useful after running bipartition_smt.py to clean up the original
files and keep only the processed bipartitioned versions.

Usage::

    $ python scripts/cleanup_smt.py <directory>

Examples::

    $ python scripts/cleanup_smt.py inputs/RandomCoupled
    $ python scripts/cleanup_smt.py inputs/RandomCoupledUnsat

The script will:
- Find all .smt2 files in the specified directory
- Delete those that do NOT end with .bi.smt2
- Keep all .bi.smt2 files intact
- Provide feedback on the cleanup process

Requires only Python 3 (no external deps).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import os
from typing import List


def cleanup_smt_files(target_dir: pathlib.Path) -> None:
    """Delete all .smt2 files that are not .bi.smt2 files in the target directory."""
    if not target_dir.is_dir():
        raise ValueError(f"{target_dir} is not a directory")
    
    # Count files before cleanup for reporting
    all_smt_files = list(target_dir.glob("*.smt2"))
    bi_files = [f for f in all_smt_files if f.name.endswith(".bi.smt2")]
    original_files = [f for f in all_smt_files if not f.name.endswith(".bi.smt2")]
    
    print(f"Found {len(all_smt_files)} .smt2 files total:")
    print(f"  - {len(bi_files)} bipartitioned (.bi.smt2) files")
    print(f"  - {len(original_files)} original files to be deleted")
    
    if len(original_files) == 0:
        print("No original files to clean up.")
        return
    
    try:
        # Store current directory
        original_cwd = os.getcwd()
        os.chdir(target_dir)
        
        # Run the find command to delete non-bipartitioned files
        # Using find command equivalent to: find . -type f -name "*.smt2" ! -name "*.bi.smt2" -delete
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.smt2", "!", "-name", "*.bi.smt2", "-delete"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✓ Successfully deleted {len(original_files)} original .smt2 files")
        print(f"✓ Kept {len(bi_files)} bipartitioned .bi.smt2 files")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during cleanup: {e}", file=sys.stderr)
        if e.stderr:
            print(f"Error details: {e.stderr}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"✗ Unexpected error during cleanup: {e}", file=sys.stderr)
        sys.exit(4)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main(argv: List[str]) -> None:
    """Main entry point for the cleanup script."""
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print("Usage: python cleanup_smt.py <directory>")
        print()
        print("Cleans up SMT-LIB files by keeping only .bi.smt2 files")
        print("and removing all other .smt2 files in the specified directory.")
        print()
        print("Examples:")
        print("  python cleanup_smt.py inputs/RandomCoupled")
        print("  python cleanup_smt.py inputs/RandomCoupledUnsat")
        sys.exit(1)

    target_dir = pathlib.Path(argv[1])
    
    if not target_dir.exists():
        print(f"error: {target_dir} does not exist", file=sys.stderr)
        sys.exit(2)
    
    if not target_dir.is_dir():
        print(f"error: {target_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    print(f"Cleaning up SMT files in: {target_dir}")
    print()
    
    try:
        cleanup_smt_files(target_dir)
    except Exception as e:
        print(f"✗ Failed to cleanup files: {e}", file=sys.stderr)
        sys.exit(5)
    
    print()
    print("Cleanup completed successfully!")


if __name__ == "__main__":
    main(sys.argv)
