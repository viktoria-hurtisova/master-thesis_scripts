#!/usr/bin/env python3
"""orchestrate_smt_pipeline.py

Orchestrator script that runs the full SMT processing pipeline on a directory:

1. copy_unsat_files.py - Copies files with UNSAT status to a new directory
2. bipartition_smt.py - Bipartitions the SMT files into A and B formulas
3. cleanup_smt.py - Removes original files, keeping only .bi.smt2 files
4. check_formula_a_sat.py - Checks if formula A is satisfiable and removes files where A is UNSAT

Usage::

    python scripts/orchestrate_smt_pipeline.py <input_directory>
    python scripts/orchestrate_smt_pipeline.py <input_directory> --timeout 600
    python scripts/orchestrate_smt_pipeline.py <input_directory> --skip-copy
    python scripts/orchestrate_smt_pipeline.py <input_directory> -o <output_directory>

The output directory will be <input_directory>_unsat (e.g., inputs/test -> inputs/test_unsat)
unless --skip-copy is used, in which case the input directory is used directly.
If -o/--output is provided, that directory is used as the output for the copy step.

Requires only Python 3 (no external deps for this orchestrator).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def get_script_path(script_name: str) -> Path:
    """Get the full path to a script in the same directory as this orchestrator."""
    return Path(__file__).parent / script_name


def run_script(script_path: Path, args: List[str], description: str) -> bool:
    """
    Run a Python script with the given arguments.
    
    Args:
        script_path: Path to the Python script
        args: List of arguments to pass to the script
        description: Description of what this step does (for logging)
    
    Returns:
        True if successful, False otherwise
    """
    print()
    print("=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    print(f"Running: python {script_path.name} {' '.join(args)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            check=False,  # Don't raise on non-zero exit
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n⚠ Warning: Script exited with code {result.returncode}")
            # Continue anyway for some scripts that might have partial failures
            return True  # Changed to True to continue pipeline
        
        print(f"\n✓ {description} completed successfully")
        return True
        
    except FileNotFoundError:
        print(f"\n✗ Error: Script not found: {script_path}")
        return False
    except Exception as e:
        print(f"\n✗ Error running script: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the full SMT processing pipeline on a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline steps:
  1. copy_unsat_files.py   - Copy files with UNSAT status to <dir>_unsat
  2. bipartition_smt.py    - Bipartition files into A and B formulas
  3. cleanup_smt.py        - Remove original files, keep only .bi.smt2
  4. check_formula_a_sat.py - Remove files where formula A is UNSAT

Examples:
    python orchestrate_smt_pipeline.py inputs/RandomCoupled
    python orchestrate_smt_pipeline.py inputs/test --timeout 300
    python orchestrate_smt_pipeline.py inputs/test -o outputs/test_processed
        """
    )
    
    parser.add_argument(
        'input_directory',
        help='Input directory containing SMT2 files to process'
    )
    
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=900,
        help='Timeout in seconds for check_formula_a_sat.py (default: 900)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--skip-copy',
        action='store_true',
        help='Skip step 1 (copy_unsat_files) and use the input directory directly'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for copy step (default: <input_directory>_unsat)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_dir = Path(args.input_directory).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Determine working directory based on --skip-copy flag and --output argument
    if args.skip_copy:
        working_dir = input_dir
    elif args.output:
        working_dir = Path(args.output).resolve()
    else:
        working_dir = input_dir.parent / f"{input_dir.name}_unsat"
    
    print("=" * 60)
    print("SMT PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input directory:  {input_dir}")
    if args.skip_copy:
        print(f"Working directory: {working_dir} (using input directly, skipping copy)")
    elif args.output:
        print(f"Output directory: {working_dir} (user-specified)")
    else:
        print(f"Output directory: {working_dir}")
    print(f"Timeout:          {args.timeout}s")
    
    if args.skip_copy:
        # Skip step 1, use input directory directly
        print("\n⏭ Skipping step 1 (copy_unsat_files) - using input directory directly")
    else:
        # Step 1: Copy UNSAT files
        script = get_script_path("copy_unsat_files.py")
        if not run_script(script, [str(input_dir), str(working_dir)], 
                          "Copy files with UNSAT status"):
            print("\n✗ Pipeline failed at step 1 (copy_unsat_files)")
            sys.exit(1)
        
        # Check if output directory was created
        if not working_dir.exists():
            print(f"\n✗ Output directory was not created: {working_dir}")
            sys.exit(1)
    
    smt_files = list(working_dir.glob("*.smt2"))
    if not smt_files:
        print(f"\n⚠ No .smt2 files found in working directory. Pipeline complete (nothing to process).")
        sys.exit(0)
    
    print(f"\nFound {len(smt_files)} .smt2 files in working directory")
    
    # Step 2: Bipartition SMT files
    script = get_script_path("bipartition_smt.py")
    if not run_script(script, [str(working_dir)], 
                      "Bipartition SMT files into A and B formulas"):
        print("\n✗ Pipeline failed at step 2 (bipartition_smt)")
        sys.exit(1)
    
    # Step 3: Cleanup (remove original files, keep .bi.smt2)
    script = get_script_path("cleanup_smt.py")
    if not run_script(script, [str(working_dir)], 
                      "Cleanup - remove original files, keep .bi.smt2"):
        print("\n✗ Pipeline failed at step 3 (cleanup_smt)")
        sys.exit(1)
    
    # Check if there are any .bi.smt2 files to process
    bi_files = list(working_dir.glob("*.bi.smt2"))
    if not bi_files:
        print(f"\n⚠ No .bi.smt2 files found after bipartition. Pipeline complete.")
        sys.exit(0)
    
    print(f"\nFound {len(bi_files)} .bi.smt2 files to check")
    
    # Step 4: Check formula A satisfiability
    script = get_script_path("check_formula_a_sat.py")
    if not run_script(script, [str(working_dir), "-t", str(args.timeout)], 
                      "Check formula A satisfiability"):
        print("\n✗ Pipeline failed at step 4 (check_formula_a_sat)")
        sys.exit(1)
    
    # Final summary
    remaining_files = list(working_dir.glob("*.smt2"))
    
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {working_dir}")
    print(f"Remaining files:  {len(remaining_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
