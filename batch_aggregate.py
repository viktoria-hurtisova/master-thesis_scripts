"""batch_aggregate.py

Batch aggregation script that finds and processes all run_results_* directories.

This script:
1. Scans solver result directories (opensmt, mathsat, yaga) for run_results_* folders
2. Runs aggregation on each folder that doesn't already have an aggregated file
3. Optionally outputs all aggregated files to a single directory

Usage::

    # Aggregate all results to a single output directory
    $ python scripts/batch_aggregate.py results/ results/aggregated

    # Process only a specific solver's results
    $ python scripts/batch_aggregate.py results/yaga/ results/aggregated

    # Force re-aggregation even if aggregated files exist
    $ python scripts/batch_aggregate.py results/ results/aggregated --force
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

from aggregate_csv_results import aggregate_csvs, gather_csv_files


def find_run_result_dirs(base_dir: Path) -> List[Path]:
    """
    Find all run_results_* directories under the base directory.
    
    Searches recursively for directories matching the pattern.
    """
    run_dirs = sorted(base_dir.glob("**/run_results_*"))
    # Filter to only directories
    return [d for d in run_dirs if d.is_dir()]


def has_aggregated_file(run_dir: Path, output_dir: Path) -> bool:
    """Check if the output directory already has an aggregated CSV file for this run."""
    expected_file = output_dir / f"{run_dir.name}_aggregated.csv"
    return expected_file.exists()


def batch_aggregate(
    base_dir: Path,
    output_dir: Path,
    force: bool = False
) -> tuple[int, int, int]:
    """
    Aggregate all run_results_* directories under base_dir.
    
    Args:
        base_dir: Base directory to search for run_results_* folders
        output_dir: Output directory for all aggregated files
        force: If True, re-aggregate even if aggregated file exists
        
    Returns:
        Tuple of (processed_count, skipped_count, error_count)
    """
    run_dirs = find_run_result_dirs(base_dir)
    
    if not run_dirs:
        print(f"No run_results_* directories found under {base_dir}")
        return 0, 0, 0
    
    print(f"Found {len(run_dirs)} run_results_* directories")
    print("=" * 60)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for run_dir in run_dirs:
        relative_path = run_dir.relative_to(base_dir)
        
        # Check if already aggregated (unless force is set)
        if not force and has_aggregated_file(run_dir, output_dir):
            print(f"[SKIP] {relative_path} (already aggregated)")
            skipped += 1
            continue
        
        # Check if there are CSV files to aggregate
        csv_files = gather_csv_files(run_dir)
        if not csv_files:
            print(f"[SKIP] {relative_path} (no CSV files)")
            skipped += 1
            continue
        
        try:
            output_file = aggregate_csvs(run_dir, output_dir)
            print(f"[OK]   {relative_path} -> {output_file.name} ({len(csv_files)} files)")
            processed += 1
        except Exception as e:
            print(f"[ERR]  {relative_path}: {e}")
            errors += 1
    
    return processed, skipped, errors


def main(argv: List[str]) -> int:
    """Main function for batch aggregation."""
    parser = argparse.ArgumentParser(
        description="Batch aggregate all run_results_* directories"
    )
    parser.add_argument(
        "base_dir",
        help="Base directory to search for run_results_* folders"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for all aggregated files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-aggregation even if aggregated files exist"
    )
    
    args = parser.parse_args(argv)
    
    base_dir = Path(args.base_dir).expanduser().resolve()
    
    if not base_dir.is_dir():
        sys.exit(f"Error: {base_dir} is not a directory")
    
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed, skipped, errors = batch_aggregate(base_dir, output_dir, args.force)
    
    print("=" * 60)
    print(f"Summary: {processed} processed, {skipped} skipped, {errors} errors")
    
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

