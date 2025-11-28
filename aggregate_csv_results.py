"""aggregate_csv_results.py

A script to aggregate multiple CSV result files into a single file.

This script:
1. Takes a directory containing CSV files as input.
2. Reads all CSV files in the directory.
3. Aggregates their contents into a single CSV file.
4. Saves the result as {directory_name}_aggregated.csv inside the directory.

Usage::

    $ python scripts/aggregate_csv_results.py results/run_results_123/yaga
    
This will create results/run_results_123/yaga/yaga_aggregated.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import List


def gather_csv_files(directory: Path) -> List[Path]:
    """Return a sorted list of CSV files in the directory (excluding aggregated files)."""
    files = sorted(
        f for f in directory.glob("*.csv") 
        if not f.name.endswith("_aggregated.csv")
    )
    return files


def aggregate_csvs(input_dir: Path) -> Path:
    """
    Aggregate all CSV files in the input directory into a single file.
    
    Args:
        input_dir: Directory containing CSV files to aggregate
        
    Returns:
        Path to the created aggregated file
    """
    csv_files = gather_csv_files(input_dir)
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    # Output file named after the directory
    output_file = input_dir / f"{input_dir.name}_aggregated.csv"
    
    all_rows: List[List[str]] = []
    header: List[str] = []
    
    for csv_file in csv_files:
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if not rows:
                continue
                
            file_header = rows[0]
            file_data = rows[1:]
            
            # Use the first file's header as the canonical header
            if not header:
                header = file_header
            elif file_header != header:
                print(f"Warning: {csv_file.name} has different header: {file_header}")
                print(f"         Expected: {header}")
                # Still include data, assuming columns are in same order
            
            all_rows.extend(file_data)
    
    # Write aggregated file
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    
    return output_file


def main(argv: List[str]) -> int:
    """Main function to aggregate CSV files."""
    parser = argparse.ArgumentParser(
        description="Aggregate CSV result files into a single file"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing CSV files to aggregate"
    )
    
    args = parser.parse_args(argv)
    
    input_dir = Path(args.input_dir).expanduser().resolve()
    
    if not input_dir.is_dir():
        sys.exit(f"Error: {input_dir} is not a directory")
    
    try:
        output_file = aggregate_csvs(input_dir)
        csv_files = gather_csv_files(input_dir)
        print(f"Aggregated {len(csv_files)} CSV files into: {output_file}")
        return 0
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

