"""run_interpolation.py

A generic orchestrator to run SMT solvers (MathSAT, Yaga, OpenSMT) on SMT-LIB benchmarks
and collect interpolation results.

This script:
1. Runs the specified solver on a set of input .smt2 files.
2. Captures the result (SAT/UNSAT/UNKNOWN), execution time, and generated interpolant.
3. Logs all results to a CSV file and a detailed text file.

Usage::

    $ python scripts/run_interpolation.py mathsat inputs/benchmark_dir -t 600
    $ python scripts/run_interpolation.py yaga inputs/single_file.smt2

The script tracks run numbers automatically in the output directory.
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Import solver definitions from solvers.py
# Assumes solvers.py is in the same directory or in the python path
try:
    from solvers import InterpolantSolver, create_solver, defined_solvers
except ImportError:
    # If running from a different directory, try adding the script directory to path
    sys.path.append(str(Path(__file__).resolve().parent))
    from solvers import InterpolantSolver, create_solver, defined_solvers

# =========================
# Orchestrator
# =========================

def process_file(path: str, solver: InterpolantSolver, timeout: int = 600) -> Dict[str, Any]:
    """
    Process a single SMT file with a single solver and return results.
    
    Returns:
        Dict containing all the results and metadata for logging
    """
    file_path = Path(path)
    
    result = {
        'file_name': file_path.name,
        'solver_name': solver.name,
        'success': False,
        'error_message': None,
        'solver_run': {},  # Result of the solver run
    }
    
    try:
        # Run the solver on the input file
        print(f"Running {solver.name}...")
        result_val, interpolant, time_val, stdout, stderr = solver.run(path, timeout)
        
        print(f"Result: {solver.name}={result_val} ({time_val:.3f}s)")
        
        # Store individual solver run results
        result['solver_run'] = {
            'solver': solver.name, 
            'input_file': file_path.name, 
            'time_seconds': f"{time_val:.6f}", 
            'result': result_val,
            'interpolant_produced': interpolant is not None,
            'error': None
        }
        
        # If UNSAT, should produce interpolant
        if result_val == "unsat":
            if interpolant is None:
                error_msg = f"{solver.name} did not produce an interpolant for UNSAT formula"
                print(f"ERROR: {error_msg}")
                result['error_message'] = error_msg
                result['solver_run']['error'] = error_msg
                # We still return the result so it gets logged, but success remains False
                return result
            
        # Store detailed results for file output
        result['detailed_results'] = {
            'file_name': file_path.name,
            'result': result_val,
            'solver_time': time_val,
            'solver_interpolant': interpolant
        }
        
        result['success'] = True
        return result
        
    except Exception as e:
        error_msg = f"Exception during processing: {e}"
        result['error_message'] = error_msg
        result['solver_run'] = {
            'solver': solver.name, 
            'input_file': file_path.name, 
            'time_seconds': "0.000000", 
            'result': "error",
            'interpolant_produced': False,
            'error': error_msg
        }
        return result


def write_results(file_result: Dict[str, Any], solver_csv_writer: csv.writer, detailed_file) -> None:
    """
    Write the results from process_file to CSV files and append to detailed results file.
    
    Args:
        file_result: Dictionary containing all results from process_file
        solver_csv_writer: CSV writer for individual solver runs
        detailed_file: Open file handle for detailed results
    """
    # Write solver run to CSV
    run = file_result['solver_run']
    if run:
        solver_csv_writer.writerow([
            run['solver'], 
            run['input_file'], 
            run['time_seconds'], 
            run['result'], 
            run['interpolant_produced'],
            run['error']
        ])
    
    # Write to detailed results file
    detailed_file.write(f"=== {file_result['file_name']} ===\n")
    detailed_file.write(f"Solver: {file_result['solver_name']}\n")
    detailed_file.write(f"Success: {file_result['success']}\n")
    
    if file_result['error_message']:
        detailed_file.write(f"Error: {file_result['error_message']}\n")
    
    # Write detailed results if available
    if 'detailed_results' in file_result:
        details = file_result['detailed_results']
        detailed_file.write(f"Result: {details['result']}\n")
        detailed_file.write(f"Time: {details['solver_time']:.6f}s\n")
        
        if details['solver_interpolant']:
            detailed_file.write(f"Interpolant: {details['solver_interpolant']}\n")
        else:
            detailed_file.write("Interpolant: None\n")
    
    detailed_file.write("\n")  # Add blank line between entries
    detailed_file.flush()  # Ensure data is written immediately


def get_next_run_number(output_dir: Path) -> int:
    """
    Get the next run number by checking existing files in the output directory.
    
    Args:
        output_dir: Directory to check for existing run files
        
    Returns:
        Next available run number
    """
    # Look for existing files with pattern *_run_N.* 
    existing_files = list(output_dir.glob("*_run_*.csv")) + list(output_dir.glob("*_run_*.txt"))
    
    if not existing_files:
        return 1
    
    # Extract run numbers from existing files
    run_numbers = []
    for file_path in existing_files:
        # Look for pattern like "correctness_solver_runs_run_3.csv"
        parts = file_path.stem.split('_run_')
        if len(parts) == 2:
            try:
                run_num = int(parts[1])
                run_numbers.append(run_num)
            except ValueError:
                continue
    
    return max(run_numbers) + 1 if run_numbers else 1


def gather_inputs(inputs_field: str) -> List[Path]:
    """Return a list of *.smt2 files to feed to the solver."""
    target = Path(inputs_field).expanduser().resolve()
    if target.is_dir():
        files = sorted(target.glob("*.smt2"))
        if not files:
            sys.exit(f"Error: No .smt2 files found in directory {target}")
        return files
    elif target.is_file():
        return [target]
    else:
        sys.exit(f"Error: Input path does not exist: {target}")


def main(argv: List[str]) -> int:
    """Main function to run a single SMT solver on inputs."""
    parser = argparse.ArgumentParser(
        description="Run an SMT solver on input files"
    )
    parser.add_argument(
        "solver", 
        choices=defined_solvers,
        help="Solver name (mathsat, yaga, opensmt)"
    )
    parser.add_argument(
        "inputs", 
        help="Path to input files or directory containing .smt2 files"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output directory for results (default: creates 'results' folder in current directory)"
    )
    parser.add_argument(
        "-t", "--timeout", 
        type=int, 
        default=600,
        help="Timeout in seconds for each solver execution (default: 600 = 10 minutes)"
    )
    
    args = parser.parse_args(argv)
    
    # Create solver instance
    try:
        solver = create_solver(args.solver)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error creating solver: {e}")
    
    # Get input files
    input_files = gather_inputs(args.inputs)
    
    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path.cwd() / "results"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get next run number
    run_number = get_next_run_number(output_dir)
    
    # Set up output file paths with run numbers
    solver_csv_path = output_dir / f"solver_runs_run_{run_number}.csv"
    detailed_results_path = output_dir / f"detailed_results_run_{run_number}.txt"
    
    print(f"Solver             : {args.solver}")
    print(f"Input files        : {len(input_files)} files")
    print(f"Output directory   : {output_dir}")
    print(f"Run number         : {run_number}")
    print(f"Timeout per solver : {args.timeout} seconds ({args.timeout/60:.1f} minutes)")
    print(f"Solver runs CSV    : {solver_csv_path}")
    print(f"Detailed results   : {detailed_results_path}")
    print(f"Processing files in: {args.inputs}\n")
    
    # Prepare CSV writer and detailed results file
    with solver_csv_path.open("w", newline="", encoding="utf-8") as solver_csv_file, \
         detailed_results_path.open("w", encoding="utf-8") as detailed_file:
        
        solver_csv_writer = csv.writer(solver_csv_file)
        
        # Always write headers since we're creating new files
        solver_csv_writer.writerow(["solver", "input_file", "time_seconds", "result", "interpolant_produced", "error"])
        
        # Write header to detailed results file
        detailed_file.write(f"Detailed Results - Run {run_number}\n")
        detailed_file.write(f"Solver: {args.solver}\n")
        detailed_file.write(f"Generated for {len(input_files)} input files\n")
        detailed_file.write("=" * 50 + "\n\n")
        
        failures = 0
        
        # Process each file
        for smt_file in input_files:
            print(f"=== Processing {smt_file.name} ===")
            print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Process the file and get results
            file_result = process_file(str(smt_file), solver, args.timeout)
            
            # Write results to CSV files and detailed results file
            write_results(file_result, solver_csv_writer, detailed_file)
            
            # Flush CSV files after each file
            solver_csv_file.flush()
            
            # Track failures and provide feedback
            if not file_result['success']:
                failures += 1
                print(f"FAILURE: {smt_file.name}")
                if file_result['error_message']:
                    print(f"  Error: {file_result['error_message']}")
            else:
                print(f"SUCCESS: {smt_file.name}")
            print()
    
    print(f"\nSummary: {failures} failures out of {len(input_files)} files")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
