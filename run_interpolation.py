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

Results are organized by solver name and run number:
    results/<solver_name>/run_results_<num>/
    
Example: results/mathsat/run_results_1/, results/mathsat/run_results_2/, etc.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import sys
import threading
import time
import subprocess
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
        result_val, interpolant, time_val, stdout, stderr = solver.run(path, timeout)
        
        # Store solver output
        result['solver_stdout'] = stdout
        result['solver_stderr'] = stderr
        
        # Store individual solver run results
        result['solver_run'] = {
            'solver': solver.name, 
            'input_file': file_path.name, 
            'time_seconds': f"{time_val:.6f}", 
            'result': result_val,
            'interpolant_produced': interpolant is not None,
            'error': None
        }
        
        # Check for errors in stderr
        if stderr and stderr.strip():
            error_msg = f"Solver produced error output in stderr"
            result['error_message'] = error_msg
            result['solver_run']['result'] = "stderr_error"
            result['solver_run']['error'] = error_msg
            # Store detailed results for file output even on error
            result['detailed_results'] = {
                'file_name': file_path.name,
                'result': "stderr_error",
                'solver_time': time_val,
                'solver_interpolant': interpolant
            }
            return result
        
        # Check for errors in stdout
        if stdout and "error" in stdout.lower():
            error_msg = f"Solver produced error in stdout"
            result['error_message'] = error_msg
            result['solver_run']['result'] = "stdout_error"
            result['solver_run']['error'] = error_msg
            # Store detailed results for file output even on error
            result['detailed_results'] = {
                'file_name': file_path.name,
                'result': "stdout_error",
                'solver_time': time_val,
                'solver_interpolant': interpolant
            }
            return result
        
        # If UNSAT, should produce interpolant
        if result_val == "unsat":
            if interpolant is None:
                error_msg = f"{solver.name} did not produce an interpolant for UNSAT formula"
                result['error_message'] = error_msg
                result['solver_run']['result'] = "no_interpolant"
                result['solver_run']['error'] = error_msg
                # Store detailed results for file output even on error
                result['detailed_results'] = {
                    'file_name': file_path.name,
                    'result': "no_interpolant",
                    'solver_time': time_val,
                    'solver_interpolant': None
                }
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
        
    except subprocess.TimeoutExpired:
        error_msg = f"Solver execution timed out after {timeout} seconds"
        result['error_message'] = error_msg
        result['solver_stdout'] = result.get('solver_stdout', "")
        result['solver_stderr'] = result.get('solver_stderr', "")
        result['solver_run'] = {
            'solver': solver.name, 
            'input_file': file_path.name, 
            'time_seconds': f"{float(timeout):.6f}", 
            'result': "timeout",
            'interpolant_produced': False,
            'error': error_msg
        }
        return result

    except Exception as e:
        error_msg = f"Exception during processing: {e}"
        result['error_message'] = error_msg
        result['solver_stdout'] = result.get('solver_stdout', "")
        result['solver_stderr'] = result.get('solver_stderr', "")
        result['solver_run'] = {
            'solver': solver.name, 
            'input_file': file_path.name, 
            'time_seconds': "0.000000", 
            'result': "error",
            'interpolant_produced': False,
            'error': error_msg
        }
        return result


def write_results(file_result: Dict[str, Any], output_dir: Path, detailed: bool = False) -> None:
    """
    Write the results from process_file to individual CSV and optionally detailed files.
    
    Args:
        file_result: Dictionary containing all results from process_file
        output_dir: Directory to save the results
        detailed: Whether to write detailed results file (default: False)
    """
    file_name = file_result['file_name']
    base_name = Path(file_name).stem
    
    csv_path = output_dir / f"{base_name}_results.csv"

    # Write solver run to CSV
    run = file_result['solver_run']
    if run:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["solver", "input_file", "time_seconds", "result", "interpolant_produced", "error"])
            writer.writerow([
                run['solver'], 
                run['input_file'], 
                run['time_seconds'], 
                run['result'], 
                run['interpolant_produced'],
                run['error']
            ])
    
    # Write detailed results file only if requested
    if detailed:
        detailed_path = output_dir / f"{base_name}_results_detailed.txt"
        with detailed_path.open("w", encoding="utf-8") as detailed_file:
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
                    interpolant = str(details['solver_interpolant'])
                    if len(interpolant) > 1000:
                        interpolant = interpolant[:1000] + "... <truncated>"
                    detailed_file.write(f"Interpolant: {interpolant}\n")
                else:
                    detailed_file.write("Interpolant: None\n")
            elif file_result['solver_run']:
                # Fallback to solver_run info for timeout/exception cases
                run = file_result['solver_run']
                detailed_file.write(f"Result: {run['result']}\n")
                detailed_file.write(f"Time: {run['time_seconds']}s\n")
                detailed_file.write(f"Interpolant: None\n")
            
            # Write solver output (stdout and stderr)
            detailed_file.write("\n--- Solver Output (STDOUT) ---\n")
            solver_stdout = file_result.get('solver_stdout', "")
            if solver_stdout:
                detailed_file.write(solver_stdout)
            else:
                detailed_file.write("(empty)\n")
            
            detailed_file.write("\n--- Solver Output (STDERR) ---\n")
            solver_stderr = file_result.get('solver_stderr', "")
            if solver_stderr:
                detailed_file.write(solver_stderr)
            else:
                detailed_file.write("(empty)\n")



def gather_inputs(inputs_field: str) -> List[Path]:
    """Return a list of *.smt2 files to feed to the solver.
    
    Recursively searches subdirectories when a directory is provided.
    """
    target = Path(inputs_field).expanduser().resolve()
    if target.is_dir():
        files = sorted(target.rglob("*.smt2"))
        if not files:
            sys.exit(f"Error: No .smt2 files found in directory {target}")
        return files
    elif target.is_file():
        return [target]
    else:
        sys.exit(f"Error: Input path does not exist: {target}")


def process_single_input(
    smt_file: Path, 
    solver_name: str, 
    timeout: int, 
    run_dir: Path, 
    print_lock: threading.Lock, 
    detailed: bool = False
) -> bool:
    """
    Process a single input file: create solver, run processing, write results, log output.
    Returns True if failure occurred, False otherwise (to sum failures).
    """
    # Create solver instance per thread to ensure safety/independence
    try:
        solver = create_solver(solver_name)
    except Exception as e:
        with print_lock:
            print(f"Error creating solver for {smt_file.name}: {e}")
        return True  # Failure
    
    with print_lock:
        print(f"=== Processing {smt_file.name} ===")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run the main processing logic
    file_result = process_file(str(smt_file), solver, timeout)
    
    # Write results to disk (thread-safe as files are distinct per input)
    write_results(file_result, run_dir, detailed)
    
    success = file_result['success']
    
    with print_lock:
        # Print result summary
        if not success:
            print(f"FAILURE: {smt_file.name}")
            if file_result['error_message']:
                print(f"  Error: {file_result['error_message']}")
        else:
            print(f"SUCCESS: {smt_file.name}")
        print()  # Blank line between entries
        
    return not success


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
    parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Enable detailed results logging to a text file (default: disabled)"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs/threads to use (default: 1)"
    )
    
    args = parser.parse_args(argv)
    
    # Check if solver creation works (just to fail early if solver is missing)
    try:
        # We just check if we can create one, but we'll create fresh ones in threads
        _ = create_solver(args.solver)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error creating solver: {e}")
    
    # Get input files
    input_files = gather_inputs(args.inputs)
    
    # Set up output directory
    if args.output:
        base_output_dir = Path(args.output)
    else:
        base_output_dir = Path.cwd() / "results"
    
    # Create base output directory if it doesn't exist
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create solver-specific directory (e.g., results/mathsat/)
    solver_dir = base_output_dir / args.solver
    solver_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run directory with timestamp (e.g., results/mathsat/run_results_1733567890/)
    timestamp = int(time.time())
    run_dir = solver_dir / f"run_results_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Solver             : {args.solver}")
    print(f"Input files        : {len(input_files)} files")
    print(f"Processing files in: {args.inputs}")
    print(f"Output directory   : {run_dir}")
    print(f"Run ID             : {timestamp}")
    print(f"Timeout per solver : {args.timeout} seconds ({args.timeout/60:.1f} minutes)")
    print(f"Detailed output    : {'enabled' if args.detailed else 'disabled'}")
    print(f"Parallel jobs      : {args.jobs}")
    print()
    
    failures = 0
    
    # Lock for synchronizing print output
    print_lock = threading.Lock()

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_input, 
                smt_file, 
                args.solver, 
                args.timeout, 
                run_dir, 
                print_lock, 
                args.detailed
            ): smt_file
            for smt_file in input_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            smt_file = future_to_file[future]
            try:
                is_failure = future.result()
                if is_failure:
                    failures += 1
            except Exception as exc:
                with print_lock:
                    print(f"Generated an exception for {smt_file.name}: {exc}")
                failures += 1

    print(f"\nSummary: {failures} failures out of {len(input_files)} files")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
