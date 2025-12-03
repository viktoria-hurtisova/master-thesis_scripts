"""yaga_interpolant_verification.py

A script to verify Craig's interpolants produced by the Yaga solver.

This script:
1. Runs Yaga on the provided SMT-LIB files.
2. If the result is UNSAT and an interpolant is produced, it extracts the interpolant.
3. Creates a verification query using Z3 to check the two interpolation conditions:
   - A => I
   - I => not B
4. Reports verification success/failure.

Usage::

    $ python scripts/yaga_interpolant_verification.py inputs/
    $ python scripts/yaga_interpolant_verification.py inputs/single_file.smt2 --timeout 600
    
The script creates a 'results/run_<timestamp>' directory with individual CSV summaries
and detailed logs for each input file.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
import sys
import time
import concurrent.futures
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from solvers import InterpolantSolver, MathSat, Yaga, OpenSMT, Z3, create_solver, load_config, defined_solvers
except ImportError:
    from scripts.solvers import InterpolantSolver, MathSat, Yaga, OpenSMT, Z3, create_solver, load_config, defined_solvers

# =========================
# Verification
# =========================

def _strip_named_wrapper(assert_body: str) -> str:
    """
    Remove SMT-LIB attribute wrapper: (! phi :named A)
    Returns the inner phi when possible; otherwise returns the input unchanged.
    """
    # Single-line, simple pattern removal
    m = re.match(r"^\(!\s*(.+?)\s+:?(?:named)\s+[AB]\s*\)$", assert_body.strip())
    if m:
        return m.group(1).strip()
    return assert_body.strip()

def create_verification_input_file(source_path: str, interpolant: str) -> str:
    """
    Create a verification SMT-LIB file that checks Craig's interpolant conditions for a single interpolant I.
    If SAT, I is NOT a valid interpolant; if UNSAT, I IS a valid interpolant:
      1) A ∧ ¬I (counterexample to A ⇒ I)
      2) I ∧ B (counterexample to I ⇒ ¬B)
    The function extracts A and B from the original file using :named or :interpolation-group attributes.
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a new file path for the comparison
    source_path_obj = Path(source_path)
    timestamp = int(time.time())
    # Use thread ID or similar to avoid collision if multiple threads process same file?
    # Actually, create_verification_input_file creates a new file.
    # If we run multiple runs in parallel on SAME file, timestamp might collide.
    # Adding a random suffix or thread ID would be safer.
    # But the logic runs one file once per run.
    # Parallelization is across *different* files.
    output_path = source_path_obj.with_name(f"{source_path_obj.stem}.verification_{timestamp}_{threading.get_ident()}.smt2")

    # process the file to extract A and B
    lines = content.split('\n')
    processed_lines: List[str] = []
    a_formula = ""
    b_formula = ""

    for line in lines:
        stripped = line.strip()
        # Collect A/B assertions (best-effort, single-line asserts)
        if stripped.startswith('(assert'):
            # capture body between (assert ...)
            m = re.match(r"^\(assert\s+(.*)\)\s*$", stripped)
            if not m:
                # fallback: skip multi-line asserts
                continue
            body = m.group(1)
            is_a = (':named A' in stripped)
            is_b = (':named B' in stripped)
            core = _strip_named_wrapper(body)
            if is_a:
                a_formula = core
            elif is_b:
                b_formula = core
            # do not passthrough assert lines
            continue
        if '(set-info :status' in stripped:
            continue
        if '(check-sat' in stripped:
            # we'll add our own checks later
            continue
        if '(exit)' in stripped:
            continue
        processed_lines.append(line)

    processed_lines.append(f'(assert (and (=> {a_formula} {interpolant}) (=> {interpolant} (not {b_formula}))))')
    processed_lines.append('(check-sat)')
    processed_lines.append('(exit)')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    return str(output_path)

def verify_interpolant(file_path: str, interpolant: str, timeout: int = 600) -> Tuple[bool, Optional[str], str, str]:
    """
    Verify a single interpolant against Craig's conditions using Z3 by checking:
    A ⇒ I ∧ I ⇒ ¬B is SAT

    Returns tuple (is_verified, z3_output, z3_stdout, z3_stderr) where z3_output contains stdout and stderr combined.
    
    Raises:
        RuntimeError: If Z3 execution fails
    """
    z3_output = None
    z3_stdout = ""
    z3_stderr = ""
    verification_file_path = None
    try:
        verification_file_path = create_verification_input_file(file_path, interpolant)
        z3_config_path = load_config("z3")
        z3_solver = Z3(z3_config_path)
        
        # Use solver.run to execute the verification
        sat_result, _, _, stdout, stderr = z3_solver.run(verification_file_path, timeout)
        z3_stdout = stdout
        z3_stderr = stderr

        # Check if Z3 output contains error
        if stdout and "error" in stdout.lower():
            error_msg = f"Z3 solver ended with an error in stdout: {stdout}"
            raise RuntimeError(error_msg)

        # Verify that  A ⇒ I ∧ I ⇒ ¬B	
        is_verified = (sat_result == "sat")
        # print(f"Verification result: {sat_result}")
        
        # Combine stdout and stderr for z3_output
        z3_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}" if (stdout or stderr) else None
        
        return is_verified, z3_output, z3_stdout, z3_stderr
        
    except Exception as e:
        error_msg = f"Z3 verification failed: {str(e)}"
        
        # Re-raise the exception so it can be caught in process_file
        raise RuntimeError(error_msg) from e
    finally:
        # delete the verification file
        if verification_file_path:
            try:
                os.unlink(verification_file_path)
            except OSError:
                pass

# =========================
# Orchestrator
# =========================

def process_file(path: str, solver: InterpolantSolver, timeout: int = 600) -> Dict[str, Any]:
    """
    Process a single SMT file with one solver and verify its interpolant (if any).
    """
    file_path = Path(path)

    result: Dict[str, Any] = {
        'file_name': file_path.name,
        'solver_name': solver.name,
        'success': False,
        'error_message': None,
        'solver_runs': [],
        'verification_result': None
    }

    try:
        # print(f"Running {solver.name}...")
        sat_res, interpolant, run_time, solver_stdout, solver_stderr = solver.run(path, timeout)
        # print(f"Result: {solver.name}={sat_res} ({run_time:.3f}s)")

        # Store solver output
        result['solver_stdout'] = solver_stdout
        result['solver_stderr'] = solver_stderr

        result['solver_runs'] = [
            {'solver': solver.name, 'input_file': file_path.name, 'time_seconds': f"{run_time:.6f}", 'result': sat_res}
        ]

        if sat_res == 'sat':
            if interpolant is not None:
                result['verification_result'] = 'error_interpolant_for_sat'
                result['error_message'] = 'Interpolant produced for SAT formula'
                return result
            # SAT and no interpolant: valid as no verification required
            result['verification_result'] = 'sat_no_interpolant'
            result['detailed_results'] = {
                'file_name': file_path.name,
                'result': sat_res,
                'solver_time': run_time,
                'interpolant': None
            }
            result['success'] = True
            return result

        if sat_res == 'unsat':
            if interpolant is None:
                result['verification_result'] = 'no_interpolant_produced'
                result['error_message'] = f"{solver.name} did not produce an interpolant"
                return result
            # Verify interpolant
            display_interpolant = interpolant
            if display_interpolant and len(display_interpolant) > 1000:
                display_interpolant = display_interpolant[:1000] + "... <truncated>"
            # print(f"Interpolant: {display_interpolant}")
            # print("Verifying interpolant...")
            try:
                is_verified, z3_output, z3_stdout, z3_stderr = verify_interpolant(path, interpolant, timeout)
                result['verification_result'] = 'verified' if is_verified else 'not_verified'
                # print(f"Interpolant is verified: {is_verified}")
                result['detailed_results'] = {
                    'file_name': file_path.name,
                    'result': sat_res,
                    'solver_time': run_time,
                    'interpolant': interpolant
                }
                # Store Z3 output
                result['z3_stdout'] = z3_stdout
                result['z3_stderr'] = z3_stderr
                if z3_output:
                    result['z3_verification_output'] = z3_output
                result['success'] = True
                return result
            except Exception as e:
                result['verification_result'] = 'z3_error'
                result['error_message'] = f"Z3 verification failed: {e}"
                result['z3_verification_output'] = str(e)
                result['detailed_results'] = {
                    'file_name': file_path.name,
                    'result': sat_res,
                    'solver_time': run_time,
                    'interpolant': interpolant
                }
                return result

        # Unknown
        result['verification_result'] = f"unknown_result_{sat_res}"
        result['error_message'] = f"Unknown result: {sat_res}"
        return result

    except subprocess.TimeoutExpired as e:
        # Handle timeout from solver execution
        result['verification_result'] = 'solver_timed_out'
        result['error_message'] = f"Solver execution timed out after {timeout} seconds"
        # Ensure solver output fields exist even on timeout
        if 'solver_stdout' not in result:
            result['solver_stdout'] = ""
        if 'solver_stderr' not in result:
            result['solver_stderr'] = ""
        
        # Add entry to solver_runs so it gets written to CSV
        result['solver_runs'] = [
            {'solver': solver.name, 'input_file': file_path.name, 'time_seconds': f"{timeout}.000000", 'result': 'timeout'}
        ]
        return result
    except RuntimeError as e:
        # Check if the runtime error is a wrapped timeout
        if "timed out" in str(e):
            result['verification_result'] = 'solver_timed_out'
            result['error_message'] = str(e)
            # Ensure solver output fields exist even on timeout
            if 'solver_stdout' not in result:
                result['solver_stdout'] = ""
            if 'solver_stderr' not in result:
                result['solver_stderr'] = ""
            
            # Add entry to solver_runs so it gets written to CSV
            result['solver_runs'] = [
                {'solver': solver.name, 'input_file': file_path.name, 'time_seconds': f"{timeout}.000000", 'result': 'timeout'}
            ]
            return result
            
        # Check if the runtime error is due to solver stderr output (from solvers.py)
        if "produced error output" in str(e):
            result['verification_result'] = 'solver_ended_with_error'
            result['error_message'] = str(e)
            # Ensure solver output fields exist even on error
            if 'solver_stdout' not in result:
                result['solver_stdout'] = ""
            if 'solver_stderr' not in result:
                # The exception message contains the stderr content
                result['solver_stderr'] = str(e)
            
            # Add entry to solver_runs so it gets written to CSV (with 0 time)
            result['solver_runs'] = [
                {'solver': solver.name, 'input_file': file_path.name, 'time_seconds': "0.000000", 'result': 'error'}
            ]
            return result

        # Other runtime errors
        result['verification_result'] = f"exception_{str(e).replace(',', ';')}" 
        result['error_message'] = f"Exception during processing: {e}"
        # Ensure solver output fields exist even on error
        if 'solver_stdout' not in result:
            result['solver_stdout'] = ""
        if 'solver_stderr' not in result:
            result['solver_stderr'] = ""
            
        # Add entry to solver_runs so it gets written to CSV (with 0 time)
        result['solver_runs'] = [
            {'solver': solver.name, 'input_file': file_path.name, 'time_seconds': "0.000000", 'result': 'error'}
        ]
        return result
    except Exception as e:
        result['verification_result'] = f"exception_{str(e).replace(',', ';')}" 
        result['error_message'] = f"Exception during processing: {e}"
        # Ensure solver output fields exist even on error
        if 'solver_stdout' not in result:
            result['solver_stdout'] = ""
        if 'solver_stderr' not in result:
            result['solver_stderr'] = ""
            
        # Add entry to solver_runs so it gets written to CSV (with 0 time)
        result['solver_runs'] = [
            {'solver': solver.name, 'input_file': file_path.name, 'time_seconds': "0.000000", 'result': 'error'}
        ]
        return result


def write_results(file_result: Dict[str, Any], output_dir: Path, detailed: bool = False) -> None:
    """
    Write the results from process_file to individual CSV and detailed files in the output directory.
    
    Args:
        file_result: Dictionary containing all results from process_file
        output_dir: Directory to save the results
        detailed: If True, also write a detailed text file with full output
    """
    file_name = file_result['file_name']
    # Create filenames based on input filename
    base_name = Path(file_name).stem
    
    csv_path = output_dir / f"{base_name}_results.csv"
    detailed_path = output_dir / f"{base_name}_results_detailed.txt"
    
    # Write solver runs to CSV with verification result
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["input_file", "time_seconds", "verification_result"])
        for run in file_result['solver_runs']:
            writer.writerow([
                run['input_file'],
                run['time_seconds'],
                file_result['verification_result']
            ])
    
    # Write to detailed results file only if detailed flag is set
    if not detailed:
        return
    
    with detailed_path.open("w", encoding="utf-8") as detailed_file:
        detailed_file.write(f"=== {file_result['file_name']} ===\n")
        detailed_file.write(f"Solver: {file_result['solver_name']}\n")
        detailed_file.write(f"Success: {file_result['success']}\n")
        detailed_file.write(f"Verification Result: {file_result['verification_result']}\n")
        
        if file_result['error_message']:
            detailed_file.write(f"Error: {file_result['error_message']}\n")
        
        # Write detailed results if available
        if 'detailed_results' in file_result:
            details = file_result['detailed_results']
            detailed_file.write(f"SAT Result: {details['result']}\n")
            detailed_file.write(f"{file_result['solver_name']} time: {details['solver_time']:.6f}s\n")
            
            interpolant = str(details['interpolant'])
            # No truncation as requested
            detailed_file.write(f"Interpolant: {interpolant}\n")
        
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
        
        # Write z3 verification output if interpolant was verified or not verified
        if 'z3_stdout' in file_result or 'z3_stderr' in file_result:
            detailed_file.write("\n--- Z3 Verification Output (STDOUT) ---\n")
            z3_stdout = file_result.get('z3_stdout', "")
            if z3_stdout:
                detailed_file.write(z3_stdout)
            else:
                detailed_file.write("(empty)\n")
            
            detailed_file.write("\n--- Z3 Verification Output (STDERR) ---\n")
            z3_stderr = file_result.get('z3_stderr', "")
            if z3_stderr:
                detailed_file.write(z3_stderr)
            else:
                detailed_file.write("(empty)\n")
        
        # Write z3 error if Z3 verification failed
        if file_result['verification_result'] == 'z3_error' and 'z3_verification_output' in file_result:
            detailed_file.write("\n--- Z3 Verification Error ---\n")
            detailed_file.write(file_result['z3_verification_output'])
            detailed_file.write("\n")


def gather_inputs(inputs_field: str) -> List[Path]:
    """Return a list of *.smt2 files to feed to the solver.
    
    Recursively searches subdirectories when a directory is provided.
    Excludes temp files from preprocessing (patterns like *_pre_*.smt2 and *.verification_*.smt2).
    """
    target = Path(inputs_field).expanduser().resolve()
    
    # Pattern to exclude temp/preprocessing files
    temp_patterns = ['_pre_', '.verification_']
    
    def is_temp_file(f: Path) -> bool:
        return any(pattern in f.name for pattern in temp_patterns)
    
    if target.is_dir():
        all_files = sorted(target.rglob("*.smt2"))
        files = [f for f in all_files if not is_temp_file(f)]
        if not files:
            sys.exit(f"Error: No .smt2 files found in directory {target}")
        return files
    elif target.is_file():
        if is_temp_file(target):
            sys.exit(f"Error: Input file appears to be a temp file: {target}")
        return [target]
    else:
        sys.exit(f"Error: Input path does not exist: {target}")


def process_single_input(smt_file: Path, solver_name: str, timeout: int, run_dir: Path, print_lock: threading.Lock, detailed: bool = False) -> bool:
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

    # Run the main processing logic (prints are suppressed in this function now)
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
        print() # Blank line between entries
        
    return not success


def main(argv: List[str]) -> int:
    """Main function to run single-solver interpolant verification."""
    parser = argparse.ArgumentParser(
        description="Verify Craig's interpolant for a single SMT solver"
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
        help="Enable detailed results logging to a text file"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs/threads to use (default: 1)"
    )
    
    args = parser.parse_args(argv)
    
    # Check if solver creation works (just to fail early if yaga is missing)
    try:
        # We just check if we can create one, but we'll create fresh ones in threads
        _ = create_solver("yaga")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error creating solvers: {e}")
    
    # Get input files
    input_files = gather_inputs(args.inputs)
    
    # Set up output directory
    if args.output:
        base_output_dir = Path(args.output)
    else:
        base_output_dir = Path.cwd() / "results"
    
    # Create base output directory if it doesn't exist
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run identifier based on timestamp
    run_id = int(time.time())
    run_dir = base_output_dir / f"run_verification_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Verifying solver   : yaga")
    print(f"Input files        : {len(input_files)} files")
    print(f"Processing files in: {args.inputs}")
    print(f"Output directory   : {run_dir}")
    print(f"Run ID             : {run_id}")
    print(f"Timeout per solver : {args.timeout} seconds ({args.timeout/60:.1f} minutes)")
    print(f"Parallel jobs      : {args.jobs}")
    print()
    
    failures = 0
    
    # Lock for synchronizing print output
    print_lock = threading.Lock()

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_input, smt_file, "yaga", args.timeout, run_dir, print_lock, args.detailed): smt_file
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
