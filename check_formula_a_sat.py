"""
Script to check satisfiability of Formula A in SMT2 files.
It creates a temporary file containing only Formula A (removing Formula B assertions)
and runs the Z3 solver on it.

Behavior:
- If Formula A is UNSAT or solver errors -> Original file is DELETED.
- If Formula A is SAT -> Original file is KEPT.
- If solver times out -> Original file is KEPT, and its path is appended to 'timed_out_files.txt' in the input directory.

Can process a single file or a directory of .smt2 files.
"""

import argparse
import sys
import os
from pathlib import Path
import time
from typing import List

# Add the directory containing this script to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from solvers import Z3, load_config
except ImportError:
    # Fallback if running from root directory
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
    from solvers import Z3, load_config

def create_a_only_file(source_path: str) -> str:
    """
    Reads the source SMT2 file, filters out lines containing ':named B',
    and writes the result to a new file.
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_lines = []
    for line in lines:
        # Remove lines containing :named B as requested
        if ':named B' in line:
            continue
        output_lines.append(line)

    source_path_obj = Path(source_path)
    # Create a new filename indicating it's for checking A
    # Using a unique timestamp to avoid collisions in parallel runs or rapid re-runs
    timestamp = int(time.time() * 1000)
    output_path = source_path_obj.with_name(f"{source_path_obj.stem}.check_a_{timestamp}.smt2")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    return str(output_path)

def check_a_sat(file_path: str, timeout_log: str, timeout: int = 900) -> str:
    """
    Runs Z3 on the file containing only Formula A.
    Returns status: 'deleted', 'kept', 'timeout', or 'error'.
    """
    temp_file = None
    try:
        # Create the file with only Formula A
        temp_file = create_a_only_file(file_path)
        # print(f"Created temporary file: {temp_file}")

        # Load Z3 configuration
        config_path = load_config("z3")
        solver = Z3(config_path)
        
        print(f"Checking Formula A in {Path(file_path).name} (Timeout: {timeout}s)...")
        
        # Run the solver
        # run() returns: (sat_result, interpolant, elapsed, stdout, stderr)
        sat_result, _, run_time, stdout, stderr = solver.run(temp_file, timeout)
        
        print(f"Result: {sat_result.upper()} (Time: {run_time:.3f}s)")
        
        # Debug output if needed, keeping it minimal for batch processing
        if stderr:
             print(f"Solver stderr: {stderr.strip()}")

        if sat_result.lower() == 'unsat':
            print(f"-> Formula A is UNSAT. Deleting original file.")
            try:
                os.remove(file_path)
                print(f"Successfully deleted {file_path}")
                return "deleted"
            except OSError as e:
                print(f"Error deleting original file: {e}")
                return "error"
        elif stderr:
            print(f"-> Solver finished with error. Deleting original file.")
            try:
                os.remove(file_path)
                return "deleted"
            except OSError as e:
                print(f"Error deleting original file: {e}")
                return "error"
        else:
            print(f"-> Formula A is {sat_result.upper()}. Keeping file.")
            return "kept"

    except Exception as e:
        if "timed out" in str(e).lower():
            print(f"-> Solver timed out. Keeping file and logging to {timeout_log}")
            try:
                # Ensure directory exists (should exist as it's input dir or parent)
                log_path = Path(timeout_log)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(timeout_log, "a") as f:
                    f.write(f"{file_path}\n")
                return "timeout"
            except Exception as io_err:
                print(f"Error writing to timeout log: {io_err}")
                return "error"

        print(f"Error processing {file_path}: {e}")
        print(f"-> Deleting file due to exception.")
        try:
            os.remove(file_path)
            return "deleted"
        except OSError as del_err:
            print(f"Error deleting original file: {del_err}")
            return "error"
    finally:
        # Cleanup the temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                # print(f"Removed temporary file: {temp_file}")
            except OSError as e:
                print(f"Error removing temporary file: {e}")

def gather_inputs(input_path: str) -> List[Path]:
    """Return a list of *.smt2 files from file or directory."""
    target = Path(input_path).expanduser().resolve()
    if target.is_dir():
        files = sorted(target.glob("*.smt2"))
        return files
    elif target.is_file():
        return [target]
    else:
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Check satisfiability of Formula A (removing :named B assertions) using Z3. Deletes file if A is UNSAT."
    )
    parser.add_argument("input_path", help="Path to the input SMT2 file or directory")
    parser.add_argument(
        "-t", "--timeout", 
        type=int, 
        default=900, 
        help="Timeout in seconds (default: 900)"
    )
    
    args = parser.parse_args()
    
    files = gather_inputs(args.input_path)
    
    if not files:
        print(f"Error: No input files found at '{args.input_path}'")
        sys.exit(1)

    # Determine timeout log file path
    input_path_obj = Path(args.input_path).expanduser().resolve()
    if input_path_obj.is_dir():
        timeout_log = input_path_obj / "_timed_out_files.txt"
    else:
        timeout_log = input_path_obj.parent / "_timed_out_files.txt"

    stats = {
        "total": len(files),
        "deleted": 0,
        "kept": 0,
        "error": 0,
        "timeout": 0
    }
    
    print(f"Found {len(files)} files to process.")
    print(f"Timeout log will be written to: {timeout_log}")
    print("=" * 60)
    
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {f.name}")
        status = check_a_sat(str(f), str(timeout_log), args.timeout)
        stats[status] += 1
        print("-" * 60)

    print("\n" + "="*30)
    print("STATISTICS")
    print("="*30)
    print(f"Total files processed: {stats['total']}")
    print(f"Files deleted (UNSAT/Error): {stats['deleted']}")
    print(f"Files kept (SAT/Other):      {stats['kept']}")
    print(f"Files timed out:             {stats['timeout']}")
    print(f"Errors encountered:          {stats['error']}")
    print("="*30)

if __name__ == "__main__":
    main()
