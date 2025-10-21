from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# notes
#  you can form a new formula by XORing them together (φ₁ ⊕ φ₂) 
# and then use a SAT (Satisfiability) solver to check 
# if this new formula is unsatisfiable. If φ₁ ⊕ φ₂ is unsatisfiable, 
# it means there's no truth assignment that makes the formulas differ, 
# thus they are equivalent

class InterpolantSolver(ABC):
    """
    Base interface for interpolant-producing solvers.
    Implement these THREE methods in concrete subclasses.
    """
    name: str
    solver_path: str
    pass_via_stdin: bool

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.name = config.get('solver_name', 'unknown')
        self.solver_path = config.get('solver_path', '')
        self.pass_via_stdin = config.get('pass_via_stdin', False)

    def run(self, input_path: str, timeout: int = 900) -> Tuple[str, str, float]:
        """
        Run the solver on the input file and return result, interpolant, and time.
        
        Args:
            input_path: Path to the input SMT file
            timeout: Timeout in seconds (default: 900 = 15 minutes)
        
        Returns:
            Tuple[str, str, float]: (sat/unsat result, interpolant or None, execution time)
        """
        
        try:
            # Preprocess the input file for this solver
            processed_path = self._preprocess(input_path)
            
            start_time = time.perf_counter()
            
            # Run the solver
            if self.pass_via_stdin:
                with open(processed_path, 'r', encoding='utf-8') as fp:
                    result = subprocess.run(
                        [self.solver_path, "-input=smt2"],
                        stdin=fp,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=timeout
                    )
            else:
                result = subprocess.run(
                    [self.solver_path, processed_path],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout
                )
            
            elapsed = time.perf_counter() - start_time
            
            # Parse the result
            stdout_lower = result.stdout.lower()
            if "unsat" in stdout_lower:
                sat_result = "unsat"
                # Extract and postprocess the interpolant
                interpolant = self._postprocess(result.stdout)
            elif "sat" in stdout_lower:
                sat_result = "sat"
                interpolant = None
            else:
                sat_result = "unknown"
                interpolant = None

            #print(f"Result: {result}")
            #print(f"Interpolant: {interpolant}")

            # Clean up temporary file if it was created
            if processed_path != input_path:
                try:
                    os.unlink(processed_path)
                except OSError:
                    pass  # Ignore cleanup errors
            
            return sat_result, interpolant, elapsed
            
        except subprocess.TimeoutExpired as e:
            elapsed = time.perf_counter() - start_time
            raise RuntimeError(f"Solver execution timed out after {timeout} seconds: {e}") from e
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            raise RuntimeError(f"Solver execution failed: {e}") from e

    @abstractmethod
    def _preprocess(self, input_path: str) -> str:
        """
        PRIVATE: create and return path to a NEW file adjusted for this solver.
        Must NOT modify the original file.
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, raw_output: str) -> str:
        """
        PRIVATE: transform the solver's raw output (stdout/stderr/files)
        into a valid SMT-LIB Bool term string. No asserts, just the term.
        """
        raise NotImplementedError

class MathSat(InterpolantSolver):

    def __init__(self, config_path: str):
        super().__init__(config_path)

    def _preprocess(self, input_path: str) -> str:
        """
        Preprocess SMT file for MathSAT interpolant generation:
        - Add (set-option :produce-interpolants true) at the start
        - Add (get-interpolant (A)) after (check-sat)
        - Replace :named with :interpolation-group
        """
        # Read the original file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a regular, inspectable file next to the input
        source_path = Path(input_path)
        timestamp = int(time.time())
        output_path = source_path.with_name(f"{source_path.stem}.{self.name}_pre_{timestamp}.smt2")

        with open(output_path, 'w', encoding='utf-8') as f:
            # Split content into lines for processing
            lines = content.split('\n')
            processed_lines = []
            
            # Add the produce-interpolants option at the beginning
            processed_lines.append('(set-option :produce-interpolants true)')
            processed_lines.append('')
            
            # Process each line
            for line in lines:
                # Replace :named with :interpolation-group
                processed_line = line.replace(':named', ':interpolation-group')
                processed_lines.append(processed_line)
                
                # If this line contains (check-sat), add the get-interpolant command after it
                if '(check-sat)' in processed_line:
                    processed_lines.append('(get-interpolant (A))')
            
            # Write all processed lines
            f.write('\n'.join(processed_lines))

        return str(output_path)

    def _postprocess(self, raw_output: str) -> str:
        """
        Postprocess MathSAT output to extract interpolant:
        - Expects exactly two lines: sat/unsat result and interpolant
        - Extract interpolant from the second line
        - Replace (to_real <num>) with just the number within larger expressions
        """
        import re
        
        lines = raw_output.strip().split('\n')
        
        # Filter out empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            return None
        
        # First line should be sat/unsat, second line should be interpolant
        interpolant_line = non_empty_lines[1]
        
        # Replace (to_real <num>) with just the number
        # Pattern to match (to_real <number>) where number can be decimal, scientific notation, or fraction
        pattern = r'\(to_real\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+/\d+)\)'
        interpolant = re.sub(pattern, r'\1', interpolant_line)
        
        return interpolant.strip()

class Yaga(InterpolantSolver):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def _preprocess(self, input_path: str) -> str:
        """
        Preprocess SMT file for Yaga interpolant generation:
        - Copy all lines from the original file
        - Add (get-interpolant A B) after (check-sat)
        """
        # Read the original file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a regular, inspectable file next to the input
        source_path = Path(input_path)
        timestamp = int(time.time())
        output_path = source_path.with_name(f"{source_path.stem}.{self.name}_pre_{timestamp}.smt2")

        with open(output_path, 'w', encoding='utf-8') as f:
            # Split content into lines for processing
            lines = content.split('\n')
            processed_lines = []
            
            # Process each line
            for line in lines:
                processed_lines.append(line)
                
                # If this line contains (check-sat), add the get-interpolant command after it
                if '(check-sat)' in line:
                    processed_lines.append('(get-interpolant A B)')
            
            # Write all processed lines
            f.write('\n'.join(processed_lines))

        return str(output_path)

    def _postprocess(self, raw_output: str) -> str:
        """
        Postprocess Yaga output to extract interpolant:
        - Returns the second line from the input
        """
        lines = raw_output.strip().split('\n')
        
        # Filter out empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            return None
        
        # Return the second line
        return non_empty_lines[1].strip()

class OpenSMT(InterpolantSolver):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def _preprocess(self, input_path: str) -> str:
        """
        Preprocess SMT file for OpenSMT interpolant generation:
        - Add (set-option :produce-interpolants 1) at the start
        - Add (set-option :certify-interpolants 1) at the start
        - Add (get-interpolants A B) after (check-sat) where A and B are the named assertions
        - Ensure file ends with LF
        """
        # Read the original file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a regular, inspectable file next to the input
        source_path = Path(input_path)
        timestamp = int(time.time())
        output_path = source_path.with_name(f"{source_path.stem}.{self.name}_pre_{timestamp}.smt2")

        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            # Split content into lines for processing
            lines = content.split('\n')
            processed_lines = []
            
            # Add the required options at the beginning
            processed_lines.append('(set-option :produce-interpolants 1)')
            processed_lines.append('(set-option :certify-interpolants 1)')
            processed_lines.append('')
            
            # Process each line
            for line in lines:
                processed_lines.append(line)
                
                # If this line contains (check-sat), add the get-interpolants command after it
                if '(check-sat)' in line:
                    processed_lines.append('(get-interpolants A B)')
            
            # Join all lines and ensure it ends with LF
            result_content = '\n'.join(processed_lines)
            if not result_content.endswith('\n'):
                result_content += '\n'
            
            f.write(result_content)

        return str(output_path)

    def _postprocess(self, raw_output: str) -> str:
        """
        Postprocess OpenSMT output to extract interpolant:
        - Expects exactly two lines: sat/unsat result and interpolant
        - Extract interpolant from the second line
        - Remove outermost brackets if they exist
        - Replace num1/num2 format with (/ num1 num2)
        """
        lines = raw_output.strip().split('\n')
        
        # Filter out empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            return None
        
        # First line should be sat/unsat, second line should be interpolant
        interpolant = non_empty_lines[1]
        
        # Remove outermost brackets if they exist
        interpolant = interpolant.strip()
        if interpolant.startswith('(') and interpolant.endswith(')'):
            interpolant = interpolant[1:-1]
        
        # Replace num1/num2 format with (/ num1 num2)
        # Pattern to match numbers in fraction format (e.g., 1/2, 3/4, etc.)
        pattern = r'(\d+)/(\d+)'
        interpolant = re.sub(pattern, r'(/ \1 \2)', interpolant)
        
        return interpolant.strip()

class Z3(InterpolantSolver):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def _preprocess(self, input_path: str) -> str:
        #TODO: implement
        return input_path

    def _postprocess(self, raw_output: str) -> str:
        # TODO: there will be no postprocessing for z3
        return raw_output.strip()   

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
    Create a verification SMT-LIB file that checks Craig's interpolant conditions for a single interpolant I:
      1) ¬A v I is sat (i.e., A ⇒ I)
      2) ¬I v ¬B is sat (i.e., I ⇒ ¬B)
    The function extracts A and B from the original file using :named or :interpolation-group attributes.
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a new file path for the comparison
    source_path_obj = Path(source_path)
    timestamp = int(time.time())
    output_path = source_path_obj.with_name(f"{source_path_obj.stem}.verification_{timestamp}.smt2")

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
        processed_lines.append(line)

    processed_lines.append(f'(assert (or (not {a_formula}) {interpolant}))')
    processed_lines.append(f'(assert (or (not {interpolant}) (not {b_formula})))')
    processed_lines.append('(check-sat)')
    processed_lines.append('(exit)')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    return str(output_path)

def verify_interpolant(file_path: str, interpolant: str, timeout: int = 900) -> bool:
    """
    Verify a single interpolant against Craig's conditions using Z3 by checking:
      (¬A v I) ∧ (¬I v ¬B) is sat (i.e., A ⇒ I ∧ I ⇒ ¬B)

    Returns True if the formula satisfies Craig's conditions.
    """
    try:
        verification_file_path = create_verification_input_file(file_path, interpolant)
        z3_config_path = load_config("z3")
        z3_solver = Z3(z3_config_path)
        
        # Step 3: Run Z3 solver on the verification file
        result, _, _ = z3_solver.run(verification_file_path, timeout)
        
        # Step 4: Process the output
        # If Z3 returns "sat", it means the Craig's conditions are satisfied
        
        is_verified = (result == "sat")
        
        # Clean up the temporary verification file
        try:
            os.unlink(verification_file_path)
        except OSError:
            pass  # Ignore cleanup errors
        
        return is_verified
        
    except Exception as e:
        print(f"ERROR: Exception during interpolant verification: {e}")
        try:
            if 'verification_file_path' in locals():
                os.unlink(verification_file_path)
        except OSError:
            pass
        return False

# =========================
# Orchestrator
# =========================

def process_file(path: str, solver: InterpolantSolver, timeout: int = 900) -> Dict[str, Any]:
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
        print(f"Running {solver.name}...")
        sat_res, interpolant, run_time = solver.run(path, timeout)
        print(f"Result: {solver.name}={sat_res} ({run_time:.3f}s)")

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
            print("Verifying interpolant...")
            is_verified = verify_interpolant(path, interpolant, timeout)
            result['verification_result'] = 'verified' if is_verified else 'not_verified'
            print("Interpolant is verified" if is_verified else "Interpolant is NOT VERIFIED")
            result['detailed_results'] = {
                'file_name': file_path.name,
                'result': sat_res,
                'solver_time': run_time,
                'interpolant': interpolant
            }
            result['success'] = True
            return result

        # Unknown
        result['verification_result'] = f"unknown_result_{sat_res}"
        result['error_message'] = f"Unknown result: {sat_res}"
        return result

    except Exception as e:
        print(f"ERROR: Exception during processing: {e}")
        result['verification_result'] = f"exception_{str(e).replace(',', ';')}"
        result['error_message'] = f"Exception: {e}"
        return result


def write_results(file_result: Dict[str, Any], solver_csv_writer: csv.writer, 
                 verification_csv_writer: csv.writer, detailed_file) -> None:
    """
    Write the results from process_file to CSV files and append to detailed results file.
    
    Args:
        file_result: Dictionary containing all results from process_file
        solver_csv_writer: CSV writer for individual solver runs
        verification_csv_writer: CSV writer for interpolant verifications
        detailed_file: Open file handle for detailed results
    """
    # Write solver runs to CSV
    for run in file_result['solver_runs']:
        solver_csv_writer.writerow([run['solver'], run['input_file'], run['time_seconds'], run['result']])
    
    # Write verification result to CSV
    verification_csv_writer.writerow([
        file_result['file_name'], 
        file_result['solver_name'], 
        file_result['verification_result']
    ])
    
    # Write to detailed results file
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
        detailed_file.write(f"Interpolant: {details['interpolant']}\n")
    
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


defined_solvers = ["mathsat", "yaga", "opensmt"]

def load_config(config_name: str) -> str:
    """Load config file path for a given solver name."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "scripts" / "configs" / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return str(config_path)


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


def create_solver(solver_name: str) -> InterpolantSolver:
    """Factory function to create solver instances."""
    config_path = load_config(solver_name)
    
    if solver_name == "mathsat":
        return MathSat(config_path)
    elif solver_name == "yaga":
        return Yaga(config_path)
    elif solver_name == "opensmt":
        return OpenSMT(config_path)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def main(argv: List[str]) -> int:
    """Main function to run single-solver interpolant verification."""
    parser = argparse.ArgumentParser(
        description="Verify Craig's interpolant for a single SMT solver"
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
        default=900,
        help="Timeout in seconds for each solver execution (default: 900 = 15 minutes)"
    )
    
    args = parser.parse_args(argv)
    
    # Create solver instance
    try:
        solver_inst = create_solver(args.solver)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error creating solvers: {e}")
    
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
    solver_csv_path = output_dir / f"correctness_solver_runs_run_{run_number}.csv"
    verification_csv_path = output_dir / f"correctness_verification_run_{run_number}.csv"
    detailed_results_path = output_dir / f"detailed_results_run_{run_number}.txt"
    
    print(f"Verifying solver   : {args.solver}")
    print(f"Input files        : {len(input_files)} files")
    print(f"Output directory   : {output_dir}")
    print(f"Run number         : {run_number}")
    print(f"Timeout per solver : {args.timeout} seconds ({args.timeout/60:.1f} minutes)")
    print(f"Solver runs CSV    : {solver_csv_path}")
    print(f"Verification CSV   : {verification_csv_path}")
    print(f"Detailed results   : {detailed_results_path}\n")
    
    # Prepare CSV writers and detailed results file
    with solver_csv_path.open("w", newline="", encoding="utf-8") as solver_csv_file, \
         verification_csv_path.open("w", newline="", encoding="utf-8") as verification_csv_file, \
         detailed_results_path.open("w", encoding="utf-8") as detailed_file:
        
        solver_csv_writer = csv.writer(solver_csv_file)
        verification_csv_writer = csv.writer(verification_csv_file)
        
        # Always write headers since we're creating new files
        solver_csv_writer.writerow(["solver", "input_file", "time_seconds", "result"])
        verification_csv_writer.writerow(["file_name", "solver", "verification_result"])
        
        # Write header to detailed results file
        detailed_file.write(f"Detailed Results - Run {run_number}\n")
        detailed_file.write(f"Solver: {args.solver}\n")
        detailed_file.write(f"Generated for {len(input_files)} input files\n")
        detailed_file.write("=" * 50 + "\n\n")
        
        failures = 0
        
        # Process each file
        for smt_file in input_files:
            print(f"=== Processing {smt_file.name} ===")
            
            # Process the file and get results
            file_result = process_file(str(smt_file), solver_inst, args.timeout)
            
            # Write results to CSV files and detailed results file
            write_results(file_result, solver_csv_writer, verification_csv_writer, detailed_file)
            
            # Flush CSV files after each file
            solver_csv_file.flush()
            verification_csv_file.flush()
            
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
