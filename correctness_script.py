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
            
            # Check for errors in stderr
            if result.stderr and result.stderr.strip():
                error_msg = f"{self.name} produced error output: {result.stderr}"
                # Clean up temporary file if it was created
                if processed_path != input_path:
                    try:
                        os.unlink(processed_path)
                    except OSError:
                        pass
                raise RuntimeError(error_msg)
            
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

def create_comparison_input_file(source_path: str, interpolant_A: str, interpolant_B: str) -> str:
    """
    Create a comparison input file that replaces all assert statements with a single XOR assertion.
    
    Args:
        source_path: Path to the original SMT file
        interpolant_A: First interpolant to compare
        interpolant_B: Second interpolant to compare
        
    Returns:
        Path to the new comparison file
    """
    # Read the original file
    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a new file path for the comparison
    source_path_obj = Path(source_path)
    timestamp = int(time.time())
    output_path = source_path_obj.with_name(f"{source_path_obj.stem}.comparison_{timestamp}.smt2")
    
    # Split content into lines for processing
    lines = content.split('\n')
    processed_lines = []
    
    # Process each line
    for line in lines:
        # Check if line starts with (assert
        if line.strip().startswith('(assert'):
            # Skip this line - we'll add our own assert at the end
            continue
        # Skip lines containing (set-info :status
        elif '(set-info :status' in line:
            # Skip this line - don't copy it to the file
            continue
        else:
            if '(check-sat)' in line:
                processed_lines.append(f"(assert (and (or {interpolant_A} (not {interpolant_B})) (or (not {interpolant_A}) {interpolant_B})))")
            # Keep all other lines as they are
            processed_lines.append(line)
    
  
    # Write the processed content to the new file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    return str(output_path)

def compare_interpolants(file_path: str, interpolant_A: str, interpolant_B: str, timeout: int = 900) -> bool:
    """
    Compare two interpolants for equivalence using Z3 solver.
    
    Args:
        file_path: Path to the original SMT file
        interpolant_A: First interpolant to compare
        interpolant_B: Second interpolant to compare
        
    Returns:
        Tuple[bool, str]: (True if interpolants are equivalent, Z3 output for debugging)
    """
    try:
        # Step 1: Create the comparison input file with XOR assertion
        comparison_file_path = create_comparison_input_file(file_path, interpolant_A, interpolant_B)
        
        # Step 2: Initialize Z3 solver
        z3_config_path = load_config("z3")
        z3_solver = Z3(z3_config_path)
        
        # Step 3: Run Z3 solver on the comparison file
        result, _, _ = z3_solver.run(comparison_file_path, timeout)
        
        # Step 4: Process the output
        # If Z3 returns "unsat", it means the XOR of the two interpolants is unsatisfiable,
        # which means the interpolants are equivalent

        are_equivalent = (result != "unsat")
        
        # Clean up the temporary comparison file
        try:
            os.unlink(comparison_file_path)
        except OSError:
            pass  # Ignore cleanup errors
        
        return are_equivalent
        
    except Exception as e:
        error_msg = f"Exception during interpolant comparison: {e}"
        # Clean up comparison file if it exists
        try:
            if 'comparison_file_path' in locals():
                os.unlink(comparison_file_path)
        except OSError:
            pass
        raise RuntimeError(error_msg)

# =========================
# Orchestrator
# =========================

def process_file(path: str, solvers: List[InterpolantSolver], timeout: int = 900) -> Dict[str, Any]:
    """
    Process a single SMT file with multiple solvers and return results.
    
    Returns:
        Dict containing all the results and metadata for logging
    """
    file_path = Path(path)
    
    if len(solvers) != 2:
        raise ValueError("Exactly two solvers are required for comparison")
    
    solver_a, solver_b = solvers
    
    result = {
        'file_name': file_path.name,
        'solver_a_name': solver_a.name,
        'solver_b_name': solver_b.name,
        'success': False,
        'error_message': None,
        'solver_runs': [],  # List of individual solver run results
        'comparison_result': None  # Result of interpolant comparison
    }
    
    try:
        # Run both solvers on the input file
        print(f"Running {solver_a.name}...")
        result_a, interpolant_a, time_a = solver_a.run(path, timeout)
        
        print(f"Running {solver_b.name}...")
        result_b, interpolant_b, time_b = solver_b.run(path, timeout)
        
        print(f"Results: {solver_a.name}={result_a} ({time_a:.3f}s), {solver_b.name}={result_b} ({time_b:.3f}s)")
        
        # Store individual solver run results
        result['solver_runs'] = [
            {'solver': solver_a.name, 'input_file': file_path.name, 'time_seconds': f"{time_a:.6f}", 'result': result_a},
            {'solver': solver_b.name, 'input_file': file_path.name, 'time_seconds': f"{time_b:.6f}", 'result': result_b}
        ]
        
        # Both solvers must agree on satisfiability
        if result_a != result_b:
            print(f"ERROR: Solvers disagree on satisfiability - {solver_a.name}: {result_a}, {solver_b.name}: {result_b}")
            result['comparison_result'] = "disagreement_on_satisfiability"
            result['error_message'] = f"Solvers disagree: {solver_a.name}={result_a}, {solver_b.name}={result_b}"
            return result
        
        # If the formula is SAT, no interpolant should be produced
        if result_a == "sat":
            if interpolant_a is not None or interpolant_b is not None:
                print("ERROR: Interpolants produced for SAT formula")
                result['comparison_result'] = "error_interpolant_for_sat"
                result['error_message'] = "Interpolants produced for SAT formula"
                return result
            print("Both solvers correctly determined SAT (no interpolant needed)")
            result['comparison_result'] = "both_sat_no_interpolant"
            
            # Store detailed results for SAT case
            result['detailed_results'] = {
                'file_name': file_path.name,
                'result': result_a,
                'solver_a_time': time_a,
                'solver_b_time': time_b,
                'interpolants_equal': None,  # No interpolants for SAT
                'solver_a_interpolant': None,
                'solver_b_interpolant': None
            }
            
            result['success'] = True
            return result
        
        # If UNSAT, both should produce interpolants
        if result_a == "unsat":
            if interpolant_a is None:
                print(f"ERROR: {solver_a.name} did not produce an interpolant for UNSAT formula")
                result['comparison_result'] = f"no_interpolant_from_{solver_a.name}"
                result['error_message'] = f"{solver_a.name} did not produce an interpolant"
                return result
            if interpolant_b is None:
                print(f"ERROR: {solver_b.name} did not produce an interpolant for UNSAT formula")
                result['comparison_result'] = f"no_interpolant_from_{solver_b.name}"
                result['error_message'] = f"{solver_b.name} did not produce an interpolant"
                return result
            
            # Compare the interpolants
            print("Comparing interpolants...")
            
            are_equal = compare_interpolants(path, interpolant_a, interpolant_b, timeout)
            result['comparison_result'] = str(are_equal).lower()
            
            if are_equal:
                print("Interpolants are equal")
            else:
                print("Interpolants are different")
            
            # Store detailed results for file output
            result['detailed_results'] = {
                'file_name': file_path.name,
                'result': result_a,
                'solver_a_time': time_a,
                'solver_b_time': time_b,
                'interpolants_equal': are_equal,
                'solver_a_interpolant': interpolant_a,
                'solver_b_interpolant': interpolant_b
            }
            
            result['success'] = True
            return result
        
        # Unknown result
        print(f"ERROR: Unknown result: {result_a}")
        result['comparison_result'] = f"unknown_result_{result_a}"
        result['error_message'] = f"Unknown result: {result_a}"
        return result
        
    except Exception as e:
        result['comparison_result'] = f"exception_{str(e).replace(',', ';')}"
        result['error_message'] = f"Exception during processing: {e}"
        return result


def write_results(file_result: Dict[str, Any], solver_csv_writer: csv.writer, 
                 comparison_csv_writer: csv.writer, detailed_file) -> None:
    """
    Write the results from process_file to CSV files and append to detailed results file.
    
    Args:
        file_result: Dictionary containing all results from process_file
        solver_csv_writer: CSV writer for individual solver runs
        comparison_csv_writer: CSV writer for interpolant comparisons
        detailed_file: Open file handle for detailed results
    """
    # Write solver runs to CSV
    for run in file_result['solver_runs']:
        solver_csv_writer.writerow([run['solver'], run['input_file'], run['time_seconds'], run['result']])
    
    # Write comparison result to CSV
    comparison_csv_writer.writerow([
        file_result['file_name'], 
        file_result['solver_a_name'], 
        file_result['solver_b_name'], 
        file_result['comparison_result']
    ])
    
    # Write to detailed results file
    detailed_file.write(f"=== {file_result['file_name']} ===\n")
    detailed_file.write(f"Solvers: {file_result['solver_a_name']} vs {file_result['solver_b_name']}\n")
    detailed_file.write(f"Success: {file_result['success']}\n")
    detailed_file.write(f"Comparison Result: {file_result['comparison_result']}\n")
    
    if file_result['error_message']:
        detailed_file.write(f"Error: {file_result['error_message']}\n")
    
    # Write detailed results if available (for both SAT and UNSAT cases)
    if 'detailed_results' in file_result:
        details = file_result['detailed_results']
        detailed_file.write(f"SAT Result: {details['result']}\n")
        detailed_file.write(f"{file_result['solver_a_name']} time: {details['solver_a_time']:.6f}s\n")
        detailed_file.write(f"{file_result['solver_b_name']} time: {details['solver_b_time']:.6f}s\n")
        
        # Handle interpolant information (may be None for SAT cases)
        if details['interpolants_equal'] is not None:
            detailed_file.write(f"Interpolants equal: {details['interpolants_equal']}\n")
            detailed_file.write(f"{file_result['solver_a_name']} interpolant: {details['solver_a_interpolant']}\n")
            detailed_file.write(f"{file_result['solver_b_name']} interpolant: {details['solver_b_interpolant']}\n")
        else:
            detailed_file.write("Interpolants: N/A (SAT case)\n")
    
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
    """Main function to run correctness verification between two solvers."""
    parser = argparse.ArgumentParser(
        description="Verify interpolant correctness between two SMT solvers"
    )
    parser.add_argument(
        "solver1", 
        choices=defined_solvers,
        help="First solver name (mathsat, yaga, opensmt)"
    )
    parser.add_argument(
        "solver2", 
        choices=defined_solvers,
        help="Second solver name (mathsat, yaga, opensmt)"
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
    
    if args.solver1 == args.solver2:
        sys.exit("Error: Cannot compare a solver with itself")
    
    # Create solver instances
    try:
        solver1 = create_solver(args.solver1)
        solver2 = create_solver(args.solver2)
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
    comparison_csv_path = output_dir / f"correctness_interpolant_comparison_run_{run_number}.csv"
    detailed_results_path = output_dir / f"detailed_results_run_{run_number}.txt"
    
    print(f"Comparing solvers  : {args.solver1} vs {args.solver2}")
    print(f"Input files        : {len(input_files)} files")
    print(f"Output directory   : {output_dir}")
    print(f"Run number         : {run_number}")
    print(f"Timeout per solver : {args.timeout} seconds ({args.timeout/60:.1f} minutes)")
    print(f"Solver runs CSV    : {solver_csv_path}")
    print(f"Comparison CSV     : {comparison_csv_path}")
    print(f"Detailed results   : {detailed_results_path}\n")
    
    # Prepare CSV writers and detailed results file
    with solver_csv_path.open("w", newline="", encoding="utf-8") as solver_csv_file, \
         comparison_csv_path.open("w", newline="", encoding="utf-8") as comparison_csv_file, \
         detailed_results_path.open("w", encoding="utf-8") as detailed_file:
        
        solver_csv_writer = csv.writer(solver_csv_file)
        comparison_csv_writer = csv.writer(comparison_csv_file)
        
        # Always write headers since we're creating new files
        solver_csv_writer.writerow(["solver", "input_file", "time_seconds", "result"])
        comparison_csv_writer.writerow(["file_name", "solver1", "solver2", "interpolants_equal"])
        
        # Write header to detailed results file
        detailed_file.write(f"Detailed Results - Run {run_number}\n")
        detailed_file.write(f"Solvers: {args.solver1} vs {args.solver2}\n")
        detailed_file.write(f"Generated for {len(input_files)} input files\n")
        detailed_file.write("=" * 50 + "\n\n")
        
        failures = 0
        
        # Process each file
        for smt_file in input_files:
            print(f"=== Processing {smt_file.name} ===")
            
            # Process the file and get results
            file_result = process_file(str(smt_file), [solver1, solver2], args.timeout)
            
            # Write results to CSV files and detailed results file
            write_results(file_result, solver_csv_writer, comparison_csv_writer, detailed_file)
            
            # Flush CSV files after each file
            solver_csv_file.flush()
            comparison_csv_file.flush()
            
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
