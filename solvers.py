from __future__ import annotations
import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

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

    def run(self, input_path: str, timeout: int = 900) -> Tuple[str, str, float, str, str]:
        """
        Run the solver on the input file and return result, interpolant, time, stdout, and stderr.
        
        Args:
            input_path: Path to the input SMT file
            timeout: Timeout in seconds (default: 900 = 15 minutes)
        
        Returns:
            Tuple[str, str, float, str, str]: (sat/unsat result, interpolant or None, execution time, stdout, stderr)
        """
        
        processed_path = input_path  # Initialize to avoid undefined variable in finally
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
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            # print(f"Result: {result}") # Optional: Keep logging if needed, or remove for cleaner library code
            
            # Check for errors in stderr
            if result.stderr and result.stderr.strip():
                error_msg = f"{self.name} produced error output: {result.stderr}"
                # We might want to log this instead of printing in a library
                # print(error_msg) 
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

            return sat_result, interpolant, elapsed, stdout, stderr
            
        except subprocess.TimeoutExpired as e:
            elapsed = time.perf_counter() - start_time
            raise RuntimeError(f"Solver execution timed out after {timeout} seconds: {e}") from e
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            # Return empty strings for stdout/stderr on exception
            return "unknown", None, elapsed, "", str(e)
        finally:
            # Clean up temporary file if it was created (happens regardless of success/failure)
            if processed_path != input_path:
                try:
                    os.unlink(processed_path)
                except OSError:
                    pass  # Ignore cleanup errors

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
        
        # Replace (to_real <something>) with just the something
        pattern = r'\(to_real\s+((?:\(-\s*\d+\)|\d+))\)'
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
        #TODO: there will be no preprocessing for z3
        return input_path

    def _postprocess(self, raw_output: str) -> str:
        # TODO: there will be no postprocessing for z3
        return raw_output.strip()   

defined_solvers = ["mathsat", "yaga", "opensmt"]

def load_config(config_name: str) -> str:
    """Load config file path for a given solver name."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "scripts" / "configs" / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return str(config_path)

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

