from __future__ import annotations
import argparse
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
from typing import Dict, List, Optional, Tuple

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

    def run(self, input_path: str) -> Tuple[str, str, float]:
        #TODO: implement

        # run the preprocessing function
        # run the solver
        # run the postprocessing function
        # delete the temporary file
        
        # return sat/unsat, the interpolant and the time it took to run the solver

        # if the result of the run of the solver is sat, the interpolant is null,the time it took to run the solver and no postprocessing is done
        
        raise NotImplementedError

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

    def run(self, input_path: str) -> Tuple[str, float]:
        #TODO: run the base implementation of the solver

    def _preprocess(self, input_path: str) -> str:
        # TODO: implement
        return input_path

    def _postprocess(self, raw_output: str) -> str:
        # TODO: implement
        return raw_output

class Yaga(InterpolantSolver):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def run(self, input_path: str) -> Tuple[str, float]:
        #TODO: run the base implementation of the solver

    def _preprocess(self, input_path: str) -> str:
        #TODO: implement
        return input_path

    def _postprocess(self, raw_output: str) -> str:
        #TODO: implement

        return raw_output    

# =========================
# Verification
# =========================

def create_verification_input_file(source_path: str, interpolant_A: str, interpolant_B: str) -> str:
    #TODO: implement
    return source_path

def verify_interpolant(file_path: str, interpolant_A: str, interpolant_B: str) -> bool:
    #TODO: implement

    # we are using z3 as the verification solver
    # we need to initialize the z3 solver
    # we need to create the verification file
    return True

# =========================
# Orchestrator
# =========================

def process_file(path: str, solvers: List[InterpolantSolver], outdir: str) -> int:
    #TODO: implement

    # for each solver in the solvers list
    # run the solver.run function
    # verify the interpolant
    # if the interpolant is not valid, return 1
    # if the interpolant is valid, return 0
    
    return 0

def main(argv: List[str]) -> int:
    #TODO: implement

    # on the input we have names of two solvers and the folder with the inputs
    # the names of the solvers must be two of the following: mathsat, yaga, opensmt
    # we need to run the process_file function for each file in the inputs folder

    # for each file in the inputs folder
    # run the process_file function

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
