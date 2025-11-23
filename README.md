# Scripts

- `bipartition_smt.py` -> takes an `.smt2` file and divide the input formula into two formulas. One formula has attribute `:named` set to **A** and the second to **B**. For every `<file>`.smt2 the script creates `<file>`.bi.smt2. It takes two arguments: the source and destination folders.
- `cleanup_smt.py` -> A utility script to clean up SMT-LIB files in a directory by keeping only the bipartitioned (.bi.smt2) files and removing all other .smt2 files.
- `copy_unsat_files.py` -> Script to copy files that contain the line `(set-info :status unsat)` from source to destination folder.
- `run_interpolation.py` -> Runs a selected solver's interpolation on selected files.
- `run_solver_generic.py` -> Generic SMT-LIB solver runner driven by a JSON config file. It executes a solver on a set of input files and produces a CSV of results.
- `analyze_smt2_formula.py` -> Processes SMT2 files to count variables and clauses. It uses Z3 to convert the formula to CNF (via `tseitin-cnf`) and counts the resulting goals.
- `yaga_interpolant_verification.py` -> Verifies interpolants produced by a solver (specifically targeted at Yaga's output) using Z3. It checks if the interpolant satisfies Craig's interpolation conditions.
