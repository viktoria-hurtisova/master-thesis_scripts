# Scripts

- `bipartition_smt.py` -> takes an `.smt2` file and divide the input formula into two formulas. One formula has attribute `:named` set to **A** and the second to **B**. For every <file>.smt2 the script creates <file>.bi.smt2. It takes two arguments: the source and destination folders.
- `cleanup_smt.py` -> A utility script to clean up SMT-LIB files in a directory by keeping only the bipartitioned (.bi.smt2) files and removing all other .smt2 files.
- 'copy_unsat_files`-> Script to copy files that contain the line '(set-info :status unsat)' from source to destination folder.
- `correctness_script.py` ->  run two solvers on the same input files and verify that the resulting interpolants are equal.
- `run_solver_*.py` -> run the solver on the input folder
