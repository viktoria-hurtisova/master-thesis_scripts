import subprocess
import sys
from pathlib import Path
import time
import csv


def main() -> None:
    """Run MathSAT solver on every .smt2 file in the inputs directory and
    aggregate all outputs into output.txt.
    """

    # Determine project root (two levels up from this script).
    project_root = Path(__file__).resolve().parent.parent

    # ---------------------------------------------------------------------
    # Paths – resolve Yaga solver binary and input/output locations
    # ---------------------------------------------------------------------

    inputs_dir = project_root / "inputs/RandomCoupled"
    output_path = project_root / "output.txt"
    csv_path = project_root / "results_yaga.csv"
    write_header = not csv_path.exists()

    # Open CSV in append mode
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["solver", "input_file", "time_seconds", "result"])

    # Detect platform-specific executable name
    exec_name = "smt.exe" if sys.platform.startswith("win") else "smt"

    # Prefer release build, fall back to debug build if necessary
    release_solver = project_root / "yaga" / "cmake-build-release-wsl" / exec_name
    debug_solver = project_root / "yaga" / "cmake-build-debug-wsl" / exec_name

    solver_path = release_solver if release_solver.exists() else debug_solver

    # Basic validation.
    if not solver_path.exists():
        sys.stderr.write(f"Error: Solver not found at {solver_path}\n")
        sys.exit(1)
    if not inputs_dir.exists() or not any(inputs_dir.glob("*.smt2")):
        sys.stderr.write(f"Error: No .smt2 input files found in {inputs_dir}\n")
        sys.exit(1)

    print(f"Using solver: {solver_path}")
    print(f"Reading inputs from: {inputs_dir}")
    print(f"Aggregating outputs into: {output_path}\n")

    with output_path.open("w", encoding="utf-8") as outfile:
        for input_file in sorted(inputs_dir.glob("*.smt2")):
            print(f"Processing {input_file.name} ...")

            start_time = time.perf_counter()

            # Invoke Yaga with the input file *as an argument* (Yaga reads the
            # file directly – no need to feed it via stdin).
            result = subprocess.run(
                [str(solver_path), str(input_file)],
                capture_output=True,
                text=True,
                check=False,
            )

            elapsed = time.perf_counter() - start_time

            # Determine SAT/UNSAT status from solver output
            status = "unsat" if "unsat" in result.stdout.lower() else "sat"

            # Log to CSV
            csv_writer.writerow(["yaga", input_file.name, f"{elapsed:.6f}", status])
            csv_file.flush()

            # Write a separator and the solver output to the aggregate file.
            outfile.write(f"=== {input_file.name} ===\n")
            outfile.write(f"Time: {elapsed:.3f} seconds\n")
            outfile.write(result.stdout)
            if result.stderr:
                outfile.write("--- stderr ---\n")
                outfile.write(result.stderr)
            outfile.write("\n\n")

            print(f"Finished {input_file.name} in {elapsed:.3f} seconds")

    print("Done. All outputs aggregated to output.txt")
    csv_file.close()


if __name__ == "__main__":
    main()
