import subprocess
import sys
from pathlib import Path
import time
import csv


def main() -> None:
    """Run Z3 solver on every .smt2 file in the inputs directory and
    aggregate all outputs into output.txt.
    """

    # Determine project root (two levels up from this script).
    project_root = Path(__file__).resolve().parent.parent

    
    # Path to the Z3 executable (Windows build)
    solver_path = project_root / "z3" / "bin" / "z3.exe"
    inputs_dir = project_root / "inputs/RandomCoupled"
    output_path = project_root / "output.txt"
    csv_path = project_root / "results_z3.csv"
    write_header = not csv_path.exists()

    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["solver", "input_file", "time_seconds", "result"])
    
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
            
            # Start timing
            start_time = time.perf_counter()
            
            # Use a project-relative path so it never starts with a leading slash.
            rel_input_path = input_file.relative_to(project_root)

            # Call Z3 and work in the project root so the relative path resolves.
            result = subprocess.run(
                [str(solver_path), str(rel_input_path)],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - start_time

            # Determine SAT/UNSAT status and log to CSV
            status = "unsat" if "unsat" in result.stdout.lower() else "sat"
            csv_writer.writerow(["z3", input_file.name, f"{elapsed:.6f}", status])
            csv_file.flush()

            # Write a separator and the solver output to the aggregate file.
            outfile.write(f"=== {input_file.name} ===\n")
            outfile.write(f"Time: {elapsed:.3f} seconds\n")
            outfile.write(result.stdout)
            if result.stderr:
                outfile.write("--- stderr ---\n")
                outfile.write(result.stderr)
            outfile.write("\n\n")

            # Print timing info to console as well
            print(f"Finished {input_file.name} in {elapsed:.3f} seconds")

    print("Done. All outputs aggregated to output.txt")
    csv_file.close()


if __name__ == "__main__":
    main()
