import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def load_config(path: Path) -> dict:
    """Load and validate the JSON config file."""
    if not path.exists():
        sys.exit(f"Error: Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fp:
            cfg = json.load(fp)
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Failed to parse JSON config – {e}")

    required = {"solver_name", "solver_path"}
    missing = required - cfg.keys()
    if missing:
        sys.exit(f"Error: Missing required config fields: {', '.join(sorted(missing))}")

    return cfg


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


def main() -> None:
    """Generic SMT-LIB solver runner driven by a JSON config file.

    Command-line arguments:
      - config : Path to JSON config file
      - inputs : Path to input files or directory containing .smt2 files

    Required JSON fields:
      - solver_name  : A label written into CSV (e.g. "mathsat", "z3", "yaga")
      - solver_path  : Absolute or relative path to the solver executable

    Optional JSON fields:
      - pass_via_stdin : true/false (default: true for MathSAT, false otherwise)
    """
    parser = argparse.ArgumentParser(description="Run an SMT solver based on JSON config")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("inputs", help="Path to input files or directory containing .smt2 files")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)

    solver_name: str = cfg["solver_name"].lower()
    solver_path = Path(cfg["solver_path"]).expanduser().resolve()
    if not solver_path.exists():
        sys.exit(f"Error: Solver executable not found at {solver_path}")

    input_files = gather_inputs(args.inputs)

    project_root = Path(__file__).resolve().parent.parent

    # Heuristic: MathSAT expects SMT-LIB on stdin when "-input=smt2" flag is used.
    default_stdin = solver_name.startswith("mathsat")
    pass_via_stdin: bool = cfg.get("pass_via_stdin", default_stdin)

    result_csv_path = project_root / f"results_{solver_name}.csv"

    print(f"Using solver       : {solver_path}")
    print(f"CSV results log    : {result_csv_path}\n")

    # Prepare CSV writer.
    write_header = not result_csv_path.exists()
    csv_file = result_csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["solver", "input_file", "time_seconds", "result"])

    # Run the solver over each input file.
    for smt_file in input_files:
        print(f"=== {smt_file.name} ===")
        start = time.perf_counter()

        if pass_via_stdin:
            with smt_file.open("r", encoding="utf-8") as fp:
                result = subprocess.run(
                    [str(solver_path), "-input=smt2"],
                    stdin=fp,
                    capture_output=True,
                    text=True,
                    check=False,
                )
        else:
            # Provide path relative to project root to avoid leading slash on
            # Windows/WSL where absolute POSIX paths confuse z3.exe.
            try:
                rel_input = smt_file.relative_to(project_root)
            except ValueError:
                # Fallback to name only if relativity fails.
                rel_input = smt_file.name

            result = subprocess.run(
                [str(solver_path), str(rel_input)],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

        elapsed = time.perf_counter() - start
        status = "unsat" if "unsat" in result.stdout.lower() else "sat"
        csv_writer.writerow([solver_name, smt_file.name, f"{elapsed:.6f}", status])
        csv_file.flush()

        # Aggregate textual output.

        print(f"Time: {elapsed:.3f} seconds")
        print(result.stdout)
        if result.stderr:
            print("--- stderr ---")
            print(result.stderr)

        print(f"done in {elapsed:.3f}s\n")

    print("\nAll tasks completed.")
    csv_file.close()


if __name__ == "__main__":
    main()
