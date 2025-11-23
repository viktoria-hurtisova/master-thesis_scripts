import argparse
import csv
import os
import json
import subprocess
import sys
import re
import tempfile
from pathlib import Path

def load_config(path: Path) -> dict:
    """Load and validate the JSON config file."""
    if not path.exists():
        sys.exit(f"Error: Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fp:
            cfg = json.load(fp)
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Failed to parse JSON config – {e}")

    required = {"solver_path"}
    missing = required - cfg.keys()
    if missing:
        sys.exit(f"Error: Missing required config fields: {', '.join(sorted(missing))}")

    return cfg

def count_declare_fun(content):
    """Counts occurrences of 'declare-fun'."""
    return content.count("declare-fun")

def modify_content(content):
    """Replaces (check-sat) with (apply tseitin-cnf)."""
    # Using replace for simplicity.
    # We want to replace the whole command.
    # It might be (check-sat) or (check-sat  ) etc.
    # Regex is safer.
    pattern = r"\(\s*check-sat\s*\)"
    replacement = "(apply tseitin-cnf)"
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        # Let's assume valid SMT2 files with check-sat.
        pass
    
    return new_content

def parse_sexpr(token_iter):
    """Parses a single S-expression from a token iterator."""
    sexpr = []
    try:
        for token in token_iter:
            if token == '(':
                sexpr.append(parse_sexpr(token_iter))
            elif token == ')':
                return sexpr
            else:
                sexpr.append(token)
    except StopIteration:
        pass
    return sexpr

def tokenize(text):
    """Tokenizes S-expressions, handling simple parentheses."""
    # Add spaces around parens and split
    # This is a naive tokenizer that assumes no string literals with parens.
    return text.replace('(', ' ( ').replace(')', ' ) ').split()

def parse_z3_output_clauses(output):
    """Parses Z3 output to count clauses in goals."""
    tokens = tokenize(output)
    if not tokens:
        return 0
    
    iter_tokens = iter(tokens)
    roots = []
    
    # Parse all top-level S-expressions
    try:
        while True:
            t = next(iter_tokens)
            if t == '(':
                roots.append(parse_sexpr(iter_tokens))
            # Ignore atoms at top level (usually solver prints nothing else top-level except errors or parens)
    except StopIteration:
        pass
        
    total_clauses = 0
    
    for root in roots:
        # We look for a list starting with "goals"
        if isinstance(root, list) and len(root) > 0 and root[0] == 'goals':
            # Iterate over 'goal' children
            for goal in root[1:]:
                if isinstance(goal, list) and len(goal) > 0 and goal[0] == 'goal':
                    # Inside a goal
                    # Structure: ('goal', clause1, clause2, ..., key, val, key, val)
                    # We iterate elements skipping the first 'goal'
                    i = 1
                    while i < len(goal):
                        item = goal[i]
                        
                        # Check if it is a keyword
                        if isinstance(item, str) and item.startswith(':'):
                            # It's a keyword like :precision
                            # Skip this and the next item (value)
                            i += 2
                        else:
                            # It's a clause
                            total_clauses += 1
                            i += 1
                            
    return total_clauses

def main():
    parser = argparse.ArgumentParser(description="Process SMT2 files to count vars and clauses via Z3.")
    parser.add_argument("input_path", help="Input directory or single .smt2 file")
    parser.add_argument("--config", default="scripts/configs/z3.json", help="Path to JSON config file")
    parser.add_argument("--output", default="results/analysis_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input {input_path} does not exist.")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config).expanduser().resolve()
    
    print(f"Loading config from: {cfg_path}")
    cfg = load_config(cfg_path)
    
    # Resolve solver path relative to CWD (which matches run_solver_generic logic)
    # If the config says "z3/linux/bin/z3", and we run from root, it finds it.
    # Note: run_solver_generic uses .expanduser().resolve() on the string path
    solver_path = Path(cfg["solver_path"]).expanduser().resolve()
    
    if not solver_path.exists():
        # Try relative to project root if simple resolve failed
        solver_path_rel = project_root / cfg["solver_path"]
        if solver_path_rel.exists():
            solver_path = solver_path_rel
        else:
             # If still not found, warn but proceed (subprocess might fail)
             print(f"Warning: Solver executable not found at {solver_path}")

    print(f"Using Z3 at: {solver_path}")
    
    results = []
    
    files = []
    if input_path.is_dir():
        files = list(input_path.glob("*.smt2"))
        print(f"Found {len(files)} .smt2 files in directory.")
    elif input_path.is_file():
        files = [input_path]
        print(f"Processing single file: {input_path.name}")
    else:
        print(f"Error: {input_path} is neither a file nor a directory.")
        sys.exit(1)
    
    for i, file_path in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {file_path.name}")
        
        try:
            # 1. Read content
            content = file_path.read_text(encoding='utf-8')
            
            # 2. Count declare-fun
            num_vars = count_declare_fun(content)
            
            # 3. Modify content
            new_content = modify_content(content)
            
            # 4. Run Z3
            # Create temp file
            # We put temp file in the same dir or temp dir? 
            # Same dir ensures relative paths in imports (if any) work, but temp dir is cleaner.
            # SMT2 files usually don't import unless using include?
            # I'll use a standard temp file.
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False, encoding='utf-8') as tmp:
                tmp.write(new_content)
                tmp_path = tmp.name
                
            try:
                # Run Z3
                # Capture stdout
                cmd = [str(solver_path), tmp_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Warning: Z3 returned non-zero exit code for {file_path.name}")
                    # print(result.stderr)
                
                # 5. Parse output
                num_clauses = parse_z3_output_clauses(result.stdout)
                
                results.append([file_path.name, num_vars, num_clauses])
                
            finally:
                # Cleanup temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            results.append([file_path.name, "error", "error"])

    # 6. Write CSV
    try:
        output_path = Path(args.output)
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'num_variables', 'num_clauses'])
            writer.writerows(results)
        print(f"Results written to {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    main()

