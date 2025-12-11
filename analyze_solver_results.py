"""
Script to analyze solver benchmark results and create comparison scatter plots.

Reads CSV files with structure:
    solver, input_file, time_seconds, result, interpolant_produced, error

Uses a theory mapping file with structure:
    theory, mode, filename

Creates scatter plots per theory:
    - QF_LRA: yaga vs opensmt, yaga vs mathsat
    - QF_UFLRA: yaga vs mathsat
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
from pathlib import Path


def get_time_buckets(timeout):
    """Generate time bucket boundaries based on timeout value."""
    return [0.01, 0.1, 1, 10, timeout]


def get_bucket_labels(timeout):
    """Generate bucket labels based on timeout value."""
    return [
        r'$\langle 0.01, 0.1)$',
        r'$\langle 0.1, 1)$',
        r'$\langle 1, 10)$',
        f'$\\langle 10, {timeout})$',
        'timeout'
    ]


def get_bucket_labels_simple(timeout):
    """Generate simple bucket labels for table display."""
    timeout_str = str(int(timeout)) if timeout == int(timeout) else str(timeout)
    return [
        r'$\langle$0.01, 0.1)',
        r'$\langle$0.1, 1)',
        r'$\langle$1, 10)',
        f'$\\langle$10, {timeout_str})',
        'timeout'
    ]


def load_theory_mapping(theory_file):
    """
    Load the theory mapping CSV file.
    
    Args:
        theory_file: Path to CSV file with columns: theory, mode, filename
        
    Returns:
        DataFrame with theory and filename columns
    """
    df = pd.read_csv(theory_file)
    df.columns = df.columns.str.strip()
    df['filename'] = df['filename'].str.strip()
    df['theory'] = df['theory'].str.strip()
    
    print(f"Loaded theory mapping: {len(df)} files")
    print(f"Theories found: {df['theory'].unique()}")
    
    return df[['theory', 'filename']]


def load_and_process_data(csv_directory, theory_df):
    """
    Load CSV files from a directory and average the time_seconds for each solver/input_file combination.
    
    Args:
        csv_directory: Path to directory containing CSV files
        theory_df: DataFrame with theory mapping
        
    Returns:
        DataFrame with averaged results including theory column
    """
    # Find all CSV files in the directory
    csv_dir = Path(csv_directory)
    csv_files = list(csv_dir.glob('*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_directory}")
    
    print(f"Found {len(csv_files)} CSV files in {csv_directory}")
    
    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            dfs.append(df)
            print(f"Loaded: {csv_file} ({len(df)} rows)")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        raise ValueError("No CSV files could be loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")
    
    # Strip whitespace from string columns
    if 'solver' in combined_df.columns:
        combined_df['solver'] = combined_df['solver'].str.strip().str.lower()
    if 'input_file' in combined_df.columns:
        combined_df['input_file'] = combined_df['input_file'].str.strip()
    if 'result' in combined_df.columns:
        combined_df['result'] = combined_df['result'].str.strip().str.lower()
    
    # Count errors per solver BEFORE filtering (excluding timeout)
    error_mask = combined_df['result'].isin(['stdout_error', 'unknown'])
    error_counts = combined_df[error_mask].groupby('solver').size().to_dict()
    
    # Filter out rows with stdout_error or unknown results
    initial_count = len(combined_df)
    combined_df = combined_df[~combined_df['result'].isin(['stdout_error', 'unknown'])]
    filtered_count = initial_count - len(combined_df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with stdout_error or unknown results")
    
    # Group by solver and input_file, calculate mean of time_seconds
    averaged_df = combined_df.groupby(['solver', 'input_file']).agg({
        'time_seconds': 'mean',
        'result': 'first',  # Take the first result (assuming consistent results)
        'interpolant_produced': 'first',
        'error': 'first'
    }).reset_index()
    
    # Join with theory mapping
    averaged_df = averaged_df.merge(
        theory_df, 
        left_on='input_file', 
        right_on='filename', 
        how='left'
    )
    
    # Check for files without theory mapping
    unmapped = averaged_df['theory'].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} entries have no theory mapping")
    
    print(f"Unique solver/file combinations after averaging: {len(averaged_df)}")
    print(f"Solvers found: {averaged_df['solver'].unique()}")
    
    return averaged_df, error_counts


def get_time_bucket(time_val, timeout):
    """Get the bucket index for a time value."""
    time_buckets = get_time_buckets(timeout)
    if time_val >= timeout:
        return len(time_buckets) - 1  # timeout bucket (last bucket)
    for i, threshold in enumerate(time_buckets[1:], 1):
        if time_val < threshold:
            return i - 1
    return len(time_buckets) - 2


def create_comparison_table(x_times, y_times, solver_x, solver_y, timeout):
    """
    Create a comparison table showing distribution of results across time buckets.
    
    Returns a 2D array with counts.
    """
    time_buckets = get_time_buckets(timeout)
    n_buckets = len(time_buckets)
    table = np.zeros((n_buckets, n_buckets), dtype=int)
    
    for x_t, y_t in zip(x_times, y_times):
        x_bucket = get_time_bucket(x_t, timeout)
        y_bucket = get_time_bucket(y_t, timeout)
        table[y_bucket, x_bucket] += 1
    
    return table


def generate_latex_table(table, solver_x, solver_y, timeout):
    """Generate LaTeX code for the comparison table.
    
    Note: Requires \\usepackage{diagbox} in LaTeX preamble.
    """
    time_buckets = get_time_buckets(timeout)
    n_buckets = len(time_buckets)
    bucket_labels = get_bucket_labels_simple(timeout)
    
    latex = []
    latex.append(r'\begin{tabular}{||c||' + 'c' * n_buckets + r'||}')
    latex.append(r'\hline')
    
    # Header row with diagbox and column labels
    header = r'\diagbox{' + solver_y.capitalize() + r'[s]}{' + solver_x.capitalize() + r'[s]} & ' + ' & '.join(bucket_labels) + r' \\ \hline \hline'
    latex.append(header)
    
    # Data rows with row labels (smallest bucket at top, timeout at bottom)
    for i in range(n_buckets):
        row_label = bucket_labels[i]
        row_data = ' & '.join(str(table[i, j]) for j in range(n_buckets))
        latex.append(f'{row_label} & {row_data} ' + r'\\ \hline')
    
    latex.append(r'\end{tabular}')
    
    return '\n'.join(latex)


def create_scatter_plot(df, solver_x, solver_y, output_path, title_suffix="", timeout=300):
    """
    Create a scatter plot comparing two solvers with log scale and different markers.
    
    Args:
        df: DataFrame with averaged results
        solver_x: Name of solver for x-axis
        solver_y: Name of solver for y-axis
        output_path: Path to save the plot
        title_suffix: Optional suffix for the title (e.g., theory name)
        timeout: Timeout value in seconds
        
    Returns:
        LaTeX table string or None if no data
    """
    # Get data for each solver
    solver_x_data = df[df['solver'] == solver_x.lower()].set_index('input_file')
    solver_y_data = df[df['solver'] == solver_y.lower()].set_index('input_file')
    
    # Find common input files
    common_files = solver_x_data.index.intersection(solver_y_data.index)
    
    if len(common_files) == 0:
        print(f"Warning: No common files found between {solver_x} and {solver_y}")
        return None
    
    print(f"\nCreating scatter plot: {solver_x} vs {solver_y} {title_suffix}")
    print(f"Common files: {len(common_files)}")
    
    # Get time and result values for common files
    x_times = solver_x_data.loc[common_files, 'time_seconds'].values
    y_times = solver_y_data.loc[common_files, 'time_seconds'].values
    x_results = solver_x_data.loc[common_files, 'result'].values
    y_results = solver_y_data.loc[common_files, 'result'].values
    
    # Determine result category for each point
    # Use the result from solver_x (or combine both)
    categories = []
    for x_res, y_res, x_t, y_t in zip(x_results, y_results, x_times, y_times):
        if x_t >= timeout or y_t >= timeout:
            categories.append('timeout')
        elif x_res == 'sat' or y_res == 'sat':
            categories.append('sat')
        elif x_res == 'unsat' or y_res == 'unsat':
            categories.append('unsat')
        else:
            categories.append('unknown')
    
    categories = np.array(categories)
    
    # Create the plot with clean style
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Define markers and colors for each category
    # Using filled markers with transparency for better visibility
    marker_styles = {
        'sat': {'marker': 's', 'color': '#2ecc71', 'alpha': 0.7, 'label': 'SAT', 'size': 40},
        'unsat': {'marker': 'o', 'color': '#1a5276', 'alpha': 0.6, 'label': 'UNSAT', 'size': 40},
        'timeout': {'marker': '^', 'color': '#e74c3c', 'alpha': 0.8, 'label': 'timeout', 'size': 50},
        'unknown': {'marker': '*', 'color': '#e67e22', 'alpha': 0.9, 'label': 'error', 'size': 120}
    }
    
    # Plot each category separately
    for cat, style in marker_styles.items():
        mask = categories == cat
        if np.sum(mask) > 0:
            ax.scatter(x_times[mask], y_times[mask], 
                      marker=style['marker'], 
                      c=style['color'],
                      alpha=style['alpha'],
                      s=style['size'], label=style['label'], edgecolors='white', linewidths=0.5)
    
    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Format axis ticks to show decimal values instead of exponents
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    
    # Determine axis limits
    min_val = 0.01
    max_val = max(max(x_times), max(y_times), timeout) * 1.5
    
    # Add diagonal line (y = x)
    ax.plot([min_val, max_val], [min_val, max_val], color='#bdc3c7', linewidth=1.2, linestyle='--', label='y = x')
    
    # Set axis limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4, color='gray')
    
    # Labels and title
    ax.set_xlabel(f'{solver_x} running time [s]', fontsize=12)
    ax.set_ylabel(f'{solver_y} running time [s]', fontsize=12)
    
    # Add legend
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray', fancybox=True, fontsize=10)
    
    # Make it square
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {output_path}")
    
    # Print some statistics
    faster_x = np.sum(x_times < y_times)
    faster_y = np.sum(y_times < x_times)
    equal = np.sum(x_times == y_times)
    
    print(f"  {solver_x} faster: {faster_x} ({100*faster_x/len(common_files):.1f}%)")
    print(f"  {solver_y} faster: {faster_y} ({100*faster_y/len(common_files):.1f}%)")
    print(f"  Equal: {equal}")
    
    # Generate comparison table
    table = create_comparison_table(x_times, y_times, solver_x, solver_y, timeout)
    latex_table = generate_latex_table(table, solver_x, solver_y, timeout)
    
    return latex_table


def main():
    parser = argparse.ArgumentParser(
        description='Analyze solver benchmark results and create comparison scatter plots.'
    )
    parser.add_argument(
        'csv_directory',
        help='Directory containing CSV files to process'
    )
    parser.add_argument(
        '-t', '--theory-file',
        required=True,
        help='CSV file mapping filenames to theories (columns: theory, mode, filename)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=None,
        help='Directory to save output plots (default: same as input directory)'
    )
    parser.add_argument(
        '--prefix',
        default='solver_comparison',
        help='Prefix for output file names (default: solver_comparison)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=300,
        help='Timeout value in seconds (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Use input directory as output directory if not specified
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.csv_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load theory mapping
    print("=" * 60)
    print("Loading theory mapping...")
    print("=" * 60)
    theory_df = load_theory_mapping(args.theory_file)
    
    # Load and process data
    print("\n" + "=" * 60)
    print("Loading and processing CSV files...")
    print("=" * 60)
    df, error_counts = load_and_process_data(args.csv_directory, theory_df)
    
    # Collect all LaTeX tables
    latex_tables = []
    
    # Create scatter plots per theory
    print("\n" + "=" * 60)
    print("Creating scatter plots...")
    print("=" * 60)
    
    # QF_LRA: yaga vs opensmt, yaga vs mathsat
    df_qf_lra = df[df['theory'] == 'QF_LRA']
    if len(df_qf_lra) > 0:
        print(f"\n--- QF_LRA ({len(df_qf_lra)} entries) ---")
        
        latex = create_scatter_plot(
            df_qf_lra, 
            'Yaga', 
            'OpenSMT',
            output_dir / f'{args.prefix}_QF_LRA_yaga_vs_opensmt.png',
            title_suffix='QF_LRA',
            timeout=args.timeout
        )
        if latex:
            latex_tables.append(f"% QF_LRA: Yaga vs OpenSMT\n{latex}")
        
        latex = create_scatter_plot(
            df_qf_lra, 
            'YAGA', 
            'MathSAT',
            output_dir / f'{args.prefix}_QF_LRA_yaga_vs_mathsat.png',
            title_suffix='QF_LRA',
            timeout=args.timeout
        )
        if latex:
            latex_tables.append(f"% QF_LRA: Yaga vs MathSAT\n{latex}")
    else:
        print("\nNo QF_LRA data found")
    
    # QF_UFLRA: yaga vs mathsat only
    df_qf_uflra = df[df['theory'] == 'QF_UFLRA']
    if len(df_qf_uflra) > 0:
        print(f"\n--- QF_UFLRA ({len(df_qf_uflra)} entries) ---")
        
        latex = create_scatter_plot(
            df_qf_uflra, 
            'yaga', 
            'mathsat',
            output_dir / f'{args.prefix}_QF_UFLRA_yaga_vs_mathsat.png',
            title_suffix='QF_UFLRA',
            timeout=args.timeout
        )
        if latex:
            latex_tables.append(f"% QF_UFLRA: Yaga vs MathSAT\n{latex}")
    else:
        print("\nNo QF_UFLRA data found")
    
    # Combined QF_LRA + QF_UFLRA: yaga vs mathsat
    df_combined = df[df['theory'].isin(['QF_LRA', 'QF_UFLRA'])]
    if len(df_combined) > 0:
        print(f"\n--- Combined QF_LRA + QF_UFLRA ({len(df_combined)} entries) ---")
        
        latex = create_scatter_plot(
            df_combined, 
            'Yaga', 
            'MathSAT',
            output_dir / f'{args.prefix}_combined_yaga_vs_mathsat.png',
            title_suffix='QF_LRA + QF_UFLRA',
            timeout=args.timeout
        )
        if latex:
            latex_tables.append(f"% Combined QF_LRA + QF_UFLRA: Yaga vs MathSAT\n{latex}")
    else:
        print("\nNo combined data found")
    
    # Save averaged data to CSV
    output_csv = output_dir / f'{args.prefix}_averaged_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nSaved averaged results to: {output_csv}")
    
    # Save LaTeX tables to file
    if latex_tables:
        latex_file = output_dir / f'{args.prefix}_latex_tables.txt'
        with open(latex_file, 'w') as f:
            f.write('\n\n'.join(latex_tables))
            
            # Add error counts per solver at the end
            f.write('\n\n% Error counts per solver (excluding timeout):\n')
            all_solvers = df['solver'].unique()
            for solver in sorted(all_solvers):
                count = error_counts.get(solver, 0)
                f.write(f'% {solver.capitalize()}: {count}\n')
        print(f"Saved LaTeX tables to: {latex_file}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
