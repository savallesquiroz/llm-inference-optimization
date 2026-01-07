"""
Analysis and Visualization for LLM Optimization Experiments
Generates plots and LaTeX tables for your report
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


class OptimizationAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11

    def analyze_quantization(self):
        """Analyze quantization format comparison"""
        csv_file = self.results_dir / 'quantization_comparison.csv'
        if not csv_file.exists():
            print(f"✗ File not found: {csv_file}")
            return

        df = pd.read_csv(csv_file)
        if df.empty:
            print("✗ No rows in quantization data")
            return

        print(f"  Analyzing {len(df)} quantization runs...")

        # Extract quantization format from model name
        df['quant_format'] = df['model'].str.extract(r"\.([Qq]\d+_[^.]*)\.", expand=False)
        df['quant_format'] = df['quant_format'].fillna(df['model'])

        # Check if memory has actual data
        has_memory = 'memory_mb' in df.columns and df['memory_mb'].sum() > 0

        # Adjust subplot count based on data availability
        n_plots = 2 if has_memory else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(12 if has_memory else 6, 5))
        if n_plots == 1:
            axes = [axes]  # Make single axis iterable

        # Plot 1: Tokens per second
        ax1 = axes[0]
        df.groupby('quant_format')['tokens_per_sec'].mean().plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Inference Speed by Quantization Format')
        ax1.set_xlabel('Quantization Format')
        ax1.set_ylabel('Tokens/Second')
        ax1.legend().remove()

        # Plot 2: Speed vs other metrics (if memory available)
        if has_memory:
            ax2 = axes[1]
            ax2.scatter(df['memory_mb'], df['tokens_per_sec'], s=100, c='green', alpha=0.6)
            for idx, row in df.iterrows():
                ax2.annotate(str(row['quant_format']),
                            (row['memory_mb'], row['tokens_per_sec']),
                            fontsize=8)
            ax2.set_xlabel('Memory (MB)')
            ax2.set_ylabel('Tokens/Second')
            ax2.set_title('Speed vs Memory')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'quantization_analysis.png', dpi=300)
        print(f"✓ Saved: quantization_analysis.png")

        # Generate LaTeX table
        self.generate_latex_table(df, 'quantization_table.tex',
                                 ['quant_format', 'tokens_per_sec'],
                                 'Quantization Format Comparison')

    def analyze_context_length(self):
        """Analyze context length scaling"""
        csv_file = self.results_dir / 'context_length_scaling.csv'
        if not csv_file.exists():
            print(f"✗ File not found: {csv_file}")
            return

        df = pd.read_csv(csv_file)
        if df.empty:
            print("✗ No rows in context length data")
            return

        print(f"  Analyzing {len(df)} context length runs...")

        # Sort by n_tokens
        if 'n_tokens' in df.columns:
            df = df.sort_values('n_tokens')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Tokens/sec vs context length
        ax1 = axes[0]
        ax1.plot(df['n_tokens'], df['tokens_per_sec'],
                 marker='o', linewidth=2, markersize=8, color='darkblue')
        ax1.set_xlabel('Number of Tokens Generated')
        ax1.set_ylabel('Tokens/Second')
        ax1.set_title('Inference Speed vs Context Length')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Total time
        ax2 = axes[1]
        if 'prompt_time_ms' in df.columns and 'generation_time_ms' in df.columns:
            total_time = (df['prompt_time_ms'] + df['generation_time_ms']) / 1000.0
            ax2.plot(df['n_tokens'], total_time,
                     marker='s', linewidth=2, markersize=8, color='crimson')
            ax2.set_ylabel('Total Time (seconds)')
        else:
            ax2.plot(df['n_tokens'], df['total_time_ms'] / 1000.0,
                     marker='s', linewidth=2, markersize=8, color='crimson')
            ax2.set_ylabel('Total Time (seconds)')
        
        ax2.set_xlabel('Number of Tokens Generated')
        ax2.set_title('Total Inference Time vs Context Length')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'context_length_analysis.png', dpi=300)
        print(f"✓ Saved: context_length_analysis.png")

        # Generate LaTeX table
        self.generate_latex_table(df, 'context_length_table.tex',
                                 ['n_tokens', 'tokens_per_sec', 'total_time_ms'],
                                 'Context Length Scaling Results')

    def analyze_thread_scaling_detailed(self):
        """Enhanced thread scaling analysis with statistics"""
        csv_file = self.results_dir / 'thread_scaling_detailed.csv'
        if not csv_file.exists():
            print(f"✗ File not found: {csv_file}")
            return

        df = pd.read_csv(csv_file)
        if df.empty:
            print("✗ No rows in thread scaling data")
            return

        print(f"  Analyzing {len(df)} thread scaling runs...")

        # Calculate statistics per thread count
        stats = df.groupby('threads')['tokens_per_sec'].agg(['mean', 'std', 'min', 'max'])
        stats = stats.sort_index()

        if len(stats) < 2:
            print(f"⚠ Only {len(stats)} thread configuration(s) in data; skipping speedup plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Mean performance with error bars
        ax1 = axes[0]
        ax1.errorbar(stats.index, stats['mean'], yerr=stats['std'].fillna(0),
                     marker='o', linewidth=2, markersize=8, capsize=5,
                     color='navy', ecolor='lightblue')
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Tokens/Second')
        ax1.set_title('Thread Scaling with Variability')
        ax1.grid(True, alpha=0.3)

        # Add efficiency line
        if stats['mean'].iloc[0] > 0:
            ax1_twin = ax1.twinx()
            efficiency = (stats['mean'] / stats.index) / (stats['mean'].iloc[0])
            ax1_twin.plot(stats.index, efficiency,
                          color='red', linestyle='--', marker='s', alpha=0.7, label='Efficiency')
            ax1_twin.set_ylabel('Parallel Efficiency', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')

        # Plot 2: Speedup vs ideal
        ax2 = axes[1]
        if stats['mean'].iloc[0] > 0:
            speedup = stats['mean'] / stats['mean'].iloc[0]
            ideal_speedup = stats.index / stats.index[0]

            ax2.plot(stats.index, speedup, marker='o', linewidth=2,
                     label='Actual Speedup', color='green')
            ax2.plot(stats.index, ideal_speedup, linestyle='--', linewidth=2,
                     label='Ideal (Linear)', color='gray', alpha=0.7)
            ax2.set_xlabel('Number of Threads')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title('Speedup: Actual vs Ideal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'thread_scaling_detailed.png', dpi=300)
        print(f"✓ Saved: thread_scaling_detailed.png")

        # Generate LaTeX table with statistics
        stats_df = stats.reset_index()
        self.generate_latex_table(stats_df, 'thread_scaling_stats_table.tex',
                                 ['threads', 'mean', 'std', 'min', 'max'],
                                 'Thread Scaling Statistics')

    def analyze_model_size(self):
        """Analyze different model sizes"""
        csv_file = self.results_dir / 'model_size_comparison.csv'
        if not csv_file.exists():
            print(f"✗ File not found: {csv_file}")
            return

        df = pd.read_csv(csv_file)
        if df.empty:
            print("✗ No rows in model size data")
            return

        print(f"  Analyzing {len(df)} model size runs...")

        # Check if memory has actual data
        has_memory = 'memory_mb' in df.columns and df['memory_mb'].sum() > 0

        n_plots = 2 if has_memory else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(12 if has_memory else 6, 5))
        if n_plots == 1:
            axes = [axes]

        # Plot 1: Performance by model size
        ax1 = axes[0]
        df.plot(x='model_size', y='tokens_per_sec', kind='bar', ax=ax1, color='purple', legend=False)
        ax1.set_title('Inference Speed by Model Size')
        ax1.set_xlabel('Model Size')
        ax1.set_ylabel('Tokens/Second')

        # Plot 2: Memory usage (if available)
        if has_memory:
            ax2 = axes[1]
            df.plot(x='model_size', y='memory_mb', kind='bar', ax=ax2, color='orange', legend=False)
            ax2.set_title('Memory Usage by Model Size')
            ax2.set_xlabel('Model Size')
            ax2.set_ylabel('Memory (MB)')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_size_analysis.png', dpi=300)
        print(f"✓ Saved: model_size_analysis.png")

        # Generate LaTeX table
        self.generate_latex_table(df, 'model_size_table.tex',
                                 ['model_size', 'tokens_per_sec'],
                                 'Model Size Comparison')

    def generate_latex_table(self, df, filename, columns, caption):
        """Generate LaTeX table from dataframe"""
        output_path = self.results_dir / filename

        # Filter columns that exist
        columns = [col for col in columns if col in df.columns]
        df_filtered = df[columns].copy()

        # Format numbers
        for col in df_filtered.columns:
            if np.issubdtype(df_filtered[col].dtype, np.number):
                df_filtered[col] = df_filtered[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{tab:{filename.replace('.tex', '')}}}\n"
        latex += "\\begin{NiceTabular}{" + "l" * len(columns) + "}\n"
        latex += "\\CodeBefore\n"
        latex += "\\rowcolors{2}{gray!10}{white}\n"
        latex += "\\Body\n"
        latex += "\\toprule\n"

        # Header
        headers = [col.replace('_', ' ').title() for col in columns]
        latex += " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\\n"
        latex += "\\midrule\n"

        # Data rows
        for _, row in df_filtered.iterrows():
            latex += " & ".join(str(row[col]) for col in columns) + " \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{NiceTabular}\n"
        latex += "\\end{table}\n"

        with open(output_path, 'w') as f:
            f.write(latex)

        print(f"✓ Saved LaTeX table: {filename}")

    def create_comprehensive_comparison(self):
        """Create a summary visualization of all experiments"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Collect summary data from available CSVs
        summaries = []
        categories = []
        details = []  # For debug

        # Try to load each experiment's best result
        experiments = [
            ('quantization_comparison.csv', 'Quantization'),
            ('context_length_scaling.csv', 'Context Length'),
            ('thread_scaling_detailed.csv', 'Thread Scaling'),
            ('model_size_comparison.csv', 'Model Size'),
        ]

        for csv_file, label in experiments:
            path = self.results_dir / csv_file
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if df.empty:
                        details.append(f"  {label}: CSV empty")
                    elif 'tokens_per_sec' not in df.columns:
                        details.append(f"  {label}: no tokens_per_sec column")
                    else:
                        avg_tps = df['tokens_per_sec'].mean()
                        if pd.isna(avg_tps) or avg_tps == 0:
                            details.append(f"  {label}: tokens_per_sec all zeros/NaN")
                        else:
                            summaries.append(avg_tps)
                            categories.append(label)
                            details.append(f"  {label}: {avg_tps:.2f} t/s")
                except Exception as e:
                    details.append(f"  {label}: Error - {e}")
            else:
                details.append(f"  {label}: File not found")

        # Print debug info
        print("\n📊 Comprehensive Comparison Debug:")
        for detail in details:
            print(detail)

        if summaries:
            # Create bar chart of average performance
            colors = ['steelblue', 'coral', 'green', 'purple'][:len(summaries)]
            bars = ax.bar(categories, summaries, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Average Tokens/Second', fontsize=12, fontweight='bold')
            ax.set_xlabel('Experiment Type', fontsize=12, fontweight='bold')
            ax.set_title('Performance Summary Across Experiments', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, (cat, val) in enumerate(zip(categories, summaries)):
                ax.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Set y-axis to start from 0 for better visualization
            ax.set_ylim(bottom=0)
        else:
            # Create a placeholder/info visualization if no data
            ax.text(0.5, 0.7, '⚠ No experiment data available', ha='center', va='center',
                    fontsize=14, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.5, 'Check that CSV files exist in results/ directory\nand contain non-zero tokens_per_sec values.',
                    ha='center', va='center', fontsize=11, transform=ax.transAxes, style='italic')
            ax.text(0.5, 0.25, f'Checked: {", ".join(exp[1] for exp in experiments)}',
                    ha='center', va='center', fontsize=9, transform=ax.transAxes, color='gray')
            ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_comparison.png', dpi=300)
        print(f"✓ Saved: comprehensive_comparison.png")

    def run_all_analyses(self):
        """Run all analysis and generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING ANALYSIS AND VISUALIZATIONS")
        print("="*60 + "\n")

        self.analyze_quantization()
        self.analyze_context_length()
        self.analyze_thread_scaling_detailed()
        self.analyze_model_size()
        self.create_comprehensive_comparison()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nAll outputs saved to: {self.results_dir}")
        print("\nGenerated files:")
        print("  - PNG visualizations for your report")
        print("  - LaTeX tables ready to paste into your document")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    analyzer = OptimizationAnalyzer(results_dir="results")
    analyzer.run_all_analyses()