# neugenes/processing-scripts/generate_histogram.py
"""
Brain Region CSV Processor and Histogram Generator
Command-line tools for processing brain imaging data
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json
from typing import Union, List, Optional

# ============================================================================
# FUNCTION 1: CSV PROCESSING
# ============================================================================

def process_brain_regions_csv(
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    regions_to_remove: Optional[List[str]] = None,
    regions_to_reset: Optional[List[str]] = None,
    reset_value: float = 0.3,
    reset_threshold: float = 1.0,
    reset_comparison: str = '>',
    verbose: bool = True
) -> dict:
    """Process brain region CSV: remove regions and/or reset values."""
    
    try:
        # Load data
        df = pd.read_csv(input_csv_path)
        original_shape = df.shape
        
        if verbose:
            print(f"Loaded CSV: {input_csv_path}", file=sys.stderr)
            print(f"Shape: {original_shape}", file=sys.stderr)
        
        # Identify region columns
        region_columns = [col for col in df.columns if col != 'Filename']
        modifications_made = False
        
        # Reset regions
        if regions_to_reset:
            modifications_made = True
            
            # Handle 'all' case
            if len(regions_to_reset) == 1 and regions_to_reset[0].lower() == 'all':
                regions_to_reset = region_columns.copy()
            
            valid_regions = [r for r in regions_to_reset if r in df.columns]
            
            if valid_regions:
                comparisons = {
                    '>': lambda x, t: x > t,
                    '<': lambda x, t: x < t,
                    '>=': lambda x, t: x >= t,
                    '<=': lambda x, t: x <= t,
                    '==': lambda x, t: x == t,
                    '!=': lambda x, t: x != t
                }
                
                compare_func = comparisons.get(reset_comparison)
                if not compare_func:
                    raise ValueError(f"Invalid comparison: {reset_comparison}")
                
                total_changes = 0
                for region in valid_regions:
                    mask = compare_func(df[region], reset_threshold)
                    n_changes = mask.sum()
                    total_changes += n_changes
                    
                    if n_changes > 0:
                        df.loc[mask, region] = reset_value
                        if verbose:
                            print(f"Reset {region}: {n_changes} values", file=sys.stderr)
                
                if verbose:
                    print(f"Total values reset: {total_changes}", file=sys.stderr)
        
        # Remove regions
        if regions_to_remove:
            modifications_made = True
            valid_removals = [r for r in regions_to_remove if r in df.columns and r != 'Filename']
            
            if valid_removals:
                df = df.drop(columns=valid_removals)
                if verbose:
                    print(f"Removed {len(valid_removals)} region(s)", file=sys.stderr)
        
        # Save CSV
        if modifications_made:
            if output_csv_path is None:
                base_name = os.path.splitext(input_csv_path)[0]
                output_csv_path = f"{base_name}_processed.csv"
            
            df.to_csv(output_csv_path, index=False)
            
            if verbose:
                print(f"Saved: {output_csv_path}", file=sys.stderr)
        
        # Return result as JSON
        result = {
            'success': True,
            'output_path': output_csv_path,
            'original_shape': list(original_shape),
            'new_shape': list(df.shape),
            'modifications_made': modifications_made
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# FUNCTION 2: HISTOGRAM GENERATION
# ============================================================================

def generate_brain_region_histogram(
    csv_path: str,
    output_image_path: str,
    remove_top_n: int = 0,
    exclude_regions: Optional[List[str]] = None,
    top_n_display: int = 30,
    power_exponent: float = 3.0,
    apply_normalization: bool = True,
    bar_color: str = 'gray',
    title: Optional[str] = None,
    xlabel: str = 'Measured Fluorescence Intensity',
    figsize_width: int = 10,
    figsize_height: int = 12,
    verbose: bool = True
) -> dict:
    """Generate horizontal bar histogram of brain region expression."""
    try:
        # Load data
        df = pd.read_csv(csv_path)
        
        if verbose:
            print(f"Loaded CSV: {csv_path}", file=sys.stderr)
            print(f"Shape: {df.shape}", file=sys.stderr)
        
        # Identify region columns
        region_columns = [col for col in df.columns if col != 'Filename']
        
        # Calculate region sums
        region_sums = df[region_columns].sum()
        scaled_sums = (region_sums / region_sums.max()) * 100
        sorted_regions = scaled_sums.sort_values(ascending=False)
        
        # Exclude specific regions
        if exclude_regions:
            valid_exclusions = [r for r in exclude_regions if r in sorted_regions.index]
            if valid_exclusions:
                sorted_regions = sorted_regions.drop(valid_exclusions)
                if verbose:
                    print(f"Excluded {len(valid_exclusions)} region(s)", file=sys.stderr)
        
        # Remove top N
        if remove_top_n > 0:
            if verbose:
                print(f"Removing top {remove_top_n} regions", file=sys.stderr)
            sorted_regions = sorted_regions.iloc[remove_top_n:]
        
        # Select top N for display
        sorted_regions = sorted_regions.sort_values(ascending=True)
        top_regions = sorted_regions.tail(top_n_display)
        
        # Apply power transformation
        if power_exponent != 1.0:
            top_regions_transformed = np.power(top_regions, power_exponent)
            
            if apply_normalization:
                top_regions_transformed = (top_regions_transformed / top_regions_transformed.max()) * 100
            
            if verbose:
                print(f"Applied transformation: ^{power_exponent}", file=sys.stderr)
        else:
            top_regions_transformed = top_regions
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        
        y_positions = np.arange(len(top_regions_transformed))
        ax.barh(
            y_positions, 
            top_regions_transformed.values, 
            color=bar_color, 
            edgecolor='black', 
            linewidth=0.5
        )
        
        # Labels and title
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_regions_transformed.index, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        
        if title is None:
            title = f'Regional Expression Levels (n={df.shape[0]})'
            if remove_top_n > 0:
                title += f', excluding top {remove_top_n}'
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Styling
        if apply_normalization:
            ax.set_xlim(0, 100)
        
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if verbose:
            print(f"Saved histogram: {output_image_path}", file=sys.stderr)
        
        # Return result as JSON
        result = {
            'success': True,
            'output_path': output_image_path,
            'n_regions_displayed': len(top_regions_transformed),
            'min_value': float(top_regions_transformed.min()),
            'max_value': float(top_regions_transformed.max())
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# FUNCTION 3: RENORMALIZATION
# ============================================================================

def renormalize_csv(
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    weights_json_path: Optional[str] = None,
    acronym_to_id_map: Optional[dict] = None,
    stabilizing_parameter: float = 0.3,
    verbose: bool = True
) -> dict:
    """
    Renormalize brain region CSV by applying calibration based on structure weights.
    """
    try:
        # Load CSV data
        df = pd.read_csv(input_csv_path)
        
        if verbose:
            print(f"Loaded CSV: {input_csv_path}", file=sys.stderr)
            print(f"Shape: {df.shape}", file=sys.stderr)
        
        # Load structure weights
        if weights_json_path is None:
            return {
                'success': False,
                'error': 'weights_json_path is required for renormalization'
            }
        
        if not os.path.exists(weights_json_path):
            return {
                'success': False,
                'error': f'Weights file not found: {weights_json_path}'
            }
        
        with open(weights_json_path, 'r') as f:
            structure_weights = json.load(f)
        
        if verbose:
            print(f"Loaded weights from: {weights_json_path}", file=sys.stderr)
            print(f"Number of structures in weights: {len(structure_weights)}", file=sys.stderr)
        
        # INVERT THE ACRONYM MAP (id -> acronym becomes acronym -> id)
        if acronym_to_id_map:
            # The input map is id -> acronym, so we need to flip it
            inverted_map = {v: k for k, v in acronym_to_id_map.items()}
            acronym_to_id_map = inverted_map
            if verbose:
                print(f"Inverted acronym map, sample: {list(acronym_to_id_map.items())[:5]}", file=sys.stderr)
        
        # Identify region columns (all except 'Filename')
        region_columns = [col for col in df.columns if col != 'Filename']
        
        # Normalize the weights
        weight_values = [float(v) for v in structure_weights.values()]
        min_val = min(weight_values)
        max_val = max(weight_values)
        
        normalized_weights = {
            key: (float(value) - min_val) / (max_val - min_val) 
            for key, value in structure_weights.items()
        }
        
        if verbose:
            print(f"Weight normalization range: [{min_val}, {max_val}]", file=sys.stderr)
        
        # Apply calibration to each region column
        calibrated_df = df.copy()
        calibrated_count = 0
        skipped_count = 0
        
        for region in region_columns:
            # Find the structure ID for this region
            struct_id = None
            
            # Try 1: Look up via acronym map
            if acronym_to_id_map and region in acronym_to_id_map:
                struct_id = str(acronym_to_id_map[region])
            
            # Try 2: Check if region name is directly in weights
            if struct_id is None and region in normalized_weights:
                struct_id = region
            
            # Try 3: Check if region (as string) is a key when converted
            if struct_id is None and str(region) in normalized_weights:
                struct_id = str(region)
            
            if struct_id and struct_id in normalized_weights:
                norm_weight = normalized_weights[struct_id]
                
                if norm_weight != 0:
                    # Apply transformation: value / weight + stabilizing_parameter
                    calibrated_df[region] = df[region] / norm_weight + stabilizing_parameter
                    calibrated_count += 1
                    if verbose and calibrated_count <= 5:  # Show first 5 matches
                        print(f"✓ Calibrated '{region}' using struct_id '{struct_id}': weight={norm_weight:.4f}", file=sys.stderr)
                else:
                    # Weight is 0, keep original values
                    if verbose:
                        print(f"Warning: Zero weight for region '{region}', keeping original values", file=sys.stderr)
                    skipped_count += 1
            else:
                # No matching weight found
                skipped_count += 1
                if verbose and skipped_count <= 10:  # Only show first 10 to avoid spam
                    print(f"Warning: No weight found for region '{region}' (tried struct_id='{struct_id}')", file=sys.stderr)
        
        if verbose:
            print(f"Calibrated {calibrated_count} regions", file=sys.stderr)
            if skipped_count > 0:
                print(f"Skipped {skipped_count} regions (no matching weights or zero weight)", file=sys.stderr)
        
        # Generate output path if not provided
        if output_csv_path is None:
            base_name = os.path.splitext(input_csv_path)[0]
            output_csv_path = f"{base_name}_renorm.csv"
        
        # Save calibrated data
        calibrated_df.to_csv(output_csv_path, index=False)
        
        if verbose:
            print(f"Saved renormalized CSV: {output_csv_path}", file=sys.stderr)
        
        return {
            'success': True,
            'output_path': output_csv_path,
            'original_shape': list(df.shape),
            'regions_calibrated': calibrated_count,
            'regions_skipped': skipped_count,
            'stabilizing_parameter': stabilizing_parameter
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Brain Region CSV Processor and Histogram Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process CSV - reset regions
  python %(prog)s process input.csv --output output.csv --reset IAD AV nVI --reset-value 0.3 --reset-threshold 1.0
  
  # Process CSV - remove regions
  python %(prog)s process input.csv --remove IAD AV CP
  
  # Generate histogram
  python %(prog)s histogram input.csv --output histogram.png --remove-top-n 2 --top-n-display 30
  
  # Generate histogram with exclusions
  python %(prog)s histogram input.csv --output hist.png --exclude IAD AV --power 2.5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ========== PROCESS COMMAND ==========
    process_parser = subparsers.add_parser('process', help='Process CSV (reset/remove regions)')
    process_parser.add_argument('input', help='Input CSV file path')
    process_parser.add_argument('--output', help='Output CSV file path (auto-generated if not provided)')
    process_parser.add_argument('--reset', nargs='+', help='Regions to reset (or "all")')
    process_parser.add_argument('--reset-value', type=float, default=0.3, help='Value to reset to (default: 0.3)')
    process_parser.add_argument('--reset-threshold', type=float, default=1.0, help='Threshold for reset (default: 1.0)')
    process_parser.add_argument('--reset-comparison', default='>', choices=['>', '<', '>=', '<=', '==', '!='],
                               help='Comparison operator (default: >)')
    process_parser.add_argument('--remove', nargs='+', help='Regions to remove entirely')
    process_parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    # ========== HISTOGRAM COMMAND ==========
    hist_parser = subparsers.add_parser('histogram', help='Generate histogram visualization')
    hist_parser.add_argument('input', help='Input CSV file path')
    hist_parser.add_argument('--output', required=True, help='Output image file path')
    hist_parser.add_argument('--remove-top-n', type=int, default=0, help='Remove top N regions (default: 0)')
    hist_parser.add_argument('--exclude', nargs='+', help='Specific regions to exclude')
    hist_parser.add_argument('--top-n-display', type=int, default=30, help='Number of regions to display (default: 30)')
    hist_parser.add_argument('--power', type=float, default=3.0, help='Power transformation exponent (default: 3.0)')
    hist_parser.add_argument('--no-normalize', action='store_true', help='Skip normalization after transformation')
    hist_parser.add_argument('--bar-color', default='gray', help='Bar color (default: gray)')
    hist_parser.add_argument('--title', help='Custom plot title')
    hist_parser.add_argument('--xlabel', default='GFP Fluorescence (% of region)', help='X-axis label')
    hist_parser.add_argument('--width', type=int, default=10, help='Figure width in inches (default: 10)')
    hist_parser.add_argument('--height', type=int, default=12, help='Figure height in inches (default: 12)')
    hist_parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    # ========== RENORMALIZE COMMAND ==========
    renorm_parser = subparsers.add_parser('renormalize', help='Renormalize CSV using structure weights')
    renorm_parser.add_argument('input', help='Input CSV file path')
    renorm_parser.add_argument('--weights', required=True, help='Path to weights JSON file')
    renorm_parser.add_argument('--output', help='Output CSV file path (auto-generated if not provided)')
    renorm_parser.add_argument('--acronym-map', help='Path to JSON file mapping acronyms to structure IDs')
    renorm_parser.add_argument('--stabilizing-param', type=float, default=0.3, 
                               help='Stabilizing parameter for calibration (default: 0.3)')
    renorm_parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'process':
        result = process_brain_regions_csv(
            input_csv_path=args.input,
            output_csv_path=args.output,
            regions_to_remove=args.remove,
            regions_to_reset=args.reset,
            reset_value=args.reset_value,
            reset_threshold=args.reset_threshold,
            reset_comparison=args.reset_comparison,
            verbose=not args.quiet
        )
    
    elif args.command == 'histogram':
        result = generate_brain_region_histogram(
            csv_path=args.input,
            output_image_path=args.output,
            remove_top_n=args.remove_top_n,
            exclude_regions=args.exclude,
            top_n_display=args.top_n_display,
            power_exponent=args.power,
            apply_normalization=not args.no_normalize,
            bar_color=args.bar_color,
            title=args.title,
            xlabel=args.xlabel,
            figsize_width=args.width,
            figsize_height=args.height,
            verbose=not args.quiet
        )

    elif args.command == 'renormalize':
        # Load acronym map if provided
        acronym_map = None
        if args.acronym_map:
            with open(args.acronym_map, 'r') as f:
                acronym_map = json.load(f)
        
        result = renormalize_csv(
            input_csv_path=args.input,
            output_csv_path=args.output,
            weights_json_path=args.weights,
            acronym_to_id_map=acronym_map,
            stabilizing_parameter=args.stabilizing_param,
            verbose=not args.quiet
        )
    
    # Output result as JSON to stdout
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()