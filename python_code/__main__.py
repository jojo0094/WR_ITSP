"""
Entry point for running the python_code module directly.
Usage: python -m python_code
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Information-Theoretic Sensor Placement for Sewer Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with flow data
  python -m python_code --max-sensors 250
  
  # Run with water depth data
  python -m python_code --use-depth --max-sensors 100
  
  # Include estimation (memory intensive)
  python -m python_code --max-sensors 250 --run-estimation
  
  # Run simple example with synthetic data
  python -m python_code --example
        """
    )
    
    parser.add_argument('--data-path', type=str, default='Data',
                       help='Path to data directory (default: Data)')
    parser.add_argument('--results-path', type=str, default='Results_folder_python',
                       help='Path to results directory (default: Results_folder_python)')
    parser.add_argument('--max-sensors', type=int, default=250,
                       help='Maximum number of sensors (default: 250, must be divisible by 25)')
    parser.add_argument('--use-depth', action='store_true',
                       help='Use water depth data instead of flow data')
    parser.add_argument('--run-estimation', action='store_true',
                       help='Run estimation algorithms (memory intensive)')
    parser.add_argument('--example', action='store_true',
                       help='Run simple example with synthetic data')
    
    args = parser.parse_args()
    
    if args.example:
        print("\nRunning simple example with synthetic data...\n")
        try:
            from .example_simple import *
        except ImportError:
            # If running as script directly
            import os
            os.chdir(os.path.dirname(__file__))
            exec(open('example_simple.py').read())
    else:
        from .run_code import run_main
        
        run_main(
            data_path=args.data_path,
            results_path=args.results_path,
            max_number_sensors=args.max_sensors,
            use_flow_data=not args.use_depth,
            run_estimation_flag=args.run_estimation
        )

if __name__ == '__main__':
    main()
