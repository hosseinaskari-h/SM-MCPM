"""
SM-MCPM Main Entry Point
Run full algorithm or specific phases
"""

import argparse
from pathlib import Path
from config import load_config
from controller import Controller
from visualizer import export_final_state


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SM-MCPM: State-Modulated Monte Carlo Physarum Machine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run full algorithm with default config
  python main.py --config configs/test.yaml   # Use custom config
  python main.py --phase 1                    # Run only Phase 1
  python main.py --output-dir results/exp1    # Custom output directory
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file (default: configs/default.yaml)')
    parser.add_argument('--output-dir', type=str, default='outputs/full_run',
                       help='Output directory for results (default: outputs/full_run)')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', 'full'],
                       default='full', help='Which phase to run (default: full)')
    
    args = parser.parse_args()
    
    print("SM-MCPM Algorithm Runner")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Phase: {args.phase}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create controller
    controller = Controller(config)
    
    # Run requested phase(s)
    if args.phase == '1':
        print("\nRunning Phase 1 only...\n")
        phase1_results = controller.run_phase1()
        
        results = {
            'phase1': phase1_results,
            'nodes': [],
            'config': config
        }
    
    elif args.phase == '2':
        print("\nRunning Phase 1 + 2...\n")
        phase1_results = controller.run_phase1()
        nodes = controller.run_phase2(
            initial_death_map=phase1_results['death_map'],
            initial_trail_map=phase1_results['trail_map']
        )
        
        results = {
            'phase1': phase1_results,
            'nodes': nodes,
            'config': config
        }
    
    elif args.phase == '3':
        print("\nRunning all phases (1, 2, 3)...\n")
        results = controller.run_full_algorithm()
    
    else:  # 'full'
        print("\nRunning full algorithm (all 3 phases)...\n")
        results = controller.run_full_algorithm()
    
    # Export results
    print("\nExporting results...")
    export_final_state(results, args.output_dir)
    
    print(" COMPLETE!")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - death_map.png")
    print(f"  - vein_network.png")
    print(f"  - summary.png")
    if results.get('nodes'):
        print(f"  - nodes.json ({len(results['nodes'])} nodes)")
    if results.get('phase3', {}).get('metrics'):
        print(f"  - metrics.csv")


if __name__ == "__main__":
    main()