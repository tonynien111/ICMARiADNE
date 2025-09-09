#!/usr/bin/env python3
"""
Script to collect expert demonstrations for behavior cloning
This can be run separately before training to pre-collect expert data
"""

import os
import sys
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from expert.expert_data_collector import collect_expert_data_script
from parameter import *


def main():
    parser = argparse.ArgumentParser(description='Collect expert demonstrations for behavior cloning')
    parser.add_argument('--episodes', type=int, default=EXPERT_EPISODES,
                       help=f'Number of episodes to collect (default: {EXPERT_EPISODES})')
    parser.add_argument('--output', type=str, default=BC_DEMONSTRATIONS_PATH,
                       help=f'Output file path (default: {BC_DEMONSTRATIONS_PATH})')
    parser.add_argument('--force', action='store_true',
                       help='Force recollection even if file exists')
    
    args = parser.parse_args()
    
    print("Expert Data Collection Script")
    print("=" * 40)
    print(f"Episodes to collect: {args.episodes}")
    print(f"Output path: {args.output}")
    
    # Check if file exists
    if os.path.exists(args.output) and not args.force:
        print(f"\nExpert demonstrations already exist at {args.output}")
        print("Use --force to overwrite or specify different --output path")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Temporarily modify parameters
    global EXPERT_EPISODES, BC_DEMONSTRATIONS_PATH
    original_episodes = EXPERT_EPISODES
    original_path = BC_DEMONSTRATIONS_PATH
    
    # Update parameters
    import parameter
    parameter.EXPERT_EPISODES = args.episodes
    parameter.BC_DEMONSTRATIONS_PATH = args.output
    
    try:
        # Collect data
        collector = collect_expert_data_script()
        
        if collector and len(collector.expert_demonstrations) > 0:
            print(f"\n✅ Successfully collected {len(collector.expert_demonstrations)} demonstrations")
            print(f"💾 Saved to: {args.output}")
            
            # Show statistics
            stats = collector.get_dataset_stats()
            print(f"\n📊 Dataset Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
                
        else:
            print("❌ Failed to collect expert demonstrations")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original parameters
        parameter.EXPERT_EPISODES = original_episodes
        parameter.BC_DEMONSTRATIONS_PATH = original_path
    
    print("\nDone.")


if __name__ == "__main__":
    main()