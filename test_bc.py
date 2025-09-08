#!/usr/bin/env python3
"""
Test script for behavior cloning integration with ARiADNE
"""

import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import PolicyNet
from behavior_cloning import run_behavior_cloning, test_behavior_cloning
from expert.expert_data_collector import collect_expert_data_script
from parameter import *


def test_expert_data_collection():
    """Test expert data collection"""
    print("=" * 60)
    print("TESTING EXPERT DATA COLLECTION")
    print("=" * 60)
    
    try:
        collector = collect_expert_data_script()
        if collector and len(collector.expert_demonstrations) > 0:
            print(f"✓ Successfully collected {len(collector.expert_demonstrations)} demonstrations")
            return True
        else:
            print("✗ Failed to collect expert demonstrations")
            return False
    except Exception as e:
        print(f"✗ Error in expert data collection: {e}")
        return False


def test_full_bc_pipeline():
    """Test the full behavior cloning pipeline"""
    print("=" * 60)
    print("TESTING FULL BEHAVIOR CLONING PIPELINE")  
    print("=" * 60)
    
    try:
        # Set device
        device = torch.device('cuda' if USE_GPU_GLOBAL and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create policy network
        policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
        print("✓ Policy network created")
        
        # Create tensorboard writer
        writer = SummaryWriter('test_bc_pipeline')
        print("✓ Tensorboard writer created")
        
        # Test the full pipeline
        success = run_behavior_cloning(policy_net, device, writer)
        
        writer.close()
        
        if success:
            print("✓ Full behavior cloning pipeline completed successfully!")
            return True
        else:
            print("✗ Full behavior cloning pipeline failed")
            return False
            
    except Exception as e:
        print(f"✗ Error in behavior cloning pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_validation():
    """Test that all required parameters are set correctly"""
    print("=" * 60)
    print("TESTING PARAMETER VALIDATION")
    print("=" * 60)
    
    required_params = [
        'USE_BEHAVIOR_CLONING',
        'EXPERT_EPISODES', 
        'BC_EPOCHS',
        'BC_BATCH_SIZE',
        'BC_LR',
        'BC_VALIDATION_SPLIT',
        'BC_PATIENCE',
        'BC_DEMONSTRATIONS_PATH',
        'BC_COLLECT_DATA_ON_START',
        'BC_MAX_STEPS_PER_EPISODE'
    ]
    
    missing_params = []
    for param in required_params:
        if not hasattr(sys.modules['parameter'], param):
            missing_params.append(param)
    
    if missing_params:
        print(f"✗ Missing parameters: {missing_params}")
        return False
    else:
        print("✓ All required parameters found")
        
        # Print parameter values
        print("\nParameter values:")
        for param in required_params:
            value = getattr(sys.modules['parameter'], param)
            print(f"  {param}: {value}")
        
        return True


def main():
    """Main test function"""
    print("Starting Behavior Cloning Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Parameter validation
    test_results.append(("Parameter validation", test_parameter_validation()))
    
    # Test 2: Expert data collection
    test_results.append(("Expert data collection", test_expert_data_collection()))
    
    # Test 3: Full BC pipeline
    test_results.append(("Full BC pipeline", test_full_bc_pipeline()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Behavior cloning integration is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)