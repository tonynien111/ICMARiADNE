# Behavior Cloning Integration for ARiADNE

This documentation describes the behavior cloning (BC) integration added to the ARiADNE reinforcement learning system.

## Overview

Behavior cloning has been integrated to pretrain the PolicyNet with expert demonstrations before starting reinforcement learning training. This helps the agent start with non-random policy weights, leading to faster convergence and better performance.

## Architecture

### Components Added

1. **Expert Planner** (`expert/expert_planner.py`)
   - Uses OR-Tools VRP solver for optimal path planning
   - Implements sophisticated coverage path planning algorithm
   - Handles multi-robot scenarios with utility-based node selection

2. **Expert Data Collector** (`expert/expert_data_collector.py`)
   - Collects expert demonstrations using the ExpertPlanner
   - Stores state-action pairs in format compatible with PolicyNet
   - Handles timeouts and error recovery during data collection

3. **Behavior Cloning Trainer** (`behavior_cloning.py`)
   - Supervised learning trainer for PolicyNet
   - Cross-entropy loss with early stopping
   - Train/validation split with performance monitoring

4. **Integration Scripts**
   - `collect_expert_data.py`: Standalone script for expert data collection
   - `test_bc.py`: Comprehensive test suite for BC integration

### Key Features

- **Expert Policy**: OR-Tools based optimal path planning
- **Robust Data Collection**: Timeout handling and fallback strategies  
- **Early Stopping**: Validation-based training termination
- **Tensorboard Logging**: Training progress visualization
- **Flexible Parameters**: Configurable via `parameter.py`

## Configuration

### New Parameters in `parameter.py`

```python
# Behavior Cloning parameters
USE_BEHAVIOR_CLONING = True  # Enable BC pretraining
EXPERT_EPISODES = 50  # Number of expert episodes to collect
BC_EPOCHS = 100  # Number of BC epochs
BC_BATCH_SIZE = 64  # Batch size for BC training
BC_LR = 1e-4  # Learning rate for BC
BC_VALIDATION_SPLIT = 0.2  # Validation split ratio
BC_PATIENCE = 10  # Early stopping patience
BC_DEMONSTRATIONS_PATH = 'expert_demonstrations.pkl'  # Data file path
BC_COLLECT_DATA_ON_START = True  # Collect data if not found
BC_MAX_STEPS_PER_EPISODE = 50  # Max steps per expert episode
```

## Usage

### Method 1: Automatic Integration (Recommended)
Just run the main driver as usual. BC pretraining will run automatically before RL training:

```bash
python driver.py
```

The system will:
1. Check for existing expert demonstrations
2. Collect new demonstrations if needed (or if `BC_COLLECT_DATA_ON_START=True`)
3. Run behavior cloning pretraining
4. Continue with normal RL training

### Method 2: Separate Data Collection
Collect expert data separately for reuse:

```bash
# Collect 100 episodes of expert data
python collect_expert_data.py --episodes 100 --output my_expert_data.pkl

# Use collected data in training by updating parameter.py:
# BC_DEMONSTRATIONS_PATH = 'my_expert_data.pkl'
# BC_COLLECT_DATA_ON_START = False

python driver.py
```

### Method 3: Testing and Validation
Run comprehensive tests:

```bash
# Test the full BC pipeline
python test_bc.py

# Test only behavior cloning training
python behavior_cloning.py

# Test only expert data collection  
python expert/expert_data_collector.py
```

## Expert Policy Details

The expert policy uses a sophisticated approach:

1. **Node Utility Calculation**: Based on observable frontiers within sensor range
2. **VRP Optimization**: OR-Tools solves vehicle routing problem for optimal coverage
3. **Iterative Refinement**: Multiple iterations to find best viewpoint selection
4. **Fallback Strategies**: Greedy approaches when VRP solver fails
5. **Adaptive Termination**: Early stopping when exploration goals achieved

### Expert Decision Process:
1. Identify nodes with positive utility (unobserved frontiers)
2. Select viewpoints using utility-weighted sampling
3. Solve VRP for optimal path between viewpoints
4. Execute first waypoint in optimal path
5. Update world state and repeat

## Integration with RL Training

The BC integration follows this sequence:

1. **Initialization**: Create PolicyNet, Q-networks, ICM (if enabled)
2. **BC Pretraining**: Run behavior cloning if enabled and not loading existing model
3. **RL Training**: Continue with normal SAC training using BC-initialized policy

The BC-pretrained policy provides:
- Better initial exploration strategies
- Faster convergence in early RL training
- More stable learning curves
- Higher sample efficiency

## File Structure

```
ICM/
├── expert/
│   ├── __init__.py
│   ├── expert_planner.py       # OR-Tools based expert policy
│   ├── expert_data_collector.py # Expert demonstration collection
│   └── ortools_solver.py       # VRP solver implementation
├── behavior_cloning.py         # BC trainer implementation  
├── collect_expert_data.py      # Standalone data collection script
├── test_bc.py                  # BC integration test suite
├── driver.py                   # Main training script (modified)
├── parameter.py                # Parameters (extended with BC params)
└── README_BC.md               # This documentation
```

## Dependencies

New dependencies added:
- `ortools`: For VRP optimization
- `scikit-learn`: For train/test splits

Install with:
```bash
pip install ortools scikit-learn
```

## Performance Tips

1. **Expert Data Quality**: More expert episodes generally improve BC performance
2. **BC Epochs**: Use early stopping rather than fixed epochs
3. **Batch Size**: Larger batches (64-128) work well for graph attention networks
4. **Learning Rate**: Start with 1e-4, adjust based on convergence
5. **Validation Split**: 20% validation is usually sufficient

## Troubleshooting

### Common Issues:

1. **OR-Tools Import Error**:
   ```bash
   pip install ortools
   ```

2. **Expert Data Collection Hangs**:
   - Check timeout settings in `expert_data_collector.py`
   - Reduce `BC_MAX_STEPS_PER_EPISODE`
   - Enable fallback strategies

3. **BC Training Not Converging**:
   - Increase `BC_EPOCHS`
   - Adjust `BC_LR` (try 1e-5 or 1e-3)
   - Check expert data quality

4. **Memory Issues**:
   - Reduce `BC_BATCH_SIZE`
   - Reduce `EXPERT_EPISODES`
   - Use CPU instead of GPU for data collection

### Debug Mode:

Enable detailed logging by setting in `parameter.py`:
```python
BC_COLLECT_DATA_ON_START = True  # Always collect fresh data
BC_EPOCHS = 10  # Quick testing
```

## Future Enhancements

Potential improvements:
- **GAIL Integration**: Replace BC with Generative Adversarial Imitation Learning
- **DAgger**: Interactive expert querying during RL training  
- **Multi-Task BC**: Train on multiple environments simultaneously
- **BC Fine-tuning**: Periodic BC updates during RL training
- **Expert Policy Ensemble**: Combine multiple expert strategies

## References

- Original ARiADNE paper: "ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration"
- OR-Tools documentation: https://developers.google.com/optimization
- Behavior Cloning: "Efficient Training of Artificial Neural Networks for Autonomous Navigation" (Pomerleau, 1991)
