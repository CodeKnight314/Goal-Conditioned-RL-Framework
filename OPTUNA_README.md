# Optuna Hyperparameter Optimization for Panda Manipulation RL

This repository now includes an automated hyperparameter optimization system using [Optuna](https://optuna.org/) to find the best combination of hyperparameters for your reinforcement learning model.

## Overview

The optimization system targets hyperparameters marked with `#OPT Target` in your `config_her.yaml` file:

- `gradient_step`: Number of gradient steps per training cycle
- `hidden_dim`: Hidden layer dimension for neural networks
- `actor_lr`: Actor learning rate
- `actor_lr_min`: Minimum actor learning rate for scheduling
- `critic_lr`: Critic learning rate  
- `critic_lr_min`: Minimum critic learning rate for scheduling
- `noise_std`: Standard deviation for exploration noise (0.1 to 0.7, multiples of 0.1)
- `policy_noise`: Policy noise for target policy smoothing
- `tau`: Soft update coefficient for target networks

## Files Added

1. **`optuna_optimization.py`** - Main optimization script with discrete hyperparameter search spaces (now runs training directly!)
2. **`src/train_for_optuna.py`** - Training wrapper (kept for compatibility, but main optimization now runs directly)
3. **`run_optimization_example.py`** - Example script showing how to run optimization
4. **`requirements.txt`** - Updated with Optuna dependency
5. **`src/__init__.py`** - Makes src a proper Python package

## ✨ New Features

- **Real-time Progress Visualization**: See tqdm progress bars during training for each trial
- **Direct Training Execution**: No more subprocess overhead - runs training directly in Python
- **Success Rate Optimization**: Both train methods now return success rate instead of reward
- **Better Error Handling**: More detailed error messages and stack traces for debugging

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Example Optimization
```bash
python run_optimization_example.py
```

This will run a quick optimization with 20 trials and 5 epochs per trial for the "reach" task.

### 3. Custom Optimization
```bash
python optuna_optimization.py \
    --config src/config/config_her.yaml \
    --output optuna_results \
    --task reach \
    --trials 50 \
    --epochs 10 \
    --envs 32 \
    --seed 1898
```

## Command Line Options

- `--config`: Path to base config file (default: `src/config/config_her.yaml`)
- `--output`: Output directory for results (default: `optuna_results`)  
- `--task`: Task to optimize for: `reach`, `push`, `slide`, `pickplace` (default: `reach`)
- `--trials`: Number of optimization trials (default: 50)
- `--epochs`: Epochs per trial for faster optimization (default: 10)
- `--envs`: Number of parallel environments (default: 32)
- `--seed`: Random seed for reproducibility (default: 1898)

## Output Structure

After optimization, you'll find these files in your output directory:

```
optuna_results/
├── trial_0000/               # Individual trial directories
│   ├── config_her.yaml      # Trial-specific config
│   ├── actor.pth            # Trained model weights
│   └── ...
├── trial_0001/
├── ...
├── optimized_config_her.yaml # Best configuration found
├── optimization_results.yaml # Summary of optimization results
└── optuna_study.db          # Optuna database for analysis
```

## Hyperparameter Search Spaces

The optimization uses discrete search spaces for better reproducibility:

- **gradient_step**: [20, 30, 40, 50, 60]
- **hidden_dim**: [32, 64, 128, 256]  
- **actor_lr**: [0.0001, 0.0003, 0.001, 0.003, 0.01]
- **actor_lr_min**: [0.00001, 0.00003, 0.0001, 0.0003]
- **critic_lr**: [0.0001, 0.0003, 0.001, 0.003, 0.01]
- **critic_lr_min**: [0.00001, 0.00003, 0.0001, 0.0003]
- **noise_std**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
- **policy_noise**: [0.1, 0.15, 0.2, 0.25, 0.3]
- **tau**: [0.001, 0.003, 0.005, 0.01, 0.02]

## Understanding Results

### optimization_results.yaml
```yaml
best_success_rate: 0.8534      # Best success rate achieved
best_hyperparameters:          # Optimal hyperparameter combination
  gradient_step: 40
  hidden_dim: 128
  actor_lr: 0.001
  # ... etc
best_trial_number: 23          # Which trial achieved the best result
total_trials: 50               # Total trials completed
```

### Using Optimized Configuration

After optimization, use the best configuration for full training:

```bash
python -m src.main \
    --c optuna_results/optimized_config_her.yaml \
    --o final_training \
    --id reach \
    --her
```

## Advanced Usage

### Resuming Optimization

The optimization uses SQLite database storage, so you can resume interrupted studies:

```bash
# This will continue from where it left off
python optuna_optimization.py --output optuna_results --trials 100
```

### Analyzing Results with Optuna Dashboard

Install and run the Optuna dashboard for interactive analysis:

```bash
pip install optuna-dashboard
optuna-dashboard optuna_results/optuna_study.db
```

### Custom Search Spaces

To modify the hyperparameter search spaces, edit the `suggest_hyperparameters` method in `optuna_optimization.py`:

```python
def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
    return {
        'gradient_step': trial.suggest_categorical('gradient_step', [30, 40, 50]),  # Custom range
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128]),          # Fewer options
        # ... etc
    }
```

## Tips for Effective Optimization

1. **Start Small**: Begin with fewer trials (10-20) and fewer epochs (5-10) to test the system
2. **Scale Gradually**: Increase trials and epochs based on available computational resources
3. **Monitor Progress**: Check trial directories to ensure training is working correctly
4. **Use Success Rate**: The optimization maximizes success rate, which is most meaningful for goal-conditioned tasks
5. **Compare Results**: Always compare optimized results with your baseline configuration

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
2. **Training Failures**: Check individual trial directories for error logs
3. **No Success Rate**: Ensure the HER environment is being used (`--her` flag)
4. **Memory Issues**: Reduce `--envs` parameter or `--epochs` per trial

### Debug Mode

For debugging, you can run a single trial manually:

```bash
python src/train_for_optuna.py \
    --c src/config/config_her.yaml \
    --o debug_trial \
    --id reach \
    --her \
    --no-wandb
```

## Performance Considerations

- Each trial creates a separate directory and trains a model
- **Real-time Progress**: You can now see tqdm progress bars showing epoch/cycle progress for each trial
- Reduce `--epochs` and `--envs` for faster trials during initial testing
- Training runs directly in Python (no subprocess overhead)
- Wandb logging is automatically disabled during optimization
- Consider running on GPU-enabled machines for faster training

## Contributing

To add new hyperparameters to optimize:

1. Mark them with `#OPT Target` in `config_her.yaml`
2. Add them to the `suggest_hyperparameters` method
3. Update the `create_trial_config` method to apply the new parameters

Happy optimizing! 🐼🤖 