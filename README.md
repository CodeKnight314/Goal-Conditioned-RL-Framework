# Panda Manipulation RL

This project implements a reinforcement learning environment for controlling a Panda robot arm for manipulation tasks using TD3 (Twin Delayed Deep Deterministic Policy Gradient) and SAC (Soft Actor-Critic) algorithms with optional Hindsight Experience Replay (HER).

## Features

- **Multiple RL Algorithms**: Support for both TD3 and SAC algorithms
- **Hindsight Experience Replay (HER)**: Improved sample efficiency for sparse reward environments
- **Multiple Tasks**: Support for reach, push, slide, and pick-and-place manipulation tasks
- **Parallel Environments**: Vectorized environment support for faster training
- **WandB Integration**: Comprehensive logging and visualization
- **Configurable**: Flexible configuration system for hyperparameter tuning

## Usage

### Training with TD3 + HER (default)

```bash
python -m src.main --id reach --c src/config/config_her.yaml --o ./outputs/td3_her_reach --her --agent TD3 --mode train --nenv 32
```

### Training with SAC + HER (recommended for sparse rewards)

```bash
python -m src.main --id reach --c src/config/config_sac_her.yaml --o ./outputs/sac_her_reach --her --agent SAC --mode train --nenv 32
```

### Testing a trained model

```bash
python -m src.main --id reach --c src/config/config_sac_her.yaml --o ./outputs/test_videos --w ./outputs/sac_her_reach/best_weights --agent SAC --mode test --neps 10
```

## Configuration Files

- `src/config/config.yaml`: Standard TD3 configuration
- `src/config/config_her.yaml`: TD3 with HER configuration  
- `src/config/config_sac_her.yaml`: SAC with HER configuration (new)

## SAC vs TD3

**SAC (Soft Actor-Critic)**:
- Stochastic policy with entropy regularization
- Better exploration and sample efficiency
- Automatic temperature (entropy) tuning
- More stable training
- Recommended for sparse reward environments

**TD3 (Twin Delayed DDPG)**:
- Deterministic policy with noise injection
- Target policy smoothing
- Delayed policy updates
- Good for continuous control tasks

## Command Line Arguments

- `--id`: Task type (reach, push, slide, pickplace)
- `--c`: Path to configuration file
- `--o`: Output directory for models and logs
- `--w`: Path to pretrained weights (for testing/resuming)
- `--agent`: Agent type (TD3 or SAC)
- `--mode`: Mode (train or test)
- `--nenv`: Number of parallel environments
- `--her`: Enable Hindsight Experience Replay
- `--verbose`: Enable verbose logging
- `--no-wandb`: Disable WandB logging

## Example Training Commands

### SAC + HER for different tasks:

```bash
# Reach task
python -m src.main --id reach --c src/config/config_sac_her.yaml --o ./outputs/sac_reach --her --agent SAC

# Push task  
python -m src.main --id push --c src/config/config_sac_her.yaml --o ./outputs/sac_push --her --agent SAC

# Pick and place task
python -m src.main --id pickplace --c src/config/config_sac_her.yaml --o ./outputs/sac_pickplace --her --agent SAC
```