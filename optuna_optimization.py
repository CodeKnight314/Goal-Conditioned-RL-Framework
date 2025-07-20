#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization Script for Panda Manipulation RL

This script optimizes the hyperparameters marked with '#OPT Target' in config_her.yaml
using discrete search spaces and creates separate config files for each trial.
"""

import os
import sys
import yaml
import argparse
import optuna
from typing import Dict, Any
import logging
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, os.getcwd())

# Import the environment classes
from src.env import PandasEnv, PandasHEREnv
from src.utils import load_her_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping for task names to environment IDs
MAPPING = {
    "reach": "FetchReachDense-v4",
    "push": "FetchPushDense-v4",
    "slide": "FetchSlideDense-v4",
    "pickplace": "FetchPickAndPlaceDense-v4",
}

HER_MAPPING = {
    "reach": "FetchReach-v4",
    "push": "FetchPush-v4",
    "slide": "FetchSlide-v4",
    "pickplace": "FetchPickAndPlace-v4",
}

class OptunaHyperparameterOptimizer:
    def __init__(self, base_config_path: str, output_dir: str, task: str = "reach", 
                 max_epochs: int = 10, num_envs: int = 32, seed: int = 1898):
        """
        Initialize the Optuna hyperparameter optimizer.
        
        Args:
            base_config_path: Path to the base config_her.yaml file
            output_dir: Directory to store optimization results and trial configs
            task: Task type (reach, push, slide, pickplace)
            max_epochs: Number of epochs to train for each trial (reduced for faster optimization)
            num_envs: Number of parallel environments
            seed: Random seed for reproducibility
        """
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.task = task
        self.max_epochs = max_epochs
        self.num_envs = num_envs
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        logger.info(f"Initialized optimizer with base config: {base_config_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Task: {task}, Max epochs per trial: {max_epochs}")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space using discrete values.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # Define discrete search spaces for parameters marked with '#OPT Target'
        hyperparams = {
            # Training parameters
            'gradient_step': trial.suggest_categorical('gradient_step', [20, 30, 40, 50, 60]),
            
            # Agent architecture
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            
            # Learning rates
            'actor_lr': trial.suggest_categorical('actor_lr', [0.0001, 0.0003, 0.001, 0.003, 0.01]),
            'actor_lr_min': trial.suggest_categorical('actor_lr_min', [0.00001, 0.00003, 0.0001, 0.0003]),
            'critic_lr': trial.suggest_categorical('critic_lr', [0.0001, 0.0003, 0.001, 0.003, 0.01]),
            'critic_lr_min': trial.suggest_categorical('critic_lr_min', [0.00001, 0.00003, 0.0001, 0.0003]),
            
            # Noise parameters (discrete values between 0.1 and 0.7, multiples of 0.1)
            'noise_std': trial.suggest_categorical('noise_std', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            'policy_noise': trial.suggest_categorical('policy_noise', [0.1, 0.15, 0.2, 0.25, 0.3]),
            
            # Soft update parameter
            'tau': trial.suggest_categorical('tau', [0.001, 0.003, 0.005, 0.01, 0.02]),
        }
        
        return hyperparams

    def create_trial_config(self, trial_number: int, hyperparams: Dict[str, Any]) -> str:
        """
        Create a config file for the current trial with suggested hyperparameters.
        
        Args:
            trial_number: Current trial number
            hyperparams: Dictionary of hyperparameters
            
        Returns:
            Path to the created config file
        """
        # Create trial directory
        trial_dir = self.output_dir / f"trial_{trial_number:04d}"
        trial_dir.mkdir(exist_ok=True)
        
        # Copy base config and modify hyperparameters
        trial_config = self.base_config.copy()
        
        # Reduce max_epoch for faster optimization
        trial_config['max_epoch'] = self.max_epochs
        
        # Update hyperparameters marked with '#OPT Target'
        trial_config['gradient_step'] = hyperparams['gradient_step']
        trial_config['agent']['hidden_dim'] = hyperparams['hidden_dim']
        trial_config['agent']['actor_lr'] = hyperparams['actor_lr']
        trial_config['agent']['actor_lr_min'] = hyperparams['actor_lr_min']
        trial_config['agent']['critic_lr'] = hyperparams['critic_lr']
        trial_config['agent']['critic_lr_min'] = hyperparams['critic_lr_min']
        trial_config['agent']['noise_std'] = hyperparams['noise_std']
        trial_config['agent']['policy_noise'] = hyperparams['policy_noise']
        trial_config['agent']['tau'] = hyperparams['tau']
        
        # Save trial config
        config_path = trial_dir / "config_her.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(trial_config, f, default_flow_style=False)
        
        logger.info(f"Created config for trial {trial_number} at {config_path}")
        return str(config_path)

    def run_training(self, trial_number: int, config_path: str) -> float:
        """
        Run training for a single trial directly using environment classes.
        
        Args:
            trial_number: Current trial number
            config_path: Path to trial config file
            
        Returns:
            Final success rate achieved during training
        """
        trial_dir = self.output_dir / f"trial_{trial_number:04d}"
        
        try:
            logger.info(f"Starting training for trial {trial_number}")
            logger.info(f"Config: {config_path}")
            logger.info(f"Output: {trial_dir}")
            
            # Create environment directly using HER (since we're optimizing HER configs)
            env = PandasHEREnv(
                env_id=HER_MAPPING[self.task],
                seed=self.seed,
                config=config_path,
                num_envs=self.num_envs,
                weights=None,
                verbose=False,  # Reduce verbosity during optimization
                use_wandb=False  # Disable wandb during optimization
            )
            
            # Run training - this will show tqdm progress bars
            final_success_rate = env.train(str(trial_dir))
            
            logger.info(f"Trial {trial_number} completed - Success rate: {final_success_rate:.4f}")
            return final_success_rate
            
        except Exception as e:
            logger.error(f"Error running trial {trial_number}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Success rate to maximize
        """
        # Get suggested hyperparameters
        hyperparams = self.suggest_hyperparameters(trial)
        
        # Log hyperparameters
        logger.info(f"Trial {trial.number} hyperparameters: {hyperparams}")
        
        # Create trial config
        config_path = self.create_trial_config(trial.number, hyperparams)
        
        # Run training and get success rate
        success_rate = self.run_training(trial.number, config_path)
        
        # Log trial result
        trial.set_user_attr("config_path", config_path)
        trial.set_user_attr("hyperparameters", hyperparams)
        
        logger.info(f"Trial {trial.number} completed with success rate: {success_rate}")
        
        return success_rate

    def optimize(self, n_trials: int = 50, study_name: str = "panda_hyperopt"):
        """
        Run the optimization study.
        
        Args:
            n_trials: Number of trials to run
            study_name: Name for the Optuna study
        """
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{self.output_dir}/optuna_study.db",
            load_if_exists=True
        )
        
        logger.info(f"Starting optimization with {n_trials} trials")
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Save results
        self.save_results(study)
        
        return study

    def save_results(self, study: optuna.Study):
        """
        Save optimization results and best configuration.
        
        Args:
            study: Completed Optuna study
        """
        # Save best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            "best_success_rate": best_value,
            "best_hyperparameters": best_params,
            "best_trial_number": study.best_trial.number,
            "total_trials": len(study.trials)
        }
        
        results_path = self.output_dir / "optimization_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Create optimized config
        optimized_config = self.base_config.copy()
        optimized_config['gradient_step'] = best_params['gradient_step']
        optimized_config['agent']['hidden_dim'] = best_params['hidden_dim']
        optimized_config['agent']['actor_lr'] = best_params['actor_lr']
        optimized_config['agent']['actor_lr_min'] = best_params['actor_lr_min']
        optimized_config['agent']['critic_lr'] = best_params['critic_lr']
        optimized_config['agent']['critic_lr_min'] = best_params['critic_lr_min']
        optimized_config['agent']['noise_std'] = best_params['noise_std']
        optimized_config['agent']['policy_noise'] = best_params['policy_noise']
        optimized_config['agent']['tau'] = best_params['tau']
        
        # Restore original max_epoch for final config
        optimized_config['max_epoch'] = self.base_config['max_epoch']
        
        optimized_config_path = self.output_dir / "optimized_config_her.yaml"
        with open(optimized_config_path, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        
        logger.info(f"Optimization completed!")
        logger.info(f"Best success rate: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Optimized config saved to: {optimized_config_path}")


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for Panda Manipulation RL")
    parser.add_argument("--config", type=str, default="src/config/config_her.yaml", 
                       help="Path to base config file")
    parser.add_argument("--output", type=str, default="optuna_results", 
                       help="Output directory for optimization results")
    parser.add_argument("--task", type=str, default="reach", 
                       choices=["reach", "push", "slide", "pickplace"],
                       help="Task to optimize for")
    parser.add_argument("--trials", type=int, default=50, 
                       help="Number of optimization trials")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of epochs per trial (reduced for faster optimization)")
    parser.add_argument("--envs", type=int, default=32, 
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=1898, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = OptunaHyperparameterOptimizer(
        base_config_path=args.config,
        output_dir=args.output,
        task=args.task,
        max_epochs=args.epochs,
        num_envs=args.envs,
        seed=args.seed
    )
    
    # Run optimization
    study = optimizer.optimize(n_trials=args.trials)
    
    print(f"\nOptimization completed!")
    print(f"Best success rate: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 