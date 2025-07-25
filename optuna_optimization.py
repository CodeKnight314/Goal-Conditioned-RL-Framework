#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization Script for Panda Manipulation RL

This script optimizes hyperparameters for different agent types (TD3/SAC) and environment types (regular/HER)
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

sys.path.insert(0, os.getcwd())

from src.env import PandasEnv, PandasHEREnv
from src.utils import load_config, load_her_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                 agent_type: str = "TD3", use_her: bool = False, max_epochs: int = 10, 
                 num_envs: int = 32, seed: int = 1898):
        """
        Initialize the Optuna hyperparameter optimizer.
        
        Args:
            base_config_path: Path to the base config file
            output_dir: Directory to store optimization results and trial configs
            task: Task type (reach, push, slide, pickplace)
            agent_type: Agent type (TD3 or SAC)
            use_her: Whether to use HER environment
            max_epochs: Number of epochs to train for each trial (reduced for faster optimization)
            num_envs: Number of parallel environments
            seed: Random seed for reproducibility
        """
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.task = task
        self.agent_type = agent_type
        self.use_her = use_her
        self.max_epochs = max_epochs
        self.num_envs = num_envs
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load base config using appropriate loader
        if self.use_her:
            self.base_config = load_her_config(base_config_path)
        else:
            self.base_config = load_config(base_config_path)
        
        logger.info(f"Initialized optimizer with base config: {base_config_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Task: {task}, Agent: {agent_type}, HER: {use_her}, Max epochs per trial: {max_epochs}")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space using discrete values, adapted for agent type.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        hyperparams = {
            'gradient_step': trial.suggest_categorical('gradient_step', [1, 2, 4, 8, 16, 32, 40]),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            'actor_lr': trial.suggest_categorical('actor_lr', [0.0001, 0.00025, 0.0003, 0.001, 0.003]),
            'actor_lr_min': trial.suggest_categorical('actor_lr_min', [0.00001, 0.00003, 0.0001, 0.00025]),
            'critic_lr': trial.suggest_categorical('critic_lr', [0.0001, 0.00025, 0.0003, 0.001, 0.003]),
            'critic_lr_min': trial.suggest_categorical('critic_lr_min', [0.00001, 0.00003, 0.0001, 0.00025]),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        
        # Add agent-specific hyperparameters
        if self.agent_type == "SAC":
            hyperparams['alpha_lr'] = trial.suggest_categorical('alpha_lr', [0.0001, 0.0003, 0.001, 0.003])
            hyperparams['tau'] = trial.suggest_categorical('tau', [0.001, 0.003, 0.005, 0.01, 0.02])
        elif self.agent_type == "TD3":
            hyperparams['policy_noise'] = trial.suggest_categorical('policy_noise', [0.1, 0.2, 0.3])
            hyperparams['noise_clamp'] = trial.suggest_categorical('noise_clamp', [0.3, 0.5, 0.7])
            hyperparams['ac_update_freq'] = trial.suggest_categorical('ac_update_freq', [1, 2, 4])
        
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
        
        # Update general hyperparameters
        trial_config['gradient_step'] = hyperparams['gradient_step']
        trial_config['agent']['hidden_dim'] = hyperparams['hidden_dim']
        trial_config['agent']['actor_lr'] = hyperparams['actor_lr']
        trial_config['agent']['actor_lr_min'] = hyperparams['actor_lr_min']
        trial_config['agent']['critic_lr'] = hyperparams['critic_lr']
        trial_config['agent']['critic_lr_min'] = hyperparams['critic_lr_min']
        trial_config['agent']['batch_size'] = hyperparams['batch_size']
        
        # Update agent-specific hyperparameters
        if self.agent_type == "SAC":
            trial_config['agent']['alpha_lr'] = hyperparams['alpha_lr']
            trial_config['agent']['tau'] = hyperparams['tau']
        elif self.agent_type == "TD3":
            trial_config['agent']['policy_noise'] = hyperparams['policy_noise']
            trial_config['agent']['noise_clamp'] = hyperparams['noise_clamp']
            trial_config['agent']['ac_update_freq'] = hyperparams['ac_update_freq']
        
        # Set max_epoch for faster optimization (for HER environments that have this parameter)
        if self.use_her and 'max_epoch' in trial_config:
            trial_config['max_epoch'] = self.max_epochs
        
        # Save trial config
        config_filename = "config_her.yaml" if self.use_her else "config.yaml"
        config_path = trial_dir / config_filename
        with open(config_path, 'w') as f:
            yaml.dump(trial_config, f, default_flow_style=False)
        
        logger.info(f"Created config for trial {trial_number} at {config_path}")
        return str(config_path)

    def run_training(self, trial_number: int, config_path: str) -> float:
        """
        Run training for a single trial using the appropriate environment class.
        
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
            
            # Choose environment mapping based on HER flag
            env_mapping = HER_MAPPING if self.use_her else MAPPING
            env_id = env_mapping[self.task]
            
            # Create appropriate environment
            if self.use_her:
                env = PandasHEREnv(
                    env_id=env_id,
                    seed=self.seed,
                    config=config_path,
                    num_envs=self.num_envs,
                    weights=None,
                    verbose=False,
                    use_wandb=False,
                    agent_type=self.agent_type
                )
            else:
                env = PandasEnv(
                    env_id=env_id,
                    seed=self.seed,
                    config=config_path,
                    num_envs=self.num_envs,
                    weights=None,
                    verbose=False,
                    use_wandb=False,
                    agent_type=self.agent_type
                )
            
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
        hyperparams = self.suggest_hyperparameters(trial)
        
        logger.info(f"Trial {trial.number} hyperparameters: {hyperparams}")
        config_path = self.create_trial_config(trial.number, hyperparams)
        
        success_rate = self.run_training(trial.number, config_path)
        
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
        study_name_full = f"{study_name}_{self.agent_type}_{self.task}_{'her' if self.use_her else 'regular'}"
        
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name_full,
            storage=f"sqlite:///{self.output_dir}/optuna_study.db",
            load_if_exists=True
        )
        
        logger.info(f"Starting optimization with {n_trials} trials")
        logger.info(f"Study name: {study_name_full}")
        
        study.optimize(self.objective, n_trials=n_trials)
        self.save_results(study)
        
        return study

    def save_results(self, study: optuna.Study):
        """
        Save optimization results and best configuration.
        
        Args:
            study: Completed Optuna study
        """
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            "best_success_rate": best_value,
            "best_hyperparameters": best_params,
            "best_trial_number": study.best_trial.number,
            "total_trials": len(study.trials),
            "agent_type": self.agent_type,
            "task": self.task,
            "use_her": self.use_her
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
        optimized_config['agent']['batch_size'] = best_params['batch_size']
        
        # Add agent-specific optimized parameters
        if self.agent_type == "SAC":
            optimized_config['agent']['alpha_lr'] = best_params['alpha_lr']
            optimized_config['agent']['tau'] = best_params['tau']
        elif self.agent_type == "TD3":
            optimized_config['agent']['policy_noise'] = best_params['policy_noise']
            optimized_config['agent']['noise_clamp'] = best_params['noise_clamp']
            optimized_config['agent']['ac_update_freq'] = best_params['ac_update_freq']
        
        # Restore original max_epoch for HER environments
        if self.use_her and 'max_epoch' in self.base_config:
            optimized_config['max_epoch'] = self.base_config['max_epoch']
        
        config_filename = f"optimized_config_{'her' if self.use_her else 'regular'}_{self.agent_type.lower()}_{self.task}.yaml"
        optimized_config_path = self.output_dir / config_filename
        with open(optimized_config_path, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        
        logger.info(f"Optimization completed!")
        logger.info(f"Best success rate: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Optimized config saved to: {optimized_config_path}")


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for Panda Manipulation RL")
    parser.add_argument("--config", type=str, default="src/config/config_sac_reach.yaml", 
                       help="Path to base config file")
    parser.add_argument("--output", type=str, default="optuna_results", 
                       help="Output directory for optimization results")
    parser.add_argument("--task", type=str, default="reach", 
                       choices=["reach", "push", "slide", "pickplace"],
                       help="Task to optimize for")
    parser.add_argument("--agent", type=str, default="SAC", 
                       choices=["TD3", "SAC"],
                       help="Agent type to optimize for")
    parser.add_argument("--her", action="store_true", 
                       help="Use HER environment instead of regular environment")
    parser.add_argument("--trials", type=int, default=50, 
                       help="Number of optimization trials")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of epochs per trial (reduced for faster optimization)")
    parser.add_argument("--envs", type=int, default=32, 
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=1898, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    optimizer = OptunaHyperparameterOptimizer(
        base_config_path=args.config,
        output_dir=args.output,
        task=args.task,
        agent_type=args.agent,
        use_her=args.her,
        max_epochs=args.epochs,
        num_envs=args.envs,
        seed=args.seed
    )
    
    study = optimizer.optimize(n_trials=args.trials)
    
    print(f"\nOptimization completed!")
    print(f"Best success rate: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 