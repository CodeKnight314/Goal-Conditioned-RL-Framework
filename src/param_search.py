import optuna
from src.env import GoalEnvHER
import os
import yaml
import argparse
from src.main import HER_MAPPING


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(path: str, config: dict):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def objective(
    trial: optuna.trial.Trial,
    agent_type: str,
    env_id: str,
    config_path: str,
    save_path: str,
    alpha: bool = False,
):
    param_ranges = {
        "actor_lr": {"low": 1e-6, "high": 1e-3, "log": True},
        "critic_lr": {"low": 1e-6, "high": 1e-3, "log": True},
        "hidden_dim": [64, 128, 192, 256, 320, 384, 448, 512],
        "gamma": {"low": 0.9, "high": 0.99, "step": 0.01},
        "k": {"low": 2, "high": 16},
    }

    actor_lr = trial.suggest_float("actor_lr", **param_ranges["actor_lr"])
    critic_lr = trial.suggest_float("critic_lr", **param_ranges["critic_lr"])
    hidden_dim = trial.suggest_categorical("hidden_dim", param_ranges["hidden_dim"])
    gamma = trial.suggest_float("gamma", **param_ranges["gamma"])
    k = trial.suggest_int("k", **param_ranges["k"])

    default_config = load_config(config_path)

    default_config["max_epoch"] = 8
    default_config["agent"]["actor_lr"] = actor_lr
    default_config["agent"]["actor_lr_min"] = actor_lr
    default_config["agent"]["critic_lr"] = critic_lr
    default_config["agent"]["critic_lr_min"] = critic_lr
    default_config["agent"]["hidden_dim"] = hidden_dim
    default_config["agent"]["gamma"] = gamma
    default_config["agent"]["k_future"] = k
    if alpha:
        alpha_lr = trial.suggest_float("alpha_lr", low=1e-6, high=1e-3, log=True)
        default_config["agent"]["alpha_lr"] = alpha_lr

    trial_dir = os.path.join(save_path, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    file_path = os.path.join(trial_dir, f"{agent_type}_config.yaml")
    save_config(file_path, default_config)

    def on_epoch(epoch: int, score: float):
        trial.report(score, step=epoch)

        if trial.should_prune():
            print(f"Trial {trial.number} pruned.")
            raise optuna.TrialPruned()

    env = GoalEnvHER(
        env_id,
        seed=1898,
        config=file_path,
        num_envs=16,
        verbose=False,
        agent_type=agent_type,
        her=True,
        use_wandb=False,
    )

    try:
        results = env.train(trial_dir, on_epoch)
    except optuna.TrialPruned:
        env._finalise_training(trial_dir)
        raise
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with Optuna"
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["SAC", "TD3", "TQC", "DDPG"],
        help="Agent type",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["reach", "push", "slide", "pickplace"],
        help="Environment task",
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--study-name", type=str, default=None, help="Optuna study name"
    )
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")

    args = parser.parse_args()

    config_path = f"src/config/{args.agent}/config_{args.agent.lower()}_{args.env}.yaml"
    save_path = f"../param_search/{args.agent}_{args.env}"

    os.makedirs(save_path, exist_ok=True)

    study_name = args.study_name or f"{args.agent}_{args.env}_optimization"

    if args.storage is None:
        db_path = os.path.join(save_path, "optuna_study.db")
        args.storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
    )

    alpha_agents = {"SAC", "TQC"}
    use_alpha = args.agent in alpha_agents

    env_id = HER_MAPPING[args.env]

    def print_best_trial():
        print("\n" + "=" * 50)
        print("Current Best Trial Summary")
        print("=" * 50)
        print(f"Best Trial Number: {study.best_trial.number}")
        print(f"Best Score: {study.best_value:.4f}")
        print("\nBest Parameters:")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
        print("=" * 50 + "\n")

    def callback(study: optuna.study.Study, trial: optuna.trial.Trial):
        print_best_trial()

    study.optimize(
        lambda trial: objective(
            trial, args.agent, env_id, config_path, save_path, use_alpha
        ),
        n_trials=args.trials,
        callbacks=[callback],
    )

    print("\nOptimization Completed!")
    print_best_trial()

    best_config_path = os.path.join(
        save_path, f"trial_{study.best_trial.number:04d}", f"{args.agent}_config.yaml"
    )
    final_config_path = os.path.join(
        save_path, f"best_{args.agent.lower()}_{args.env}_config.yaml"
    )

    best_config = load_config(best_config_path)
    save_config(final_config_path, best_config)
    print(f"Best configuration saved to: {final_config_path}")


if __name__ == "__main__":
    main()
