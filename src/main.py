import argparse
from src.env import GoalEnvHER

HER_MAPPING = {
    "reach": "PandaReach-v3",
    "push": "PandaPush-v3",
    "slide": "PandaSlide-v3",
    "pickplace": "PandaPickAndPlace-v3",
}


def main(args):
    env = GoalEnvHER(
        env_id=HER_MAPPING[args.id],
        seed=args.seed,
        config=args.c,
        num_envs=args.nenv,
        weights=args.w,
        verbose=args.verbose,
        use_wandb=not args.no_wandb,
        agent_type=args.agent,
        her=args.her,
    )

    if args.mode == "train":
        env.train(args.o)
    elif args.mode == "test":
        env.test(args.o, args.neps)
    else:
        raise ValueError("[ERROR] invalid mode chosen for running main script")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Helper for Pandas Environment")
    parser.add_argument(
        "--id", type=str, default="reach", choices=list(HER_MAPPING.keys())
    )
    parser.add_argument(
        "--c", type=str, required=True, help="Path to config file for environment"
    )
    parser.add_argument(
        "--o", type=str, required=True, help="Path to output directory for environment"
    )
    parser.add_argument(
        "--w",
        type=str,
        help="Path to directory containing actor.pth, critic_1.pth, critic_2.pth",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode to run environment in",
    )
    parser.add_argument(
        "--nenv",
        type=int,
        default=32,
        help="Number of parallel environments to run concurrently",
    )
    parser.add_argument(
        "--neps", type=int, default=10, help="Number of episodes to run for testing"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging and rendering"
    )
    parser.add_argument(
        "--her", action="store_true", help="Initialize HER based environment"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="TD3",
        choices=["TD3", "SAC", "TQC", "DDPG"],
        help="Agent type to use",
    )
    parser.add_argument(
        "--seed", type=int, default=1898, help="Seed for reproducibility"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    main(args)
