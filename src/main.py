import argparse
from src.env import PandasEnv, PandasHEREnv

MAPPING = {
    "reach": "PandaReachDense-v3",
    "push": "PandaPushDense-v3",
    "slide": "PandaSlideDense-v3",
    "pickplace": "PandaPickAndPlaceDense-v3",
    "stack": "PandaStackDense-v3",
}

HER_MAPPING = {
    "reach": "PandaReach-v3",
    "push": "PandaPush-v3",
    "slide": "PandaSlide-v3",
    "pickplace": "PandaPickAndPlace-v3",
    "stack": "PandaStack-v3",
}

def main(args):
    if args.her: 
        env = PandasHEREnv(HER_MAPPING[args.id], args.seed, args.c, args.nenv, args.w, args.verbose, not args.no_wandb)
    else:
        env = PandasEnv(MAPPING[args.id], args.seed, args.c, args.nenv, args.w, args.verbose, not args.no_wandb)
        
    if args.mode == "train":
        env.train(args.o)
    elif args.mode == "test":
        env.test(args.o, args.neps)
    else: 
        raise ValueError("[ERROR] invalid mode chosen for running main script")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Helper for Pandas Environment")
    parser.add_argument("--id", type=str, default="reach", choices=list(MAPPING.keys()))
    parser.add_argument("--c", type=str, required=True, help="Path to config file for environment")
    parser.add_argument("--o", type=str, required=True, help="Path to output directory for environment")
    parser.add_argument("--w", type=str, help="Path to directory containing actor.pth, critic_1.pth, critic_2.pth")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode to run environment in")
    parser.add_argument("--nenv", type=int, default=32, help="Number of parallel environments to run concurrently")
    parser.add_argument("--neps", type=int, default=10, help="Number of episodes to run for testing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and rendering")
    parser.add_argument("--her", action="store_true", help="Initialize HER based environment")
    parser.add_argument("--seed", type=int, default=1898, help="Seed for reproducibility")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    main(args)