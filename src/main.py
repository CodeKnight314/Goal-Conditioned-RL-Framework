import argparse
from src.env import PandasEnv

MAPPING = {
    "reach": "PandaReachJointsDense-v3",
    "push": "PandaPushJointsDense-v3",
    "slide": "PandaSlideJointsDense-v3",
    "pickplace": "PandaPickAndPlaceJointsDense-v3",
    "stack": "PandaStackJointsDense-v3",
}

def main(args):
    env = PandasEnv(MAPPING[args.id], args.c, args.n, args.w, args.verbose)
    if args.mode == "train":
        env.train(args.o)
    elif args.mode == "test":
        env.test(args.o, args.num_episodes)
    else: 
        raise ValueError("[ERROR] invalid mode chosen for running main script")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Helper for Pandas Environment")
    parser.add_argument("--id", type=str, default="reach", choices=list(MAPPING.keys()))
    parser.add_argument("--c", type=str, required=True, help="Path to config file for environment")
    parser.add_argument("--o", type=str, required=True, help="Path to output directory for environment")
    parser.add_argument("--w", type=str, help="Path to directory containing actor.pth, critic_1.pth, critic_2.pth")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode to run environment in")
    parser.add_argument("--n", type=int, default=32, help="Number of parallel environments to run concurrently")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run for testing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and rendering")
    args = parser.parse_args()