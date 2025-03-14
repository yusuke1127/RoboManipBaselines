import argparse
import importlib
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a meta argument parser for the teleop switching between different environments. The actual arguments are handled by another internal argument parser.",
        add_help=False,
    )
    parser.add_argument("env", type=str, help="environment")
    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and continue"
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.help:
        parser.print_help()
        sys.argv += ["--help"]

    if "Isaac" in args.env:
        from isaacgym import (
            gymapi,  # noqa: F401
            gymtorch,  # noqa: F401
            gymutil,  # noqa: F401
        )
    if args.env.endswith("Vec"):
        from robo_manip_baselines.teleop import TeleopBaseVec as TeleopBase
    else:
        from robo_manip_baselines.teleop import TeleopBase

    operation_module = importlib.import_module(
        f"robo_manip_baselines.envs.operation.Operation{args.env}"
    )
    OperationEnvClass = getattr(operation_module, f"Operation{args.env}")

    class Teleop(OperationEnvClass, TeleopBase):
        pass

    if args.config is None:
        config = {}
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    teleop = Teleop(**config)
    teleop.run()


if __name__ == "__main__":
    main()
