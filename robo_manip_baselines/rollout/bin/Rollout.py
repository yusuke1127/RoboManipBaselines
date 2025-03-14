import argparse
import importlib
import re
import sys

import yaml


def camel_to_snake(name):
    """Converts camelCase or PascalCase to snake_case (also converts the first letter to lowercase)"""
    name = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", name
    )  # Insert '_' between a lowercase/number and an uppercase letter
    name = re.sub(
        r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name
    )  # Insert '_' between consecutive uppercase letters followed by a lowercase letter
    name = name[0].lower() + name[1:]  # Convert the first letter to lowercase
    return name.lower()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a meta argument parser for the rollout switching between different policies and environments. The actual arguments are handled by another internal argument parser.",
        add_help=False,
    )
    parser.add_argument(
        "policy",
        type=str,
        choices=["Mlp", "Sarnn", "Act", "DiffusionPolicy"],
        help="policy",
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

    operation_module = importlib.import_module(
        f"robo_manip_baselines.envs.operation.Operation{args.env}"
    )
    OperationEnvClass = getattr(operation_module, f"Operation{args.env}")

    policy_module = importlib.import_module(
        f"robo_manip_baselines.{camel_to_snake(args.policy)}"
    )
    RolloutPolicyClass = getattr(policy_module, f"Rollout{args.policy}")

    # The order of parent classes must not be changed in order to maintain the method resolution order (MRO)
    class Rollout(OperationEnvClass, RolloutPolicyClass):
        pass

    if args.config is None:
        config = {}
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    rollout = Rollout(**config)
    rollout.run()


if __name__ == "__main__":
    main()
