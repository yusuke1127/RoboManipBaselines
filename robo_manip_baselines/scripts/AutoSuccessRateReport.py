import argparse
import os
import re
import subprocess
import tempfile
import venv


class AutoSuccessRateReport:
    REPOSITORY_NAME = "RoboManipBaselines"
    GIT_CLONE_URL = f"https://github.com/isri-aist/{REPOSITORY_NAME}.git"
    THIRD_PARTY_PATHS = {
        "Sarnn": ["eipl"],
        "Act": ["act", "detr"],
        "DiffusionPolicy": ["diffusion_policy"],
    }

    def __init__(self, policy, env, commit_id, dataset_url):
        self.policy = policy
        self.env = env
        self.commit_id = commit_id
        self.dataset_url = dataset_url

    @classmethod
    def run_command(cls, command, cwd=None):
        print(f"[{cls.__name__}] Executing command: {command}", flush=True)
        with subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

    def start(self):
        temp_dir = tempfile.mkdtemp()
        print(f"[{self.__class__.__name__}] Temporary directory created at: {temp_dir}")

        repository_dir = os.path.join(temp_dir, self.REPOSITORY_NAME)
        self.run_command(
            ["git", "clone", "--recursive", self.GIT_CLONE_URL, repository_dir]
        )

        self.run_command(
            ["git", "switch", "-c", self.commit_id],
            cwd=repository_dir,
        )

        venv_dir = os.path.join(temp_dir, "venv")
        venv.create(venv_dir, with_pip=True)
        venv_python = os.path.join(venv_dir, "bin", "python")
        print(f"[{self.__class__.__name__}] Virtual environment created at: {venv_dir}")

        # TODO:
        #     apt install is assumed to have already been finished,
        #     but this should be checked and terminated if not installed.

        # common install
        self.run_command(
            [venv_python, "-m", "pip", "install", "-e", "."],
            cwd=repository_dir,
        )

        # install of each policy
        if self.policy in self.THIRD_PARTY_PATHS:
            self.run_command(
                [
                    venv_python,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    f'.[{camel_to_snake(self.policy).replace("_","-")}]',
                ],
                cwd=repository_dir,
            )
            self.run_command(
                [venv_python, "-m", "pip", "install", "-e", "."],
                cwd=os.path.join(
                    repository_dir, "third_party", *self.THIRD_PARTY_PATHS[self.policy]
                ),
            )


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


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a parser for the evaluation based on a specific commit.",
    )
    parser.add_argument(
        "policy",
        type=str,
        choices=["Mlp", "Sarnn", "Act", "DiffusionPolicy"],
        help="policy",
    )
    parser.add_argument(
        "env",
        type=str,
        help="environment",
    )
    parser.add_argument("-c", "--commit_id", type=str, required=True)
    parser.add_argument(
        "-d",
        "--dataset_url",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    success_report = AutoSuccessRateReport(
        args.policy, args.env, args.commit_id, args.dataset_url
    )
    success_report.start()
