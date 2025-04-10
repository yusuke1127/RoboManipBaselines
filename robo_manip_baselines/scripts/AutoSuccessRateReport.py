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
        self.dataset_url = self.adjust_dataset_url(dataset_url)
        self.repository_tmp_dir = os.path.join(tempfile.mkdtemp(), self.REPOSITORY_NAME)
        self.venv_python = os.path.join(tempfile.mkdtemp(), "venv/bin/python")
        self.dataset_temp_dir = tempfile.mkdtemp()

    @classmethod
    def adjust_dataset_url(cls, dataset_url):

        # escape '&'
        if "&" in dataset_url and "\\&" not in dataset_url:
            dataset_url = dataset_url.replace("&", "\\&")

        # dl=0 → dl=1
        if dataset_url.endswith("dl=0"):
            print(f"[{cls.__name__}] The URL ends with 'dl=0'. Changing it to 'dl=1'.")
            dataset_url = dataset_url[: -len("dl=0")] + "dl=1"

        assert dataset_url.endswith(
            "dl=1"
        ), f"[{cls.__name__}]Error: The URL '{dataset_url}' does not end with 'dl=1'."
        return dataset_url

    @classmethod
    def run_command(cls, command, cwd=None):
        print(f"[{cls.__name__}] Executing command: {command}", flush=True)
        with subprocess.Popen(
            command,
            cwd=cwd,
            shell=False,  # secure default: using list avoids shell injection risks
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
        # git clone
        self.run_command(
            ["git", "clone", "--recursive", self.GIT_CLONE_URL, self.repository_tmp_dir]
        )
        self.run_command(
            ["git", "switch", "-c", self.commit_id],
            cwd=self.repository_tmp_dir,
        )

        # venv creation
        venv.create(os.path.join(self.venv_python, "../../../venv/"), with_pip=True)

        # TODO:
        #     apt install is assumed to have already been finished,
        #     but this should be checked and terminated if not installed.

        # install common
        self.run_command(
            [self.venv_python, "-m", "pip", "install", "-e", "."],
            cwd=self.repository_tmp_dir,
        )

        # install each policy
        if self.policy in self.THIRD_PARTY_PATHS:
            self.run_command(
                [
                    self.venv_python,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    ".[" + camel_to_snake(self.policy).replace("_", "-") + "]",
                ],
                cwd=self.repository_tmp_dir,
            )
            self.run_command(
                [self.venv_python, "-m", "pip", "install", "-e", "."],
                cwd=os.path.join(
                    self.repository_tmp_dir,
                    "third_party/",
                    *self.THIRD_PARTY_PATHS[self.policy],
                ),
            )

        # download dataset
        wget_zip_path = os.path.join(self.dataset_temp_dir, "dataset.zip")
        self.run_command(["wget", "-O", wget_zip_path, self.dataset_url])
        self.run_command(
            [
                "unzip",
                "-d",
                os.path.join(self.dataset_temp_dir, "dataset/"),
                wget_zip_path,
            ]
        )

        # train
        self.run_command(
            [
                self.venv_python,
                os.path.join(self.repository_tmp_dir, "bin/Train.py"),
                self.policy,
                self.env,
                "--dataset_dir",
                os.path.join(self.dataset_temp_dir, "dataset/<dataset_name>"),
                "--checkpoint_dir",
                os.path.join(
                    self.repository_tmp_dir, "checkpoint_dir/", self.policy, self.env
                ),
            ]
        )

        # rollout
        self.run_command(
            [
                self.venv_python,
                os.path.join(self.repository_tmp_dir, "bin/Rollout.py"),
                self.policy,
                self.env,
                "--checkpoint_dir",
                os.path.join(
                    self.repository_tmp_dir,
                    "checkpoint_dir/",
                    self.policy,
                    self.env,
                    "policy_last.ckpt",
                ),
            ]
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
