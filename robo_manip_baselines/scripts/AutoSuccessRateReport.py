"""Script for evaluation based on a specific commit."""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import venv


class AutoSuccessRateReport:
    """Class for generating success rate reports based on commit evaluation."""

    REPOSITORY_NAME = "RoboManipBaselines"
    THIRD_PARTY_PATHS = {
        "Sarnn": ["eipl"],
        "Act": ["act", "detr"],
        "DiffusionPolicy": ["diffusion_policy"],
    }

    def __init__(
        self,
        policy,
        env,
        commit_id,
        repository_owner_name=None,
    ):
        """Initialize the instance with default or provided configurations."""
        self.policy = policy
        self.env = env
        self.commit_id = commit_id
        self.repository_owner_name = repository_owner_name

        self.repository_tmp_dir = os.path.join(tempfile.mkdtemp(), self.REPOSITORY_NAME)
        self.venv_python = os.path.join(tempfile.mkdtemp(), "venv/bin/python")
        self.dataset_temp_dir = tempfile.mkdtemp()

    @classmethod
    def exec_command(cls, command, cwd=None):
        """Execute a shell command, optionally in the specified working directory."""
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

    def git_clone(self):
        """Clone the target Git repository into a temporary directory."""
        git_clone_url = (
            f"https://github.com/{self.repository_owner_name}/"
            + f"{self.REPOSITORY_NAME}.git"
        )
        self.exec_command(
            [
                "git",
                "clone",
                "--recursive",
                git_clone_url,
                self.repository_tmp_dir,
            ],
        )
        self.exec_command(
            ["git", "switch", "-c", self.commit_id], cwd=self.repository_tmp_dir
        )

    def install_common(self):
        """Install common dependencies required for the environment."""
        self.exec_command(
            [self.venv_python, "-m", "pip", "install", "-e", "."],
            cwd=self.repository_tmp_dir,
        )

    def install_each_policy(self):
        """Install dependencies specific to each policy."""
        if self.policy in self.THIRD_PARTY_PATHS:
            self.exec_command(
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
            self.exec_command(
                [self.venv_python, "-m", "pip", "install", "-e", "."],
                cwd=os.path.join(
                    self.repository_tmp_dir,
                    "third_party/",
                    *self.THIRD_PARTY_PATHS[self.policy],
                ),
            )

    @classmethod
    def adjust_dataset_url(cls, dataset_url):
        """Adjust and normalize the dataset URL for compatibility."""
        # Remove '\&'
        for _ in range(len(dataset_url)):
            if r"\&" not in dataset_url:
                break

            # With subprocess.Popen(shell=False), use '&' without escape.
            dataset_url = dataset_url.replace(r"\&", r"&")

        # dl=0 → dl=1
        if dataset_url.endswith("dl=0"):
            print(f"[{cls.__name__}] The URL ends with 'dl=0'. Changing it to 'dl=1'.")
            dataset_url = dataset_url[: -len("dl=0")] + "dl=1"

        assert dataset_url.endswith(
            "dl=1"
        ), f"[{cls.__name__}] Error: The URL '{dataset_url}' does not end with 'dl=1'."
        return dataset_url

    def download_dataset(self, dataset_url):
        """Download the dataset from the specified URL."""
        zip_filename = "dataset.zip"
        self.exec_command(
            [
                "wget",
                "-O",
                os.path.join(self.dataset_temp_dir, zip_filename),
                self.adjust_dataset_url(dataset_url),
            ],
        )
        try:
            self.exec_command(
                [
                    "unzip",
                    "-d",
                    "dataset/",
                    zip_filename,
                ],
                cwd=self.dataset_temp_dir,
            )
        except subprocess.CalledProcessError as e:
            e_stderr = e.stderr
            if e_stderr:
                e_stderr = e_stderr.strip()
            sys.stderr.write(
                f"[self.__class__.__name__] Warning: Command failed with return code "
                f"{e.returncode}. {e_stderr}\n"
            )
        rmb_items_count = len(
            glob.glob(os.path.join(self.dataset_temp_dir, "dataset", "*.rmb"))
        )
        print(
            f"[self.__class__.__name__] {rmb_items_count} rmb items have been unzipped."
        )

    def train(self, args_file_train):
        """Execute the training process using the specified arguments file."""
        command = [
            self.venv_python,
            os.path.join(self.repository_tmp_dir, "robo_manip_baselines/bin/Train.py"),
            self.policy,
            "--dataset_dir",
            os.path.join(
                self.dataset_temp_dir,
                "dataset/",
            ),
            "--checkpoint_dir",
            os.path.join(
                self.repository_tmp_dir,
                "robo_manip_baselines/checkpoint_dir/",
                self.policy,
                self.env,
            ),
        ]
        if args_file_train:
            command.append("@" + args_file_train)
        self.exec_command(command)

    def rollout(self, args_file_rollout, rollout_duration, rollout_world_idx_list):
        """Execute the rollout using the provided configuration and duration."""
        for world_idx in rollout_world_idx_list or [None]:
            command = [
                self.venv_python,
                os.path.join(
                    self.repository_tmp_dir, "robo_manip_baselines/bin/Rollout.py"
                ),
                self.policy,
                self.env,
                "--checkpoint",
                os.path.join(
                    self.repository_tmp_dir,
                    "robo_manip_baselines/checkpoint_dir/",
                    self.policy,
                    self.env,
                    "policy_last.ckpt",
                ),
                "--duration",
                f"{rollout_duration}",
            ]
            if world_idx is not None:
                command.extend(["--world_idx", f"{world_idx}"])
            if args_file_rollout:
                command.append("@" + args_file_rollout)
            self.exec_command(command)

    def start(
        self,
        dataset_url,
        args_file_train,
        args_file_rollout,
        rollout_duration,
        rollout_world_idx_list,
    ):
        """Start all required processes."""
        self.git_clone()
        venv.create(os.path.join(self.venv_python, "../../../venv/"), with_pip=True)

        # TODO:
        #     apt install is assumed to have already been finished,
        #     but this should be checked and terminated if not installed.

        self.install_common()
        self.install_each_policy()

        self.download_dataset(dataset_url)
        self.train(args_file_train)
        self.rollout(args_file_rollout, rollout_duration, rollout_world_idx_list)


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
    """Parse and return the command-line arguments."""
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
        "-u",
        "--repository_owner_name",
        type=str,
        default="isri-aist",
        help="github repository owner name (user or organization)",
    )
    parser.add_argument("-d", "--dataset_url", type=str, required=True)
    parser.add_argument("--args_file_train", type=str, required=False)
    parser.add_argument("--args_file_rollout", type=str, required=False)
    parser.add_argument("--rollout_duration", type=float, required=False, default=30.0)
    parser.add_argument(
        "--rollout_world_idx_list",
        type=int,
        nargs="*",
        help="list of world indexes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    success_report = AutoSuccessRateReport(
        args.policy, args.env, args.commit_id, args.repository_owner_name
    )
    success_report.start(
        args.dataset_url,
        args.args_file_train,
        args.args_file_rollout,
        args.rollout_duration,
        args.rollout_world_idx_list,
    )
