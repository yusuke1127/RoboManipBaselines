import argparse
import glob
import os
import re
import subprocess
import sys
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

    def __init__(
        self,
        policy,
        env,
        commit_id,
        dataset_url,
        args_file_train=None,
        args_file_rollout=None,
        rollout_duration=30.0,
        rollout_world_idx_list=None,
    ):
        self.policy = policy
        self.env = env
        self.commit_id = commit_id
        self.dataset_url = self.adjust_dataset_url(dataset_url)

        self.args_file_train = args_file_train
        self.args_file_rollout = args_file_rollout
        self.rollout_duration = rollout_duration
        self.rollout_world_idx_list = rollout_world_idx_list

        self.repository_tmp_dir = os.path.join(tempfile.mkdtemp(), self.REPOSITORY_NAME)
        self.venv_python = os.path.join(tempfile.mkdtemp(), "venv/bin/python")
        self.dataset_temp_dir = tempfile.mkdtemp()

    @classmethod
    def adjust_dataset_url(cls, dataset_url):
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

    @classmethod
    def exec_command(cls, command, cwd=None):
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
        self.exec_command(
            [
                "git",
                "clone",
                "--recursive",
                self.GIT_CLONE_URL,
                self.repository_tmp_dir,
            ],
        )
        self.exec_command(
            ["git", "switch", "-c", self.commit_id], cwd=self.repository_tmp_dir
        )

    def install_common(self):
        self.exec_command(
            [self.venv_python, "-m", "pip", "install", "-e", "."],
            cwd=self.repository_tmp_dir,
        )

    def install_each_policy(self):
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

    def download_dataset(self):
        zip_filename = "dataset.zip"
        self.exec_command(
            [
                "wget",
                "-O",
                os.path.join(self.dataset_temp_dir, zip_filename),
                self.dataset_url,
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

    def train(self):
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
        if self.args_file_train:
            command.append("@" + self.args_file_train)
        self.exec_command(command)

    def rollout(self):
        for world_idx in self.rollout_world_idx_list or [None]:
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
                self.rollout_duration,
            ]
            if world_idx:
                command.extend(["--world_idx", world_idx])
            if self.args_file_rollout:
                command.append("@" + self.args_file_rollout)
            self.exec_command(command)

    def start(self):
        self.git_clone()
        venv.create(os.path.join(self.venv_python, "../../../venv/"), with_pip=True)

        # TODO:
        #     apt install is assumed to have already been finished,
        #     but this should be checked and terminated if not installed.

        self.install_common()
        self.install_each_policy()

        self.download_dataset()
        self.train()
        self.rollout()


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
    parser.add_argument("-d", "--dataset_url", type=str, required=True)
    parser.add_argument("--args_file_train", type=str, required=False)
    parser.add_argument("--args_file_rollout", type=str, required=False)
    parser.add_argument("--rollout_duration", type=float, required=False)
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
        args.policy,
        args.env,
        args.commit_id,
        args.dataset_url,
        args.args_file_train,
        args.args_file_rollout,
        args.rollout_duration,
    )
    success_report.start()
