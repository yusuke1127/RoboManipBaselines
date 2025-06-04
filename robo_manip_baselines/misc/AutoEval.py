import argparse
import datetime
import fcntl
import glob
import os
import re
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path
from urllib.parse import urlparse

import schedule

LOCK_FILE_PATH = str(
    Path("/tmp")
    / f"{'_'.join(Path(__file__).resolve().parts[-4:-1] + (Path(__file__).resolve().stem,))}.lock"
)
ROLLOUT_RESULT_PRINT_PATTERN = re.compile(r"Rollout result: (success|failure)")


class AutoEval:
    """Class that automatically performs installation, training, and rollout."""

    REPOSITORY_NAME = "RoboManipBaselines"
    THIRD_PARTY_PATHS = {
        "Sarnn": ["eipl"],
        "Act": ["act", "detr"],
        "DiffusionPolicy": ["diffusion_policy"],
        "MtAct": ["roboagent"],
    }
    APT_REQUIRED_PACKAGE_NAMES = [
        "libosmesa6-dev",
        "libgl1-mesa-glx",
        "libglfw3",
        "patchelf",
    ]

    def __init__(
        self,
        policy,
        env,
        commit_id,
        repository_owner_name=None,
        target_dir=None,
        input_checkpoint_file=None,
        is_rollout_disabled=False,
    ):
        """Initialize the instance with default or provided configurations."""
        self.policy = policy
        self.env = env
        self.commit_id = commit_id
        self.repository_owner_name = repository_owner_name
        self.is_rollout_disabled = is_rollout_disabled

        self.repository_dir = self.resolve_repository_path(target_dir)
        self.input_checkpoint_file = input_checkpoint_file
        venv_dir = os.path.join(self.repository_dir, "..", "venv")
        self.venv_python = os.path.join(venv_dir, "bin", "python")
        venv.create(venv_dir, with_pip=True)
        self.dataset_dir = None

        self.result_datetime_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "result/",
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    @classmethod
    def resolve_repository_path(cls, target_dir):
        """Resolve a suitable repository path under the given or temporary directory."""

        # normalize input path by removing trailing separator if provided
        normalized_path = target_dir.rstrip(os.sep) if target_dir is not None else None

        # determine final_repository_path: either use normalized_path if it matches pattern, else create new temp dir
        should_create_new_tempdir = True
        if normalized_path is not None:
            if os.path.basename(
                normalized_path
            ) == cls.REPOSITORY_NAME and re.fullmatch(
                r"^RmbAutoEval_\d{8}_\d{6}_.+$",
                os.path.basename(os.path.dirname(normalized_path)),
            ):
                final_repository_path = normalized_path
                should_create_new_tempdir = False

        if should_create_new_tempdir:
            # generate timestamped prefix and create temporary directory
            datetime_now = datetime.datetime.now()
            prefix = f"RmbAutoEval_{datetime_now:%Y%m%d_%H%M%S}_"
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=target_dir)
            print(f"[{cls.__name__}] Temporary directory created: {temp_dir}")

            # append repository name and return full path
            final_repository_path = os.path.join(temp_dir, cls.REPOSITORY_NAME)

        # extract base directory name and check if it is a file path
        basename = os.path.basename(final_repository_path)
        _, ext = os.path.splitext(basename)
        if ext:
            raise IOError(f"File paths are not allowed: {target_dir}")

        # verify that parent directory exists
        parent_dir = os.path.dirname(final_repository_path)
        assert os.path.isdir(
            parent_dir
        ), f"[{cls.__name__}] Parent directory does not exist: {parent_dir}"

        # return normalized and validated repository path
        return final_repository_path

    @classmethod
    def exec_command(cls, command, cwd=None, stdout_line_match_pattern=None):
        """Execute a shell command, optionally in the specified working directory,
        and return lines from standard output that match the given regex pattern."""
        print(
            f"[{cls.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing command: {' '.join(command)}",
            flush=True,
        )

        matched_results = []

        with subprocess.Popen(
            command,
            cwd=cwd,
            shell=False,  # secure default: using list avoids shell injection risks
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for stdout_line in process.stdout:
                print(stdout_line, end="", flush=True)
                if stdout_line_match_pattern:
                    match = stdout_line_match_pattern.match(stdout_line)
                    if match:
                        matched_results.append(match)

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

        return matched_results

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
                self.repository_dir,
            ],
        )
        if self.commit_id is not None:
            self.exec_command(
                ["git", "switch", "--detach", self.commit_id],
                cwd=self.repository_dir,
            )

    def check_apt_packages_installed(self, package_names):
        """Check if required APT packages are installed."""
        for pkg in package_names:
            dpkg_query_result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Status}", pkg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if (
                dpkg_query_result.returncode != 0
                or b"install ok installed" not in dpkg_query_result.stdout
            ):
                raise AssertionError("APT package not installed: " + pkg)

    def install_common(self):
        """Install common dependencies required for the environment."""
        self.exec_command(
            [self.venv_python, "-m", "pip", "install", "-e", "."],
            cwd=self.repository_dir,
        )

    def install_each_policy(self):
        """Install dependencies specific to each policy."""
        self.exec_command(
            [
                self.venv_python,
                "-m",
                "pip",
                "install",
                "-e",
                ".[" + camel_to_snake(self.policy).replace("_", "-") + "]",
            ],
            cwd=self.repository_dir,
        )

        if self.policy in self.THIRD_PARTY_PATHS:
            self.exec_command(
                [self.venv_python, "-m", "pip", "install", "-e", "."],
                cwd=os.path.join(
                    self.repository_dir,
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

        # dl=0 -> dl=1
        if dataset_url.endswith("dl=0"):
            print(
                f"[{cls.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] The URL ends with 'dl=0'. Changing it to 'dl=1'."
            )
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
                os.path.join(self.dataset_dir, zip_filename),
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
                cwd=self.dataset_dir,
            )
        except subprocess.CalledProcessError as e:
            e_stderr = e.stderr
            if e_stderr:
                e_stderr = e_stderr.strip()
            sys.stderr.write(
                f"[{self.__class__.__name__}] Warning: Command failed with return code "
                f"{e.returncode}. {e_stderr}\n"
            )
        rmb_items_count = len(
            glob.glob(os.path.join(self.dataset_dir, "**", "*.rmb"), recursive=True)
        )
        print(
            f"[{self.__class__.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {rmb_items_count} data files in RMB format have been unzipped."
        )

    def get_dataset(self, input_dataset_location):
        if bool(urlparse(input_dataset_location).scheme):
            self.dataset_dir = os.path.join(
                self.repository_dir, "robo_manip_baselines/dataset"
            )
            if not os.path.isdir(self.dataset_dir):
                raise FileNotFoundError(
                    f"expected dataset directory not found: {self.dataset_dir}"
                )
            self.download_dataset(input_dataset_location)
        elif os.path.isdir(input_dataset_location):
            self.dataset_dir = input_dataset_location
            return
        else:
            raise ValueError(f"Invalid: {input_dataset_location=}.")

    def train(self, args_file_train, seed):
        """Execute the training process using the specified arguments file."""
        assert self.dataset_dir

        if seed == -1:
            seed = int(time.time() * 1000000) % (2**32)
            print(
                f"[{self.__class__.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {seed=}"
            )

        command = [
            self.venv_python,
            os.path.join(self.repository_dir, "robo_manip_baselines/bin/Train.py"),
            self.policy,
            "--dataset_dir",
            self.dataset_dir,
            "--checkpoint_dir",
            os.path.join(
                self.repository_dir,
                "robo_manip_baselines/checkpoint",
                self.policy,
                self.env,
            ),
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        if args_file_train:
            command.append("@" + args_file_train)

        self.exec_command(command)

    def rollout(
        self,
        args_file_rollout,
        rollout_duration,
        rollout_world_idx_list,
        input_checkpoint_file,
    ):
        """Execute the rollout using the provided configuration and duration."""

        task_success_list = []
        if self.input_checkpoint_file:
            input_checkpoint_file = self.input_checkpoint_file
        else:
            input_checkpoint_file = os.path.join(
                self.repository_dir,
                "robo_manip_baselines/checkpoint",
                self.policy,
                self.env,
                "policy_last.ckpt",
            )
        if not os.path.isfile(input_checkpoint_file):
            raise FileNotFoundError(
                f"checkpoint file not found: {input_checkpoint_file}"
            )
        for world_idx in rollout_world_idx_list or [None]:
            command = [
                self.venv_python,
                os.path.join(
                    self.repository_dir, "robo_manip_baselines/bin/Rollout.py"
                ),
                self.policy,
                self.env,
                "--checkpoint",
                input_checkpoint_file,
                "--duration",
                f"{rollout_duration}",
                "--no_plot",
                "--no_render",
                "--save_last_image",
                "--output_image_dir",
                os.path.join(
                    self.repository_dir,
                    "robo_manip_baselines/checkpoint_dir/",
                    self.policy,
                    self.env,
                ),
            ]
            if world_idx is not None:
                command.extend(["--world_idx", f"{world_idx}"])
            if args_file_rollout:
                command.append("@" + args_file_rollout)

            reward_statuses = self.exec_command(
                command, stdout_line_match_pattern=ROLLOUT_RESULT_PRINT_PATTERN
            )
            assert (
                len(reward_statuses) == 1
            ), f"[{self.__name__}] {len(reward_statuses)=}"
            assert reward_statuses[0].group(1) in (
                "success",
                "failure",
            ), f"[{self.__name__}] {reward_statuses[0].group(1)=}"

            task_success_list.append(int(reward_statuses[0].group(1) == "success"))

        return task_success_list

    def save_result(self, task_success_list):
        """Save task_success_list."""

        output_dir_path = os.path.join(self.result_datetime_dir, self.policy, self.env)
        os.makedirs(output_dir_path, exist_ok=True)
        output_file_path = os.path.join(output_dir_path, "task_success_list.txt")

        if os.path.exists(output_file_path):
            base, ext = os.path.splitext(output_file_path)
            max_attempts = 100
            for counter in range(1, max_attempts + 1):
                new_file_path = f"{base}_old_{counter}{ext}"
                if not os.path.exists(new_file_path):
                    os.rename(output_file_path, new_file_path)
                    print(
                        f"[{self.__class__.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Existing file renamed to: {new_file_path}"
                    )
                    break
            else:
                raise RuntimeError(
                    f"Exceeded {max_attempts} attempts to rename existing file. "
                    f"Too many conflicting versions exist in: {output_dir_path}"
                )

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, task_success_list)))
        print(
            f"[{self.__class__.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] File has been saved: {output_file_path}"
        )

    def start(
        self,
        input_dataset_location,
        input_checkpoint_file,
        args_file_train,
        args_file_rollout,
        rollout_duration,
        rollout_world_idx_list,
        seed=None,
    ):
        """Start all required processes."""
        with open(LOCK_FILE_PATH, "w", encoding="utf-8") as lock_file:
            print(f"[{self.__class__.__name__}] Lock file: {LOCK_FILE_PATH}")
            print(f"[{self.__class__.__name__}] Attempting to acquire lock...")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            print(f"[{self.__class__.__name__}] Lock acquired. Starting processing...")
            try:
                self.git_clone()
                venv.create(
                    os.path.join(self.venv_python, "../../../venv/"), with_pip=True
                )
                self.check_apt_packages_installed(self.APT_REQUIRED_PACKAGE_NAMES)

                self.install_common()
                self.install_each_policy()

                self.get_dataset(input_dataset_location)
                self.train(args_file_train, seed)
                if not self.is_rollout_disabled:
                    task_success_list = self.rollout(
                        args_file_rollout,
                        rollout_duration,
                        rollout_world_idx_list,
                        input_checkpoint_file,
                    )
                    self.save_result(task_success_list)

                print(f"[{self.__class__.__name__}] Processing completed.")

            finally:
                print(f"[{self.__class__.__name__}] Releasing lock.")


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
        "policies",
        type=str,
        nargs="+",
        choices=["Mlp", "Sarnn", "Act", "DiffusionPolicy", "MtAct"],
        help="policies",
    )
    parser.add_argument(
        "env",
        type=str,
        help="environment",
    )
    parser.add_argument("-c", "--commit_id", type=str, required=False, default=None)
    parser.add_argument(
        "--target_dir",
        type=str,
        required=False,
        default=None,
        help="base directory used throughout program for repository clone, virtual environment, dataset, and result outputs",
    )
    parser.add_argument(
        "-u",
        "--repository_owner_name",
        type=str,
        default="isri-aist",
        help="github repository owner name (user or organization)",
    )
    parser.add_argument(
        "-d",
        "--dataset_location",
        "--input_dataset_location",
        type=str,
        dest="input_dataset_location",
        required=True,
        help="specify URL of online storage or local directory",
    )
    parser.add_argument(
        "-k",
        "--input_checkpoint_file",
        type=str,
        required=False,
        help="specify checkpoint file to use",
    )
    parser.add_argument("--args_file_train", type=str, required=False)
    parser.add_argument("--args_file_rollout", type=str, required=False)
    parser.add_argument(
        "--rollout_duration",
        type=float,
        required=False,
        default=30.0,
        help="duration of rollout in seconds, disable rollout step if value is zero or negative",
    )
    parser.add_argument(
        "--rollout_world_idx_list",
        type=int,
        nargs="*",
        required=True,
        help="list of world indicies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="random seed; use -1 to generate different value on each run",
    )
    parser.add_argument(
        "-t",
        "--daily_schedule_time",
        type=str,
        required=False,
        metavar="HH:MM",
        help="daily schedule time, for example 18:30",
    )
    parsed_args = parser.parse_args()

    if parsed_args.daily_schedule_time:
        if not re.fullmatch(
            r"(?:[01]\d|2[0-3]):[0-5]\d", parsed_args.daily_schedule_time
        ):
            parser.error(
                f"Invalid time format for --daily_schedule_time: "
                f"'{parsed_args.daily_schedule_time}'. Expected HH:MM (00-23,00-59)."
            )

    return parsed_args


if __name__ == "__main__":
    args = parse_argument()

    is_rollout_disabled = args.rollout_duration <= 0.0
    if is_rollout_disabled:
        print(
            f"[{AutoEval.__name__}] rollout step is disabled because {args.rollout_duration=} (<= 0)."
        )

    def run_once():
        """Execute the start function."""
        for policy in args.policies:
            auto_eval = AutoEval(
                policy,
                args.env,
                args.commit_id,
                args.repository_owner_name,
                args.target_dir,
                args.input_checkpoint_file,
                is_rollout_disabled,
            )
            auto_eval.start(
                args.input_dataset_location,
                args.input_checkpoint_file,
                args.args_file_train,
                args.args_file_rollout,
                args.rollout_duration,
                args.rollout_world_idx_list,
                args.seed,
            )

    # Immediate execution mode (when -t is not specified)
    if not args.daily_schedule_time:
        run_once()
        print(f"[{AutoEval.__name__}] Completed one-time run. Exiting.")
        sys.exit(0)

    # Scheduling mode
    schedule.every().day.at(args.daily_schedule_time).do(run_once)
    print(
        f"[{AutoEval.__name__}] Scheduled daily run at {args.daily_schedule_time}. Waiting...",
        flush=True,
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n[{AutoEval.__name__}] Scheduler stopped by user.")
