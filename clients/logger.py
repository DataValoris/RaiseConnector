import datetime
import os
import platform
import re
import sys
import traceback

import pkg_resources
from psutil import Process, virtual_memory, cpu_percent


class LoggerWritter(object):
    _cfg = {}
    cmd_args = {}
    filename = "log.txt"
    agents_part_translate = {".config": "c", "_weights.h5": "w", ".genotype": "g"}

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def init_project(self, cfg, cmd_args):
        self._cfg = cfg
        self.cmd_args = cmd_args

    @staticmethod
    def print_dependencies(f):
        f.write(f"<Packages usage>\n")
        with open("./requirements.txt", "r") as reqs:
            requirements = reqs.readlines()
        package_version = [req.strip().split(";")[0] for req in requirements if req.strip()]
        packages = [re.split(r"[~=<>]=", req.strip()) for req in package_version if req.strip()]
        for package in packages:
            name = package[0].strip()
            try:
                version = pkg_resources.get_distribution(package[0]).version
            except pkg_resources.DistributionNotFound:
                version = "Not installed"
            correct_version = package[1].strip()
            f.write(f"\t| {name} == {version} (need {correct_version})\n")

    def print_agents(self, f, **kwargs):
        f.write(f"<Local agents>\n")
        try:
            agents_dir = os.path.join(os.path.abspath(kwargs.get("agents_path")), str(kwargs.get("project_id")))
            files: list = [f for f in os.listdir(agents_dir) if os.path.isfile(os.path.join(agents_dir, f))]
            files_ids = list(
                set(
                    "InitAgent_" + el.split("_")[1] if el.startswith("InitAgent_") else el.split("_")[0]
                    for el in set([file.split(".")[0] for file in files])
                )
            )
            for agent_id in files_ids:
                temp_files = [file for file in files if agent_id in file]
                for tr_k, tr_v in self.agents_part_translate.items():
                    temp_files = [tr_v if el == agent_id + tr_k else el for el in temp_files]
                f.write(f"\t{agent_id} ({'+'.join(temp_files)})\n")
        except FileNotFoundError as e:
            f.write(f"\tIncorrect data in configuration file\n" f"\t{e}\n")

    @staticmethod
    def print_hardware_info(f):
        mem = virtual_memory()
        f.write(
            f"<Hardware info>\n"
            f"\tPlatform: {platform.platform()}\n"
            f"\tRAM: {Process(os.getpid()).memory_info()[0] / 2. ** 20:.1f}/{mem.total / 2 ** 20:.1f}\n"
            f"\tCPU %: {cpu_percent(percpu=True)}\n"
        )

    @staticmethod
    def print_datetime_now(f):
        f.write(f"<{datetime.datetime.now()}>\n")

    def print_script_usage(self, f, **kwargs):
        f.write(
            f"<Script usage>\n"
            f"\tConfig: '{kwargs.get('config') or self.cmd_args['config_file']}'\n"
            f"\tTrainer_cl: '{kwargs.get('trainer_cl') or self.cmd_args['train_file']}'\n"
            f"\tSeed: {kwargs.get('seed') or self.cmd_args['set_seed']}\n"
            f"\tProject ID: {kwargs.get('project_id') or self._cfg['project_id']}\n"
            f"\tPython v{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n"
            f"\tConnector v{kwargs.get('client_version') or self._cfg['client_version']}\n"
        )

    @staticmethod
    def print_error(f):
        f.write(f"<Error>\n" f"{traceback.format_exc()}\n")

    def full_stamp(self, **kwargs):
        with open(self.filename, "a") as f:
            self.print_datetime_now(f)
            self.print_hardware_info(f)
            self.print_script_usage(f, **kwargs)
            self.print_dependencies(f)
            self.print_agents(f, **kwargs)
            self.print_error(f)

    def local_stamp(self, **kwargs):
        with open(self.filename, "a") as f:
            self.print_datetime_now(f)
            self.print_script_usage(f, **kwargs)
            self.print_dependencies(f)
            self.print_error(f)


Logger = LoggerWritter()
