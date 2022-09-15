import argparse
import importlib.util
import json
import os
from configparser import ConfigParser, NoOptionError
from functools import singledispatch

import numpy as np


class ConnectorError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", dest="config_file", help="configuration file", required=True)
    parser.add_argument(
        "--train_file",
        dest="train_file",
        help="python module, which performs train and return " "train and validation score",
        required=True,
    )
    parser.add_argument(
        "--project_id", dest="project_id", help="id of project that will optimize", required=False
    )
    parser.add_argument(
        "--token", dest="token", help="user token", required=False
    )
    parser.add_argument(
        "--set_seed", dest="set_seed", help="value for reproducibility results for the same source model", type=int
    )
    cmd_args = parser.parse_args()
    config = ConfigParser()
    config.read(cmd_args.config_file)

    spec = importlib.util.spec_from_file_location("train", cmd_args.train_file)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    return config, train.Trainer, cmd_args.set_seed, cmd_args


def make_dir_if_not_exists(name):
    if not os.path.exists(name):
        os.makedirs(name)


def make_directories(dirs):
    for directory in dirs:
        make_dir_if_not_exists(directory)


def read_model(path):
    with open(path) as f:
        return json.load(f)


def create_KeyMap(Genotype):
    Map = dict()
    # print("Genotype['Sensors']:",Genotype['Sensors'])
    for SNodeDict in Genotype["Sensors"]:
        Coordinate = SNodeDict["Coordinate"]
        Version = SNodeDict["Version"]
        # print("SCoordinate:",Coordinate,"Version:",Version)
        Map[Coordinate] = coord2key(Coordinate, Version)

    for ANodeDict in Genotype["Actuators"]:
        Coordinate = ANodeDict["Coordinate"]
        Version = ANodeDict["Version"]
        # print("Coordinate:",Coordinate,"Version:",Version)
        Map[Coordinate] = coord2key(Coordinate, Version)

    for Layer in Genotype["Topology"]:
        for NodeDict in Layer:
            Coordinate = NodeDict["Coordinate"]
            Version = NodeDict["Version"]
            # print("ACoordinate:",Coordinate,"Version:",Version)
            Map[Coordinate] = coord2key(Coordinate, Version)
    return Map


def coord2key(Coordinate, Version):
    """Due to the use of cartesian style Ids, to keep track of width and depth, similar to DXNN,
    the coord2key converts the coordinate and version number of the node to a string.

    """
    # print("Coordinate:",Coordinate,"Version:",Version)
    (x, y) = Coordinate
    # print("Version:",Version)
    Key = "DV_" + str(round(1000 * x)) + "_" + str(round(1000 * y)) + "_" + str(Version)
    return Key


def check_connector(api, client_version):
    """Check connector version"""
    is_current_client_version = api.check_client_version(client_version)
    if is_current_client_version:
        print(f"Connector version: {client_version}")
    else:
        raise ConnectorError("Your connector version isn't latest, please upload new version connector")


def create_token_and_project_id(config, config_path, token, project_id):
    config_changed = False
    if "token" not in config["DEFAULT"] or token:
        config["DEFAULT"]["token"] = token or input("Please enter a token\n")
        config_changed = True
    if "project_id" not in config["DEFAULT"] or project_id:
        config["DEFAULT"]["project_id"] = project_id or input("Please enter a project ID\n")
        config_changed = True
    if config_changed:
        with open(config_path, 'w') as new_config:
            config.write(new_config)


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


@singledispatch
def serializer(obj):
    return str(obj)


@serializer.register(np.float32)
def tj_float32(val):
    return np.float64(val)


class JSONSerializer:
    @staticmethod
    def dumps(obj):
        return json.dumps(obj, default=serializer)

    @staticmethod
    def dump(obj, fp):
        return json.dump(obj, fp, default=serializer)

    @staticmethod
    def loads(s):
        return json.loads(s)

    @staticmethod
    def load(fp):
        return json.load(fp)


def get_getche():
    import platform

    if platform.system() == "Windows":
        from msvcrt import getche as _getche
    else:
        try:
            from getch import getche as _getche
        except ImportError:

            def _getche():
                from getkey import getkey, keys

                key = getkey()
                print(key)
                return key

    return _getche


getche = get_getche()
