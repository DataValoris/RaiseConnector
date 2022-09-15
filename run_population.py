import datetime
import faulthandler
import gc
import os
import random
import signal
import sys
import time

import numpy as np

from clients.api import RequestToApi
from clients.client import METRIC_MAE, Client, create_config
from clients.kerases import get_framework
from clients.logger import Logger
from clients.utils import (
    ConnectorError,
    check_connector,
    make_directories,
    parse_args,
    read_model,
    create_token_and_project_id
)
from examples.itrainer import ITrainer

process_exit_flag = False

x_train, y_train, x_test, y_test = None, None, None, None

DEL_AGENTS_MODE_ALL = "all"
DEL_AGENTS_MODE_BEST = "best"


def exit_gracefully(signum, frame):
    signal.signal(signal.SIGINT, original_sigint)
    try:
        time.sleep(0.5)
        if input("\n\nDo you want to stop train process? (y/n)> ").lower().startswith("y"):
            global process_exit_flag
            process_exit_flag = True

            print("Process will be stopped after current cycle finish.")
            time.sleep(3)
        else:
            signal.signal(signal.SIGINT, exit_gracefully)
    except KeyboardInterrupt:
        print("\nProcess was killed!")
        sys.exit(1)


def terminate_gracefully(signum, frame):
    # for handle Aborted by OS
    print("Some critical error, process will be stopped by your OS. Please rerun program.")
    faulthandler.enable()
    sys.stdout.flush()
    global process_exit_flag
    process_exit_flag = True


def print_lib_versions():
    import re
    from pkg_resources import get_distribution, DistributionNotFound

    with open("./requirements.txt", "r") as f:
        requirements = f.readlines()
    package_version = [req.strip().split(";")[0] for req in requirements if req.strip()]
    packages = [re.split(r"[~=<>]=", req.strip()) for req in package_version if req.strip()]
    print("+---- Packages info -----------------------------------------------------------")
    for package in packages:
        name = package[0].strip()
        try:
            version = get_distribution(package[0]).version
        except DistributionNotFound:
            version = "Not installed"
        correct_version = package[1].strip()
        print(f"| {name} == {version} (need {correct_version})")
    print("+------------------------------------------------------------------------------\n")


def get_tolerance(metric, initial_score, project_tolerance):
    if metric == METRIC_MAE:
        return initial_score + project_tolerance / 100
    else:
        return max(initial_score - project_tolerance / 100, 0.0)


def run_population(project_config, trainer_class, seed_value=None):
    global process_exit_flag
    print_lib_versions()

    token = project_config.get("DEFAULT", "token")
    project_id = int(project_config.get("DEFAULT", "project_id"))
    agents_dir = project_config.get("DEFAULT", "agents_path")
    population_dir = project_config.get("DEFAULT", "mutants_path")
    url = project_config.get("DEFAULT", "url")

    api = RequestToApi(url, token)
    check_connector(api, project_config.get("DEFAULT", "client_version"))
    project = api.get_project(project_id=project_id)

    # population_size = project["mutantsPerCycle"]
    framework = project["connector"]
    mode = project["optimizationMode"]
    del_agents_mode = project["savingAgentsMode"]
    client_model_config_path = project["initialConfigPath"]
    client_model_weights_path = project["initialWeightPath"]

    keras = get_framework(framework)
    trainer: ITrainer = trainer_class(keras=keras)
    client = Client(keras=keras)

    agents_dir = os.path.join(os.path.abspath(agents_dir), str(project_id) + os.sep)
    projects_dir = os.path.join(os.path.abspath(population_dir), str(project_id) + os.sep)
    make_directories([agents_dir])

    gc.enable()

    cycle_index = project["mutantsEvaluated"]
    if cycle_index != 0:
        if project["status"] != "quota" or project["mutantsEvaluated"] < project["mutantsQuota"]:
            print("Continue process.")
        else:
            print("Quota reached!")
        client.check_project(project, framework, mode)
        initial_mutant = api.get_mutant(mutant_id=project["initialMutantId"])
        tolerance = get_tolerance(project["metric"], initial_mutant["score"], project["tolerance"])
    else:
        print("Start process.")

        if seed_value:
            os.environ["PYTHONHASHSEED"] = str(seed_value)
            random.seed(seed_value)
            np.random.seed(seed_value)

        client_model_config = read_model(client_model_config_path)
        model, client_model_config = client.check_framework(client_model_config, client_model_weights_path, framework)
        evaluation_score = float(trainer.evaluate_func(model))
        train_score = evaluation_score
        validation_params_count = model.count_params()

        initial_mutant = api.create_mutant(
            request_params={
                "score": evaluation_score,
                "size": validation_params_count,
                "generation": 0,
                "project_id": project_id,
            }
        )
        tolerance = get_tolerance(project["metric"], initial_mutant["score"], project["tolerance"])

        selection_type, selection_parameter, neural_efficiency = client.get_selection_type(project["growthControl"])

        project = api.modify_project(
            project_id=project_id,
            request_params={
                "initialMutantId": initial_mutant["id"],
                "agentsDir": agents_dir,
                "mutantsPath": projects_dir,
            },
        )
        # for case if doesnt need all processing
        if (
            project["scoreLimit"]
            and evaluation_score > project["scoreLimit"]
            or project["sizeLimit"]
            and project["sizeLimit"] > validation_params_count
        ):
            api.modify_project(request_params={"status": "completed"}, project_id=project_id)
            print("Initial score or size is better than the holding condition!")
            return

        response = api.init_project(client_model_config, project_id, initial_mutant)
        initial_mutant, genotype = response["initial_mutant"], response["genotype"]
        client_config = create_config(
            Framework=project["connector"],
            ModelConfigPath=client_model_config_path,
            ModelWeightsPath=client_model_weights_path,
            AgentsDIR=agents_dir,
            ProjectsDIR=projects_dir,
            HOFSize=project["hofSize"],
            PopSize=project["mutantsPerCycle"],
            SelectionType=selection_type,
            SelectionParameter=selection_parameter,
            NeuralEffeciency=neural_efficiency,
            metrics=project["metric"],
        )
        api.start_project(
            project_id=project_id,
            instance=client.get_population_from_genotype(
                genotype=genotype,
                config=client_config,
                train_fitness=train_score,
                validation_fitness=evaluation_score,
                project_id=project_id,
            ),
        )
    project_cycles_info = []

    api.modify_project(request_params={"status": "in_progress"}, project_id=project_id)

    cycle_score_mutant, cycle_size_mutant = None, None

    while True:
        keras.backend.clear_session()
        project = api.get_project(project_id=project_id)
        cycle_max_generation = project["maxGeneration"]
        train_time = None

        if mode == "population":
            instance = api.modify_genotype(project_id=project_id)

            cycle_index = instance["Config"]["CycleIndex"]
            print(f"CYCLE: {cycle_index}")
            project_cycles_info.append({"cycle_id": cycle_index})

            instance, train_time = client.train_generation(
                population=instance, train_function=trainer.train_func, evaluate_function=trainer.evaluate_func
            )
            response = api.send_selection_cycle_result(
                project_id,
                request_params={"instance": instance, "initial_mutant": initial_mutant, "tolerance": tolerance},
            )
            instance = response["instance"]
            cycle_score_mutant = response["score_mutant"]
            cycle_size_mutant = response["size_mutant"]
            cycle_max_generation = response["max_generation"]
            if del_agents_mode == DEL_AGENTS_MODE_BEST:
                client.remove_unused_agents(instance)

        # save trained cycle info and max generation
        cycles_info = {
            "index": cycle_index,
            "trainTime": int(train_time),
            "scoreChampionId": cycle_score_mutant.get("id"),
            "sizeChampionId": cycle_size_mutant.get("id"),
            "projectId": project["id"],
        }
        api.send_cycle_info(request_params=cycles_info)

        project_cycles_info[-1] = {
            "cycle_id": cycle_index,
            "mutants_count": project.get("mutantsEvaluated"),
            "train_time": int(train_time),
            "timestamp": str(datetime.datetime.now()),
            "best_score": {
                "mutant_id": cycle_score_mutant["agentId"] if cycle_score_mutant else initial_mutant["agentId"],
                "mutant_score": cycle_score_mutant["score"],
                "mutant_size": cycle_score_mutant["size"] if cycle_score_mutant else initial_mutant["size"],
            },
        }
        print("############################### FITNESS PROGRESSION ##############################")
        print([x["best_score"]["mutant_score"] for x in project_cycles_info])
        print(
            "ITERATION: %d \nCURRENT MUTANT FITNESS: %.4f\nCURRENT MUTANT ID: %s"
            % (cycle_index, cycle_score_mutant["score"], cycle_score_mutant["agentId"])
        )
        print("##################################################################################")
        gc.collect(generation=2)

        project = (
            api.modify_project(project_id=project_id, request_params={"maxGeneration": cycle_max_generation})
            if cycle_max_generation > project["maxGeneration"]
            else api.get_project(project_id=project_id)
        )

        if project["status"] in ("completed", "on_hold") or process_exit_flag:
            break

    if process_exit_flag:
        project = api.modify_project(project_id=project_id, request_params={"status": "on_hold"})

    size_champion = (
        api.get_mutant(mutant_id=project["sizeChampionId"]) if project["sizeChampionId"] else initial_mutant
    )
    score_champion = (
        api.get_mutant(mutant_id=project["scoreChampionId"]) if project["scoreChampionId"] else initial_mutant
    )
    print("##################################################################################")
    if score_champion:
        print(
            "SCORE CHAMPION FITNESS: %.4f\nSCORE CHAMPION ID: %s\nSCORE CHAMPION SIZE: %d\n"
            % (score_champion["score"], score_champion["agentId"], score_champion["size"])
        )
    if size_champion:
        print(
            "SIZE CHAMPION FITNESS: %.4f\nSIZE CHAMPION ID: %s\nSIZE CHAMPION SIZE: %d"
            % (size_champion["score"], size_champion["agentId"], size_champion["size"])
        )
    print("##################################################################################")


if __name__ == "__main__":
    config, trainer_cl, seed, args = parse_args()
    try:
        # handle ctrl+C input for stop the process gracefully
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, exit_gracefully)
        signal.signal(signal.SIGABRT, terminate_gracefully)
        signal.signal(signal.SIGSEGV, terminate_gracefully)

        create_token_and_project_id(config, args.config_file, args.token, args.project_id)
        Logger.init_project(cfg=dict(config.items("DEFAULT")), cmd_args=vars(args))
        run_population(project_config=config, trainer_class=trainer_cl, seed_value=seed)
    except Exception as err:
        status = "error"
        if isinstance(err, KeyboardInterrupt):
            status = "on_hold"
        elif isinstance(err, ConnectorError):
            print(
                "###################################################################################################"
            )
            print(err)
            sys.exit(0)
        Logger.full_stamp(
            config=args.config_file,
            trainer_cl=args.train_file,
            seed=args.set_seed,
            project_id=config and config.get("DEFAULT", "project_id"),
            client_version=config and config.get("DEFAULT", "client_version"),
            agents_path=config and config.get("DEFAULT", "agents_path"),
        )
        if not isinstance(err, ConnectionError):
            RequestToApi(config.get("DEFAULT", "url"), config.get("DEFAULT", "token")).modify_project(
                project_id=int(config.get("DEFAULT", "project_id")), request_params={"status": status}
            )
        raise err
