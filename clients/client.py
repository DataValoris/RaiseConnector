import os
import signal
import sys
import time
from multiprocessing import Process

from clients.logger import Logger

try:
    import tensorflow as tf
except ImportError:
    tf = None

from clients.models import AdvancedModel
from clients.utils import JSONSerializer as json, create_dir_if_not_exists, getche

METRIC_ACC = "acc"
METRIC_MAE = "mae"

DEL_AGENTS_MODE_ALL = "all"
DEL_AGENTS_MODE_BEST = "best"


class CtrlNInterrupt(KeyboardInterrupt):
    pass


class Client:
    def __init__(self, keras):
        self.keras = keras

    def get_custom_keras_attr(self, item):
        return {
            "bottleneck_block": self.create_bottleneck_block,
            "naive_inception_block": self.create_naive_inception_block,
            "inception_block": self.create_inception_block,
            "dense_block": self.create_dense_block,
        }.get(item, False) or self.keras.layers.__getattribute__(item)

    def check_model(self, model, path_weights):
        if isinstance(model, self.keras.models.Sequential):
            print("'Sequential model' is deprecated.")
            model = self.sequential2functional(model)
        else:
            print("Functional model")
        model.load_weights(path_weights)
        return model, model.to_json()

    def sequential2functional(self, seq_model):
        """This function converts of a sequential model to a functional model with Keras
        and give the equivalent functional model.

        """
        input_layer = self.keras.layers.Input(batch_shape=seq_model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in seq_model.layers:
            prev_layer = layer(prev_layer)
        func_model = self.keras.models.Model([input_layer], [prev_layer])
        return func_model

    def check_framework(self, json_model, path_weights, framework):
        """Method for checking framework by loaded model"""
        try:
            if tf is not None:
                model = self.keras.models.model_from_json(json_model, custom_objects={"tf": tf})
            else:
                print("No tensorflow module. Loading model in unsafe mode.")
                model = self.keras.models.model_from_json(json_model)
        except ValueError as e:
            raise ValueError("Not keras model! Check model_path in config file!") from e
        except Exception:
            raise ValueError("Framework %s is not supported! Now using %s!" % (framework, self.keras.__name__))
        try:
            return self.check_model(model, path_weights)
        except ValueError as e:
            raise ValueError("Check model_path and weights_path in config file!") from e

    def create_bottleneck_block(self, X, Parameters, NodeName):
        X = self.keras.layers.Conv2D(**Parameters[0], name=NodeName + "_bottleneck_1")(X)
        X = self.keras.layers.Conv2D(**Parameters[1], name=NodeName + "_bottleneck_2")(X)
        X = self.keras.layers.Conv2D(**Parameters[2], name=NodeName + "_bottleneck_3")(X)
        return X

    def create_naive_inception_block(self, X, Parameters, NodeName):
        a = self.keras.layers.Conv2D(**Parameters[0], name=NodeName + "_naive_inception_1")(X)
        b = self.keras.layers.Conv2D(**Parameters[1], name=NodeName + "_naive_inception_2")(X)
        c = self.keras.layers.Conv2D(**Parameters[2], name=NodeName + "_naive_inception_3")(X)
        d = self.keras.layers.MaxPooling2D(**Parameters[3], name=NodeName + "_naive_inception_4")(X)
        return self.keras.layers.concatenate([a, b, c, d], name=NodeName + "_naive_inception_5")

    def create_inception_block(self, X, Parameters, NodeName):
        a = self.keras.layers.Conv2D(**Parameters[0], name=NodeName + "_inception_1")(X)
        b = self.keras.layers.Conv2D(**Parameters[1], name=NodeName + "_inception_2")(X)
        b = self.keras.layers.Conv2D(**Parameters[2], name=NodeName + "_inception_3")(b)
        c = self.keras.layers.Conv2D(**Parameters[3], name=NodeName + "_inception_4")(X)
        c = self.keras.layers.Conv2D(**Parameters[4], name=NodeName + "_inception_5")(c)
        d = self.keras.layers.MaxPooling2D(**Parameters[5], name=NodeName + "_inception_6")(X)
        d = self.keras.layers.Conv2D(**Parameters[6], name=NodeName + "_inception_7")(d)
        return self.keras.layers.concatenate([a, b, c, d], name=NodeName + "_inception_8")

    def create_dense_block(self, X, Parameters, NodeName):
        i_itex = X
        for i in range(len(Parameters)):
            x_iter = self.keras.layers.Conv2D(**Parameters[i], name=NodeName + "_dense_1_" + str(i))(i_itex)
            i_itex = self.keras.layers.concatenate([i_itex, x_iter], name=NodeName + "_dense_2_" + str(i))
        return i_itex

    @staticmethod
    def async_get_key(pid):
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            while True:
                try:
                    key = ord(getche())
                except OverflowError:
                    key = None
                if key == 14:  # Ctrl + N
                    os.kill(pid, signal.SIGTERM)
                    print("\n\nCurrent mutant was skipped, moving on to next agent.\n")
                    sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)

    @staticmethod
    def break_the_training(signum, frame):
        raise CtrlNInterrupt

    def clever_thread_train(self, genotype, config, train_function, evaluate_function):
        training_process = None
        try:
            training_process = Process(target=self.async_get_key, args=(os.getpid(),))
            training_process.start()
            try:
                signal.signal(signal.SIGTERM, self.break_the_training)
                agent = AdvancedModel(genotype, client=self)
                AgentsDIR = os.path.abspath(config["AgentsDIR"]) + os.sep
                try:
                    agent.Model.load_weights(f"{AgentsDIR}{genotype['ParentId']}_weights.h5", by_name=True)
                except Exception:
                    Logger.local_stamp()
                    print(f"####Could not load from parent: {genotype['ParentId']} using new weights####")
                self.print_model(agent.Model)
                train_fitness, test_fitness = train_function(agent.Model)
                validation_fitness = evaluate_function(agent.Model)
                genotype["TrnFitness"] = train_fitness
                genotype["TstFitness"] = test_fitness
                genotype["ValFitness"] = validation_fitness
                self.save_Agent(agent, AgentsDIR)
                print(f"Good topology for {genotype['Id']} and parent {genotype['ParentId']}")
                print("Done, moving on to next agent.")
            except Exception as e:
                Logger.local_stamp()
                print(
                    "Invalid topology. Can't load %s using parent %s. Trying a different topology."
                    % (str(genotype["Id"]), str(genotype["ParentId"]))
                )
                print(f"Exception raised: {e}")
                agent = AdvancedModel(genotype, client=self)
                agent.Model.summary()
                genotype["TrnFitness"] = 0
                genotype["TstFitness"] = 0
                print("Done, moving on to next agent.")
            return genotype
        finally:
            if training_process and training_process.is_alive():
                training_process.kill()

    def train_generation(self, population, train_function, evaluate_function):
        """Trains agents in the Population['StagedAgents'] on the provided data"""
        print(f"Training new generation of agents in population: {population['Id']}.population")
        start_time = time.time()

        config = population["Config"]
        population_size = len(population["StagedAgents"])
        print(f"TOTAL STAGED AGENTS: {population_size}")
        while True:
            evaluate_agents = []
            for agent_index, genotype in enumerate(population["StagedAgents"]):
                print(f"#### TRAINING AGENT#:{agent_index + 1} OF TOTAL: {population_size} ####")
                try:
                    evaluate_agents.append(
                        self.clever_thread_train(genotype, config, train_function, evaluate_function)
                    )
                except CtrlNInterrupt:
                    pass
                print("#################################################")
            if len(evaluate_agents):
                break
            else:
                print(
                    "\nWARNING: You missed all the mutants. "
                    "The training will start over. "
                    "Please train at least one mutant\n"
                )
                input("Press `Enter` to start the train process of the current cycle from the beginning\n")
        population["EvaluatedAgents"] = evaluate_agents
        population["StagedAgents"] = []
        train_time = time.time() - start_time
        return population, train_time

    def get_population_from_genotype(self, genotype, config, train_fitness, validation_fitness, project_id="Test_Id"):
        Weights_Path = config["ModelWeightsPath"]
        AgentsDIR = os.path.abspath(config["AgentsDIR"]) + os.sep
        Derived_Agent = AdvancedModel(genotype, client=self)
        Derived_Agent.Model.load_weights(Weights_Path)
        print("#### DERIVED MODEL ####")
        print(Derived_Agent.Model.summary())
        self.save_Agent(Derived_Agent, AgentsDIR)
        Derived_Agent.Genotype["TrnFitness"] = train_fitness
        Derived_Agent.Genotype["TstFitness"] = validation_fitness
        Population = self.create_PopulationDict(config, [Derived_Agent.Genotype], project_id)
        return Population

    def remove_unused_agents(self, population):
        agents_dir = os.path.abspath(population["AgentsDIR"]) + os.sep
        files = [f for f in os.listdir(agents_dir) if os.path.isfile(os.path.join(agents_dir, f))]
        for Id in population["HOFIds"]:
            files = [file for file in files if Id not in file]
        ids_for_cleaning = set([file.split(".")[0] for file in files])
        ids_for_cleaning = set(
            "InitAgent_" + el.split("_")[1] if el.startswith("InitAgent_") else el.split("_")[0]
            for el in ids_for_cleaning
        )
        for _id in ids_for_cleaning:
            self.remove_Agent(_id, agents_dir)

    def create_Layer(self, Node_Type, WrappedNode_Type, Parameters, NodeName, X):
        if Node_Type in self.get_BlockList():
            Output = self.get_custom_keras_attr(Node_Type)(X, Parameters, NodeName)
        else:
            if WrappedNode_Type == -1:
                Output = self.get_custom_keras_attr(Node_Type)(**Parameters, name=NodeName)(X)
            else:
                Output = self.get_custom_keras_attr(Node_Type)(
                    self.get_custom_keras_attr(WrappedNode_Type)(**Parameters), name=NodeName
                )(X)
        return Output

    @staticmethod
    def get_BlockList():
        return [
            "bottleneck_block",
            "naive_inception_block",
            "inception_block",
            "residual_block",
            "resnext_block",
            "dense_block",
            "squeeze_block",
            "NASNet_block",
            "inv_residual_block",
        ]

    @staticmethod
    def get_selection_type(selection_type):
        """Get type of selection
        :param selection_type: str with 'UnScaled', 'SizeOptimizer' or 'SizeScaled'
        :return: tuple with some data
        >>> Client.get_selection_type('UnScaled,0.0,0.5')
        ('UnScaled', 0.0, 0.5)
        >>> Client.get_selection_type('SizeScaled,0.1,0.1')
        ('SizeScaled', 0.1, 0.1)
        """
        data = selection_type.split(",")
        if len(data) == 3:
            return data[0], float(data[1]), float(data[2])
        return data[0], float(data[1])

    @staticmethod
    def print_model(model):
        print("\nMODEL LAYER DISPLAY: \n")
        model.summary()
        print("\nPARAMETER COUNT: %d\n\n" % model.count_params())

    @staticmethod
    def check_project(project_settings, framework, mode):
        if framework.lower().strip() != project_settings["connector"]:
            raise ValueError(
                "Framework in config file is %s, but must be %s." % (framework, project_settings["connector"])
            )
        if mode.lower().strip() != project_settings["optimizationMode"]:
            raise ValueError(
                "Mode in config file is %s, but must be %s." % (mode, project_settings["optimizationMode"])
            )

    @staticmethod
    def fix_coordinates(Genotype):
        """When storing the genotype in json, the tuples get converted to lists.
        This function fixes this by going through the coordinates in the genotype and
        converting them back to tuples when the genotype is initially loaded from file.

        """
        for SNode in Genotype["Sensors"]:
            SNode["Coordinate"] = (SNode["Coordinate"][0], SNode["Coordinate"][1])
        for ANode in Genotype["Actuators"]:
            ANode["Coordinate"] = (ANode["Coordinate"][0], ANode["Coordinate"][1])
            ANode["FromCoordinate"] = (ANode["FromCoordinate"][0], ANode["FromCoordinate"][1])
        for Layer in Genotype["Topology"]:
            for NodeDict in Layer:
                NodeDict["Coordinate"] = (NodeDict["Coordinate"][0], NodeDict["Coordinate"][1])
                NodeDict["From"] = [(X, Y) for (X, Y) in NodeDict["From"]]

    @staticmethod
    def create_PopulationDict(config, InitGenotypePool=None, Id="Test_Id"):
        """Creates the initial Population dictionary"""
        Population = {
            "Id": Id,
            "Config": config,
            "HOF": [],
            "StagedAgents": [],
            "EvaluatedAgents": [],
            "InitGenotypePool": InitGenotypePool or [],
            "ExploredPhylogenies": [],
            "PhylogeneticProgression": [],
        }
        if not InitGenotypePool:
            Population["PhylogeneticProgression"] = []
        else:
            PhylProgStep = []
            for Genotype in InitGenotypePool:
                PhylProgStep.append(
                    [
                        Genotype["Id"],
                        Genotype["Topology"],
                        Genotype["TrnFitness"],
                        Genotype["ValFitness"],
                        Genotype["TstFitness"],
                    ]
                )
            Population["PhylogeneticProgression"].append([PhylProgStep, time.ctime()])
        return Population

    def save_Agent(self, Agent, DIR="./Agents/"):
        """Saves the Agent(AdvancedModel) to file, by seperately saving the model's subparts (Weights,Config,Genotype).
        The subparts are saved in json format, with their respective extensions (weights, genotype, config).

        """
        if self.keras.__version__[-2:] == "tf":
            Agent.Model.save_weights(DIR + Agent.Id + "_weights.h5", overwrite=True, save_format="h5")
        else:
            Agent.Model.save_weights(DIR + Agent.Id + "_weights.h5", overwrite=True)
        with open(DIR + Agent.Id + ".genotype", "w") as f:
            json.dump(Agent.Genotype, f)

        json_string = Agent.Model.to_json()
        with open(DIR + Agent.Id + ".config", "w") as f:
            json.dump(json_string, f)

    @staticmethod
    def remove_Agent(Id, DIR="./Agents/"):
        print(f"Removing agent with ID: {Id}")
        import os

        if os.path.exists(DIR + Id + ".genotype"):
            os.remove(DIR + Id + ".genotype")
        else:
            print("The file does not exist:", DIR + Id + ".genotype")
        if os.path.exists(DIR + Id + "_weights.h5"):
            os.remove(DIR + Id + "_weights.h5")
        else:
            print("The file does not exist:", DIR + Id + "_weights.h5")
        if os.path.exists(DIR + Id + ".config"):
            os.remove(DIR + Id + ".config")
        else:
            print("The file does not exist:", DIR + Id + ".config")


def create_config(**kw):
    __available_frameworks = ["tf.keras", "keras"]

    if "Framework" in kw and kw["Framework"] not in __available_frameworks:
        raise ValueError(f"At this time: {kw['Framework']} is not supported yet. Please contact us for tech support.")

    data = {
        "TotalMutants": 10000,
        "MutantIndex": 0,
        "PopSize": 20,
        "SeedPopSize": 1,
        "CycleLimit": 1000,
        "CycleIndex": 0,
        "FitnessGoal": 1.0,
        "LearnRate": 0.1,
        "SelectionType": "SizeOptimizer",
        "SelectionParameter": 0.05,
        "HOFSize": 20,
        "SizeScalingFactor": 0.0000,
        "TimeScalingFactor": 0.0001,
        "NeuralEffeciency": 0.5,
        "LayerSearchConstraint": "Unconstrained",
        "MutationSearchConstraint": "Unconstrained",
        "CustomLearner": "",
        "Framework": None,
        "ClientConnectionConfig": "",
        "ServerConnectionConfig": "",
        "InShape": None,
        "OVL": None,
        "ModelConfigPath": "./Agents/TestModel.config",
        "ModelWeightsPath": "./Agents/TestModel_weights.h5",
        "TrainingData_Shapes": [],
        "TrainingData_Sizes": [],
        "ValidationData_Shapes": [],
        "ValidationData_Sizes": [],
        "TrainingDataInput_Paths": ["./Agents/TestTrainingInput_Data"],
        "TrainingDataOutput_Paths": ["./Agents/TestTrainingOutput_Data"],
        "ValidationDataInput_Paths": ["./Agents/TestValidationInput_Data"],
        "ValidationDataOutput_Paths": ["./Agents/TestValidationOutput_Data"],
        "TestDataInput_Paths": ["./Agents/TestTestInput_Data"],
        "TestDataOutput_Paths": ["./Agents/TestTestOutput_Data"],
        "AgentsDIR": "./Agents/",
        "ProjectsDIR": "./Populations/",
        "Phase": "Init",
        "Optimizer": "adam",
        "Loss": "categorical_crossentropy",
        "TestData_Shapes": [],
        "TestData_Sizes": [],
        "CPU_Capacity": [],
        "GPU_Capacity": [],
        "Epochs": 20,
        "BatchSize": 1000,
        "Patience": 2,
        "metrics": "acc",
        "Maximizer": "void",
        "ProblemDomain": "Classification",
        "ProgressionHistory": [],
        "ClientNodePool": [],
        "LayerList": [],
        "MOList": [],
        "CustomPool": [],
        "InitNodeTypes": [],
        "UnitsPool": [],
        "FiltersPool": [],
        "StridesPool": [],
        "PoolingSizePool": [],
        "Model_Parameter_Limit": 5000000,
        "Model_LayerDepth_Limit": 100,
        "Model_Node_Limit": 100,
        "DataAggregation": True,
        "FreezeWeights": False,
        "LayerUnits": [16 * 2, 16 * 3, 16 * 4],
        "LayerFilters": [16 * 2, 16 * 3, 16 * 4],
        "LayerKernelSizes": [1, 2, 3, 4],
        "LayerStrides": [1, 2, 3],
        "LayerPoolSizes": [2],
        "UseSeedWeights": False,
        "PhylogeneticFitness": -1,
        "EarlyStop_EpochTimeDeviation": True,
        "Median_TrnEpochTimeFactor": 2,
        "Median_TrnEpochTime": -1,
        "Median_TrnEpochTime_StD": -1,
        "EarlyStop_LossDeviation": False,
        "Median_TrnLossFactor": 3,
        "Median_TrnLoss": -1,
        "Median_TrnLoss_StD": -1,
        "EarlyStop_EpochTimeThreshold": -1,
        "EarlyStop_LossThreshold": -1,
        "DynamicBaselineUpdate": True,
        "HOF_Selection_Method": "Default",
        "Dataset": "client_dataset",
    }

    for k, v in kw.items():
        if k in data:
            data[k] = v

    create_dir_if_not_exists(data["ProjectsDIR"])
    create_dir_if_not_exists(data["AgentsDIR"])

    return data
