import codecs
import copy
import marshal
import types

try:
    import tensorflow as tf  # Used for processing custom objects
except ImportError:
    tf = None

from clients.utils import create_KeyMap


class AdvancedModel:
    """Given a Genotype, this class creates an AdvancedModel (Agent),
    with a deep network model constructed based on the Genotype and
    the connection map of the NeuralTopology nodes.

    During the construction, the function automatically adds Flatten and
    Concatenate layers as needed, based on the type and number of inputs the Node has.

    """

    ALLOWED_KWARGS_FOR_LAMBDA_LAYER = {
        "function",
        "batch_size",
        "trainable",
        "input_dtype",
        "batch_input_shape",
        "input_shape",
        "weights",
        "name",
        "dtype",
    }

    def __init__(self, Genotype, client):
        # print("AdvancedModel")
        # pprint.pprint(Genotype)
        KeyMap = create_KeyMap(Genotype)
        Id = Genotype["Id"]
        NeuralTopology = Genotype["Topology"]
        Sensors = Genotype["Sensors"]
        Actuators = Genotype["Actuators"]
        self.Id = Id
        self.NeuralTopology = NeuralTopology
        self.Sensors = Sensors
        self.Actuators = Actuators
        self.Genotype = Genotype
        TempDict = dict()
        Inputs = []
        Outputs = []
        for SNodeDict in Sensors:
            SNodeCoordinate = SNodeDict["Coordinate"]
            SNodeName = KeyMap[SNodeCoordinate]
            Input = client.keras.layers.Input(
                shape=self.fix_outshape(SNodeDict["OutShape"], client.keras.__name__), name=SNodeName
            )
            TempDict[SNodeCoordinate] = Input
            Inputs.append(Input)

        for Layer in NeuralTopology:
            for NodeDict in Layer:
                Node_Coordinate = NodeDict["Coordinate"]
                NodeName = KeyMap[Node_Coordinate]
                Node_Type = NodeDict["Type"]
                Parameters = NodeDict["Parameters"]
                InputList = [TempDict[FromCoordinate] for FromCoordinate in NodeDict["From"]]
                if (len(InputList) > 1) and (Node_Type != "Concatenate") and (Node_Type != "concatenate"):
                    if NodeDict["Flatten"]:
                        Acc = []
                        for i in InputList:
                            if i.shape.ndims >= 3:
                                Acc.append(client.keras.layers.Flatten()(i))
                            else:
                                Acc.append(i)
                        InputList = Acc
                    COutput = client.keras.layers.Concatenate(axis=-1, name=NodeName + "_C")(InputList)

                    if Node_Type == "TimeDistributed":
                        WrappedNode_Type = NodeDict["WrappedType"]
                        if NodeDict["Normalization"]:
                            COutput = client.keras.layers.BatchNormalization(name=NodeName + "_BatchNorm")(COutput)
                        Output = client.create_Layer(Node_Type, WrappedNode_Type, Parameters, NodeName, COutput)
                        if NodeDict["Dropout"]:
                            Output = client.keras.layers.Dropout(0.2)(Output)
                    else:
                        if NodeDict["Normalization"]:
                            COutput = client.keras.layers.BatchNormalization(name=NodeName + "_BatchNorm")(COutput)
                        Output = client.create_Layer(Node_Type, -1, Parameters, NodeName, COutput)
                        if NodeDict["Dropout"]:
                            Output = client.keras.layers.Dropout(0.2)(Output)
                        # print("NodeType:",Node_Type,"Parameters:",Parameters)
                elif (Node_Type == "Concatenate") or (Node_Type == "concatenate"):
                    # Output = LayerMap(Node_Type)(**Parameters,name=NodeName)(InputList)
                    Output = client.create_Layer(Node_Type, -1, Parameters, NodeName, InputList)
                elif (Node_Type == "Maximum") or (Node_Type == "maximum"):
                    # Output = LayerMap(Node_Type)(**Parameters,name=NodeName)(InputList)
                    Output = client.create_Layer(Node_Type, -1, Parameters, NodeName, InputList)
                elif (Node_Type == "Minimum") or (Node_Type == "minimum"):
                    # Output = LayerMap(Node_Type)(**Parameters,name=NodeName)(InputList)
                    Output = client.create_Layer(Node_Type, -1, Parameters, NodeName, InputList)
                elif (Node_Type == "Add") or (Node_Type == "add"):
                    # Output = LayerMap(Node_Type)(**Parameters,name=NodeName)(InputList)
                    Output = client.create_Layer(Node_Type, -1, Parameters, NodeName, InputList)
                elif (Node_Type == "Average") or (Node_Type == "average"):
                    # Output = LayerMap(Node_Type)(**Parameters,name=NodeName)(InputList)
                    Output = client.create_Layer(Node_Type, -1, Parameters, NodeName, InputList)
                else:  # Standard layer, which accepts a single an input from a single presynaptic layer
                    FromCoordinate = NodeDict["From"][0]
                    X = TempDict[FromCoordinate]
                    if Node_Type == "TimeDistributed":
                        WrappedNode_Type = NodeDict["WrappedType"]
                        # print("X.shape.ndims:",X.shape.ndims)
                        if NodeDict["Normalization"]:
                            X = client.keras.layers.BatchNormalization(name=NodeName + "_BatchNorm")(X)
                        # Output = LayerMap(Node_Type)(LayerMap(WrappedNode_Type)(**Parameters),name=NodeName)(X)
                        Output = client.create_Layer(Node_Type, WrappedNode_Type, Parameters, NodeName, X)
                        if NodeDict["Dropout"]:
                            Output = client.keras.layers.Dropout(0.2)(Output)
                    elif Node_Type == "Lambda":
                        if NodeDict["Normalization"]:
                            X = client.keras.layers.BatchNormalization(name=NodeName + "_BatchNorm")(X)
                        # Function = getattr(external,Parameters['function'])
                        # Output_Shape = Parameters['output_shape']
                        # Arguments = Parameters['arguments']
                        # Output = LayerMap(Node_Type)(
                        #     function=Function,
                        #     output_shape=Output_Shape,
                        #     arguments=Arguments,
                        #     name=NodeName)(X)
                        Lambda_Parameters = copy.deepcopy(Parameters)
                        code = Genotype.get("custom_functions", {}).get(Lambda_Parameters["function"])
                        Lambda_Parameters["function"] = self.code_to_function(code)
                        allowed_lambda_parameters = self.remove_not_allowed_keys(
                            Lambda_Parameters, self.ALLOWED_KWARGS_FOR_LAMBDA_LAYER
                        )

                        Output = client.create_Layer(Node_Type, -1, allowed_lambda_parameters, NodeName, X)
                        if NodeDict["Dropout"]:
                            Output = client.keras.layers.Dropout(0.2)(Output)
                    else:
                        if NodeDict["Flatten"]:
                            X = client.keras.layers.Flatten(name=NodeName + "_F")(X)
                        if NodeDict["Normalization"]:
                            X = client.keras.layers.BatchNormalization(name=NodeName + "_BatchNorm")(X)
                        # Output = LayerMap(Node_Type)(**Parameters,name=NodeName)(X)
                        Output = client.create_Layer(Node_Type, -1, Parameters, NodeName, X)
                        if NodeDict["Dropout"]:
                            Output = client.keras.layers.Dropout(0.2)(Output)
                TempDict[Node_Coordinate] = Output
        for ANodeDict in Actuators:
            FromCoordinate = ANodeDict["FromCoordinate"]
            Outputs.append(TempDict[FromCoordinate])
        model = client.keras.models.Model(inputs=Inputs, outputs=Outputs)
        self.Model = model

    @staticmethod
    def fix_outshape(OutShape, framework):
        if framework == "keras":
            return OutShape[1:]
        elif framework == "tensorflow.keras":
            return OutShape[0][1:]

    @staticmethod
    def code_to_function(code):
        raw_code = codecs.decode(code.encode("ascii"), "base64")
        if tf is not None:
            return types.FunctionType(marshal.loads(raw_code), {**globals(), "tf": tf})
        else:
            print("No tensorflow module. Loading func in unsafe mode.")
            return types.FunctionType(marshal.loads(raw_code), {**globals()})

    @staticmethod
    def remove_not_allowed_keys(d, allowed_keys):
        r = dict(d)
        for key in d.keys():
            if key not in allowed_keys:
                del r[key]
        return r
