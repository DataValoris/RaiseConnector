### rAIse client
This readme file for connector version 2.4.10

#### Dependency versions
```
Python: 3.7
tensorflow: 2.2.0
keras: 2.3.1
requests: 2.22.0
numpy: 1.19.5
```
project.cfg - file with connection settings
Change in project.cfg project_id and token:
 - url - path to api (don't change)
 - project id - Projects tab -> Engine ID in parentheses near the project name (change it!)
 - token - Users tab -> Engine Token under the phone number (can be changed!)
 - agents_path - path to folder to store agents

#### Install dependencies
```bash
pip3 install -r requirements.txt
```

#### Examples

##### Create a sample models
> You can choose the framework for creating the model by commenting the correct line in the end of file with the function
> (need use only one framework)
> * Uncomment `keras = get_keras('tf.keras')` and you get model using Tensorflow framework
> * Uncomment `keras = get_keras('keras')` and you get model using Keras framework
+ simple functional model
```
python3 examples/functional_model.py
```
+ simple sequential model
```
python3 examples/sequential_model.py
```
+ simple functional model with custom lambda function
```
python3 examples/lambda.py
```
+ simple functional model with multi-input
```
python3 examples/multi_input.py
```

#### How to run it locally
> run population (start and continue in one script)
> train_sample.py - module with your function train_model(model)/evaluate_model(model) like in train_sample.py
+ simple functional model
```
python3 run_population.py --config_file project.cfg --train_file examples/functional_model.py
```
+ simple sequential model
```
python3 run_population.py --config_file project.cfg --train_file examples/sequential_model.py
```
+ simple functional model with custom lambda function
```
python3 run_population.py --config_file project.cfg --train_file examples/lambda.py
```
+ simple functional model with multi-input
```
python3 run_population.py --config_file project.cfg --train_file examples/multi_input.py
```

#### Features of the work process
> You can skip the current mutant training and move to the next one using the `Ctrl+N` key

#### For reproducibility for the same source model
> You can add a seed flag.
```
PYTHONHASHSEED=42 python3 run_population.py --config_file project.cfg --train_file examples/functional_model.py --set_seed=42 --project_id <your_project_id>
```
