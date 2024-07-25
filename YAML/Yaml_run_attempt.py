import geoTherm as gt
import yaml
import os
thermo = gt.thermo()


# Put the file name here:
file_name = 'Model_Template.yml'
# Put the file path here. 
# Current code looks in the same folder as this script, but we'll want a "models" folder
folder = (os.path.dirname(__file__))

model_file = os.path.join(folder, file_name)
with open(model_file, "r") as stream:
    model_config = (yaml.safe_load(stream))


# Set the default input/output units 
gt.units.input = model_config['input_units']
gt.units.output = model_config['output_units']

# Fluid types are set in the YAML, but these are available if needed:
#hot_loop_fluid = model_config['hot_loop_fluid']
working_loop_fluid = model_config['working_loop_fluid']
#cooling_loop_fluid = model_config['cooling_loop_fluid']

thermo.TPY = 308, 101325, working_loop_fluid

def create_node(node, node_args):
    new_node = []
    node_call = 'gt.' + node_args['node_type'] + '(**node_args)'
    #The variable node_type only exists in the YAML file. YAML requires unique names for keys, 
    # so the type of node selected cannot be a key. 
    # The rest of this dictionary is the as-formatted inputs for the node_call function when evaluated,
    # so node_type needs to be removed from the dictionary before node_call is evaluated. 
    del node_args['node_type']
    new_node = eval(node_call)
    return new_node


#This set of for loops iterates through all of the model inputs looking for a dictionary with the 
# key "replacement_variable". Then it adds the value of that dict to a list of replacement variables,
# and removes the "replacement_variable label"
# This fuction assumes that the structure model_config['Loop_models'][model] already exists
def find_replacement_variables(model, replacement_variables):
    for node in model_config['Loop_models'][model]: 
        for node_arg in model_config['Loop_models'][model][node]:
            value = model_config['Loop_models'][model][node][node_arg]
            if isinstance(value, dict):
                if value.get('replacement_variable'):
                    actual_value = value['replacement_variable']
                    replacement_variables.append(actual_value)
                    model_config['Loop_models'][model][node][node_arg] = actual_value #overwrite one level of the dictionary
                    # this should delete the level that says "replacement variable"
                    # the node_arg should now just be a label and an expression with a matching
                    # label & expression in replacement_variables[]. After replacement_variables[] is
                    # evaluated, it can get swapped back in.

    return replacement_variables


#This fucntion iterates over the list of replacement variables, evaluates the expressions,
# and then outputs the list with the evaluated values
def evaluate_replacement_variables(replacement_variables):
    for position, item in enumerate(replacement_variables):
        if len(item.values()) == 1: #Verify that this is a dictionary with one key:value pair
            key, val = next(iter(item.items())) #Pulls dict key and value #thanks ChatGPT
            val = eval(val)
            item = {key:val}
            replacement_variables[position] = item
 
    return replacement_variables
    

def replace_model_variables(model, replacement_variables):
    for item in replacement_variables: #iterate over every replacement variable
        print('replacement items:')
        print(item)
        for node in model_config['Loop_models'][model]: 
            for node_arg in model_config['Loop_models'][model][node]:
                value = model_config['Loop_models'][model][node][node_arg] #iterate through every model node argument
                if isinstance(value, dict):   #the argument should only be a dict if it needs replacing
                    if (value.keys()) == (item.keys()):     #check that the right value is being matched
                        model_config['Loop_models'][model][node][node_arg] = list(item.values())[0]
                        #Pull the value and assign directly as the node argument

    return replacement_variables

print('_______________')

list_of_model_loops = [] #Make a list to store each loop in

for model in model_config['Loop_models']:
    #print(model)
    this_model = gt.Model([])  #make a new blank model
    list_of_model_loops.append(this_model) #add this blank model to the end of the list of models

    #Check for expressions that need to be evaluated, and then evaluate them
    replacement_variables = []
    replacement_variables = find_replacement_variables(model, replacement_variables)
    print('replacement variables found: ')
    print(replacement_variables)
    replacement_variables = evaluate_replacement_variables(replacement_variables)
    replacement_variables = replace_model_variables(model, replacement_variables)
    
    #Build the model
    for node in model_config['Loop_models'][model]: 
        node_args = model_config['Loop_models'][model][node]
        new_node = create_node(node, node_args)
        gt.Model.addNode(this_model, new_node)

    this_model.solve()
    
    print(model)
    print(this_model)