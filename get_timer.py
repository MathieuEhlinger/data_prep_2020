from ml_run_lib.bids_metaexp import BIDSAsklExperiment
import sys

#Add library to path
lib_path = 'path_to_lib'
sys.path.append(lib_path)

#Add path to original datasets
main_path='path_to_original_datasets'

#Add path to data
path_to_data = 'add_path_to_data'

#Add path to me data
path_to_me_data = 'add_path_to_me_data'

#Change to True in order to write an output
save= False

#With the new dataset
exp_bnr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_native/','askl_bids_new_native')
exp_bwr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_residuals/','askl_bids_new_residuals')
exp_bdn=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_networks/','askl_bids_new_networks')

exp_bwr.models_dict['BiDirect_New_Matched_All_Structural_residuals'].core_model.show_models()
exp_bwr.models_dict['BiDirect_New_Matched_All_Structural_residuals'].core_model.trajectory_
test = exp_bwr.models_dict['BiDirect_New_Matched_All_Structural_residuals'].core_model.cv_results_


#https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_get_pipeline_components.html#sphx-glr-examples-40-advanced-example-get-pipeline-components-py
automl=exp_bwr.models_dict['BiDirect_New_Matched_All_Structural_residuals'].core_model

automl.automl_.runhistory_.data

for run_key in automl.automl_.runhistory_.data:
    print('#########')
    print(run_key)
    print(automl.automl_.runhistory_.data[run_key])
    
run_key = list(automl.automl_.runhistory_.data.keys())[0]
run_value = automl.automl_.runhistory_.data[run_key]

print("Configuration ID:", run_key.config_id)
print("Instance:", run_key.instance_id)
print("Seed:", run_key.seed)
print("Budget:", run_key.budget)

print(automl.automl_.runhistory_.ids_config[run_key.config_id])

run_key = list(automl.automl_.runhistory_.data.keys())[0]
run_value = automl.automl_.runhistory_.data[run_key]

print("Cost:", run_value.cost)
print("Time:", run_value.time)
print("Status:", run_value.status)
print("Additional information:", run_value.additional_info)
print("Start time:", run_value.starttime)
print("End time", run_value.endtime)

losses_and_configurations = [
    (run_value.cost, run_key.config_id)
    for run_key, run_value in automl.automl_.runhistory_.data.items()
]
losses_and_configurations.sort()
print("Lowest loss:", losses_and_configurations[0][0])
print(
    "Best configuration:",
    automl.automl_.runhistory_.ids_config[losses_and_configurations[0][1]]
)

print(automl.cv_results_)

#Inspect the components of the best model
for i, (weight, pipeline) in enumerate(automl.get_models_with_weights()):
    for stage_name, component in pipeline.named_steps.items():
        if 'feature_preprocessor' in stage_name:
            print(
                "The {}th pipeline has a explained variance of {}".format(
                    i,
                    # The component is an instance of AutoSklearnChoice.
                    # Access the sklearn object via the choice attribute
                    # We want the explained variance attributed of
                    # each principal component
                    component.choice.preprocessor.explained_variance_ratio_
                )
            )