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

from ml_run_lib.bids_metaexp import BIDSAsklExperiment

#With the new dataset
exp_bnr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_native/','askl_bids_new_native')
exp_bwr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_residuals/','askl_bids_new_residuals')
exp_bdn=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_networks/','askl_bids_new_networks')

#Sex
from ml_run_lib.gender_metaexp import GenderAsklExperiment
exp_gnr=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default/','askl_gender_default')
exp_gwr=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default_1/','askl_gender_default')

#Age
from ml_run_lib.age_metaexp import AgeAsklExperiment
exp_anr=AgeAsklExperiment.load_from_pickle(path_to_me_data+'/askl_age_default_4/','askl_age_default')
exp_awr=AgeAsklExperiment.load_from_pickle(path_to_me_data+'/askl_age_default_0/','askl_age_default')

from ml_run_lib.model_producing_askl import multiproc_askl

from ml_run_lib.gender_metaexp import GenderMetaExperiment
exp=GenderMetaExperiment.load_from_pickle(path_to_me_data+'/exp_gender_New_Genme/','exp_gender_New_Genme')

from ml_run_lib.age_metaexp import AgeMetaExperiment
exp=AgeMetaExperiment.load_from_pickle(path_to_me_data+'/exp_age_New_ageme/','exp_age_New_ageme')

from ml_run_lib.age_metaexp import AgeAsklExperiment
exp_age=AgeAsklExperiment.load_from_pickle(path_to_me_data+'/askl_age_dummy_default_10/','askl_age_dummy_default')

from ml_run_lib.gender_metaexp import GenderAsklExperiment
exp=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default/','askl_gender_default')
exp=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default_1/','askl_gender_default')

X=exp.library['df_gv_na'].df_data
y=exp.target_ds
estimator=exp.models_dict['df_gv_na']

import defined_datasets.ds_normed as ds
exp=AgeAsklExperiment(library=ds.test_ds_lib)
exp=AgeAsklExperiment(dummy=True)
exp.phase_1_exp= multiproc_askl

from ml_run_lib.age_metaexp import AgeAsklExperiment
exp_age=AgeAsklExperiment.load_from_pickle(path_to_me_data+'/askl_age_default_4/','askl_age_default')

from ml_run_lib.gender_metaexp import GenderAsklExperiment
#from defined_datasets import ds_benchmark_1000B as ds
from defined_datasets import ds_benchmark_BiDirect as ds

