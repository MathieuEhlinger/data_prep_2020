from defined_datasets import ds_benchmark_BiDirect as ds
from ml_run_lib.bids_metaexp import BIDSAsklExperiment
import pandas as pd
import sys

#Add library to path
lib_path = 'path_to_lib'
sys.path.append(lib_path)

#Add path to original datasets
main_path='path_to_original_datasets'

#Add path to data
path_to_csv_data = 'add_path_to_csv_data'

#Add path to me data
path_to_me_data = 'add_path_to_me_data'


#-------------------------------------------------------------
#Current presentation and DA results are from those ME
#Already ran !


exp_bnr=BIDSAsklExperiment(library=ds.TC_Brains_lib, name='new_native')
exp_bwr=BIDSAsklExperiment(library=ds.TC_residuals_Brains_lib, name='new_residuals')
exp_bdn=BIDSAsklExperiment(library=ds.TC_Network_lib, name='new_networks')

cross_val_set=pd.read_csv(path_to_csv_data+"/csv_army_BiDirect/new_fits/re_cross_val_set.csv",index_col=0)
exploration_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_exploration_set.csv",index_col=0)

exp_bwr.train_index=exploration_set.index
exp_bnr.train_index=exploration_set.index
exp_bdn.train_index=exploration_set.index

exp_bnr.test_index=cross_val_set.index
exp_bwr.test_index=cross_val_set.index
exp_bdn.test_index=cross_val_set.index

exp_bnr.save()
exp_bwr.save()
exp_bdn.save()

#-------------------------------------------------------------
#Already ran
'''
Run with optimum hypothesis
'''

from defined_datasets import ds_benchmark_BiDirect as ds
from ml_run_lib.bids_metaexp import BIDSAsklExperimentOther
import pandas as pd

exp_bwr=BIDSAsklExperimentOther(library=ds.TC_residuals_Brains_lib, name='new_residuals_proto')

cross_val_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_cross_val_set.csv",index_col=0)
exploration_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_exploration_set.csv",index_col=0)

exp_bwr.train_index=exploration_set.index
exp_bwr.test_index=cross_val_set.index
exp_bwr.save()

#-------------------------------------------------------------

#OH = 0.65. Already ran. only GMV wr

#Seems to have prevented overfitting from the ensembles on train set! Results similar though

from defined_datasets import ds_benchmark_BiDirect as ds
from ml_run_lib.bids_metaexp import BIDSAsklExperimentOther
import pandas as pd

exp_bwr=BIDSAsklExperimentOther(library=ds.TC_residuals_Brains_lib, name='new_residuals_proto_2')

cross_val_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_cross_val_set.csv",index_col=0)
exploration_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_exploration_set.csv",index_col=0)

exp_bwr.train_index=exploration_set.index
exp_bwr.test_index=cross_val_set.index
exp_bwr.save()


#-------------------------------------------------------------
#Have to test some other settings. 

from defined_datasets import ds_benchmark_BiDirect as ds
from ml_run_lib.bids_metaexp import BIDSAsklExperimentHighThresh
import pandas as pd

exp_bwr=BIDSAsklExperimentHighThresh(library=ds.TC_residuals_Brains_lib, name='new_residuals_proto_3')

cross_val_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_cross_val_set.csv",index_col=0)
exploration_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_exploration_set.csv",index_col=0)

exp_bwr.phase_1_exp_other_args={'threadp':1, 'askl_2':False, 'askl_config':{'ensemble_size':1,'time_left_for_this_task':3600*3, 'memory_limit': 3072*4,'n_jobs':30, 'per_run_time_limit': 30*60, 
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
exp_bwr.train_index=exploration_set.index
exp_bwr.test_index=cross_val_set.index
exp_bwr.save()

#-------------------------------------------------------------
#Run with ensemble_size = 10. OH=0.70. Askl2 this time

#Currently running !

from defined_datasets import ds_benchmark_BiDirect as ds
from ml_run_lib.bids_metaexp import BIDSAsklExperimentHighThresh
import pandas as pd

exp_bwr=BIDSAsklExperimentHighThresh(library=ds.TC_residuals_Brains_lib, name='new_residuals_proto_3_2')

cross_val_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_cross_val_set.csv",index_col=0)
exploration_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_exploration_set.csv",index_col=0)

exp_bwr.phase_1_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'ensemble_size':1,'time_left_for_this_task':3600*3, 'memory_limit': 3072*4,'n_jobs':30, 'per_run_time_limit': 30*60}}
#'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
exp_bwr.train_index=exploration_set.index
exp_bwr.test_index=cross_val_set.index
exp_bwr.save()


#-------------------------------------------------------------
# Run with ensemble_size = 1. Askl2 this time

from defined_datasets import ds_benchmark_BiDirect as ds
from ml_run_lib.bids_metaexp import BIDSAsklExperimentHighThresh
import pandas as pd

exp_bwr=BIDSAsklExperimentHighThresh(library=ds.TC_residuals_Brains_lib, name='new_residuals_proto_4')

cross_val_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_cross_val_set.csv",index_col=0)
exploration_set=pd.read_csv(path_to_csv_data+"csv_army_BiDirect/new_fits/re_exploration_set.csv",index_col=0)

exp_bwr.phase_1_exp_other_args={'threadp':1, 'askl_2':False, 'askl_config':{'ensemble_size':10,'time_left_for_this_task':3600*3, 'memory_limit': 3072*4,'n_jobs':30, 'per_run_time_limit': 30*60, 
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
exp_bwr.train_index=exploration_set.index
exp_bwr.test_index=cross_val_set.index
exp_bwr.save()

