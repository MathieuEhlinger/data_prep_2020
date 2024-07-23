import sys
from defined_datasets import ds_benchmark_BiDirect as bd
from defined_datasets import ds_benchmark_1000B as brains 
from copy import deepcopy
import pandas as pd
import numpy as np

from ml_run_lib.gender_metaexp import GenderAsklExperiment

#Add library to path
lib_path = 'path_to_lib'
sys.path.append(lib_path)

#Add path to original datasets
main_path='path_to_original_datasets'

#Add path to data
path_to_data = 'add_path_to_data'

#Add path to me data
path_to_me_data = 'add_path_to_me_data'

exp_gnr=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default/','askl_gender_default')
exp_gwr=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default_1/','askl_gender_default')

from ml_run_lib.age_metaexp import AgeAsklExperiment
exp_anr=AgeAsklExperiment.load_from_pickle(path_to_me_data+'/askl_age_default_4/','askl_age_default')
exp_awr=AgeAsklExperiment.load_from_pickle(path_to_me_data+'/askl_age_default_0/','askl_age_default')

from ml_run_lib.bids_metaexp import BIDSAsklExperiment

exp_bnr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_native/','askl_bids_new_native')
exp_bwr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_residuals/','askl_bids_new_residuals')

def results_for_me(me, mebi):
  #Retest the predictors on the other dataset, without residuals, remade res only
  results=dict()
  result=0
  print('---------------------------')
  for df in [i[12:] for i in me.models_dict.keys()]:
    print('Calculating for :', df)
    del(result)
    me.models_dict['1000_Brains_'+df].fit(me.library['1000_Brains_'+df].df_data, me.library.target_ds)
    result = me.models_dict['1000_Brains_'+df].predict(mebi.library['BiDirect_New_Matched_'+df].df_data)
    results[df] = deepcopy(me.train_scorer(mebi.library.inf_ds[me.target], result))
  return results

def results_for_me_residuals_remade(me, mebi):
  #Retest the predictors on the other dataset, with residuals , remade
  results=dict()
  result=0
  print('---------------------------')
  for df in [i[12:-10] for i in me.models_dict.keys()]:
    print('Calculating for :', df)
    del(result)
    me.models_dict['1000_Brains_'+df+'_residuals'].fit(me.library['1000_Brains_'+df+'_comparison_residuals'].df_data, me.library.target_ds)
    result = me.models_dict['1000_Brains_'+df+'_residuals'].predict(mebi.library['BiDirect_New_Matched_'+df+'_comparison_residuals'].df_data)
    results[df] = deepcopy(me.train_scorer(mebi.library.inf_ds[me.target], result))
  return results


#The residuals can't be taken as such ! The same transformation needs to have been applied to both dataset - Therefore, the residual library from BiDirect
#is swapped for a residuals library calculated with the preprocessors of 1000 Brains

exp_bwr.library = bd.TC_comparison_residuals_Brains_lib
exp_gwr.library = brains.TC_comparison_residuals_Brains_lib
exp_awr.library = deepcopy(brains.TC_comparison_residuals_Brains_lib)

exp_awr.library.inf_ds['Sex']=deepcopy(brains.TC_comparison_residuals_Brains_lib.target_ds)
exp_awr.library.target_ds=deepcopy(brains.TC_comparison_residuals_Brains_lib.inf_ds['Age'])
exp_awr.library.target_ds=deepcopy(pd.DataFrame(exp_awr.library.target_ds))
exp_awr.library.inf_ds['Age']=np.nan
exp_awr.models_dict=exp_awr.salvage_gen_models_askl()[0]

from sklearn.metrics import mean_absolute_error
exp_anr.train_scorer=mean_absolute_error
exp_awr.train_scorer=mean_absolute_error

me_results_new_res={'Sex_without_residuals': results_for_me(exp_gnr,exp_bnr),\
'Sex_with_residuals': results_for_me_residuals_remade(exp_gwr,exp_bwr),\
'Age_without_residuals': results_for_me(exp_anr,exp_bnr),\
'Age_with_residuals':results_for_me_residuals_remade(exp_awr,exp_bwr)
}


bd_results = pd.DataFrame(me_results_new_res).rename({'Cortical_Thickness':'CT','All_Structural':'GMV + CT + SA','Grey_Matter_Volume':'GMV','Surface_Area':'SA'})
