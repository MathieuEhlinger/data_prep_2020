'''
This script serves the purpose of allowing to perform residuals calculation with eTIV on 1000 Brains, then standardization
Then uses the trained models on 1000 Brains to do the same on BiDirect !
'''

import sys
import main_1000BRAINS as brains 
import main_bidirect as bd
import statsmodels.api as sm
import pandas as pd
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import numpy as np

#Add library to path
lib_path = 'path_to_lib'

sys.path.append(lib_path+'/data_processing/data_prep/1000_BRAINS/')
sys.path.append(lib_path+'/data_processing/data_prep/BiDirect/')

save_brain=False
save_bd=False

print('----------------------------------------')
print('Save for 1000 Brains :', save_brain)
print('Save for BiDirect :', save_bd)
print('----------------------------------------')

def output_residual_output_model(x,y):  
  #Outputs a LinearRegression model fitted on x,y
  model = LinearRegression()
  model.fit(x, y)
  return model

def output_scaler(x):  
  #Outputs a StandardScaler model fitted on x
  model = StandardScaler(with_mean=True, with_std =True )
  model.fit(x)
  return model

#Some variables initialisation
#Models for standardscaling, models for regression and residuals calculation, and the new residuals
scaling_models, res_models = dict(), dict()

#Gather the origin dataset before transformations
Brains_Origin= {'Surface_Area':brains.df_sa, 'Cortical_Thickness':brains.df_ta, 'Grey_Matter_Volume':brains.df_gv}
BiDirect_origin= {'Surface_Area':bd.matched_sa, 'Cortical_Thickness':bd.matched_ct, 'Grey_Matter_Volume':bd.matched_gv}

#Those will be the outputed datasets after transformation ^^
Brains_output= {'Surface_Area':deepcopy(brains.df_sa), 'Cortical_Thickness':deepcopy(brains.df_ta), 'Grey_Matter_Volume':deepcopy(brains.df_gv)}
BiDirect_output= {'Surface_Area':deepcopy(bd.matched_sa), 'Cortical_Thickness':deepcopy(bd.matched_ct), 'Grey_Matter_Volume':deepcopy(bd.matched_gv)}
true_etiv= bd.df_etiv.loc[bd.rematched_df.index]['eTIV']

#Generating the 1000 Brains data data
for data_type in Brains_Origin:
  res_models[data_type]=dict()
  scaling_models[data_type]=dict()
  for column in Brains_Origin[data_type].columns:
    res_models[data_type][column]= output_residual_output_model(brains.inf_ds.eTIV.values[:, np.newaxis], Brains_Origin[data_type][column].values.reshape(-1, 1))
    predictions = res_models[data_type][column].predict(brains.inf_ds.eTIV.values[:, np.newaxis])
    predictions = Brains_Origin[data_type][column]-predictions.flatten()
    scaling_models[data_type][column]=output_scaler(predictions.values[:, np.newaxis])
    Brains_output[data_type][column]= scaling_models[data_type][column].transform(predictions.values[:, np.newaxis]).flatten()
    
#Generating the BiDirect data
for data_type in Brains_Origin:
  for column in Brains_Origin[data_type].columns:
    predictions = res_models[data_type][column].predict(true_etiv.values[:, np.newaxis])
    predictions = BiDirect_origin[data_type][column]-predictions.flatten()
    BiDirect_output[data_type][column] = scaling_models[data_type][column].transform(predictions.values[:, np.newaxis]).flatten()



#Outputted csv for 1000 Brains
Brains_output['All_Structural']=pd.concat([Brains_output['Grey_Matter_Volume'], Brains_output['Surface_Area'], Brains_output['Cortical_Thickness']],axis=1)

if save_brain: Brains_output['Cortical_Thickness'].to_csv(brains.main_path+'1000_Brains_Cortical_Thickness_comparison_residuals.csv')

if save_brain:Brains_output['Grey_Matter_Volume'].to_csv(brains.main_path+'1000_Brains_Grey_Matter_Volume_comparison_residuals.csv')

if save_brain:Brains_output['Surface_Area'].to_csv(brains.main_path+'1000_Brains_Surface_Area_comparison_residuals.csv')

if save_brain:Brains_output['All_Structural'].to_csv(brains.main_path+'1000_Brains_All_Structural_comparison_residuals.csv')

#Outputted csv for BiDirect
if save_bd:BiDirect_output['All_Structural']=pd.concat([BiDirect_output['Grey_Matter_Volume'], BiDirect_output['Surface_Area'], BiDirect_output['Cortical_Thickness']],axis=1)

if save_bd:BiDirect_output['Cortical_Thickness'].to_csv(bd.main_path+'BiDirect_New_Matched_Cortical_Thickness_comparison_residuals.csv')

if save_bd:BiDirect_output['Grey_Matter_Volume'].to_csv(bd.main_path+'BiDirect_New_Matched_Grey_Matter_Volume_comparison_residuals.csv')

if save_bd:BiDirect_output['Surface_Area'].to_csv(bd.main_path+'BiDirect_New_Matched_Surface_Area_comparison_residuals.csv')

if save_bd:BiDirect_output['All_Structural'].to_csv(bd.main_path+'BiDirect_New_Matched_All_Structural_comparison_residuals.csv')
