'''
Script designed to prepare the input datasets for the ML runs
'''

import pandas as pd
import sys
import numpy as np
from organiser import repartition

#Add library to path
lib_path = 'path_to_lib'
sys.path.append(lib_path)

#Add path to original datasets
main_path='path_to_original_datasets'

#Add path to data
path_to_data = 'add_path_to_data'

#Add path to me data
path_to_me_data = 'add_path_to_me_data'

df=pd.read_excel(main_path+"1000BRAINS-all-struct/aparc-all-stats-QC.xlsx")

#Change to True in order to write an output
save= False

if not save: print('Note : Save is set to False - Won\'t rewrite files')

#Cleaning some inputs out due to insufficient data quality in QC
forbidden_index=pd.read_excel(main_path+'1000BRAINS-all-struct/aparc-stats-all-index-QC.xlsx')
forbidden_ind = np.array([i[1:] for i in forbidden_index.values])
forbidden_index = forbidden_ind[~np.isnan(forbidden_ind)].astype(int)
sup_len=df.shape[0]+len(np.unique(forbidden_index))

print('Forbidden index and df index matches :', pd.Index.intersection(df.index,forbidden_index))

#Verifying lengths
forbidden_df=pd.read_excel(main_path+'1000BRAINS-all-struct/aparc-all-stats.xlsx')
forbidden_df_2=pd.read_excel(main_path+'1000BRAINS-all-struct/aparc-all-stats-age.xlsx')

df = df.set_index('Unnamed: 0')

df = df[~df.index.duplicated()] #<- Doesn't change anything, cause no duplicates

#group_BIDS==1 now means depressive subject
#{df['group_BIDS']=df['group_BIDS'].apply(lambda x: 1 if x==1 else 0)
df['Sex']=df['Sex'].apply(lambda x: 1 if x=='Female' else 0)
df.index.name='Subject'
#Description of initial set
#pd.concat([df_d.describe(),df_d.loc[df_d.Sex==1].describe(),df_d.loc[df_d.Sex==0].describe()]).to_csv(main_path+'description_female_male.csv')

#Generating subdataframes
df_gv =  df[[i for i in df.columns if '_volume' in i]]
if save: df_gv.to_csv(main_path+'1000_Brains_Grey_Matter_Volume.csv')
df_sa =  df[[i for i in df.columns if '_area' in i]]
if save: df_sa.to_csv(main_path+'1000_Brains_Surface_Area.csv')
df_ta =  df[[i for i in df.columns if '_thickness' in i]]
if save: df_ta.to_csv(main_path+'1000_Brains_Cortical_Thickness.csv')

df_na=pd.concat([df_gv, df_sa, df_ta],axis=1)
if save: df_na.to_csv(main_path+'1000_Brains_All_Structural.csv')

inf_ds=df[['lh.Schaefer2018_400Parcels_7Networks_gcs.volume', 'BrainSegVolNotVent', 'eTIV', 'Age', 'Sex', 'volume_mean', 'thickness_mean', 'surface_mean']]
if save: inf_ds.to_csv(main_path+'1000_Brains_general_information.csv')

#Residuals for ETIV
df_gv_residuals=df_gv.apply(lambda x:repartition.residuals_for_x(inf_ds.eTIV,x))
if save: df_gv_residuals.to_csv(main_path+'1000_Brains_Grey_Matter_Volume_residuals.csv')
df_sa_residuals=df_sa.apply(lambda x:repartition.residuals_for_x(inf_ds.eTIV,x))
if save: df_sa_residuals.to_csv(main_path+'1000_Brains_Surface_Area_residuals.csv')
df_ta_residuals=df_ta.apply(lambda x:repartition.residuals_for_x(inf_ds.eTIV,x))
if save: df_ta_residuals.to_csv(main_path+'1000_Brains_Cortical_Thickness_residuals.csv')

df_na_residuals=pd.concat([df_gv_residuals, df_sa_residuals, df_ta_residuals],axis=1)
if save: df_na_residuals.to_csv(main_path+'1000_Brains_All_Structural_residuals.csv')

#Residuals for ETIV with complement
df_gv_residuals_c=df_gv.apply(lambda x:repartition.residuals_for_x(inf_ds.eTIV,x))

df_sa_residuals_c=df_sa.apply(lambda x:repartition.residuals_for_x(inf_ds.eTIV,x))

df_ta_residuals_c=df_ta.apply(lambda x:repartition.residuals_for_x(inf_ds.eTIV,x))

df_na_residuals_c=pd.concat([df_gv_residuals, df_sa_residuals, df_ta_residuals,df[df.columns[-7:]]],axis=1)
if save: df_na_residuals_c.to_csv(main_path+'1000_Brains_All_Structural_residuals_c.csv')

df_gv_residuals_c=pd.concat([df_gv_residuals_c,df[df.columns[-7:]]],axis=1)
if save: df_gv_residuals_c.to_csv(main_path+'1000_Brains_Grey_Matter_Volume_residuals_c.csv')

df_ta_residuals_c=pd.concat([df_ta_residuals_c,df[df.columns[-7:]]],axis=1)
if save: df_ta_residuals_c.to_csv(main_path+'1000_Brains_Cortical_Thickness_residuals_c.csv')

df_sa_residuals_c=pd.concat([df_sa_residuals_c,df[df.columns[-7:]]],axis=1)
if save: df_sa_residuals_c.to_csv(main_path+'1000_Brains_Surface_Area_residuals_c.csv')