import pandas as pd
import sys
from organiser import repartition
from copy import deepcopy
import copy 

from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency

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

#Add paths prior to running script
main_read_path='add_main_read_path_struct_data'
main_path='add_main_path_global_data'
struct_path='add_main_path_global_data'


if not save: print('Note : Save is set to False - Won\'t rewrite files')
#Gathering surface area, cortical thickness, and volume data in Schaefer Parcellation
df_sa=   pd.concat([pd.read_csv(struct_path+'Schaefer2018_400Parcels_7Networks-area-lh-stats.csv', index_col=0), pd.read_csv(struct_path+'Schaefer2018_400Parcels_7Networks-area-rh-stats.csv', index_col=0)],axis=1).drop(['BrainSegVolNotVent',	'eTIV'],axis=1)
df_sa= df_sa.drop(['rh_WhiteSurfArea_area', 'lh_WhiteSurfArea_area'],axis=1)
df_ct=   pd.concat([pd.read_csv(struct_path+'Schaefer2018_400Parcels_7Networks-thickness-lh-stats.csv', index_col=0), pd.read_csv(struct_path+'Schaefer2018_400Parcels_7Networks-thickness-rh-stats.csv', index_col=0)],axis=1).drop(['BrainSegVolNotVent',	'eTIV'],axis=1)
df_ct= df_ct.drop(['lh_MeanThickness_thickness', 'rh_MeanThickness_thickness'],axis=1)
df_vol=  pd.concat([pd.read_csv(struct_path+'Schaefer2018_400Parcels_7Networks-volume-lh-stats.csv', index_col=0), pd.read_csv(struct_path+'Schaefer2018_400Parcels_7Networks-volume-rh-stats.csv', index_col=0)],axis=1).drop(['BrainSegVolNotVent',	'eTIV'],axis=1)
df_etiv=pd.read_csv(struct_path+'Schaefer2018_400Parcels_7Networks-area-rh-stats.csv', index_col=0)[['BrainSegVolNotVent',	'eTIV']]
df_sa_start_value=set(deepcopy(df_sa.index))

#Eliminating values showing a too high anomaly after IQR calculation
df_sa_out, eliminated_sa_high, eliminated_sa_low = repartition.df_after_iqr_calculations(df_sa)
df_ct_out, eliminated_ct_high, eliminated_ct_low  = repartition.df_after_iqr_calculations(df_ct)
df_vol_out, eliminated_vol_high, eliminated_vol_low = repartition.df_after_iqr_calculations(df_vol)

#Final indexes that were taken out for IQR
eliminated_index_iqr = eliminated_sa_high + eliminated_sa_low + eliminated_ct_high + eliminated_ct_low + eliminated_vol_high + eliminated_vol_low


#Data like Gender, age, BIDS, gathered from BiDirect
df=pd.read_csv(main_read_path+'bids_csv_export.csv',sep=';')
df.s0_age=df.s0_age.str.replace(',', '.').astype(float)
df['Age']=df.s0_age

#Setting the correct index
df = df.set_index('idbidirect')
df_start_index=set(df.index)

print('--------------------------------------')
print('Initial size of information dataset : ', df.shape[0])
print('First step: Elimination by IQR. Number of subjects sorted out : ', len(set(eliminated_index_iqr)))
df=df.drop(eliminated_index_iqr)
print('Size of information dataset : ', df.shape[0])
print('')
df_BIDS=pd.read_csv(main_read_path+'sensitive_subject_information_export_new.csv')
df_BIDS = df_BIDS.set_index('subject')

eliminated_indexes_others=[str(i) for i in df_BIDS.loc[df_BIDS.index.duplicated()].index]+ [str(i) for i in df.loc[df.index.duplicated()].index]
print('--------------------------------------')
print('Eliminating data with duplicates. Total : ', len(eliminated_indexes_others))
df=df.drop(eliminated_indexes_others,errors='ignore')
print('Size of information dataset : ', df.shape[0])
print('')
df = df[~df.index.duplicated()]
df_BIDS = df_BIDS[~df_BIDS.index.duplicated()]
#sub-30160'

#group_BIDS==1 now means depressive subject
df_BIDS['group_BIDS']=df_BIDS['group_BIDS'].apply(lambda x: 1 if x==1 else 0)
df['Sex']=df['sex'].apply(lambda x: 1 if x=='female' else 0)
df=df.drop('sex',axis=1)

df['BIDS']=df_BIDS['group_BIDS']

temp_index=set(df.index)
df=df.drop('s0_age',axis=1).dropna()
removed_index_2=temp_index-set(df.index)
print('--------------------------------------')
print('Second step: Elimination by missing Age, Sex or BIDS. Number of subjects sorted out : ', len(set(removed_index_2)))
df=df.drop(removed_index_2,errors='ignore')
print('Size of information dataset : ', df.shape[0])

print('')
old_index=set(df.index)
common_index=pd.Index.intersection(df.index,df_sa.index)
df=df.loc[common_index]
removed_index_3=set(old_index)-set(common_index)
print('--------------------------------------')
print('Third step: Elimination by missing match between general data and structural data. Number of subjects sorted out : ', len(set(removed_index_3)))
df=df.drop(removed_index_3,errors='ignore')
print('Size of information dataset : ', df.shape[0])

print('')

set_1=set(list(eliminated_index_iqr) + list(removed_index_2)+list(removed_index_3)+list(eliminated_indexes_others))
print('--------------------------------------')
print('The two following numbers must match :')
print('Final number of subjects removed :',len(set_1))
set_2=set(list(df_start_index) + list(df_sa_start_value))-set(df.index)
print('Control - other way of calculating, number removed : ', len(set_2))
print('--------------------------------------')

#Propensity score matching for BIDS and Sex
'''
features=['BIDS','Sex']

matched_df= repartition.propensity_score_matching(df, features)
index_for_d=copy.deepcopy(matched_df.index)
matched_df.index=([int(i[4:]) for i inrematched_df.index])
matched_df.to_csv(main_path+'BiDirect_New_Matched_General_Information.csv')
df_etiv.index=[int(i[4:]) for i in df_etiv.index]
split_1, split_2 = repartition.split_in_matching_datasets(matched_df, [['BIDS', bool],['Sex', bool], ['Age', float]]    )
split_1.to_csv(main_path+'exploration_set.csv')
split_2.to_csv(main_path+'cross_val_set.csv')
'''

#Propensity score matching for BIDS and Age + Sex
rematched_df=deepcopy(df)
rematched_df.index=([int(i[4:]) for i in rematched_df.index])
start_index_p=set(copy.deepcopy(rematched_df.index))
#features=['BIDS','Age','Sex']
propensity = LogisticRegression()
propensity = propensity.fit(rematched_df[['Sex','Age']], rematched_df.BIDS)
pscore = propensity.predict_proba(rematched_df[['Sex','Age']])[:,1]
rematched_df['Propensity'] = pscore
stuff = repartition.prop_match(rematched_df.BIDS, rematched_df.Propensity).dropna()
rematched_df=rematched_df.drop(['Propensity'],axis=1).sort_index()
rematched_df = pd.concat([rematched_df.loc[stuff.index], rematched_df.loc[[int(i) for i in stuff.tolist()]]])
index_for_d=copy.deepcopy(rematched_df.index)

print('--------------------------------------')
print('Starting propensity score matching')
last_index_p=set(list(stuff.index)+list([int(i) for i in stuff.tolist()]))
print('Before drop : ', len(start_index_p))
print('Dropped : ', len(start_index_p-last_index_p))
print('Size of both samples :  ', rematched_df.shape[0])
print('--------------------------------------')


if save: rematched_df.to_csv(main_path+'BiDirect_Rematched_General_Information.csv')
df_etiv.index=[int(i[4:]) for i in df_etiv.index]
print('Creating splits matched for BIDS, Sex & Age')
split_1, split_2 = repartition.split_in_matching_datasets(rematched_df, [['BIDS', bool],['Sex', bool], ['Age', float]]    )
if save: split_1.to_csv(main_path+'re_exploration_set.csv')
if save: split_2.to_csv(main_path+'re_cross_val_set.csv')


#Generating matched dataset
matched_sa=df_sa
matched_sa.index=[int(i[4:]) for i in matched_sa.index]
matched_sa=df_sa.loc[index_for_d]
if save: matched_sa.to_csv(main_path+'BiDirect_New_Matched_Surface_Area.csv')

matched_ct=df_ct
matched_ct.index=[int(i[4:]) for i in matched_ct.index]
matched_ct=df_ct.loc[index_for_d]
if save: matched_ct.to_csv(main_path+'BiDirect_New_Matched_Cortical_Thickness.csv')

matched_gv=df_vol
matched_gv.index=[int(i[4:]) for i in matched_gv.index]
matched_gv=df_vol.loc[index_for_d]
if save: matched_gv.to_csv(main_path+'BiDirect_New_Matched_Grey_Matter_Volume.csv')

matched_df_all_df=pd.concat([matched_gv,matched_sa,matched_ct],axis=1)
if save: matched_df_all_df.to_csv(main_path+'BiDirect_New_Matched_All_Structural.csv')

#Residuals for ETIV
matched_gv_residuals=matched_gv.apply(lambda x:repartition.residuals_for_x(df_etiv.loc[rematched_df.index].eTIV,x))
if save: matched_gv_residuals.to_csv(main_path+'BiDirect_New_Matched_Grey_Matter_Volume_residuals.csv')

matched_sa_residuals=matched_sa.apply(lambda x:repartition.residuals_for_x(df_etiv.loc[rematched_df.index].eTIV,x))
if save: matched_sa_residuals.to_csv(main_path+'BiDirect_New_Matched_Surface_Area_residuals.csv')

matched_ct_residuals=matched_ct.apply(lambda x:repartition.residuals_for_x(df_etiv.loc[rematched_df.index].eTIV,x))
if save: matched_ct_residuals.to_csv(main_path+'BiDirect_New_Matched_Cortical_Thickness_residuals.csv')

matched_df_all_df_residuals=matched_df_all_df.apply(lambda x:repartition.residuals_for_x(df_etiv.loc[rematched_df.index].eTIV,x))
if save: matched_df_all_df_residuals.to_csv(main_path+'BiDirect_New_Matched_All_Structural_residuals.csv')

confounder_df =rematched_df[['Sex','Age']]
if save: confounder_df.to_csv(main_path+'BiDirect_New_Matched_Confounders.csv')

#Checking if there are similar patients in the splits
temp_1_ind = split_1.sort_index()
temp_2_ind = split_2.sort_index()
a = temp_1_ind.index.searchsorted(temp_2_ind.index[0])
if a == 0: print('Splits do not have same subjects')
else: print('Warning: Splits have same subjects')

#Saving all those data sets
hc_df_matched=rematched_df.loc[rematched_df['BIDS']==0]
if save: hc_df_matched.to_csv(main_path+'healthy_cohort_matched.csv')
ds_df_matched=rematched_df.loc[rematched_df['BIDS']==1]
if save: ds_df_matched.to_csv(main_path+'ds_matched.csv')
description_hvd=pd.concat([hc_df_matched.describe(),ds_df_matched.describe()])
if save: description_hvd.to_csv(main_path+'hc_vs_ds_desc.csv')

description=pd.concat([rematched_df.describe(),split_1.describe(),split_2.describe()])
if save: description.to_csv(main_path+'description_of_matched_df_split_1_split_2.csv')

#x=df.Age.plot.density()
#x.figure.savefig('/home/homeGlobal/mehlinger/test_env/clean_project/data_prep/drawing.png')

cont_columns = [str(i) for i in matched_ct.columns if '_Cont_' in str(i)]
limbic_columns = [str(i) for i in matched_ct.columns if '_Limbic_' in str(i)]
Default_columns = [str(i) for i in matched_ct.columns if '_Default_' in str(i)]
SomMot_columns = [str(i) for i in matched_ct.columns if '_SomMot_' in str(i)]
dorsalattention_columns = [str(i) for i in matched_ct.columns if '_DorsAttn_' in str(i)]
ventralattention_columns = [str(i) for i in matched_ct.columns if 'VentAttn_' in str(i)]
vis_columns = [str(i) for i in matched_ct.columns if '_Vis_' in str(i)]
group = [cont_columns,limbic_columns,Default_columns,SomMot_columns,dorsalattention_columns,ventralattention_columns,vis_columns]
flat_list = [item for sublist in group for item in sublist]


df_cont= matched_ct[cont_columns]
if save: df_cont.to_csv(main_path+'BiDirect Cont Network.csv')
df_limbic=matched_ct[limbic_columns]
if save: df_limbic.to_csv(main_path+'BiDirect Limbic Network.csv')
df_default=matched_ct[Default_columns]
if save: df_default.to_csv(main_path+'BiDirect Default Network.csv')
df_sommot=matched_ct[SomMot_columns]
if save: df_sommot.to_csv(main_path+'BiDirect Somatomotor Network.csv')
df_dorsalattn=matched_ct[dorsalattention_columns]
if save: df_dorsalattn.to_csv(main_path+'BiDirect DorsalAttention Network.csv')
df_ventralatt=matched_ct[ventralattention_columns]
if save: df_ventralatt.to_csv(main_path+'BiDirect VentralAttention Network.csv')
df_vis=matched_ct[vis_columns]
if save: df_vis.to_csv(main_path+'BiDirect Visual Network.csv')

print('--------------------------------------')
print('Performing division in sub-datasets for network experiment')
print('Is every column from ct in one network ? : ', len(set(flat_list)) == matched_ct.shape[1])
print('--------------------------------------')
#Testing BiDirect sets with t-test
#This is a two-sided test for the null hypothesis that 2 
#independent samples have identical average (expected) values. 
#This test assumes that the populations have identical variances by default.
print('Performing t-test on age for HC and depressive subjects :', ttest_ind(hc_df_matched['Age'], ds_df_matched['Age']))
#Testing BiDirect sets with chi-square
'''
contingency=pd.DataFrame([
[hc_df_matched.loc[hc_df_matched['Sex']==0].shape[0],hc_df_matched.loc[hc_df_matched['Sex']==1].shape[0]],
[ds_df_matched.loc[ds_df_matched['Sex']==0].shape[0],ds_df_matched.loc[ds_df_matched['Sex']==1].shape[0]]], 
columns = ['Male','Female'], index=['HC','Depressive subjects'])
'''
contingency= pd.crosstab(rematched_df['BIDS'], rematched_df['Sex'])
c, p, dof, expected = chi2_contingency(contingency)
print('Performing chi-square on Sex for HC and depressive subjects : ', p)
print('Chi2 : ', c, ' - Degrees of freedom : ', dof)
print('Expected frequencies : ', expected)

print('--------------------------------------')
