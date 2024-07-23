import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

#Add path to save_folder
path_save_folder = 'path_ save_folder'

from ml_runs_libs.bids_metaexp import BIDSAsklExperiment

exp_bnr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_native/','askl_bids_new_native')
exp_bwr=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_residuals/','askl_bids_new_residuals')
exp_bdn=BIDSAsklExperiment.load_from_pickle(path_to_me_data+'/askl_bids_new_networks/','askl_bids_new_networks')

def change_width(ax, new_value) :
    for patch in ax.patches :
      current_width = patch.get_width()
      diff = current_width - new_value
      # we change the bar width
      patch.set_width(new_value)
      # we recenter the bar
      patch.set_x(patch.get_x() + diff * .5)

def double_ticks(ticks):
  new_ticks = np.arange(min(ticks), max(ticks)+((max(ticks)-min(ticks))/(len(ticks)-1)/2), (max(ticks)-min(ticks))/(len(ticks)-1)/2)
  return [round(i,2) for i in new_ticks]

def tenth_ticks(ticks):
  new_ticks = np.arange(min(ticks), max(ticks), (max(ticks)-min(ticks))/(len(ticks)-1)/10)
  return [round(i,2) for i in new_ticks]

def dilute_for_ticks(outnumbered_ticks, real_ticks):
  new_ticks=list()
  for tick in real_ticks:
    if tick in outnumbered_ticks:
      new_ticks.append(str(tick))
    else:
      new_ticks.append('')
  return new_ticks

def dilute_for_ticks_percent(outnumbered_ticks, real_ticks):
  new_ticks=list()
  for tick in real_ticks:
    if tick in outnumbered_ticks:
      new_ticks.append(str(int(tick*100))+'%')
    else:
      new_ticks.append('')
  return new_ticks


me_exp = exp_bnr
me_exp_r = exp_bwr
associated_dummy_score = me_exp.phase_2a_dummy_score
associated_dummy_score_r = me_exp_r.phase_2a_dummy_score
score=me_exp.phase_2a_exp_retest_score
score_r=me_exp_r.phase_2a_exp_retest_score
to_extract='test_balanced_accuracy_score'
score_legend='Balanced accuracy'
title='Depression classification performance for different modalities' 
from_these=None
x=10
library=me_exp.library
library_r=me_exp_r.library

results            =   me_exp.generate_one_result_df(score_o=score,to_extract=to_extract)
results_r          =   me_exp_r.generate_one_result_df(score_o=score_r,to_extract=to_extract)
dummy_values       =   me_exp.return_simplified_score(associated_dummy_score, to_extract)
dummy_values_r       =   me_exp_r.return_simplified_score(associated_dummy_score_r, to_extract)

best_dummy_name    =   dummy_values.sort_values('average',ascending=False).iloc[0].name
best_dummy_name_r   =  dummy_values_r.sort_values('average',ascending=False).iloc[0].name
best_dummy_score   =   me_exp.generate_one_result_df(score_o=associated_dummy_score, to_extract=to_extract)[best_dummy_name]
best_dummy_score_r   =   me_exp_r.generate_one_result_df(score_o=associated_dummy_score_r, to_extract=to_extract)[best_dummy_name_r]
columns            =   results.mean()
columns_r            =   results_r.mean()

if from_these is not None : columns            =   columns[from_these].sort_values(ascending=False).index[:x]
else :                      columns            =   columns.sort_values(ascending=False).index[:x]

if from_these is not None : columns_r            =   columns_r[from_these].sort_values(ascending=False).index[:x]
else :                      columns_r            =   columns_r.sort_values(ascending=False).index[:x]


library['BiDirect_New_Matched_All_Structural'].description='GMV + CT + SA'
library['BiDirect_New_Matched_Surface_Area'].description='SA'
library['BiDirect_New_Matched_Cortical_Thickness'].description='CT'
library['BiDirect_New_Matched_Grey_Matter_Volume'].description='GMV'
library['BiDirect_New_Matched_Confounders'].description='Confounders'

library_r['BiDirect_New_Matched_All_Structural_residuals'].description='GMV + CT + SA'
library_r['BiDirect_New_Matched_Surface_Area_residuals'].description='SA'
library_r['BiDirect_New_Matched_Cortical_Thickness_residuals'].description='CT'
library_r['BiDirect_New_Matched_Grey_Matter_Volume_residuals'].description='GMV'
library_r['BiDirect_New_Matched_Confounders'].description='Confounders'

#Renaming the columns for clarity
new_index_dict = dict()
for lib in library :
  new_index_dict[lib]=library[lib].description

results = results.rename(columns=new_index_dict)
new_columns = [new_index_dict[i] for i in columns]

new_index_dict_r = dict()
for lib in library_r :
  new_index_dict_r[lib]=library_r[lib].description

results_r = results_r.rename(columns=new_index_dict_r)
new_columns_r = [new_index_dict_r[i] for i in columns_r]

df=results
df=df[['GMV + CT + SA','GMV','CT','SA','Confounders']]
df_r=results_r
df_r=df_r[['GMV + CT + SA','GMV','CT','SA','Confounders']]
#df['Random Predictor'] = best_dummy_score
columns=list(new_columns)
columns_r=list(new_columns_r)

#All the preprocessing of the data is now done. We know have to make it look pretty
fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

#Limiting the upper size
ax.set_ylim(0,1)
ax2.set_ylim(0,1)

#Setting width of columns
width = 0.3

#Setting the y ticks as wanted
ax.yaxis.tick_left()
ax2.yaxis.tick_left()
ax.set_yticks(double_ticks(double_ticks(ax.get_yticks())))
#ax.set_yticks(double_ticks(ax.get_yticks()))
ticks=ax2.get_yticks()
ax2.set_yticks(double_ticks(double_ticks(ax2.get_yticks())))

#Naming the ticks either with their number OR with ''
ax.set_yticklabels(dilute_for_ticks_percent(double_ticks(ticks),double_ticks(double_ticks(ticks))))
ax2.set_yticklabels(dilute_for_ticks_percent(double_ticks(ticks),double_ticks(double_ticks(ticks))))

#Declaring the 2 plots
plot1=df.mean().abs().plot(kind='bar', color='royalblue', ax=ax, yerr=df[['GMV + CT + SA','GMV','CT','SA','Confounders']].std(), width=width, position=1.05,rot=0)
plot2=df_r.mean().abs().plot(kind='bar', color='navy', ax=ax2,yerr=df_r[['GMV + CT + SA','GMV','CT','SA','Confounders']].std(), width=width, position=-0.05,rot=0)

#Handling the legend
colors = {'Raw data':'royalblue', 'Residuals':'navy'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

ax.set_ylabel(score_legend)

margin = (1 - width) + width / 2
ax.set_xlim(-margin, len(df.columns) - 1 + margin)
ax2.set_xlim(-margin, len(df.columns) - 1 + margin)

myline = plt.axhline(y=best_dummy_score.mean(), color='black', linestyle = 'dotted')
labels.append('Best random classifier')
handles.append(myline)

#plt.legend(handles, labels, bbox_to_anchor=(1.04,1), loc="lower left")
plt.legend(handles, labels)
#plt.ylabel(score_legend)
plt.title(title)
plt.tight_layout()

plt.savefig(path_save_folder+'/results_bids_4.png', bbox_inches="tight")
plt.legend(handles, labels, bbox_to_anchor=(1.04,1), loc="lower left")
plt.savefig(path_save_folder+'/results_bids_4_legend.png', bbox_inches="tight")