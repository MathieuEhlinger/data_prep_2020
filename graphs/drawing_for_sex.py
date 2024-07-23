import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
#import drawing_functions

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

from ml_runs_libs.gender_metaexp import GenderAsklExperiment

exp_gnr=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default/','askl_gender_default')
exp_gwr=GenderAsklExperiment.load_from_pickle(path_to_me_data+'/askl_gender_default_1/','askl_gender_default')


def change_width(ax, new_value) :
    for patch in ax.patches :
      current_width = patch.get_width()
      diff = current_width - new_value
      # we change the bar width
      patch.set_width(new_value)
      # we recenter the bar
      patch.set_x(patch.get_x() + diff * .5)

me_exp = exp_gnr
me_exp_r = exp_gwr
associated_dummy_score = me_exp.phase_2a_dummy_score
associated_dummy_score_r = me_exp_r.phase_2a_dummy_score
score=me_exp.phase_2a_exp_retest_score
score_r=me_exp_r.phase_2a_exp_retest_score
to_extract='test_balanced_accuracy_score'
score_legend='Balanced accuracy'
title='Sex classification performance for different modalities' 
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


library['1000_Brains_All_Structural'].description='GMV + CT + SA'
library['1000_Brains_Surface_Area'].description='SA'
library['1000_Brains_Cortical_Thickness'].description='CT'
library['1000_Brains_Grey_Matter_Volume'].description='GMV'

library_r['1000_Brains_All_Structural_residuals'].description='GMV + CT + SA'
library_r['1000_Brains_Surface_Area_residuals'].description='SA'
library_r['1000_Brains_Cortical_Thickness_residuals'].description='CT'
library_r['1000_Brains_Grey_Matter_Volume_residuals'].description='GMV'

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
df=df[['GMV + CT + SA','GMV','CT','SA']]
df_r=results_r
df_r=df_r[['GMV + CT + SA','GMV','CT','SA']]
#df['Random Predictor'] = best_dummy_score
columns=list(new_columns)
columns_r=list(new_columns_r)
#All the preprocessing of the data is now done. We know have to make it look pretty
'''
#columns.append('Random Predictor')
fig, (ax1) = plt.subplots(1)
df[columns].mean().abs().plot.bar(yerr=[df[columns].std()],rot=0,fontsize=10, grid=False, width = 0.3)
ax1.set_ylim(0,12)
#change_width(ax1, .05)
ax1.axhline(y=best_dummy_score, color='r')
plt.legend()
plt.ylabel(score_legend)
plt.title(title)
plt.tight_layout()
'''
fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
ax.set_ylim(0,1)
ax2.set_ylim(0,1)
width = 0.3
ax.yaxis.tick_left()
ax2.yaxis.tick_left()


plot1=df.mean().abs().plot(kind='bar', color='royalblue', ax=ax, yerr=df[columns].std(), width=width, position=1,rot=0)
plot2=df_r.mean().abs().plot(kind='bar', color='navy', ax=ax2,yerr=df_r[columns].std(), width=width, position=0,rot=0)

colors = {'On native data':'royalblue', 'On residuals':'navy'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

ax.set_ylabel(score_legend)

margin = (1 - width) + width / 2
ax.set_xlim(-margin, len(df.columns) - 1 + margin)
ax2.set_xlim(-margin, len(df.columns) - 1 + margin)

myline = plt.axhline(y=best_dummy_score.mean(), color='b')
labels.append('Best random predictor')
handles.append(myline)

plt.legend(handles, labels)
#plt.ylabel(score_legend)
plt.title(title)
plt.tight_layout()

plt.savefig(path_save_folder+'/benchmark_sex.png')
