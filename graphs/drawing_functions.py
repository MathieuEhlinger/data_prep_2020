'''
Defining functions for drawing of different plots
'''

import os
import pandas as pd

from os import listdir
from os.path import isfile, join
import numpy as np

import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_without_deleting(me_exp, dir_path, file_name = '_my_picture', extension = '.png'):
      # Just for not overwriting things. Also checks if dir exists.
      if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
      if dir_path is None : dir_path = me_exp.meta_experience_path
       
      if not os.path.exists(dir_path+'/'+ me_exp.meta_experience_name + file_name +'_0' + extension):
            save_path = dir_path+'/'+ me_exp.meta_experience_name + file_name +'_0'+ extension
      else:
        allfiles = [f for f in listdir(dir_path) if isfile(join(dir_path+'/', f))]
        number = max([int(f.split('_')[-1][:-len(extension)]) for f in allfiles if ((me_exp.meta_experience_name + file_name) in f and extension in f)])+1
        save_path = dir_path+'/'+ me_exp.meta_experience_name + file_name +'_'+str(number)+extension
      return save_path
    
    
def generate_one_result_df(me_exp, score_o, to_extract='test_MAE' ):
  #Generating boxplots requires all results of a 10 times 10 Fold to be in a single array
  #This functions takes a aci-kit learn score and returns as one array the results of all the runs as a DataFrame
  result = dict()
  score,_ = me_exp.scan_results_for_nan_and_struct(score_o)
  for ds in score :
    result[ds] = pd.DataFrame(score[ds])[to_extract].dropna().flatten()
  
  max_length = max([len(x) for x in result.values()])
  
  for ds in result :
    if len(result[ds]) < max_length :
        needed_iterations  = max_length - len(result[ds])
        list_to_append     = [np.nan]*needed_iterations
        result[ds] = result[ds] + list_to_append
        #print(ds,' new length : ', len(result[ds]))
  
  return pd.DataFrame(result)
  
def boxplot_from_score_with_dummy(me_exp,df, best_dummy_score, dir_path=None, file_name = 'test_file',columns=None,title='', score_legend=''):
  #This function draws a boxplot using a scikit score flattened according to 'generate one result'
  #It also takes a dummy score modified the same way
  #And generates a nice plots saved under 'path' 
  
  plt.clf()
  df['Random Predictor'] = best_dummy_score
  columns=list(columns)
  columns.append('Random Predictor')
  boxplot = df[columns].boxplot(rot=45,fontsize=10, grid=False)
  
  fig     = boxplot.get_figure()
  axes    = fig.get_axes()
  
  dummy_line_middle = axes[0].axhline(df['Random Predictor'].quantile(0.5),color='grey',ls='-')
  dummy_line_middle.set_label('Median random prediction')
  
  dummy_line_middle_plus = axes[0].axhline(df['Random Predictor'].quantile(0.75),color='grey',ls='--')
  dummy_line_middle_plus.set_label('Q3 random prediction')
  
  dummy_line_middle_minus = axes[0].axhline(df['Random Predictor'].quantile(0.25),color='grey',ls='--')
  dummy_line_middle_minus.set_label('Q1 random prediction')
  
  fig.set_size_inches(12.5, 7.25)
  
  plt.legend()
  plt.ylabel(score_legend)
  plt.title(title)
  plt.tight_layout()
  path = me_exp.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
  plt.savefig(path)
  
def barplot_from_score_with_dummy(me_exp,df, best_dummy_score, dir_path=None, file_name = 'test_file',columns=None,title='', score_legend=''):
  #This function draws a barplot using a scikit score flattened according to 'generate one result'
  #It also takes a dummy score modified the same way
  #And generates a nice plots saved under 'path' 
  
  plt.clf()
  df['Random Predictor'] = best_dummy_score
  columns=list(columns)
  columns.append('Random Predictor')
  barplot = df[columns].mean().abs().plot.bar(yerr=[df[columns].std()],rot=45,fontsize=10, grid=False)
  fig     = barplot.get_figure()
  axes    = fig.get_axes()
  fig.set_size_inches(12.5, 7.25)
  
  plt.ylabel(score_legend)
  plt.title(title)
  plt.tight_layout()
  path = me_exp.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
  plt.savefig(path)

def table_plot_for_overfitting(me_exp,df_result_train, df_result_test, dir_path=None, file_name = 'test_file_tp',title='Measure of overfitting',top=None):
  #This function draws a boxplot using a scikit score flattened according to 'generate one result'
  #It also takes a dummy score modified the same way
  #And generates a nice plots saved under 'path' 
  if top is None : top = df_result_train.shape[0]
  
  plt.clf()
  
  table = pd.concat([df_result_train, df_result_test],axis=1).iloc[:top]
  table.rename(columns={ table.columns[0]: "Train Dataset Result" }, inplace = True)
  table.rename(columns={ table.columns[1]: "Test Dataset Result" }, inplace = True)
  table['Difference']  =    table["Test Dataset Result"]-table["Train Dataset Result"]
  
  #plt.figure()

  # table
  #plt.subplot(121)
  #plt.plot(table)
  cell_text = []
  for row in range(len(table)):
      cell_text.append(table.iloc[row])
  
  plt.table(cellText=cell_text, colLabels=table.columns, rowLabels=table.index)
  #plt.table(cellText=cell_text, colLabels=table.columns, rowLabels=table.index, loc='center')
  plt.axis('off')

  #plt.tight_layout()
  #plt.legend()
  #plt.title(title)
  #file_name = '_my_picture', extension = '.png'
  path = me_exp.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
  plt.savefig(path, pad_inches=1)
       

def draw_top_x_boxplots(me_exp, score, associated_dummy_score, dir_path=None, file_name='test_name',x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,library=None):
  #Takes the scikit score for a run and the associated dummy, reduces them to 1D arrays and sends them to 
  #boxplot_from_score_with_dummy for drawing.
  
  results            =   generate_one_result_df(me_exp,score_o=score,to_extract=to_extract)
  dummy_values       =   me_exp.return_simplified_score(associated_dummy_score, to_extract)
  best_dummy_name    =   dummy_values.sort_values('average',ascending=False).iloc[0].name
  best_dummy_score   =   generate_one_result_df(me_exp, score_o=associated_dummy_score, to_extract=to_extract)[best_dummy_name]
  columns            =   results.mean()
  if from_these is not None : columns            =   columns[from_these].sort_values(ascending=False).index[:x]
  else :                      columns            =   columns.sort_values(ascending=False).index[:x]
  
  new_index_dict = dict()
  for lib in library :
    new_index_dict[lib]=library[lib].description
  results = results.rename(columns=new_index_dict)
  new_columns = [new_index_dict[i] for i in columns]
  boxplot_from_score_with_dummy(me_exp, df=results,best_dummy_score=best_dummy_score, dir_path=dir_path, file_name=file_name, columns=new_columns, title=title,  score_legend=score_legend)

def draw_top_x_barplots(me_exp, score, associated_dummy_score, dir_path=None, file_name='test_name',x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,library=None):
  #Takes the scikit score for a run and the associated dummy, reduces them to 1D arrays and sends them to 
  #boxplot_from_score_with_dummy for drawing.
  
  results            =   me_exp.generate_one_result_df(score_o=score,to_extract=to_extract)
  dummy_values       =   me_exp.return_simplified_score(associated_dummy_score, to_extract)
  best_dummy_name    =   dummy_values.sort_values('average',ascending=False).iloc[0].name
  best_dummy_score   =   me_exp.generate_one_result_df(score_o=associated_dummy_score, to_extract=to_extract)[best_dummy_name]
  columns            =   results.mean()
  if from_these is not None : columns            =   columns[from_these].sort_values(ascending=False).index[:x]
  else :                      columns            =   columns.sort_values(ascending=False).index[:x]
  
  new_index_dict = dict()
  for lib in library :
    new_index_dict[lib]=library[lib].description
  results = results.rename(columns=new_index_dict)
  new_columns = [new_index_dict[i] for i in columns]
  me_exp.barplot_from_score_with_dummy(df=results,best_dummy_score=best_dummy_score, dir_path=dir_path, file_name=file_name, columns=new_columns, title=title,  score_legend=score_legend)

#generate_sub_exps

def draw_boxplots_for_these(me_exp, score, associated_dummy_score, dir_path=None, file_name='test_name',
            x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,
            split_ds_by={'All':['_'],'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['_na']},
            drop_ds={'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['df_na'],'All':['']},library=None):
  
  subexps = me_exp.generate_sub_exps(split_ds_by=split_ds_by, drop_ds=drop_ds)
  for subexp in subexps :
    me_exp.draw_top_x_boxplots(score=score, 
                    associated_dummy_score=associated_dummy_score, 
                    dir_path=dir_path, file_name = file_name+'_'+subexp,x=x,
                    title=subexp+' '+title, to_extract=to_extract,  score_legend=score_legend,
                    from_these = list(subexps[subexp].keys()), library=library)
    
def boxplots_for_refined_phases(me_exp, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
  #The two most meaningful phases for plottingg are, for now, phases 3b and 4a.
  #This function draws the plots associated to them.
  
  if plot_func is None : plot_func = draw_top_x_boxplots
  if library   is None : library   = me_exp.intermediary_dataset_3b
  path_3b = me_exp.meta_experience_path+'/plots/'+'third_phase_b'+'/'
  
  plot_func(score=me_exp.phase_3b_exp_refined_score, associated_dummy_score=me_exp.phase_3b_dummy_score, dir_path=path_3b, file_name = '3_b_boxplot_10_best_refined',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)

def return_barplots_for_these(me_exp, score, associated_dummy_score, dir_path=None, file_name='test_name',
            x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,
            split_ds_by={'All':['_'],'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['_na']},
            drop_ds={'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['df_na'],'All':['']},library=None):
  
  subexps = me_exp.generate_sub_exps(split_ds_by=split_ds_by, drop_ds=drop_ds)
  for subexp in subexps :
    me_exp.draw_top_x_barplots(score=score, 
                    associated_dummy_score=associated_dummy_score, 
                    dir_path=dir_path, file_name = file_name+'_'+subexp,x=x,
                    title=subexp+' '+title, to_extract=to_extract,  score_legend=score_legend,
                    from_these = list(subexps[subexp].keys()), library=library)
    
def barplots_for_refined_phases(me_exp, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
  #The two most meaningful phases for plottingg are, for now, phases 3b and 4a.
  #This function draws the plots associated to them.
  
  if plot_func is None : plot_func = draw_top_x_barplots
  if library   is None : library   = me_exp.intermediary_dataset_3b
  path_3b = me_exp.meta_experience_path+'/plots/'+'third_phase_b'+'/'
  
  plot_func(score=me_exp.phase_3b_exp_refined_score, associated_dummy_score=me_exp.phase_3b_dummy_score, dir_path=path_3b, file_name = '3_b_barplot_10_best_refined',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)

def repartition_histogramms_by_class(me_exp, dir_path=None, file_name='test', title='', var = 'Age',class_to_dif='Gender',df=None):
  #Histogramms of the df. Differentiate by class_to_dif
  #This function draws the plots and saves them in path.
  
  plt.clf()

  print(df)
  print(class_to_dif)
  hist = sns.pairplot(df, hue=class_to_dif, kind='reg', diag_kind='hist',size=2.5, vars=list(df.drop(class_to_dif,axis=1)))

  path = me_exp.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
  plt.savefig(path)

def hist_for_age_and_gender(me_exp, var='Age',class_to_dif='Gender', drop=[]):
  #draw repartition plots of the general information dataset
  org_df           = pd.concat([me_exp.library.inf_ds.dropna(axis=1),me_exp.library.target_ds],axis=1)
  org_df           = org_df.drop(drop,axis=1)
  df_1,  path_1    = org_df.loc[me_exp.train_index],me_exp.meta_experience_path+'/plots/'+'first_phase'+'/'
  df_2a, path_2a   = org_df.loc[me_exp.test_index], me_exp.meta_experience_path+'/plots/'+'second_phase_a'+'/'
  df_4a, path_4a   = org_df, me_exp.meta_experience_path+'/plots/'+'fourth_phase_a'+'/'
  
  me_exp.repartition_histogramms_by_class(dir_path=path_1, file_name='hist_of_rep_1', title='Repartition of train set', var=var,class_to_dif=class_to_dif,df=df_1)
  me_exp.repartition_histogramms_by_class(dir_path=path_2a, file_name='hist_of_rep_2a', title='Repartition of test set', var=var, class_to_dif=class_to_dif,df=df_2a)
  me_exp.repartition_histogramms_by_class(dir_path=path_4a, file_name='hist_of_rep_4a', title='Repartition of whole set', var=var, class_to_dif=class_to_dif,df=df_4a)
  
def boxplots_for_test_phases_raw(me_exp, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
  #The two most meaningful phases for plotting are, for now, phases 3b and 4a.
  #This function draws the plots associated to them.
  
  if plot_func is None : plot_func = me_exp.draw_top_x_boxplots
  if library   is None : library   = me_exp.library
  path_2a = me_exp.meta_experience_path+'/plots/'+'second_phase_a'+'/'
  path_3b_raw = me_exp.meta_experience_path+'/plots/'+'third_phase_b'+'/'
  path_4a = me_exp.meta_experience_path+'/plots/'+'fourth_phase_a'+'/'
  
  plot_func(score=me_exp.phase_2a_exp_retest_score, associated_dummy_score=me_exp.phase_2a_dummy_score, dir_path=path_2a, file_name = '2_a_boxplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
  plot_func(score=me_exp.phase_3b_exp_retest_score, associated_dummy_score=me_exp.phase_3b_dummy_score, dir_path=path_3b_raw, file_name = '3_b_boxplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
  plot_func(score=me_exp.phase_4a_exp_retest_score, associated_dummy_score=me_exp.phase_4a_dummy_score, dir_path=path_4a, file_name = '4_a_boxplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)      
  
def barplots_for_test_phases_raw(me_exp, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
  #The two most meaningful phases for plotting are, for now, phases 3b and 4a.
  #This function draws the plots associated to them.
  
  if plot_func is None : plot_func = draw_top_x_barplots
  if library   is None : library   = me_exp.library
  path_2a = me_exp.meta_experience_path+'/plots/'+'second_phase_a'+'/'
  path_3b_raw = me_exp.meta_experience_path+'/plots/'+'third_phase_b'+'/'
  path_4a = me_exp.meta_experience_path+'/plots/'+'fourth_phase_a'+'/'
  
  plot_func(score=me_exp.phase_2a_exp_retest_score, associated_dummy_score=me_exp.phase_2a_dummy_score, dir_path=path_2a, file_name = '2_a_barplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
  plot_func(score=me_exp.phase_3b_exp_retest_score, associated_dummy_score=me_exp.phase_3b_dummy_score, dir_path=path_3b_raw, file_name = '3_b_barplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
  plot_func(score=me_exp.phase_4a_exp_retest_score, associated_dummy_score=me_exp.phase_4a_dummy_score, dir_path=path_4a, file_name = '4_a_barplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
  
def table_plot_overfit_measurement(me_exp, to_extract='test_MAE'):
  #Two datasets are used to search for models. To measure overfit, we compare their results with fully independent datasets
  phase_1_df    = me_exp.results_refined('first_phase'   , save='10CV10_',subdir='exp_dirs', models = None)[[to_extract+'_mean']].sort_values(by=to_extract+'_mean', ascending=False)
  phase_1_df.rename(columns={ phase_1_df.columns[0]: "Other_name" }, inplace = True) 
  #\-> Names gotta be different between the two dataframes send to the plotting function
  
  phase_2a_df   = me_exp.results_refined('second_phase_a', save='10CV10_',subdir='exp_dirs', models = None)[[to_extract+'_mean']].loc[phase_1_df.index]
  me_exp.table_plot_for_overfitting(phase_1_df, phase_2a_df, dir_path=me_exp.meta_experience_path+'/plots/'+'first_phase'+'/')
  
  phase_2b_df   = me_exp.results_refined('second_phase_b', save='10CV10_',subdir='exp_dirs_2', models = me_exp.p2b_intermediaries_models)[[to_extract+'_mean']].sort_values(by=to_extract+'_mean', ascending=False)
  phase_2b_df.rename(columns={ phase_2b_df.columns[0]: "Other_name" }, inplace = True)
  #\-> Names gotta be different between the two dataframes send to the plotting function
  
  phase_3b_df   = me_exp.results_refined('third_phase_b' , save='10CV10_',subdir='exp_dirs_2', models = me_exp.p2b_intermediaries_models)[[to_extract+'_mean']].loc[phase_2b_df.index]
  me_exp.table_plot_for_overfitting(phase_2b_df,phase_3b_df, dir_path=me_exp.meta_experience_path+'/plots/'+'second_phase_a'+'/')
