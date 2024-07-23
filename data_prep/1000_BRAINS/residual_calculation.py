
import statsmodels.api as sm
from scipy.stats import pearsonr

def residuals_for_x(x,y):  
  #add constant to predictor variables
  x = sm.add_constant(x)
  
  #fit linear regression model
  model = sm.OLS(y, x).fit() 
  
  influence = model.get_influence()
  
  #obtain standardized residuals
  standardized_residuals = influence.resid_studentized_internal
  
  #display standardized residuals
  #print(standardized_residuals)
  
  return standardized_residuals
  
def residual_output_model(x,y):  
  #add constant to predictor variables
  x = sm.add_constant(x)
  
  #fit linear regression model
  model = sm.OLS(y, x).fit() 
  
  influence = model.get_influence()
  
  #obtain standardized residuals
  standardized_residuals = influence.resid_studentized_internal
  
  #display standardized residuals
  #print(standardized_residuals)
  
  return model