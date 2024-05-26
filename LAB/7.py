import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load data
df = pd.read_csv('heart_disease.csv')

# Define the structure of the Bayesian Network
model = BayesianNetwork([('age', 'heart_disease'), ('sex', 'heart_disease'), 
                         ('cp', 'heart_disease'), ('heart_disease', 'exang')])

# Fit the model using Maximum Likelihood Estimation
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)
result = inference.map_query(variables=['heart_disease'], evidence={'age': 55, 'sex': 1, 'cp': 2})
print(result)
