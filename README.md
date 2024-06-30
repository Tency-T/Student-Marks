##End to End Data Science Project

import os
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/tencytrvd/Student-Marks.mlflow'

import dagshub
dagshub.init(repo_owner='tencytrvd', repo_name='Student-Marks', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)