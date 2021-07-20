from iris.data import get_data
from iris.pipeline import create_pipeline
import joblib
from memoized_property import memoized_property

import mlflow
from  mlflow.tracking import MlflowClient

class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    def mlflow_create_run(self):
        self.mlfow_run = self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlfow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlfow_run.info.run_id, key, value)


    def save_pipeline(self):
        joblib.dump(self.pipe, 'pipeline.joblib')

    def train(self, **kwargs):
        # # get the data
        (X_train, X_test, y_train, y_test) = get_data()

        # create a pipeline
        self.pipe = create_pipeline()

        # train the model
        self.pipe.fit(X_train, y_train)

        # logging params and metrics to MLflow
        self.mlflow_create_run()
        for key, value in kwargs.items():
            self.mlflow_log_param(key, value)
        self.mlflow_log_metric("rmse", 4.5)
        # save the pipeline to a joblib
        self.save_pipeline()






