import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from google.cloud import storage

client = storage.Client()

TRACKING_SERVER_HOST = "0.0.0.0"
TRACKING_SERVER_PORT = "5000"
TRAIN_DATA_PATH = "/home/Pranoy/MLOPS-Practice/data/yellow_tripdata_2023-01.parquet"
VALID_DATA_PATH = "/home/Pranoy/MLOPS-Practice/data/yellow_tripdata_2023-02.parquet"
EXPERIMENT_NAME = "practice_prefect_nyc_experiment"


dv = DictVectorizer()
curr_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@task
def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        
        df.lpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename, engine = 'pyarrow')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df = df.sample(frac=0.01)
    df.reset_index(inplace=True)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df

@task
def vectorizer(df: pd.DataFrame):
    features = ['PU_DO', 'trip_distance']
    tdf = df[features]

    return tdf

@task
def train_model_rf_search(x_train, x_val, y_train, y_val):
    
    mlflow.sklearn.autolog()
    count = 0
    def objective(params):
        nonlocal count
        count+=1
        with mlflow.start_run(run_name=f"RandomForest_{curr_timestamp}_{count}", description=f'Random Forest at timestamp {curr_timestamp} of Run Instance {count}'):
            mlflow.set_tag("model", "rf")
            mlflow.set_tags({'FirstName':'Pranoy'
                             ,'LastName':'Dewanjee'})
            mlflow.log_param("train_data",TRAIN_DATA_PATH)
            
            rf_model = RandomForestRegressor(**params)
            rf_model.fit(x_train, y_train)
            y_pred = rf_model.predict(x_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2score = r2_score(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2score)
                
            mlflow.sklearn.log_model(rf_model, artifact_path="model")

        return {'loss': rmse, 'status': STATUS_OK}


    search_space = {
        'n_estimators' : scope.int(hp.uniform('n_estimators',10,150)),
        'max_depth' : scope.int(hp.uniform('max_depth',1,40)),
        'min_samples_leaf' : scope.int(hp.uniform('min_samples_leaf',1,10)),
        'min_samples_split' : scope.int(hp.uniform('min_samples_split',2,10)),
        'random_state' : 42
    }
    
    rstate = np.random.default_rng(42)  # for reproducible results
    best_result =  fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials(),
        rstate=rstate
    )
    return

@task
def train_model_lasso_search(x_train, x_val, y_train, y_val):

    mlflow.sklearn.autolog()
    count = 1
    for alpha in [0.01, 0.03, 0.05, 0.07, 0.09]:
        mlflow.start_run(run_name=f"Lasso_Regression_{curr_timestamp}_{count}", description=f"Lasso Regression at timestamp {curr_timestamp} of Run Instance {count}")
        mlflow.set_tag("model","lr")
        mlflow.set_tags({'FirstName':'Pranoy'
                         ,'LastName':'Dewanjee'})
        mlflow.log_param("train_data",TRAIN_DATA_PATH)

        ls = Lasso(alpha)
        ls.fit(x_train,y_train)
        y_pred = ls.predict(x_val)
        rmse = mean_squared_error(y_val,y_pred, squared = False)
        r2score = r2_score(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2score)

        mlflow.sklearn.log_model(ls,artifact_path="model")

        mlflow.end_run()
        count+=1

    return

@flow(name="RF_Lasso_Run", task_runner=SequentilTaskRunner())
def main():
    """
    Executes the training workflow
    """
    tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger = get_run_logger()

    df_train = read_dataframe(TRAIN_DATA_PATH)
    df_val = read_dataframe(VALID_DATA_PATH)
    
    x_train = vectorizer(df_train)
    x_val = vectorizer(df_val)
    
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model_rf_search(x_train, x_val, y_train, y_val)
    train_model_lasso_search(x_train, x_val, y_train, y_val)

    logger.info("Successfully executed our flow !!!")


if __name__ == "__main__":
    main()