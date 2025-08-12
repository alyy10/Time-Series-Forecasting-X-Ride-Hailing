import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime
from joblib import dump, load


def train_test_data_prep(df_train, df_test):
    df_test = df_test.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'])
    temp = pd.concat([df_train, df_test])
    temp = temp.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'])
    temp = temp.set_index(
        ['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter', 'dayofweek'])

    temp['lag_1'] = temp.groupby(level=['pickup_cluster'])[
        'request_count'].shift(1)
    temp['lag_2'] = temp.groupby(level=['pickup_cluster'])[
        'request_count'].shift(2)
    temp['lag_3'] = temp.groupby(level=['pickup_cluster'])[
        'request_count'].shift(3)
    temp['rolling_mean'] = temp.groupby(level=['pickup_cluster'])['request_count'].apply(
        lambda x: x.rolling(window=6).mean()).shift(1)

    temp = temp.reset_index(drop=False).dropna()
    temp = temp[['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter',
                 'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'request_count']]
    train1 = temp[temp.ts.dt.day <= 23]
    test1 = temp[temp.ts.dt.day > 23]

    X = train1.iloc[:, 1:-1]
    y = train1.iloc[:, -1]
    X_test = test1.iloc[:, 1:-1]
    y_test = test1.iloc[:, -1]
    return X, y, X_test, y_test


def metrics_calculate(regressor, X_test, y_test):
    y_pred = regressor.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    return rms


def model_train(X, y, X_test, y_test):
    import xgboost as xgb
    model = xgb.XGBRegressor(learning_rate=0.01, random_state=0, n_estimators=1500, max_depth=8,
                             objective="reg:squarederror")

    eval_set = [(X_test, y_test)]
    model.fit(X, y, verbose=True, eval_set=eval_set,
              early_stopping_rounds=20, eval_metric="rmse")
    print("XGBOOST Regressor")
    print("Model Score: ", model.score(X, y))
    print(
        "RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, model.predict(X))), metrics_calculate(model, X_test, y_test)))
    dump(model, '../Model/prediction_model.joblib', compress=3)


def main():
    start_time = datetime.now()
    df = pd.read_csv('../Data/Data_Prepared.csv', compression='gzip')
    df['request_count'] = pd.to_numeric(
        df['request_count'], downcast='integer')
    df.ts = pd.to_datetime(df.ts)

    # First 24days of every month in Train and last 7 days of every month in Test
    df_train = df[df.ts.dt.day <= 23]
    df_test = df[df.ts.dt.day > 23]
    print(df_train.head())

    # X, y, X_test, y_test = train_test_data_prep(df_train, df_test)
    # model_train(X, y, X_test, y_test)
    # print("Time Taken for Model Training: {}".format(datetime.now() - start_time))


if __name__ == '__main__':
    main()
