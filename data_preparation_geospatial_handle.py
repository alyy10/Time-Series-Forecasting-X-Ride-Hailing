import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.cluster import MiniBatchKMeans, KMeans
import gpxpy.geo
from datetime import datetime, timedelta
from joblib import dump, load


def min_distance(regionCenters, totalClusters):
    less_dist = []
    more_dist = []
    min_distance = np.inf  
    for i in range(totalClusters):
        good_points = 0
        bad_points = 0
        for j in range(totalClusters):
            if j != i:
                distance = gpxpy.geo.haversine_distance(latitude_1=regionCenters[i][0], longitude_1=regionCenters[i][1],
                                                        latitude_2=regionCenters[j][0], longitude_2=regionCenters[j][1])
                distance = distance / (1.60934 * 1000)  # distance from meters to miles
                min_distance = min(min_distance, distance)  # it will return minimum of "min_distance, distance".
                if distance < 2:
                    good_points += 1
                else:
                    bad_points += 1
        less_dist.append(good_points)
        more_dist.append(bad_points)
    print("On choosing a cluster size of {}".format(totalClusters))
    print("Avg. Number clusters within vicinity where inter cluster distance < 2 miles is {}".format(
        np.ceil(sum(less_dist) / len(less_dist))))
    print("Avg. Number clusters outside of vicinity where inter cluster distance > 2 miles is {}".format(
        np.ceil(sum(more_dist) / len(more_dist))))
    print("Minimum distance between any two clusters = {}".format(min_distance))
    print("-" * 10)


def makingRegions(noOfRegions, coord):
    regions = MiniBatchKMeans(n_clusters=noOfRegions, batch_size=10000, random_state=0).fit(coord)
    regionCenters = regions.cluster_centers_
    totalClusters = len(regionCenters)
    return regionCenters, totalClusters


def optimal_cluster(df):
    coord = df[["pick_lat", "pick_lng"]].values
    startTime = datetime.now()
    for i in range(10, 100, 10):
        regionCenters, totalClusters = makingRegions(i, coord)
        min_distance(regionCenters, totalClusters)
    print("Time taken = " + str(datetime.now() - startTime))


def round_timestamp_30interval(x):
    if type(x) == str:
        x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return x - timedelta(minutes=x.minute % 30, seconds=x.second, microseconds=x.microsecond)


def time_features(data) -> pd.DataFrame:
    data['mins'] = data.ts.dt.minute
    data['hour'] = data.ts.dt.hour
    data['day'] = data.ts.dt.day
    data['month'] = data.ts.dt.month
    data['dayofweek'] = data.ts.dt.dayofweek
    data['quarter'] = data.ts.dt.quarter
    return data


def main():
    start_time = datetime.now()
    df = pd.read_csv('./../Data/clean_data.csv', compression='gzip')
    optimal_cluster(df)

    # From Above Optimal Cluster Finder we found 50 is the ideal region count; distance b/w two cluster should be
    # less than 0.5miles (approx)
    coord = df[["pick_lat", "pick_lng"]].values
    regions = MiniBatchKMeans(n_clusters=50, batch_size=10000, random_state=5).fit(coord)
    df["pickup_cluster"] = regions.predict(df[["pick_lat", "pick_lng"]])

    # Model to Define pickup cluster, given latitude and longitude
    dump(regions, '../Model/pickup_cluster_model.joblib', compress=3)

    df['ts'] = np.vectorize(round_timestamp_30interval)(df['ts'])
    dataset = deepcopy(df)
    dataset = dataset[['ts', 'number', 'pickup_cluster']]
    dataset = dataset.groupby(by=['ts', 'pickup_cluster']).count().reset_index()
    dataset.columns = ['ts', 'pickup_cluster', 'request_count']

    # Adding Dummy pickup cluster -1
    # Change this Data based on your data
    l = [datetime(2020, 3, 26, 00, 00, 00) + timedelta(minutes=30 * i) for i in range(0, 48 * 365)]
    lt = []
    for x in l:
        lt.append([x, -1, 0])
    temp = pd.DataFrame(lt, columns=['ts', 'pickup_cluster', 'request_count'])
    dataset = dataset.append(temp, ignore_index=True)
    data = dataset.set_index(['ts', 'pickup_cluster']).unstack().fillna(value=0).asfreq(
        freq='30Min').stack().sort_index(level=1).reset_index()
    # Removing Dummy Cluster
    data = data[data.pickup_cluster >= 0]

    # 366days x 48 (30 mins intervals) x 50 regions
    assert len(data) == 878400
    data = time_features(data)

    data.to_csv('./../Data/Data_Prepared.csv', index=False, compression='gzip')
    print("Time Taken for Data Preparation: {}".format(datetime.now() - start_time))


if __name__ == '__main__':
    main()
