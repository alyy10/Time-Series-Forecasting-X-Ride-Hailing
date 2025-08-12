import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from datetime import datetime

geolocator = Nominatim(user_agent="OLABikes")


def basic_cleanup(df) -> pd.DataFrame:
    # remove duplicates
    df.drop_duplicates(subset=['ts', 'number'], inplace=True, keep='last')
    df.reset_index(inplace=True, drop=True)

    # remove NaN
    df['number'] = pd.to_numeric(df['number'], errors='coerce')
    df.dropna(inplace=True)

    df['number'] = pd.to_numeric(
        df['number'], errors='coerce', downcast='integer')
    df['ts'] = pd.to_datetime(df['ts'])
    return df


def time_features_add(df) -> pd.DataFrame:
    df['hour'] = df['ts'].dt.hour
    df['mins'] = df['ts'].dt.minute
    df['day'] = df['ts'].dt.day
    df['month'] = df['ts'].dt.month
    df['year'] = df['ts'].dt.year
    df['dayofweek'] = df['ts'].dt.dayofweek
    return df


def shift_booking_time(df) -> pd.DataFrame:
    # get difference of time booking b/w two consecutive rides booked by user
    df['shift_booking_ts'] = df.groupby('number')['booking_timestamp'].shift(1)
    df['shift_booking_ts'].fillna(0, inplace=True)
    df['shift_booking_ts'] = df['shift_booking_ts'].astype('int64')
    df['booking_time_diff_hr'] = round(
        (df['booking_timestamp'] - df['shift_booking_ts']) // 3600)
    df['booking_time_diff_min'] = round(
        (df['booking_timestamp'] - df['shift_booking_ts']) // 60)
    return df


def geodestic_distance(pick_lat, pick_lng, drop_lat, drop_lng) -> float:
    return round(geodesic((pick_lat, pick_lng), (drop_lat, drop_lng)).miles * 1.60934, 2)


def advance_cleanup(df) -> pd.DataFrame:
    # remove duplicate booking within 1hour from same user at same pickup lat-long
    df = df[~((df.duplicated(subset=['number', 'pick_lat', 'pick_lng'],
                             keep=False)) & (df.booking_time_diff_hr <= 1))]

    # remove demand count / repeat booking by same user within 4mins (consider multiple retry/error booking)
    df = df[(df.booking_time_diff_min >= 8)]

    # Geodesic Distance calculate
    df['geodesic_distance'] = np.vectorize(geodestic_distance)(df['pick_lat'], df['pick_lng'], df['drop_lat'],
                                                               df['drop_lng'])

    # remove ride where pickup and drop location distance is less than 50meters
    df = df[df.geodesic_distance > 0.05]

    # remove rides outside India Bounding Box
    df.reset_index(inplace=True, drop=True)
    outside_India = df[(df.pick_lat <= 6.2325274) | (df.pick_lat >= 35.6745457) | (df.pick_lng <= 68.1113787) | (
        df.pick_lng >= 97.395561) | (df.drop_lat <= 6.2325274) | (df.drop_lat >= 35.6745457) | (
        df.drop_lng <= 68.1113787) | (df.drop_lng >= 97.395561)]
    df = df[~df.index.isin(outside_India.index)].reset_index(drop=True)

    # remove rides outside KA and pickup/drop distance > 500kms
    total_ride_outside_KA = df[
        (df.pick_lat <= 11.5945587) | (df.pick_lat >= 18.4767308) | (df.pick_lng <= 74.0543908) | (
            df.pick_lng >= 78.588083) | (df.drop_lat <= 11.5945587) | (df.drop_lat >= 18.4767308) | (
            df.drop_lng <= 74.0543908) | (df.drop_lng >= 78.588083)]
    suspected_bad_rides = total_ride_outside_KA[total_ride_outside_KA.geodesic_distance > 500]

    df = df[~df.index.isin(suspected_bad_rides.index)].reset_index(drop=True)
    return df


def main():
    start_time = datetime.now()
    df = pd.read_csv('../Data/raw_data.csv',
                     low_memory=False, compression='gzip')
    print("Initial Raw Row Count: {}".format(len(df)))

    df = basic_cleanup(df)
    print("After Basic CleanUp Row Count: {}".format(len(df)))
    df = time_features_add(df)

    df.sort_values(by=['number', 'ts'], inplace=True)
    df.reset_index(inplace=True)
    df['booking_timestamp'] = df.ts.values.astype(np.int64) // 10 ** 9

    df = shift_booking_time(df)
    df = advance_cleanup(df)
    print("After Advance CleanUp Row Count: {}".format(len(df)))

    dataset = df[
        ['ts', 'number', 'pick_lat', 'pick_lng', 'drop_lat', 'drop_lng', 'geodesic_distance', 'hour', 'mins', 'day',
         'month', 'year', 'dayofweek', 'booking_timestamp', 'booking_time_diff_hr', 'booking_time_diff_min']]
    dataset.to_csv('./../Data/clean_data.csv', index=False, compression='gzip')
    print("Time Taken for Data Preprocessing: {}".format(
        datetime.now()-start_time))


if __name__ == '__main__':
    main()
