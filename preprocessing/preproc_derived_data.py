import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests
import urllib
import urllib3
from census import Census
import multiprocessing
from multiprocessing import Pool
import time
import math
import pywt
import warnings
import glob
# TODO: put this in a config file
CENSUS_KEY = 'f34e78b85a544c493c9dcdfe4c00a893238052fa'

STORE_ATTR_NAME = 'final_store_attributes.csv'
CWT_NAME = 'cwt.csv'
MOY_NAME = 'moy.csv'
YOY_NAME = 'yoy.csv'
SNOW_NAME = 'snow_feature.csv'


def create_derived_data(raw_data_obj, start_date, end_date):

    # find all current files
    file_list = []
    for file in glob.glob("*.csv"):
        file_list.append(file)

    with warnings.catch_warnings(record=True) as w:
        #get the store feature data
        print(f'Getting the store specific data.......')
        get_all_store_features(raw_data_obj.raw_store_attibutes_df,file_list)

        # get the cwt feature data
        print(f'Getting all of the CWT data.........')
        get_all_cwt(raw_data_obj.raw_invoice_df, raw_data_obj.raw_phone_df, raw_data_obj.raw_store_attibutes_df,
                    raw_data_obj.raw_holiday_df, raw_data_obj.raw_promotion_df, start_date, end_date,file_list)
        # get the moy data
        print(f'Getting the moy data.........')
        get_moy(raw_data_obj.raw_invoice_df, raw_data_obj.raw_phone_df, start_date, end_date)

        # get the yoy data
        print(f'Getting the yoy data..........')
        get_yoy(raw_data_obj.raw_invoice_df, raw_data_obj.raw_phone_df, start_date, end_date)

        # get the snow data
        print(f'Getting the snow data...........')
        get_snow_feature(raw_data_obj.raw_weather_df, raw_data_obj.raw_store_attibutes_df, start_date, end_date)



def get_moy(raw_invoice_df, raw_phone_df, start_date, end_date):
    # the start date should be the first day you would like this moy value
    # the end date shuold be the last day you would like this moy value
    # they are both in str format, e.g."2020-03-01"

    # obtain all store code
    store_code = raw_invoice_df.store_code.append(raw_phone_df.store_code)
    store_code = pd.DataFrame(store_code.unique(), columns=['store_code']).dropna()

    # create full calendar with store_code
    full_calendar = pd.merge(
        store_code.assign(key=1),
        pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=["date"]).assign(key=1),
        how='left').drop('key', 1)

    # get year, week, month for the full calendar
    full_calendar['year'] = full_calendar['date'].dt.year
    full_calendar['month'] = full_calendar['date'].dt.month
    full_calendar['week'] = full_calendar['date'].dt.week

    # merge with phone and invoice
    # forward fill missing values
    raw_volume_df = pd.merge(full_calendar, raw_invoice_df[['effective_date', 'store_code', 'actual']], how='left',
                             left_on=['store_code', 'date'], right_on=['store_code', 'effective_date']).rename(
        columns={'actual': 'invoice'}).drop('effective_date', 1).fillna(method='ffill')

    raw_volume_df = pd.merge(raw_volume_df, raw_phone_df[['effective_date', 'store_code', 'actual']], how='left',
                             left_on=['store_code', 'date'], right_on=['store_code', 'effective_date']).rename(
        columns={'actual': 'phone'}).drop('effective_date', 1).fillna(method='ffill')

    # back fill na if there is any
    raw_volume_df = raw_volume_df.fillna(method='bfill')

    # get annually, monthly, and daily median for each store (in all previous years)
    raw_volume_df = raw_volume_df.loc[raw_volume_df['year'] < (max(raw_volume_df['year'].unique()) - 1)]

    # annual sum
    total_annually_df = raw_volume_df.groupby(['store_code']).sum().reset_index().rename(
        columns={'invoice': 'total_annually_invoice', 'phone': 'total_annually_phone'})
    # monthly sum
    total_monthly_df = raw_volume_df.groupby(['store_code', 'month']).sum().reset_index().rename(
        columns={'invoice': 'total_monthly_invoice', 'phone': 'total_monthly_phone'})
    # monthly median
    average_monthly_df = raw_volume_df.groupby(['store_code', 'month']).median().reset_index().rename(
        columns={'invoice': 'average_monthly_invoice', 'phone': 'average_monthly_phone'})

    # merge back to the original data frame
    raw_volume_df = pd.merge(full_calendar,
                             total_annually_df[['store_code', 'total_annually_invoice', 'total_annually_phone']],
                             how='left',
                             on=['store_code'])
    raw_volume_df = pd.merge(raw_volume_df,
                             total_monthly_df[
                                 ['store_code', 'month', 'total_monthly_invoice', 'total_monthly_phone']],
                             how='left',
                             on=['store_code', 'month'])
    raw_volume_df = pd.merge(raw_volume_df,
                             average_monthly_df[
                                 ['store_code', 'month', 'average_monthly_invoice', 'average_monthly_phone']],
                             how='left',
                             on=['store_code', 'month'])

    # get moy
    raw_volume_df['invoice_moy_perc'] = raw_volume_df['total_monthly_invoice'] / raw_volume_df[
        'total_annually_invoice']
    raw_volume_df['phone_moy_perc'] = raw_volume_df['total_monthly_phone'] / raw_volume_df[
        'total_annually_phone']

    # get moy file
    moy = raw_volume_df[['date', 'store_code', 'invoice_moy_perc', 'average_monthly_invoice', 'phone_moy_perc',
                         'average_monthly_phone']]

    print(f"MOY output will have {len(moy['store_code'].unique())} stores")
    moy.to_csv(MOY_NAME, index=False)


def get_yoy(invoice_df, phone_df, start_date, end_date):
    invoice_df = invoice_df.rename(columns={'actual': 'invoice_actual', 'effective_date': 'date'})
    phone_df = phone_df.rename(columns={'actual': 'phone_actual', 'effective_date': 'date'})

    combine_list = np.concatenate((invoice_df.store_code.unique(), phone_df.store_code.unique()))
    combine_list = list(filter(None, combine_list))
    combine_list = np.unique(combine_list, axis=0)

    df1 = pd.merge(
        pd.DataFrame(combine_list, columns=["store_code"]).assign(key=1),
        pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=["date"]).assign(key=1),
        how='outer').drop('key', 1)

    df1 = pd.merge(df1, invoice_df[['store_code', 'date', 'invoice_actual']], on=['store_code', 'date'], how='left')
    df1 = pd.merge(df1, phone_df[['store_code', 'date', 'phone_actual']], on=['store_code', 'date'], how='left')
    df1['year'] = pd.to_datetime(df1['date']).dt.year
    df1['month'] = pd.to_datetime(df1['date']).dt.month
    df1['month'] = df1.month.map("{:02}".format)
    df1['key'] = df1['year'].astype(str) + df1['month'].astype(str)
    df1['key'] = df1['key'].astype(int)

    # invoice
    df2 = df1.groupby(['store_code', 'key'])['invoice_actual'].sum()
    df2 = df2.reset_index()
    df2.sort_values(by=['store_code', 'key'], inplace=True)
    df2['p_m'] = df2.groupby(['store_code'])['invoice_actual'].shift(1)
    df2['p_y_m'] = df2.groupby(['store_code'])['invoice_actual'].shift(13)
    df2['invoice_prev_month_yoy'] = (df2['p_m'] - df2['p_y_m']) / df2['p_y_m']
    df2['missing_invoice_prev_month_yoy'] = np.where(df2.invoice_prev_month_yoy.isnull(), 1, 0)

    # for phone
    df3 = df1.groupby(['store_code', 'key'])['phone_actual'].sum()
    df3 = df3.reset_index()
    df3.sort_values(by=['store_code', 'key'], inplace=True)
    df3['p_m'] = df3.groupby(['store_code'])['phone_actual'].shift(1)
    df3['p_y_m'] = df3.groupby(['store_code'])['phone_actual'].shift(13)
    df3['phone_prev_month_yoy'] = (df3['p_m'] - df3['p_y_m']) / df3['p_y_m']
    df3['missing_phone_prev_month_yoy'] = np.where(df3.phone_prev_month_yoy.isnull(), 1, 0)

    # merge
    df1 = pd.merge(df1, df2[['store_code', 'key', 'invoice_prev_month_yoy', 'missing_invoice_prev_month_yoy']],
                   on=['store_code', 'key'], how='left')
    df1 = pd.merge(df1, df3[['store_code', 'key', 'phone_prev_month_yoy', 'missing_phone_prev_month_yoy']],
                   on=['store_code', 'key'], how='left')
    df1['invoice_prev_month_yoy'] = df1['invoice_prev_month_yoy'].fillna(0)
    df1['phone_prev_month_yoy'] = df1['phone_prev_month_yoy'].fillna(0)
    df1 = df1[['store_code', 'date', 'invoice_prev_month_yoy', 'phone_prev_month_yoy', 'missing_invoice_prev_month_yoy',
               'missing_phone_prev_month_yoy']]

    # replace infinity with zeros
    df1 = df1.replace([np.inf, -np.inf], 0.00)

    print(f"YOY output will have {len(df1['store_code'].unique())} stores")
    df1.to_csv(YOY_NAME, index=False)


# get full calendar with store code
def get_full_calendar(raw_store_attribute_df, start_date, end_date):
    full_calendar = pd.merge(raw_store_attribute_df[['store_code']].assign(key=1),
                             pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=["date"]).assign(
                                 key=1),
                             how='left').drop('key', 1)
    return full_calendar


def get_snow_flag(raw_snow_data, full_calendar):
    # get full calendar
    raw_snow_data = pd.merge(full_calendar, raw_snow_data, how='left', on=['store_code', 'date']).fillna(0)

    # snow flag
    raw_snow_data['is_snow'] = np.where(raw_snow_data['snowfall_inches'] > 0, 1, 0)
    return raw_snow_data


def get_rolling_snow_count_weekly(raw_snow_data):
    # rolling snow count within a week
    raw_snow_data['snow_count_weekly'] = raw_snow_data['is_snow'].rolling(7, min_periods=7).sum().fillna(0).shift(
        periods=1, fill_value=0)
    return raw_snow_data


def get_first_now_of_the_year(raw_snow_data):
    # convert date time to year and month
    raw_snow_data['year'] = raw_snow_data['date'].dt.year
    raw_snow_data['month'] = raw_snow_data['date'].dt.month
    # first snow
    raw_snow_data['first_snow'] = 0

    for cur_store in raw_snow_data.store_code.unique():
        cur_weather = raw_snow_data.loc[raw_snow_data['store_code'] == cur_store]
        for cur_year in cur_weather.year.unique():
            if (sum(cur_weather.loc[(cur_weather['year'] == cur_year) & (cur_weather['month'] > 8)][
                        'snowfall_inches'].ne(0)) > 0):
                m = cur_weather.loc[(cur_weather['year'] == cur_year) & (cur_weather['month'] > 8)][
                    'snowfall_inches'].ne(0).idxmax()
                raw_snow_data['first_snow'][m] = 1

    return raw_snow_data


def get_first_snow_period(raw_snow_data):
    # calculate one week, one month within the first snow of the year
    raw_snow_data['previous_first_snow_one_week'] = raw_snow_data['first_snow'].rolling(7, min_periods=1).max().fillna(
        0).shift(periods=1, fill_value=0)
    raw_snow_data['previous_first_snow_one_month'] = raw_snow_data['first_snow'].rolling(30,
                                                                                         min_periods=1).max().fillna(
        0).shift(periods=1, fill_value=0)
    # calculate the number of days away from the first snow of the year after it happens
    raw_snow_data['previous_first_snow_two_weeks'] = raw_snow_data['first_snow'].rolling(14,
                                                                                         min_periods=1).max().fillna(
        0).shift(periods=1, fill_value=0)
    raw_snow_data['previous_first_snow_rolling_two_weeks'] = raw_snow_data['previous_first_snow_two_weeks'].rolling(14,
                                                                                                                    min_periods=1).sum().fillna(
        0).shift(periods=1, fill_value=0)
    return raw_snow_data


def get_snow_feature(raw_snow_data, raw_store_attribute_df, start_date, end_date):
    # full calendar
    full_calendar = get_full_calendar(raw_store_attribute_df, start_date, end_date)
    # snow flag
    raw_snow_data = get_snow_flag(raw_snow_data, full_calendar)
    # snow weekly count
    raw_snow_data = get_rolling_snow_count_weekly(raw_snow_data)
    # first snow of the year
    raw_snow_data = get_first_now_of_the_year(raw_snow_data)
    # first snow period
    raw_snow_data = get_first_snow_period(raw_snow_data)
    # return only relevant columns
    col = ['date', 'store_code', 'first_snow', 'previous_first_snow_rolling_two_weeks',
           'snow_count_weekly', 'previous_first_snow_one_week', 'previous_first_snow_one_month']

    print(f"Snow feature output will have {len(raw_snow_data['store_code'].unique())} stores")
    raw_snow_data[col].to_csv(SNOW_NAME, index=False)


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def make_remote_request(url: str, params: dict):
    """
    Makes the remote request
    Continues making attempts until it succeeds
    """

    count = 1
    while True:
        try:
            response = requests.get((url + urllib.parse.urlencode(params)))
        except (OSError, urllib3.exceptions.ProtocolError) as error:
            print('\n')
            print('*' * 20, 'Error Occured', '*' * 20)
            print('Number of tries:{}'.format(count))
            print('URL: {}'.format(url))
            print(error)
            print('\n')
            count += 1
            continue
        break

    return response


def elevation_function(x):
    url = 'https://nationalmap.gov/epqs/pqs.php?'
    params = {'x': x['store_longitude_degree'],
              'y': x['store_latitude_degree'],
              'units': 'Meters',
              'output': 'json'}
    result = make_remote_request(url, params)
    return result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']


def add_features(df):
    c = Census(CENSUS_KEY)

    def func(pin, code, dummy):
        pin = pin.split('-')[0]
        try:
            v = c.acs5.zipcode(code, pin)[0][code]
            return v
        except:
            return None

    df['total_population'] = df['store_postal_code'].apply(func, args=('B01001_002E', None))

    df['male_population'] = df['store_postal_code'].apply(func, args=('B01001_001E', None))

    df['female_population'] = df['store_postal_code'].apply(func, args=('B01001_002E', None))

    df['housing_unit'] = df['store_postal_code'].apply(func, args=('B25001_001E', None))

    df['aggregate_vehicles'] = df['store_postal_code'].apply(func, args=('B08015_001E', None))

    df['median income'] = df['store_postal_code'].apply(func, args=('B06011_001E', None))  # median income

    df['individuals_with_no_vehicle'] = df['store_postal_code'].apply(func, args=(
        'B08014_002E', None))  # individuals with no vehicle

    df['individuals_with_one_vehicle'] = df['store_postal_code'].apply(func, args=(
        'B08014_003E', None))  # individuals with 1 vehicle

    df['individuals_with_two_vehicle'] = df['store_postal_code'].apply(func, args=(
        'B08014_004E', None))  # individuals with 2 vehicles

    df['individuals_with_three_vehicle'] = df['store_postal_code'].apply(func, args=(
        'B08014_005E', None))  # individuals with 3 vehicles

    df['individuals_with_four_vehicle'] = df['store_postal_code'].apply(func, args=(
        'B08014_006E', None))  # individuals with 4 vehicles

    df['individuals_with_five_or_more_vehicle'] = df['store_postal_code'].apply(func, args=(
        'B08014_007E', None))  # individuals with 5 or more vehicles

    df['individuals_total'] = df['store_postal_code'].apply(func,
                                                            args=('B08014_001E', None))  # Total vehicles data available

    return df


def parallelize_dataframe(df, function):
    n_cores = (multiprocessing.cpu_count()) - 3
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(function, df_split))
    pool.close()
    pool.join()
    return df


def format_store_closing_data(store_attr):
    # format the closing dates
    store_attr["close_date"] = np.where(store_attr['store_sales_close_date'] == '9999-12-31',
                                        pd.datetime.now().strftime("%Y-%m-%d"), store_attr["store_sales_close_date"])
    store_attr['store_open_date'] = pd.to_datetime(store_attr['store_open_date'])
    store_attr['close_date'] = pd.to_datetime(store_attr['close_date'])
    store_attr['store_age'] = store_attr['close_date'] - store_attr['store_open_date']
    store_attr['store_age'] = store_attr['store_age'] / np.timedelta64(1, 'Y')
    del store_attr['close_date']

    return store_attr


def get_store_distances(store_attr):
    # format the lat/lon and distance
    data_new = store_attr[['store_code', 'store_latitude_degree', 'store_longitude_degree']].drop_duplicates()
    data_new['key'] = 1
    final_data = data_new.merge(data_new, how='left', on=['key'])
    final_data['distance_in_km'] = haversine_np(final_data['store_longitude_degree_x'],
                                                final_data['store_latitude_degree_x'],
                                                final_data['store_longitude_degree_y'],
                                                final_data['store_latitude_degree_y'])

    final_data = final_data[final_data['store_code_x'] != final_data['store_code_y']]
    final_data = final_data.groupby('store_code_x')['distance_in_km'].min()
    final_data = pd.DataFrame(final_data)
    final_data = final_data.reset_index()
    final_data.rename(columns={'store_code_x': 'store_code', 'distance_in_km': 'distance_to_nearest_store_km'},
                      inplace=True)
    store_attr = pd.merge(store_attr, final_data, on=['store_code'], how='left')

    return store_attr


def rad(coord):
    '''
    converts lat/lon coordinates to radians
    '''

    rad = math.radians(coord)

    return rad


def get_distance(lat1, lon1, lat2, lon2):
    '''
    calculates arbitrary distance measure between 2 lat/lon pairs used for nearest grid key approximation
    '''

    x = (rad(lon2) - rad(lon1)) * math.cos(rad(lat1))
    y = rad(lat2) - rad(lat1)
    d = (x ** 2 + y ** 2) ** (1 / 2)

    return d


def find_closest_on_grid(store, grid_ref):
    '''
    finds the closest NARR grid point for a given store
    '''

    dist = []
    storeid = store[0]
    lat = store[1]
    lon = store[2]

    for g in grid_ref:
        d = get_distance(lat, lon, g[1], g[2])
        dist.append(d)

    closest_ind = dist.index(min(dist))

    closest_grid_key = int(grid_ref[closest_ind][0])
    closest_lat = grid_ref[closest_ind][1]
    closest_lon = grid_ref[closest_ind][2]

    out = [closest_grid_key, closest_lat, closest_lon]

    return out


def get_historical_weather(store_attr):
    # map the grid cells to stores
    grid_keys1 = pd.read_csv("30yrs_weather_data_agg.csv")

    grid_keys = grid_keys1[['grid_key', 'latitude', 'longitude']]
    stores = store_attr[['store_code', 'store_latitude_degree', 'store_longitude_degree']]

    grid_keys = grid_keys.values.tolist()
    stores = stores.values.tolist()

    results = []

    for i, store in enumerate(stores):

        if i % 100 == 0 and i>0:
            print(f'Done getting historical weather data for {i} stores')

        closest = find_closest_on_grid(store, grid_keys)

        results.append([store[0], store[1], store[2], closest[0], closest[1], closest[2]])

    df = pd.DataFrame(results, columns=['store_code', 'store_latitude', 'store_longitude', 'grid_key', 'grid_latitude',
                                        'grid_longitude'])

    df = pd.merge(df, grid_keys1, on=['grid_key'], how='left')
    df.drop(['latitude', 'longitude'], axis=1, inplace=True)

    df.to_csv("store_hist_weather_info.csv")

    # get historical weather data
    store_hist_weather_info = pd.read_csv("store_hist_weather_info.csv")
    store_hist_weather_info = store_hist_weather_info.drop(store_hist_weather_info.columns[[0]], axis=1)
    store_hist_weather_info = store_hist_weather_info[
        ['store_code', 'avg_daily_temp', 'avg_yearly_pcp', 'avg_yearly_acc_snow', 'max_daily_temp', 'max_daily_pcp',
         'max_daily_acc_snow',
         'min_daily_temp', 'min_daily_pcp', 'min_daily_acc_snow', 'stddev_daily_temp', 'stddev_yearly_pcp',
         'stddev_yearly_acc_snow']]

    store_attr = pd.merge(store_attr, store_hist_weather_info, on=['store_code'], how='left')

    return store_attr


def get_elevation_data(store_attr):
    store_attr['elevation_meter'] = store_attr.apply(elevation_function, axis=1)
    return store_attr


def get_census_data(store_attr):
    df = store_attr[['store_code', 'store_postal_code']]
    start = time.time()
    final_census = parallelize_dataframe(df, add_features)
    print(time.time() - start)
    # final_census
    del final_census['store_postal_code']

    # fill in nan with means
    final_census.fillna((final_census.mean()), inplace=True)

    store_attr = pd.merge(store_attr, final_census, on=['store_code'], how='left')

    return store_attr


def get_all_store_features(store_attr,file_list):


    store_attr,original_store_attr,process_stores = find_stores_to_process(store_attr,file_list,STORE_ATTR_NAME)

    if process_stores:
        # format the closing dates
        store_attr = format_store_closing_data(store_attr)

        # format the lat/lon and distance
        store_attr = get_store_distances(store_attr)

        # get historical weather data
        print(f'Getting the historical weather data............')
        store_attr = get_historical_weather(store_attr)

        # get the elevation data
        print(f'Getting the elevation data..........')
        store_attr = get_elevation_data(store_attr)

        # get the census data
        print(f'Getting the census data............')
        store_attr = get_census_data(store_attr)

        if STORE_ATTR_NAME in file_list:
            store_attr = pd.concat([store_attr,original_store_attr])


        store_attr.to_csv(STORE_ATTR_NAME, index=False)


# standardize invoice per store
def standardize_data_per_store(raw_df_yearly):
    # perform a robust scaler transform of the dataset for each store
    trans = StandardScaler()
    df = raw_df_yearly
    for store_code in df['store_code'].unique():
        data = np.array(df.loc[df['store_code'] == store_code, 'actual'])
        data = data.reshape(-1, 1)
        data = trans.fit_transform(data)
        df.loc[df['store_code'] == store_code, 'standard_actual'] = data
    return df


# standardize per year
def standardize_data_per_year(raw_df):
    df = raw_df
    df['year'] = df['date'].dt.year
    df['standard_actual'] = 0
    for year in df['year'].unique():
        df_yearly = df.loc[df['year'] == year]
        df_yearly = standardize_data_per_store(df_yearly)
        df.loc[df['year'] == year, 'standard_actual'] = list(df_yearly['standard_actual'])
    return df


# calcualte cwt
def mexh_wavelet_standard_actual(scales, x, wavelet='mexh'):
    scales = np.arange(1, scales)
    coeffs, freqs = pywt.cwt(list(x['avg_standard_actual']), scales, wavelet=wavelet)
    # create scalogram
    # ax = plt.matshow(coeffs)
    cwt = pd.DataFrame(np.transpose(coeffs))
    cwt['store_code'] = list(x['store_code'])
    cwt['date'] = list(x['date'])
    cwt['date'] = pd.to_datetime(cwt['date'])
    return cwt


# calcualte cwt for all stores
def wavelet_all_stores(raw_df):
    df = raw_df
    cnt = 0
    for store_code in df['store_code'].unique():
        x = df.loc[df['store_code'] == store_code]
        if cnt == 0:
            cwt = mexh_wavelet_standard_actual(22, x)
        else:
            cwt = cwt.append(mexh_wavelet_standard_actual(22, x))
        cnt += 1
    df = pd.merge(df, cwt, how='left', left_on=['store_code', 'date'], right_on=['store_code', 'date'])
    # sort by store_id, effective_date, and actual
    df = df.sort_values(by=['store_code', 'date'])
    # drop duplicated entries
    df.drop_duplicates(['date', 'store_code'], keep='last', inplace=True)

    # use scale 3, 7, and 15 for averaging
    df['cwt'] = (df[3] + df[7] + df[15]) / 3
    df['cwt_mexh_quarter'] = df[20]
    return df[['store_code', 'date', 'season', 'cwt', 'cwt_mexh_quarter']]


def mexh_wavelet_promo_holiday(x, colname, scales=18, wavelet='mexh'):
    scales = np.arange(1, scales)
    coeffs, freqs = pywt.cwt(list(x), scales, wavelet=wavelet)

    # create scalogram
    cwt = pd.DataFrame(np.transpose(coeffs)).set_index(x.index)

    if colname == 'promotion':
        cwt['ave_cwt'] = (cwt[3] + cwt[7] + cwt[15]) / 3
    else:
        cwt['ave_cwt'] = -cwt[3]
    return cwt['ave_cwt']


def get_promotion_cwt(promo_df, start_date, end_date):
    # get full calendar to the promo
    promo_df = pd.merge(promo_df.assign(is_promo=1),
                        pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=["date"]),
                        how='right', left_on='date', right_on='date').fillna(0)
    # cwt
    promo_df = promo_df.sort_values(by='date')
    promo_cwt = mexh_wavelet_promo_holiday(promo_df['is_promo'], 'promotion')
    # add to the original data frame
    promo_df['promotion_cwt'] = list(promo_cwt)

    return promo_df


def get_holiday_cwt(holiday_df, start_date, end_date):
    # get full calendar to the promo
    holiday_df = pd.merge(holiday_df.assign(is_holiday=1),
                          pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=["date"]),
                          how='right', left_on='date', right_on='date').fillna(0)

    # cwt
    holiday_df = holiday_df.sort_values(by='date')
    holiday_cwt = mexh_wavelet_promo_holiday(holiday_df['is_holiday'], 'holiday')
    # add to the original data frame
    holiday_df['holiday_cwt'] = list(holiday_cwt)

    return holiday_df


# get the season (using solstice -- only works for the Northern Hemisphere)
def season(date):
    md = date.month * 100 + date.day
    if ((md > 320) and (md < 621)):
        s = 'spring'  # spring
    elif ((md > 620) and (md < 923)):
        s = 'summer'  # summer
    elif ((md > 922) and (md < 1223)):
        s = 'fall'  # fall
    else:
        s = 'winter'  # winter
    return s


# get season for full_calendar
def calendar_season(full_calendar):
    season_range = []
    for i in full_calendar['date']:
        season_range.append(season(i))
    full_calendar['season'] = season_range
    return full_calendar


def get_full_calendar(raw_df, start_date, end_date):
    # create full calendar with store_code

    store_code = pd.DataFrame(raw_df.store_code.unique(), columns=['store_code']).dropna()
    full_calendar = pd.merge(
        store_code.assign(key=1),
        pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=["date"]).assign(key=1),
        how='left').drop('key', 1)

    # get year, month, day for the full calendar
    full_calendar['year'] = full_calendar['date'].dt.year
    full_calendar['month'] = full_calendar['date'].dt.month
    full_calendar['day'] = full_calendar['date'].dt.day

    # get season
    full_calendar = calendar_season(full_calendar)
    return full_calendar


def merge_all_df(invoice_cwt, phone_cwt, holiday_cwt, promo_cwt):
    # rename phone column
    phone_cwt = phone_cwt.rename(columns={
        'cwt': 'phone_cwt',
        'state_mean_cwt': 'phone_state_mean_cwt',
        'cwt_mexh_quarter': 'phone_cwt_mexh_quarter',
        'winter_state_mean_cwt': 'phone_winter_state_mean_cwt',
        'fall_state_mean_cwt': 'phone_fall_state_mean_cwt',
        'state_mean_cwt_mexh_quarter': 'phone_state_mean_cwt_mexh_quarter',
        'winter_state_mean_cwt_mexh_quarter': 'phone_winter_state_mean_cwt_mexh_quarter',
        'fall_state_mean_cwt_mexh_quarter': 'phone_fall_state_mean_cwt_mexh_quarter'
    })

    # rename invoice column
    invoice_cwt = invoice_cwt.rename(columns={
        'cwt': 'invoice_cwt',
        'state_mean_cwt': 'invoice_state_mean_cwt',
        'cwt_mexh_quarter': 'invoice_cwt_mexh_quarter',
        'winter_state_mean_cwt': 'invoice_winter_state_mean_cwt',
        'fall_state_mean_cwt': 'invoice_fall_state_mean_cwt',
        'state_mean_cwt_mexh_quarter': 'invoice_state_mean_cwt_mexh_quarter',
        'winter_state_mean_cwt_mexh_quarter': 'invoice_winter_state_mean_cwt_mexh_quarter',
        'fall_state_mean_cwt_mexh_quarter': 'invoice_fall_state_mean_cwt_mexh_quarter'
    })

    # merge phone and invoice cwt
    actual_cwt = pd.merge(invoice_cwt, phone_cwt, how='outer',
                          on=['date', 'store_code', 'store_state_code', 'weather_region'])

    # identify wherther a store has phone or invoice data
    actual_cwt['is_invoice_available'] = np.where(actual_cwt['invoice_cwt'].isna(), 0, 1)
    actual_cwt['is_phone_available'] = np.where(actual_cwt['phone_cwt'].isna(), 0, 1)

    # fill na
    actual_cwt = actual_cwt.fillna(0)

    # merge with holiday and promotion
    cwt = pd.merge(actual_cwt, promo_cwt[['date', 'promotion_cwt']], how='left', on=['date'])
    cwt = pd.merge(cwt, holiday_cwt[['date', 'holiday_cwt']], how='left', on=['date'])

    # columns to be returned
    cols = ['date', 'store_code', 'invoice_cwt', 'store_state_code', 'invoice_state_mean_cwt',
            'invoice_winter_state_mean_cwt',
            'invoice_fall_state_mean_cwt', 'weather_region', 'invoice_cwt_mexh_quarter',
            'invoice_state_mean_cwt_mexh_quarter', 'invoice_winter_state_mean_cwt_mexh_quarter',
            'invoice_fall_state_mean_cwt_mexh_quarter', 'phone_cwt', 'phone_state_mean_cwt',
            'phone_winter_state_mean_cwt',
            'phone_fall_state_mean_cwt', 'phone_cwt_mexh_quarter', 'phone_state_mean_cwt_mexh_quarter',
            'phone_winter_state_mean_cwt_mexh_quarter', 'phone_fall_state_mean_cwt_mexh_quarter',
            'is_invoice_available', 'is_phone_available', 'promotion_cwt', 'holiday_cwt']
    return cwt[cols]


# populate over all dates

def get_cwt_per_store(raw_invoice_df, start_date, end_date):
    # obtain all store code
    raw_df = raw_invoice_df

    # get full calendar
    full_calendar = get_full_calendar(raw_invoice_df, start_date, end_date)

    # merge
    raw_df = pd.merge(full_calendar, raw_df, how='left',
                      left_on=['date', 'store_code'],
                      right_on=['effective_date', 'store_code']
                      ).drop(['effective_date', 'store_id', 'metric_id'], 1).fillna(method='ffill').fillna(
        method='bfill')

    # only select two year ago's data
    raw_df = raw_df.loc[raw_df['year'] < (max(raw_df['year'].unique()) - 1)]
    # standardize every year
    raw_df = standardize_data_per_year(raw_df)
    # calculate mean of the standardized data
    daily_mean = raw_df.groupby(['store_code', 'month', 'day']).sum().reset_index().rename(
        columns={'standard_actual': 'avg_standard_actual'}).drop(['actual', 'year'], 1)

    # back to the orignial dataframe
    raw_df = pd.merge(full_calendar, daily_mean, how='left',
                      on=['store_code', 'month', 'day']).fillna(method='ffill')

    # cwt
    raw_df = wavelet_all_stores(raw_df)

    return raw_df


def get_cwt_per_state(raw_df, store_attribute_df):
    # include store state code information
    raw_df = pd.merge(raw_df, store_attribute_df[['store_code', 'store_state_code']],
                      how='left', on=['store_code']).fillna(raw_df['store_code'][:1])

    # find state mean
    state_mean = raw_df.groupby(by=['store_state_code', 'date']).mean().reset_index()

    # merge to original data frame
    raw_df = pd.merge(raw_df,
                      state_mean[['store_state_code', 'date', 'cwt', 'cwt_mexh_quarter']].rename(
                          columns={'cwt_mexh_quarter': 'state_mean_cwt_mexh_quarter', 'cwt': 'state_mean_cwt'}),
                      how='left', on=['date', 'store_state_code'])

    return raw_df


def get_cwt_per_state_per_season(raw_df):
    # one hot encoding
    raw_df['fall'] = np.where(raw_df['season'] == 'fall', 1, 0)
    raw_df['winter'] = np.where(raw_df['season'] == 'winter', 1, 0)
    raw_df['fall_state_mean_cwt'] = raw_df['state_mean_cwt'] * raw_df['fall']
    raw_df['winter_state_mean_cwt'] = raw_df['state_mean_cwt'] * raw_df['winter']
    raw_df['fall_state_mean_cwt_mexh_quarter'] = raw_df['state_mean_cwt_mexh_quarter'] * raw_df['fall']
    raw_df['winter_state_mean_cwt_mexh_quarter'] = raw_df['state_mean_cwt_mexh_quarter'] * raw_df['winter']
    raw_df['weather_region'] = 'other'
    raw_df.loc[raw_df['store_state_code'] == 'CO', 'weather_region'] = 'CO'
    raw_df.loc[raw_df['store_state_code'] == 'MI', 'weather_region'] = 'MI'
    raw_df.loc[raw_df['store_state_code'] == 'MN', 'weather_region'] = 'MN'
    return raw_df


# calculat all invoice/phone related cwt columns
def get_cwt_from_actual(raw_df, store_attribute_df, start_date, end_date):
    # cwt for each store
    cwt_df = get_cwt_per_store(raw_df, start_date=start_date, end_date=end_date)

    # get cwt for each state
    cwt_df = get_cwt_per_state(cwt_df, store_attribute_df)

    # state mean cwt for fall and winter
    cwt_df = get_cwt_per_state_per_season(cwt_df)

    return cwt_df


def get_all_cwt(raw_invoice_df, raw_phone_df, store_attribute_df, holiday_df, promotion_df, start_date, end_date,file_list):

    # find what stores need to be processed
    process_stores = True
    original_df = None

    if CWT_NAME in file_list:
        original_df = pd.read_csv(CWT_NAME)
        original_store_code_list = original_df['store_code'].unique()
        new_store_code_list = store_attribute_df['store_code']
        diff_store_list = list(set(new_store_code_list) - set(original_store_code_list))
        print(f'New stores to process {diff_store_list}')

        if len(diff_store_list)>0:
            raw_invoice_df = raw_invoice_df.loc[raw_invoice_df['store_code'].isin(diff_store_list)]
            raw_phone_df = raw_phone_df.loc[raw_phone_df['store_code'].isin(diff_store_list)]
            store_attribute_df = store_attribute_df.loc[store_attribute_df['store_code'].isin(diff_store_list)]

        else:
            process_stores = False


    if process_stores:
        holiday_df['date'] = pd.to_datetime(holiday_df['HolidayDate'])
        promotion_df['date'] = pd.to_datetime(promotion_df['PromotionDate'])

        print(f'Get the invoice CWT.......')
        invoice_cwt_df = get_cwt_from_actual(raw_invoice_df, store_attribute_df, start_date, end_date)

        print(f'Get the phone CWT.......')
        phone_cwt_df = get_cwt_from_actual(raw_phone_df, store_attribute_df, start_date, end_date)

        # calculate holiday and promotion cwt
        holiday_cwt_df = get_holiday_cwt(holiday_df, start_date, end_date)
        print(f'Get the holiday CWT.......')

        promotion_cwt_df = get_promotion_cwt(promotion_df, start_date, end_date)
        print(f'Get the promo CWT.........')

        # merge all data frames
        cwt = merge_all_df(invoice_cwt_df, phone_cwt_df, holiday_cwt_df, promotion_cwt_df)

        if CWT_NAME in file_list:
            cwt = pd.concat([cwt,original_df])

        print(f"CWT output will have {len(cwt['store_code'].unique())} stores")

        cwt.to_csv(CWT_NAME, header=True, index=False)


def find_stores_to_process(df,file_list,file_name):

    process_stores = True
    original_df = None

    if file_name in file_list:
        original_df = pd.read_csv(file_name)
        original_store_code_list = original_df['store_code'].unique()
        new_store_code_list = df['store_code']
        diff_store_list = list(set(new_store_code_list) - set(original_store_code_list))

        if len(diff_store_list)>0:
            print(f'New stores to process {diff_store_list}')
            df = df.loc[df['store_code'].isin(diff_store_list)]
        else:
            process_stores = False

    return df,original_df,process_stores