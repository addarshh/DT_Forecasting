import random
import numpy as np
import pandas as pd
import sys
import calendar

sys.path.append('/DTFORECASTING/')
from preprocessing.preproc_split_data import ModelData


class SingleStoreFeatures:
    def __init__(self, preproc_config, raw_data_obj):

        # global definitions
        self.preproc_config = preproc_config
        self.raw_data_obj = raw_data_obj
        is_forecast_week = self.preproc_config.is_forecast_week

        # get store specific data
        self.get_store_specific_data()

        # set the lead time
        self.set_lead_time()

        # get invoice and phone volumes; apply pre-filter
        self.raw_invoice = self.volume_prefilter(self.raw_data_obj.invoice_df_by_store[self.preproc_config.store_id])
        self.raw_phone_calls = self.volume_prefilter(self.raw_data_obj.phone_df_by_store[self.preproc_config.store_id])

        # make the indexes
        self.make_indexes(is_forecast_week)

        # set the output variable
        self.get_output_data()

        # get the dates
        self.get_dates()

        # lagged dates for dayofweek
        self.get_lagged_dates()

        # get the phone call data
        self.get_lagged_main_vars()

        # get the weather info
        self.weather_prefilter()
        self.get_weather_data()

        # make promotional_dates
        self.get_promotional_data()

        # make holidays
        self.get_holiday_data()

        # cwt ceofficients
        self.get_ctw_coefs()

        # percentage sales for dayofweek
        self.invoice_dow_perc = self.make_relative_volume_dow(self.raw_invoice)
        self.phone_dow_perc = self.make_relative_volume_dow(self.raw_phone_calls)

        # party date
        self.get_party_data()

        # yoy vars
        self.make_yoy_vars()

        # get the snow specific features
        self.get_snow_specific_vars()

        # get calendar perc variables
        self.get_calendar_perc_vars()

        # get weather region inputs
        self.get_weather_region_vars()

        # get season variables
        self.get_seasoanl_data()

        # invoice_cwt
        self.get_actual_cwt()

        # invoice cwt lagged
        self.get_lagged_actual_cwt()

        # invoice cwt long range
        self.get_actual_cwt_long()

        # invoice cwt long range lagged
        self.get_lagged_actual_cwt_long()

        # create the empty feature df
        self.feature_df = pd.DataFrame(columns=self.preproc_config.feature_list)

        # make features
        self.make_features()

    def get_store_specific_data(self):
        self.store_specific_data = self.raw_data_obj.all_store_attibutes_df[
            self.raw_data_obj.all_store_attibutes_df.store_code == self.preproc_config.store_id]

    def set_lead_time(self):
        if self.store_specific_data['store_state_code'].any() in self.preproc_config.multi_week_forecasts_states:
            self.preproc_config.min_forecast_lead = self.preproc_config.long_min_forecast_lead
        else:
            self.preproc_config.min_forecast_lead = self.preproc_config.default_min_forecast_lead

    def volume_prefilter(self, raw_series):
        first_date = raw_series.index[0]
        first_record_date = raw_series.first_valid_index()
        last_date = raw_series.index[-1]
        last_record_date = raw_series.last_valid_index()
        last_date_dow = last_date.dayofweek
        days_to_extend = (self.preproc_config.predict_day - last_date_dow) % 7
        if self.preproc_config.run_type == 'score':
            # For scoring, make sure missing data up to the reference point for prediction is interpolated, e.g.,
            #   long weekend where Saturday's data not available on Monday
            raw_series = raw_series.reindex(
                pd.date_range(first_date, last_date + pd.Timedelta(f'{days_to_extend} days')))
        else:
            # For training, only use the available valid date range
            raw_series = raw_series.reindex(
                pd.date_range(first_record_date, last_record_date))
        raw_df = pd.DataFrame(raw_series).reset_index()
        raw_df.columns = ['date', 'volume']
        raw_df['dow'] = raw_df['date'].dt.dayofweek
        dow_dict = {}
        for day in range(7):
            # Fill missing data via bi-directional interpolation of nearest same days-of-week
            dow_dict[day] = raw_df.loc[raw_df['dow'] == day].interpolate(method='linear',
                                                                         limit_direction='both').round()
        interpolated_df = pd.concat(dow_dict.values()).sort_values('date').drop('dow', axis=1).set_index('date').fillna(
            method='ffill')  # "ffill" to populate Sunday with values from previous Saturday
        interpolated_series = pd.Series(interpolated_df['volume'])
        interpolated_series.index = interpolated_df.index
        if self.preproc_config.run_type == 'score':
            # For scoring, extend predicted date index into the future based on specified forecast lead time
            last_date_interp = interpolated_series.index[-1]
            interpolated_series = interpolated_series.fillna(method='bfill')
            interpolated_series = interpolated_series.reindex(
                pd.date_range(first_date, last_date_interp +
                              pd.Timedelta(f'{self.preproc_config.min_forecast_lead + 7} days')))
        else:
            interpolated_series = interpolated_series.reindex(pd.date_range(first_date, last_date))
        return interpolated_series

    def weather_prefilter(self):
        # Fill missing store weather data via mean imputation
        raw_precip_data = self.raw_data_obj.precip_df_by_store[self.preproc_config.store_id].reindex(self.raw_dates)
        raw_snowfall_data = self.raw_data_obj.snow_accum_df_by_store[self.preproc_config.store_id].reindex(self.raw_dates)
        precip_mean = raw_precip_data.mean()
        snowfall_mean = raw_snowfall_data.mean()
        self.raw_precip_data = raw_precip_data.fillna(precip_mean)
        self.raw_snowfall_data = raw_snowfall_data.fillna(snowfall_mean)

    def make_indexes(self, is_forecast_week):

        # day of week integer list
        if self.preproc_config.output_var == 'invoices':
            all_days_of_week = [index_f.dayofweek for index_f in self.raw_invoice.index]

        elif self.preproc_config.output_var == 'phone_calls':
            all_days_of_week = [index_f.dayofweek for index_f in self.raw_phone_calls.index]

        # get x and y indexes
        if is_forecast_week:
            day_convert_dict = {index_f: self.preproc_config.min_forecast_lead + index_f for index_f in range(7)}
            self.y_indexes = []
            self.x_indexes = []


            for count, cur_day_of_week in enumerate(all_days_of_week):

                if count > day_convert_dict[cur_day_of_week] + 1:
                    potential_index = count - day_convert_dict[cur_day_of_week]
                    # potential_lag_day = all_days_of_week[count - day_convert_dict[cur_day_of_week]]
                    if all_days_of_week[count - day_convert_dict[cur_day_of_week]] != self.preproc_config.predict_day:
                        potential_index = \
                            np.where(np.array(all_days_of_week[
                                              :count - self.preproc_config.min_forecast_lead]) == self.preproc_config.predict_day)[
                                0][-1]

                    if all_days_of_week[potential_index] != self.preproc_config.predict_day:
                        print(
                            f'ERROR! Do not have the correct lagged day for: i {count}.....day of week: {cur_day_of_week}.......lagged value {all_days_of_week[potential_index]}')

                    self.y_indexes.append(count)
                    self.x_indexes.append(potential_index)

            self.y_indexes = np.array(self.y_indexes)
            self.x_indexes = np.array(self.x_indexes)

        else:
            self.y_indexes = np.arange(0, len(all_days_of_week))[self.preproc_config.lag:]
            self.x_indexes = np.arange(0, len(all_days_of_week))[:-self.preproc_config.lag]

    def get_output_data(self):

        if self.preproc_config.output_var == 'invoices':
            self.y_data = self.raw_invoice[self.y_indexes]
            self.raw_dates = self.raw_invoice.index
            self.lagged_df = self.raw_invoice[self.x_indexes]
            self.store_list = self.raw_data_obj.invoice_df_by_store.columns

        elif self.preproc_config.output_var == 'phone_calls':
            self.y_data = self.raw_phone_calls[self.y_indexes]
            self.raw_dates = self.raw_phone_calls.index
            self.lagged_df = self.raw_phone_calls[self.x_indexes]
            self.store_list = self.raw_data_obj.phone_df_by_store.columns

    def get_dates(self):
        self.lagged_dates = self.lagged_df.index
        self.pred_dates = self.y_data.index
        self.pred_dayofweek = [index_f.dayofweek for index_f in self.pred_dates]

    def get_lagged_dates(self):
        # lagged dates for dayofweek
        self.lagged_dow_x_indexes = []
        for dow_count, dow in enumerate(self.pred_dayofweek):
            self.lagged_dow_x_indexes.append(self.x_indexes[dow_count] - self.preproc_config.perc_dow_lag_dict[dow])

    def get_lagged_main_vars(self):
        self.lagged_invoice_data = self.raw_invoice.reindex(self.raw_dates)[self.x_indexes]
        self.lagged_phone_data = self.raw_phone_calls.reindex(self.raw_dates)[self.x_indexes]

    def get_weather_data(self):
        self.lagged_precip_data = self.raw_precip_data[self.x_indexes]
        self.lagged_snowfall_data = self.raw_snowfall_data[self.x_indexes]

    def get_promotional_data(self):
        self.promotion_dummies = pd.get_dummies(
            self.raw_data_obj.promotion_df
        ).reindex(self.pred_dates).fillna(0)
        self.promotion_ind = self.promotion_dummies.sum(axis=1)

    def get_holiday_data(self):
        self.holiday_dummies = pd.get_dummies(
            self.raw_data_obj.holiday_df
        ).reindex(self.pred_dates).fillna(0)
        self.holiday_ind = self.holiday_dummies.sum(axis=1)

    def make_relative_volume_dow(self, raw_series):
        volume_series = raw_series[raw_series.first_valid_index():]
        grouper_dict = {0: 'W-MON', 1: 'W-TUES', 2: 'W-WED', 3: 'W-THU', 4: 'W-FRI', 5: 'W-SAT'}
        grouper_val = grouper_dict[self.preproc_config.predict_day]
        vol_df = pd.DataFrame(volume_series.reset_index())
        vol_df.columns = ['date', 'volume']
        vol_df['dow'] = vol_df['date'].dt.dayofweek
        first_full_week_start_idx = vol_df['dow'][vol_df['dow'].eq(self.preproc_config.predict_day)].index[0] + 1
        first_full_week_start_date = vol_df['date'][first_full_week_start_idx]
        vol_df = vol_df[first_full_week_start_idx:]
        vol_df = vol_df.loc[vol_df['dow'] != 6].drop('dow', axis=1)
        vol_df = vol_df.groupby(pd.Grouper(key='date', freq=grouper_val))['volume'] \
            .sum().reset_index().set_index('date') \
            .merge(pd.DataFrame(volume_series[first_full_week_start_date:]), how='right', left_index=True,
                   right_index=True) \
            .fillna(method='bfill')
        vol_df.columns = ['weekly_volume', 'daily_volume']
        vol_df['vol_frac'] = vol_df['daily_volume'] / vol_df['weekly_volume']
        relative_volume = vol_df['vol_frac'].reindex(self.raw_dates)[self.y_indexes]
        return relative_volume

    def get_ctw_coefs(self):

        self.promotion_cwt = self.raw_data_obj.raw_promotion_cwt_by_store[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.y_indexes]
        self.holiday_cwt = self.raw_data_obj.raw_holiday_cwt_by_store[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.y_indexes]

    def get_party_data(self):
        self.party_data = self.raw_data_obj.party_by_store[self.preproc_config.store_id].reindex(
            self.pred_dates).fillna(0)

    def make_yoy_vars(self):

        # invoices
        self.invoice_prev_month_yoy = self.raw_data_obj.invoice_prev_month_yoy[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.y_indexes]

        self.missing_invoice_prev_month_yoy = \
            self.raw_data_obj.missing_invoice_prev_month_yoy[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]

        # phones
        self.phone_prev_month_yoy = self.raw_data_obj.phone_prev_month_yoy[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.y_indexes]
        self.missing_phone_prev_month_yoy = \
            self.raw_data_obj.missing_phone_prev_month_yoy[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]

    def get_snow_specific_vars(self):

        # snow data
        # first snow in fall of a year
        self.snow_day_data = self.raw_data_obj.snow_day_by_store[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.x_indexes]

        # previous snow day
        self.previous_snow_day_data = self.raw_data_obj.previous_snow_by_store[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.x_indexes]

        # previous snow count
        self.previous_snow_count_data = \
            self.raw_data_obj.previous_snow_count_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]

        # first snow one month
        self.previous_snow_one_month_data = \
            self.raw_data_obj.previous_snow_month_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]

    def get_calendar_perc_vars(self):
        # invoice
        self.invoice_moy_perc_data = self.raw_data_obj.invoice_moy_perc_by_store[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.y_indexes]

        self.average_monthly_lagged_invoice_data = \
            self.raw_data_obj.average_monthly_invoice_actual_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]

        # phone
        self.phone_moy_perc_data = self.raw_data_obj.phone_moy_perc_by_store[self.preproc_config.store_id].reindex(
            self.raw_dates)[self.y_indexes]

        self.average_monthly_lagged_phone_data = \
            self.raw_data_obj.average_monthly_phone_actual_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]

    def get_weather_region_vars(self):
        self.weather_region_data = \
            self.raw_data_obj.raw_weather_region_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]

    def get_seasoanl_data(self):
        # season data
        self.season_data = []
        dates_to_season = self.raw_dates[self.y_indexes].month * 100 + self.raw_dates[self.y_indexes].day
        for md in dates_to_season:
            if ((md > 320) and (md < 621)):
                s = 'spring'  # spring
            elif ((md > 620) and (md < 923)):
                s = 'summer'  # summer
            elif ((md > 922) and (md < 1223)):
                s = 'fall'  # fall
            else:
                s = 'winter'  # winter
            self.season_data.append(s)

    def get_actual_cwt(self):
        # invoice
        self.invoice_cwt_per_store_data = \
            self.raw_data_obj.raw_invoice_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.invoice_cwt_per_state_data = \
            self.raw_data_obj.raw_invoice_state_mean_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.fall_invoice_cwt_per_state_data = \
            self.raw_data_obj.raw_invoice_fall_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.winter_invoice_cwt_per_state_data = \
            self.raw_data_obj.raw_invoice_winter_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        # phone
        self.phone_cwt_per_store_data = \
            self.raw_data_obj.raw_phone_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.phone_cwt_per_state_data = \
            self.raw_data_obj.raw_phone_state_mean_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.fall_phone_cwt_per_state_data = \
            self.raw_data_obj.raw_phone_fall_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.winter_phone_cwt_per_state_data = \
            self.raw_data_obj.raw_phone_winter_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]

    def get_lagged_actual_cwt(self):
        # invoice
        self.lagged_invoice_cwt_per_store_data = \
            self.raw_data_obj.raw_invoice_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_invoice_cwt_per_state_data = \
            self.raw_data_obj.raw_invoice_state_mean_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_fall_invoice_cwt_per_state_data = \
            self.raw_data_obj.raw_invoice_fall_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_winter_invoice_cwt_per_state_data = \
            self.raw_data_obj.raw_invoice_winter_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]

        # phone
        self.lagged_phone_cwt_per_store_data = \
            self.raw_data_obj.raw_phone_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_phone_cwt_per_state_data = \
            self.raw_data_obj.raw_phone_state_mean_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_fall_phone_cwt_per_state_data = \
            self.raw_data_obj.raw_phone_fall_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_winter_phone_cwt_per_state_data = \
            self.raw_data_obj.raw_phone_winter_cwt_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]

    def get_actual_cwt_long(self):
        # invoice
        self.invoice_cwt_quarterly_per_store_data = \
            self.raw_data_obj.raw_invoice_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.invoice_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_invoice_state_mean_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.fall_invoice_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_invoice_fall_state_mean_cwt_mexh_quarter_by_store[
                self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.winter_invoice_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_invoice_winter_state_mean_cwt_mexh_quarter_by_store[
                self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        # phone
        self.phone_cwt_quarterly_per_store_data = \
            self.raw_data_obj.raw_phone_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.phone_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_phone_state_mean_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.fall_phone_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_phone_fall_state_mean_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]
        self.winter_phone_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_phone_winter_state_mean_cwt_mexh_quarter_by_store[
                self.preproc_config.store_id].reindex(
                self.raw_dates)[self.y_indexes]

    def get_lagged_actual_cwt_long(self):
        # invoice
        self.lagged_invoice_cwt_quarterly_per_store_data = \
            self.raw_data_obj.raw_invoice_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_invoice_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_invoice_state_mean_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_fall_invoice_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_invoice_fall_state_mean_cwt_mexh_quarter_by_store[
                self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_winter_invoice_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_invoice_winter_state_mean_cwt_mexh_quarter_by_store[
                self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        # phone
        self.lagged_phone_cwt_quarterly_per_store_data = \
            self.raw_data_obj.raw_phone_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_phone_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_phone_state_mean_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_fall_phone_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_phone_fall_state_mean_cwt_mexh_quarter_by_store[self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]
        self.lagged_winter_phone_cwt_quarterly_per_state_data = \
            self.raw_data_obj.raw_phone_winter_state_mean_cwt_mexh_quarter_by_store[
                self.preproc_config.store_id].reindex(
                self.raw_dates)[self.x_indexes]

    def make_cwt_feat_names(self, var_name, var_type):
        return f'{var_name}_{var_type}'

    def make_features(self):

        # loop through the feature list
        for feat in self.feature_df.columns:

            if feat == 'invoice_moving_mean_dow_perc':
                for i in self.preproc_config.dow_perc_moving_avg_lag_list:
                    dow_dict = {}
                    for d in range(7):
                        dow_dict[d] = self.invoice_dow_perc.loc[self.invoice_dow_perc.index.dayofweek == d].shift(
                            2).rolling(window=i, min_periods=1).mean()
                    comb = pd.concat([dow_dict[d] for d in dow_dict.keys()]).sort_index().reindex(self.y_data.index)
                    self.feature_df[f'invoice_{i}w_moving_mean_dow_perc'] = list(comb)
                self.feature_df.drop('invoice_moving_mean_dow_perc', axis=1, inplace=True)

            elif feat == 'phone_moving_mean_dow_perc':
                for i in self.preproc_config.dow_perc_moving_avg_lag_list:
                    dow_dict = {}
                    for d in range(7):
                        dow_dict[d] = self.phone_dow_perc.loc[self.phone_dow_perc.index.dayofweek == d].shift(
                            2).rolling(window=i, min_periods=1).mean()
                    comb = pd.concat([dow_dict[d] for d in dow_dict.keys()]).sort_index().reindex(self.y_data.index)
                    self.feature_df[f'phone_{i}w_moving_mean_dow_perc'] = list(comb)
                self.feature_df.drop('phone_moving_mean_dow_perc', axis=1, inplace=True)

            elif feat == 'invoice_moving_mean_dow':
                for i in self.preproc_config.dow_volume_moving_avg_lag_list:  # rolling period in weeks
                    dow_dict = {}
                    for d in range(7):
                        dow_dict[d] = self.raw_invoice[self.raw_invoice.index.dayofweek == d].shift(2).rolling(window=i,
                                                                                                               min_periods=1).mean()
                    comb = pd.concat([dow_dict[d] for d in dow_dict.keys()]).sort_index().reindex(self.y_data.index)
                    self.feature_df[f'invoice_{i}w_moving_mean_dow'] = list(comb)
                self.feature_df.drop('invoice_moving_mean_dow', axis=1, inplace=True)

            elif feat == 'phone_moving_mean_dow':
                for i in self.preproc_config.dow_volume_moving_avg_lag_list:  # rolling period in weeks
                    dow_dict = {}
                    for d in range(7):
                        dow_dict[d] = self.raw_phone_calls[self.raw_phone_calls.index.dayofweek == d].shift(2).rolling(
                            window=i, min_periods=1).mean()
                    comb = pd.concat([dow_dict[d] for d in dow_dict.keys()]).sort_index().reindex(self.y_data.index)
                    self.feature_df[f'phone_{i}w_moving_mean_dow'] = list(comb)
                self.feature_df.drop('phone_moving_mean_dow', axis=1, inplace=True)

            elif feat == 'invoice_moving_mean':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_mean = self.raw_invoice.rolling(i, min_periods=1).mean()
                    self.feature_df[f'invoice_moving_mean{i}'] = list(
                        cur_rolling_mean.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('invoice_moving_mean', axis=1, inplace=True)

            elif feat == 'invoice_moving_max':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_max = self.raw_invoice.rolling(i, min_periods=1).max()
                    self.feature_df[f'invoice_moving_max{i}'] = list(
                        cur_rolling_max.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('invoice_moving_max', axis=1, inplace=True)

            elif feat == 'invoice_lag':
                for i in range(0, self.preproc_config.feat_invoice_lag_max):
                    temp_df = pd.DataFrame(columns=['invoice_data'])
                    temp_df['invoice_data'] = self.raw_invoice.values
                    temp_df['shifted_invoice_data'] = temp_df.shift(i)
                    cur_invoice_shift_array = np.array(temp_df['shifted_invoice_data'])
                    self.feature_df[f'invoice_lag{i}'] = cur_invoice_shift_array[self.lagged_dow_x_indexes]
                self.feature_df.drop('invoice_lag', axis=1, inplace=True)

            elif feat == 'phone_lag':
                for i in range(0, self.preproc_config.feat_phone_lag_max):
                    temp_df = pd.DataFrame(columns=['phone_data'])
                    temp_df['phone_data'] = self.raw_phone_calls.values
                    temp_df['shifted_phone_data'] = temp_df.shift(i)
                    cur_phone_shift_array = np.array(temp_df['shifted_phone_data'])
                    self.feature_df[f'phone_lag{i}'] = cur_phone_shift_array[self.lagged_dow_x_indexes]
                self.feature_df.drop('phone_lag', axis=1, inplace=True)

            elif feat == 'phone_moving_mean':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_mean = self.raw_phone_calls.rolling(i, min_periods=1).mean()
                    self.feature_df[f'phone_moving_mean{i}'] = list(
                        cur_rolling_mean.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('phone_moving_mean', axis=1, inplace=True)

            elif feat == 'phone_moving_max':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_max = self.raw_phone_calls.rolling(i, min_periods=1).max()
                    self.feature_df[f'phone_moving_max{i}'] = list(
                        cur_rolling_max.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('phone_moving_max', axis=1, inplace=True)

            elif feat == 'invoice_velocity':
                for i in self.preproc_config.velocity_lag_list:
                    cur_velocity = self.raw_invoice.diff(periods=i)
                    self.feature_df[f'invoice_velocity{i}'] = list(cur_velocity.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('invoice_velocity', axis=1, inplace=True)

            elif feat == 'phone_velocity':
                for i in self.preproc_config.velocity_lag_list:
                    cur_velocity = self.raw_phone_calls.diff(periods=i)
                    self.feature_df[f'phone_velocity{i}'] = list(cur_velocity.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('phone_velocity', axis=1, inplace=True)

            elif feat == 'precip_moving_mean':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_mean = self.raw_precip_data.rolling(i, min_periods=1).mean()
                    self.feature_df[f'precip_moving_mean{i}'] = list(
                        cur_rolling_mean.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('precip_moving_mean', axis=1, inplace=True)

            elif feat == 'snow_moving_mean':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_mean = self.raw_snowfall_data.rolling(i, min_periods=1).mean()
                    self.feature_df[f'snow_moving_mean{i}'] = list(
                        cur_rolling_mean.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('snow_moving_mean', axis=1, inplace=True)

            elif feat == 'snow_moving_max':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_max = self.raw_snowfall_data.rolling(i, min_periods=1).max()
                    self.feature_df[f'snow_moving_max{i}'] = list(
                        cur_rolling_max.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('snow_moving_max', axis=1, inplace=True)

            elif feat == 'prev_week_invoice':
                self.feature_df[feat] = list(self.lagged_invoice_data)

            elif feat == 'month':
                month_names = pd.Index(list(calendar.month_name)[1:])
                month = pd.get_dummies(pd.Series(self.y_data.index.month_name())).reindex(columns=month_names, fill_value=0)
                month.columns = [f'is_{s.lower()}' for s in month.columns]
                self.feature_df = pd.concat([self.feature_df, month], axis=1)
                self.feature_df.drop('month', axis=1, inplace=True)

            elif feat == 'dayofweek':
                dow = pd.get_dummies(pd.Series(self.y_data.index.day_name()))
                dow.columns = [f'is_{s.lower()}' for s in dow.columns]
                self.feature_df = pd.concat([self.feature_df, dow], axis=1)
                self.feature_df.drop('dayofweek', axis=1, inplace=True)

            elif feat == 'prev_week_phone':
                self.feature_df[feat] = list(self.lagged_phone_data)

            elif feat == 'prev_week_precip':
                self.feature_df[feat] = list(self.lagged_precip_data)

            elif feat == 'prev_week_snowfall':
                self.feature_df[feat] = list(self.lagged_snowfall_data)

            elif feat == 'promotion':
                self.feature_df[feat] = list(self.promotion_ind)

            elif feat == 'holiday':
                self.feature_df = pd.concat([self.feature_df, self.holiday_dummies.reset_index(drop=True)], axis=1)
                self.feature_df.drop('holiday', axis=1, inplace=True)


            elif feat == 'lat_times_lon':
                self.feature_df[feat] = float(self.store_specific_data['store_latitude_degree']) * float(
                    self.store_specific_data['store_longitude_degree'])

            elif feat == 'dayofweek_times_lat_lon':
                dow = pd.get_dummies(pd.Series(self.y_data.index.day_name())).apply(
                    lambda x: x * float(self.store_specific_data['store_latitude_degree']) * float(
                        self.store_specific_data['store_longitude_degree']))
                dow.columns = [f'is_{s.lower()}_time_lat_lon' for s in dow.columns]
                self.feature_df = pd.concat([self.feature_df, dow], axis=1)
                self.feature_df.drop('dayofweek_times_lat_lon', axis=1, inplace=True)

            elif feat == 'promotion_cwt':
                self.feature_df[feat] = list(self.promotion_cwt)

            elif feat == 'holiday_cwt':
                self.feature_df[feat] = list(self.holiday_cwt)

            elif feat == 'promo_window':
                window_df = self.promotion_dummies.rolling(window=60, center=True, min_periods=1).max()
                self.feature_df = pd.concat([self.feature_df, window_df.reset_index(drop=True)], axis=1)
                self.feature_df.drop('promo_window', axis=1, inplace=True)

            elif feat == 'holiday_window':
                window_df = self.holiday_dummies.rolling(window=60, center=True, min_periods=1).max()
                self.feature_df = pd.concat([self.feature_df, window_df.reset_index(drop=True)], axis=1)
                self.feature_df.drop('holiday_window', axis=1, inplace=True)

            elif feat in self.raw_data_obj.store_attributes_names:
                self.feature_df[feat] = float(self.store_specific_data[feat])

            elif feat == 'is_party':
                self.feature_df[feat] = list(self.party_data)

            elif feat == 'yoy_invoice':
                self.feature_df[feat] = list(self.invoice_prev_month_yoy)
                self.feature_df['missing_invoice_yoy'] = list(self.missing_invoice_prev_month_yoy)

            elif feat == 'yoy_phone':
                self.feature_df[feat] = list(self.phone_prev_month_yoy)
                self.feature_df['missing_phone_yoy'] = list(self.missing_phone_prev_month_yoy)

            elif feat == 'previous_snow_count':
                self.feature_df[feat] = list(self.previous_snow_count_data)
                self.feature_df['previous_snow_month'] = list(self.previous_snow_one_month_data)

            elif feat == 'prediction_offset':
                self.feature_df[feat] = (self.pred_dates - self.lagged_dates).days

            elif feat == 'first_snow_day':
                self.feature_df[feat] = list(self.snow_day_data)

            elif feat == 'previous_first_snow_day':
                self.feature_df[feat] = list(self.previous_snow_day_data)

            elif feat == 'moy_perc_invoice':
                self.feature_df[feat] = list(self.invoice_moy_perc_data)

            elif feat == 'moy_perc_phone':
                self.feature_df[feat] = list(self.phone_moy_perc_data)

            elif feat == 'weekly_perc_invoice':
                self.feature_df[feat] = list(self.average_monthly_lagged_invoice_data)
                self.feature_df[feat] = (self.feature_df['invoice_moving_mean7'] * 7) / (self.feature_df[feat] * 30)

            elif feat == 'weekly_perc_phone':
                self.feature_df[feat] = list(self.average_monthly_lagged_phone_data)
                self.feature_df[feat] = (self.feature_df['phone_moving_mean7'] * 7) / (self.feature_df[feat] * 30)


            elif feat == 'weather_region':

                self.feature_df[feat] = list(self.weather_region_data)
                # one hot encoding
                self.feature_df = pd.concat([self.feature_df, pd.get_dummies(self.feature_df[feat])], axis=1)

                for region in self.preproc_config.winter_stores:
                    if region not in self.feature_df.columns:
                        self.feature_df[region] = 0
                self.feature_df.drop('weather_region', 1, inplace=True)

            elif feat == 'invoice_cwt':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(self.invoice_cwt_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(self.invoice_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.fall_invoice_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.winter_invoice_cwt_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)

            elif feat == 'phone_cwt':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(self.phone_cwt_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(self.phone_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.fall_phone_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.winter_phone_cwt_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)

            elif feat == 'lagged_invoice_cwt':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(
                    self.lagged_invoice_cwt_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(
                    self.lagged_invoice_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.lagged_fall_invoice_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.lagged_winter_invoice_cwt_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)


            elif feat == 'lagged_phone_cwt':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(
                    self.lagged_phone_cwt_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(
                    self.lagged_phone_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.lagged_fall_phone_cwt_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.lagged_winter_phone_cwt_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)


            elif feat == 'invoice_cwt_long':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(
                    self.invoice_cwt_quarterly_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(
                    self.invoice_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.fall_invoice_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.winter_invoice_cwt_quarterly_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)

            elif feat == 'phone_cwt_long':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(
                    self.phone_cwt_quarterly_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(
                    self.phone_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.fall_phone_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.winter_phone_cwt_quarterly_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)

            elif feat == 'lagged_phone_cwt_long':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(
                    self.lagged_phone_cwt_quarterly_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(
                    self.lagged_phone_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.lagged_fall_phone_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.lagged_winter_phone_cwt_quarterly_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)

            elif feat == 'lagged_invoice_cwt_long':

                self.feature_df[self.make_cwt_feat_names(feat, 'per_store')] = list(
                    self.lagged_invoice_cwt_quarterly_per_store_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state')] = list(
                    self.lagged_invoice_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_fall')] = list(
                    self.lagged_fall_invoice_cwt_quarterly_per_state_data)
                self.feature_df[self.make_cwt_feat_names(feat, 'per_state_winter')] = list(
                    self.lagged_winter_invoice_cwt_quarterly_per_state_data)
                self.feature_df.drop(feat, axis=1, inplace=True)

            elif feat == 'season':

                self.feature_df[feat] = list(self.season_data)
                season_dummies = pd.get_dummies(self.feature_df[feat])

                # rename the seasons
                season_dummies = season_dummies.rename(
                    columns={"fall": "is_fall", "spring": "is_spring", "summer": "is_summer", "winter": "is_winter"})

                # one hot encoding
                self.feature_df = pd.concat([self.feature_df, season_dummies], axis=1)
                cols = ['is_spring', 'is_summer', 'is_fall', 'is_winter']
                for season in cols:
                    if season not in self.feature_df.columns:
                        self.feature_df[season] = 0
                self.feature_df.drop('season', 1, inplace=True)

            elif feat == 'invoice_moving_mean_velocity':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_mean = self.raw_invoice.rolling(i, min_periods=1).mean()
                    cur_mean_velocity = cur_rolling_mean.diff(periods=i)
                    self.feature_df[f'invoice_moving_mean_velocity{i}'] = list(
                        cur_mean_velocity.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('invoice_moving_mean_velocity', axis=1, inplace=True)

            elif feat == 'phone_moving_mean_velocity':
                for i in self.preproc_config.moving_avg_lag_list:
                    cur_rolling_mean = self.raw_phone_calls.rolling(i, min_periods=1).mean()
                    cur_mean_velocity = cur_rolling_mean.diff(periods=i)
                    self.feature_df[f'phone_moving_mean_velocity{i}'] = list(
                        cur_mean_velocity.reindex(self.raw_dates)[self.x_indexes])
                self.feature_df.drop('phone_moving_mean_velocity', axis=1, inplace=True)


class AllStoreFeatures:
    def __init__(self, raw_data_obj, preproc_config, total_num_stores=None, shuffle_stores=False, raw_store_list=[],
                 is_make_master_df=False):
        print('Creating features for all stores.......')

        self.raw_data_obj = raw_data_obj
        self.preproc_config = preproc_config
        self.total_num_stores = total_num_stores
        self.shuffle_stores = shuffle_stores
        self.raw_store_list = raw_store_list
        self.is_make_master_df = is_make_master_df

        self.make_all_store_model_data()

    def make_all_store_model_data(self):

        count = 0
        self.x_train_list = []
        self.y_train_list = []
        self.x_test_list = []
        self.y_test_list = []
        self.store_id_list = []

        self.train_store_index = []
        self.test_store_index = []

        self.raw_all_train_dates = []
        self.raw_all_test_dates = []

        self.all_train_dates = []
        self.all_test_dates = []

        self.feat_df_list = []

        if self.preproc_config.master_df_fn and not self.is_make_master_df:
            raw_master_df = pd.read_csv(self.preproc_config.master_df_fn, index_col=False)

        # get the list of stores
        # set the output variable
        if len(self.raw_store_list) > 0:
            self.store_list = self.raw_store_list
        else:
            if self.preproc_config.master_df_fn and not self.is_make_master_df:
                self.store_list = raw_master_df['store_id'].unique()

            else:

                if self.preproc_config.output_var == 'invoices':

                    self.store_list = self.raw_data_obj.invoice_df_by_store.columns
                elif self.preproc_config.output_var == 'phone_calls':
                    self.store_list = self.raw_data_obj.phone_df_by_store.columns

                # shuffle stores
                if self.shuffle_stores:
                    random.seed(0)
                    self.store_list = random.sample(list(self.store_list), self.total_num_stores)
                elif self.total_num_stores:
                    self.store_list = self.store_list[:self.total_num_stores]

        print(f'Running processing for {len(self.store_list)} stores')
        for cur_store in self.store_list:

            print(f'Getting data for store {cur_store}......')
            try:
                self.preproc_config.store_id = cur_store

                # Feature section
                if self.preproc_config.master_df_fn and not self.is_make_master_df:
                    cur_store_master_df = raw_master_df[raw_master_df.store_id == cur_store]
                    cur_store_master_df = cur_store_master_df.reset_index()
                    y_data = cur_store_master_df['y_data']
                    y_data.index = pd.to_datetime(cur_store_master_df['date'])
                    x_data = cur_store_master_df.drop(['y_data', 'date', 'store_id', 'index'], axis=1)

                else:

                    feat_obj = SingleStoreFeatures(self.preproc_config, raw_data_obj=self.raw_data_obj)
                    x_data = feat_obj.feature_df
                    y_data = feat_obj.y_data


                # filter out sundays
                not_sunday_index = x_data.index[x_data['is_sunday'] == 0]
                x_data = x_data.iloc[not_sunday_index].drop(['is_sunday'], axis=1)
                y_data = y_data.iloc[not_sunday_index]

                if self.is_make_master_df:
                    output_master_df = feat_obj.feature_df
                    output_master_df['y_data'] = np.array(feat_obj.y_data)
                    output_master_df['date'] = np.array(feat_obj.y_data.index.strftime('%Y-%m-%d'))
                    output_master_df['store_id'] = [cur_store] * len(np.array(feat_obj.y_data))
                    self.feat_df_list.append(output_master_df)
                else:
                    model_data = ModelData(x_data=x_data, y_data=y_data,
                                           preproc_config=self.preproc_config,
                                           train_perc=0.75)

                    self.all_train_dates.extend(model_data.train_dates)
                    self.all_test_dates.extend(model_data.test_dates)

                    self.raw_all_train_dates.extend(model_data.raw_train_dates)
                    self.raw_all_test_dates.extend(model_data.raw_test_dates)

                    # store the training data

                    self.x_train_list.append(model_data.x_train)
                    self.y_train_list.append(model_data.y_train)

                    # store the testing data
                    self.x_test_list.append(model_data.x_test)
                    self.y_test_list.append(model_data.y_test)

                    if len(model_data.y_train.shape) > 1:
                        self.train_store_index.extend([cur_store] * len(model_data.y_train.flatten()))
                        self.test_store_index.extend([cur_store] * len(model_data.y_test.flatten()))
                    else:
                        self.train_store_index.extend([cur_store] * len(model_data.y_train))
                        self.test_store_index.extend([cur_store] * len(model_data.y_test))

                    # store the store id
                    self.store_id_list.append(cur_store)

            except Exception as e:
                print(f'was not able to get data for store {cur_store}')
                print(f'The exception was {e}')

            count += 1

        # get the list of features
        self.feat_list = list(x_data.columns)

        if self.is_make_master_df:

            pd.concat(self.feat_df_list).to_csv(self.preproc_config.master_df_fn, index=False)
        else:
            # concate the training data
            self.y_train = np.concatenate(self.y_train_list)
            self.x_train = np.concatenate(self.x_train_list)

            # concate the testing data
            self.y_test = np.concatenate(self.y_test_list)
            self.x_test = np.concatenate(self.x_test_list)

            if any(self.raw_all_train_dates):
                self.train_mask_indexes = \
                    np.where(np.array([index_f.dayofweek for index_f in self.raw_all_train_dates]) != 6)[0]

            if any(self.raw_all_test_dates):
                self.test_mask_indexes = \
                    np.where(np.array([index_f.dayofweek for index_f in self.raw_all_test_dates]) != 6)[0]
