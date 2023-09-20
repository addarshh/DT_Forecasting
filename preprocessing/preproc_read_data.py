import sys
import pandas as pd
import copy
sys.path.append('/DTFORECASTING/')

from preprocessing.preproc_derived_data import create_derived_data

class RawData:
    def __init__(self, preproc_config, start_date='2017-01-01', end_date='2020-06-30',is_make_derived_feats=False):
        # get datasets
        self.get_invoice_daily()
        self.get_phone_daily()
        self.get_raw_store_attributes()
        self.get_weather_daily()
        self.get_promotion_calendar()
        self.get_holiday_calendar()

        if is_make_derived_feats:
            create_derived_data(self,start_date,end_date)

        self.get_all_store_attributes()
        self.get_yoy_data()
        self.get_moy_data()
        self.get_snow_feature_data()
        self.get_cwt_data()
        self.get_party_calendar_data()

    def get_invoice_daily(self):
        self.raw_invoice_df = pd.read_csv('invoice_daily_actuals.csv')

        # fix the dates
        self.raw_invoice_df['effective_date'] = pd.to_datetime(self.raw_invoice_df['effective_date'], format='%Y%m%d')

        # pivot by store
        self.invoice_df_by_store = self.raw_invoice_df.pivot(index='effective_date', columns='store_code',
                                                             values='actual')
        # sort data by store
        self.invoice_df_by_store = self.invoice_df_by_store.sort_values(by='effective_date')

    def get_phone_daily(self):
        self.raw_phone_df = pd.read_csv('phone_daily_actuals.csv')
        self.raw_phone_df['effective_date'] = pd.to_datetime(self.raw_phone_df['effective_date'], format="%Y-%m-%d")

        # pivot by store
        self.phone_df_by_store = self.raw_phone_df.pivot(index='effective_date', columns='store_code', values='actual')
        # sort data by store
        self.phone_df_by_store = self.phone_df_by_store.sort_values(by='effective_date')


    def get_promotion_calendar(self):
        self.raw_promotion_df = pd.read_csv('promotion_calendar.csv')

        self.promotion_df = copy.deepcopy(self.raw_promotion_df)
        self.promotion_df['PromotionDate'] = [
            '/'.join(cur_date.split('/')[:2]) + '/' + ''.join('20' + str(cur_date).split('/')[-1]) for cur_date in
            self.promotion_df['PromotionDate']]
        self.promotion_df['PromotionDate'] = pd.to_datetime(self.promotion_df['PromotionDate'])
        self.promotion_df.set_index('PromotionDate', inplace=True)

    def get_holiday_calendar(self):
        self.raw_holiday_df = pd.read_csv('holiday_calendar.csv')

        self.holiday_df = copy.deepcopy(self.raw_holiday_df)
        self.holiday_df['HolidayDate'] = [
            '/'.join(cur_date.split('/')[:2]) + '/' + ''.join('20' + str(cur_date).split('/')[-1]) for cur_date in
            self.raw_holiday_df['HolidayDate']]


        self.holiday_df['HolidayDate'] = pd.to_datetime(self.holiday_df['HolidayDate'])
        self.holiday_df.set_index('HolidayDate', inplace=True)


    def get_raw_store_attributes(self):
        self.raw_store_attibutes_df = pd.read_csv('store_attributes.csv')

    def get_all_store_attributes(self):
        self.all_store_attibutes_df = pd.read_csv('final_store_attributes.csv')
        self.store_attributes_names = list(self.all_store_attibutes_df.columns)

    def get_weather_daily(self):
        self.raw_weather_df = pd.read_csv('weather_daily_actuals.csv')
        self.raw_weather_df['date'] = pd.to_datetime(self.raw_weather_df['date'], format='%Y%m%d')

        # pivot by store
        self.temp_min_df_by_store = self.raw_weather_df.pivot(index='date', columns='store_code',
                                                              values='temp_min_degrees_f').fillna(0.0)
        self.temp_max_df_by_store = self.raw_weather_df.pivot(index='date', columns='store_code',
                                                              values='temp_max_degrees_f').fillna(0.0)

        self.precip_df_by_store = self.raw_weather_df.pivot(index='date', columns='store_code',
                                                            values='precipitation_inches').fillna(0.0)

        self.snowfall_df_by_store = self.raw_weather_df.pivot(index='date', columns='store_code',
                                                              values='snowfall_inches').fillna(0.0)

        self.snow_accum_df_by_store = self.raw_weather_df.pivot(index='date', columns='store_code',
                                                                values='snow_accumulation_inches').fillna(0.0)

    def get_party_calendar_data(self):
        self.raw_party_calendar_data = pd.read_csv('party_calendar.csv')
        # get only the party dates
        self.raw_party_calendar_data = self.raw_party_calendar_data.loc[self.raw_party_calendar_data['is_party'] == 1]
        # datetime
        self.raw_party_calendar_data['date'] = pd.to_datetime(self.raw_party_calendar_data['date'])
        # merge with store data
        self.raw_party_calendar_data = pd.merge(self.all_store_attibutes_df[['store_code']],
                                                self.raw_party_calendar_data[['date', 'store_code', 'is_party']],
                                                how='left', on='store_code')
        # only get the date where there is a party
        self.party_by_store = self.raw_party_calendar_data.pivot(index='date', columns='store_code',
                                                                 values='is_party')

    def get_yoy_data(self):
        self.raw_yoy_data = pd.read_csv('yoy.csv').drop_duplicates()
        self.raw_yoy_data['date'] = pd.to_datetime(self.raw_yoy_data['date'])

        self.invoice_prev_month_yoy = self.raw_yoy_data.pivot(index='date', columns='store_code',
                                                              values='invoice_prev_month_yoy')
        self.missing_invoice_prev_month_yoy = self.raw_yoy_data.pivot(index='date', columns='store_code',
                                                                      values='missing_invoice_prev_month_yoy')

        self.phone_prev_month_yoy = self.raw_yoy_data.pivot(index='date', columns='store_code',
                                                            values='phone_prev_month_yoy')
        self.missing_phone_prev_month_yoy = self.raw_yoy_data.pivot(index='date', columns='store_code',
                                                                    values='missing_phone_prev_month_yoy')

    def get_moy_data(self):
        self.raw_moy_data = pd.read_csv('moy.csv')
        self.raw_moy_data['date'] = pd.to_datetime(self.raw_moy_data['date'])
        self.average_monthly_invoice_actual_by_store = self.raw_moy_data.pivot(index='date',
                                                                                        columns='store_code',
                                                                                        values='average_monthly_invoice')
        self.average_monthly_phone_actual_by_store = self.raw_moy_data.pivot(index='date',
                                                                                      columns='store_code',
                                                                                      values='average_monthly_phone')
        self.invoice_moy_perc_by_store = self.raw_moy_data.pivot(index='date', columns='store_code',
                                                                          values='invoice_moy_perc')
        self.phone_moy_perc_by_store = self.raw_moy_data.pivot(index='date', columns='store_code',
                                                                        values='phone_moy_perc')

    def get_snow_feature_data(self):
        self.raw_snow_feature_data = pd.read_csv('snow_feature.csv')
        # convert to datetime
        self.raw_snow_feature_data['date'] = pd.to_datetime(self.raw_snow_feature_data['date'], format='%Y/%m/%d')
        # pivot by store
        self.snow_day_by_store = self.raw_snow_feature_data.pivot(index='date', columns='store_code',
                                                                  values='first_snow')
        self.previous_snow_by_store = self.raw_snow_feature_data.pivot(index='date', columns='store_code',
                                                                       values='previous_first_snow_rolling_two_weeks')
        self.previous_snow_count_by_store = self.raw_snow_feature_data.pivot(index='date', columns='store_code',
                                                                             values='snow_count_weekly')
        self.previous_snow_week_by_store = self.raw_snow_feature_data.pivot(index='date', columns='store_code',
                                                                            values='previous_first_snow_one_week')
        self.previous_snow_month_by_store = self.raw_snow_feature_data.pivot(index='date', columns='store_code',
                                                                             values='previous_first_snow_one_month')

    def get_cwt_data(self):
        self.raw_cwt_data = pd.read_csv('cwt.csv')
        # datetime
        self.raw_cwt_data['date'] = pd.to_datetime(self.raw_cwt_data['date'],
                                                   format='%Y/%m/%d')
        # pivot by store
        # higher sampling frequency
        self.raw_invoice_cwt_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                values='invoice_cwt')
        self.raw_invoice_state_mean_cwt_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                           values='invoice_state_mean_cwt')
        self.raw_invoice_winter_cwt_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                       values='invoice_winter_state_mean_cwt')
        self.raw_invoice_fall_cwt_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                     values='invoice_fall_state_mean_cwt')
        self.raw_phone_cwt_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                              values='phone_cwt')
        self.raw_phone_state_mean_cwt_by_store = self.raw_cwt_data.pivot(index='date',
                                                                         columns='store_code',
                                                                         values='phone_state_mean_cwt')
        self.raw_phone_winter_cwt_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                     values='phone_winter_state_mean_cwt')
        self.raw_phone_fall_cwt_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                   values='phone_fall_state_mean_cwt')
        # lower sampling frequency
        self.raw_invoice_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                             values='invoice_cwt_mexh_quarter')
        self.raw_invoice_state_mean_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date',
                                                                                        columns='store_code',
                                                                                        values='invoice_state_mean_cwt_mexh_quarter')
        self.raw_invoice_winter_state_mean_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date',
                                                                                               columns='store_code',
                                                                                               values='invoice_winter_state_mean_cwt_mexh_quarter')
        self.raw_invoice_fall_state_mean_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date',
                                                                                             columns='store_code',
                                                                                             values='invoice_fall_state_mean_cwt_mexh_quarter')
        self.raw_phone_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date', columns='store_code',
                                                                           values='phone_cwt_mexh_quarter')
        self.raw_phone_state_mean_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date',
                                                                                      columns='store_code',
                                                                                      values='phone_state_mean_cwt_mexh_quarter')
        self.raw_phone_winter_state_mean_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date',
                                                                                             columns='store_code',
                                                                                             values='phone_winter_state_mean_cwt_mexh_quarter')
        self.raw_phone_fall_state_mean_cwt_mexh_quarter_by_store = self.raw_cwt_data.pivot(index='date',
                                                                                           columns='store_code',
                                                                                           values='phone_fall_state_mean_cwt_mexh_quarter')
        # winter_region
        self.raw_weather_region_by_store = self.raw_cwt_data.pivot(index='date',
                                                                   columns='store_code',
                                                                   values='weather_region')
        # promotion
        self.raw_promotion_cwt_by_store = self.raw_cwt_data.pivot(index='date',
                                                                   columns='store_code',
                                                                   values='promotion_cwt')
        # holiday
        self.raw_holiday_cwt_by_store = self.raw_cwt_data.pivot(index='date',
                                                                   columns='store_code',
                                                                   values='holiday_cwt')
