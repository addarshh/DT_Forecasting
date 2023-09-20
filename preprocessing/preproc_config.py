import multiprocessing
import yaml
import os


class PreProcConfig:
    def __init__(self, store_id=None, feature_list=[], is_single_store=False, output_var='invoices', main_dir=None,
                 is_forecast_week=True, is_run_parallel=False, is_create_master_df=False, run_type='full_pipeline',
                 master_df_fn=None, min_forecast_lead=None, ymal_fn='config.yml', model_fn_dict={},data_folder='data'):

        # set the directory
        self.main_dir = main_dir
        working_dir = f'{self.main_dir}{data_folder}/'
        os.chdir(working_dir)

        # get the config
        self.ymal_fn = ymal_fn
        self.read_yml()

        # set main parameters
        self.store_id = store_id
        self.feature_list = feature_list
        self.is_single_store = is_single_store
        self.output_var = output_var
        self.is_forecast_week = is_forecast_week
        self.is_run_parallel = is_run_parallel
        self.run_type = run_type
        self.min_forecast_lead = min_forecast_lead
        self.model_fn_dict = model_fn_dict

        # set the master df file name
        if is_create_master_df:
            self.master_df_fn = f'{self.main_dir}data/{self.run_type}_{self.output_var}_master_df.csv'

        else:

            if master_df_fn:
                self.master_df_fn = master_df_fn
            else:
                self.master_df_fn = None

        # store specific features
        self.store_specific_vars = ['avg_daily_temp',
                                    'avg_yearly_pcp',
                                    'avg_yearly_acc_snow',
                                    'max_daily_acc_snow',
                                    'max_daily_pcp',
                                    'stddev_daily_temp',
                                    'stddev_yearly_pcp',
                                    'stddev_yearly_acc_snow',
                                    'elevation_meter',
                                    'total_population',
                                    'median income',
                                    'store_age']

        # main feature list
        if self.output_var == 'invoices':
            self.output_plot_var = 'Invoices'

            self.feature_list = [
                'prev_week_invoice',
                'prev_week_phone',
                'invoice_moving_mean',
                'invoice_moving_max',
                'invoice_velocity',
                'invoice_moving_mean_dow',
                'invoice_moving_mean_dow_perc',
                'invoice_lag',
                'holiday',
                'holiday_cwt',
                'promotion',
                'promotion_cwt',
                'promo_window',
                'month',
                'dayofweek',
                'prev_week_precip',
                'precip_moving_mean',
                'prev_week_snowfall',
                'snow_moving_mean',
                'snow_moving_max',
                'is_party',
                'yoy_invoice',
                'prediction_offset',
                'previous_snow_count',
                'moy_perc_invoice',
                'weekly_perc_invoice',
                'season',
                'invoice_cwt',
                'lagged_invoice_cwt',
                'invoice_cwt_long',
                'lagged_invoice_cwt_long',
                'invoice_moving_mean_velocity'
            ]

            self.feature_list.extend(self.store_specific_vars)



        elif self.output_var == 'phone_calls':
            self.output_plot_var = 'Phone Calls'
            self.feature_list = [
                'prev_week_phone',
                'prev_week_invoice',
                'phone_moving_mean',
                'phone_moving_max',
                'phone_velocity',
                'phone_moving_mean_dow',
                'phone_moving_mean_dow_perc',
                'phone_lag',
                'holiday',
                'holiday_cwt',
                'promotion',
                'promotion_cwt',
                'promo_window',
                'month',
                'dayofweek',
                'prev_week_precip',
                'precip_moving_mean',
                'prev_week_snowfall',
                'snow_moving_mean',
                'snow_moving_max',
                'is_party',
                'yoy_phone',
                'prediction_offset',
                'previous_snow_count',
                'moy_perc_phone',
                'weekly_perc_phone',
                'season',
                'phone_cwt',
                'lagged_phone_cwt',
                'phone_cwt_long',
                'lagged_phone_cwt_long',
                'phone_moving_mean_velocity'
            ]
            self.feature_list.extend(self.store_specific_vars)

        # make lag dictionary
        self.perc_dow_lag_dict = {
            index_f: int(self.predict_day - index_f) if self.predict_day - index_f >= 0 else int(5 + abs(
                self.predict_day - index_f)) for index_f in range(7)}

        # get the number of cpus
        try:
            self.num_cpus = int(multiprocessing.cpu_count() / 2)
        except NotImplementedError:
            self.num_cpus = 2  # arbitrary default

        # check if we need to get UQ
        if 'lower_ci_model_fn' in list(self.model_fn_dict.keys()) or 'upper_ci_model_fn' in list(
                self.model_fn_dict.keys()):
            self.is_output_uq = True
        else:
            self.is_output_uq = False

    def read_yml(self):

        with open(self.ymal_fn, 'r') as stream:
            try:
                yml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # get the feature variables
        self.velocity_lag_list = list(yml_dict['feature_variables']['velocity_lag_list'])
        self.dow_volume_moving_avg_lag_list = list(yml_dict['feature_variables']['dow_volume_moving_avg_lag_list'])
        self.dow_perc_moving_avg_lag_list = list(yml_dict['feature_variables']['dow_perc_moving_avg_lag_list'])
        self.moving_avg_lag_list = list(yml_dict['feature_variables']['moving_avg_lag_list'])
        self.feat_invoice_lag_max = int(yml_dict['feature_variables']['feat_invoice_lag_max'])
        self.feat_phone_lag_max = int(yml_dict['feature_variables']['feat_phone_lag_max'])

        # get the lag variables
        self.lag = int(yml_dict['lag_variables']['lag'])
        self.max_lag = int(yml_dict['lag_variables']['max_lag'])
        self.predict_day = int(yml_dict['lag_variables']['predict_day'])
        self.default_min_forecast_lead = int(yml_dict['lag_variables']['default_min_forecast_lead'])
        self.long_min_forecast_lead = int(yml_dict['lag_variables']['long_min_forecast_lead'])

        # get the misc variables
        self.metric_end_dates = list(yml_dict['misc_variables']['metric_end_dates'])
        self.winter_stores = list(yml_dict['misc_variables']['winter_stores'])
        self.ci_perc = float(yml_dict['misc_variables']['ci_perc'])
        self.uq_models = list(yml_dict['misc_variables']['uq_models'])
        self.multi_week_forecasts_states = list(yml_dict['misc_variables']['multi_week_forecasts_states'])
