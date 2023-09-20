from preprocessing.preproc_read_data import RawData
from exploratory_analysis.preproc_seasonality import make_output_data
from preprocessing.preproc_config import PreProcConfig
from preprocessing.preproc_features import SingleStoreFeatures

# TODO: put this in an object
is_seasonally_adjust = False
is_fit_single_method = True
is_run_single_store = True
is_forecast_week = True

model_abbrev = 'ert'
output_var = 'invoices'

# set the current store
cur_store = 'WIG 01'
# cur_store = 'OHC 02'
# cur_store = 'TXH 22'
# cur_store = 'COD 12'
# cur_store = 'COD 10'
# cur_store = 'NVL 04'
# cur_store = 'CAL 21'
# cur_store = 'TXD 57'
# cur_store = 'TXA 05'

# set the config
moving_avg_lag_list = [7]
velocity_lag_list = [7]
feat_invoice_lag_max = 7
feat_list = ['dow_perc','month', 'dayofweek', 'prev_week_invoice', 'prev_week_phone', 'prev_week_precip', 'prev_week_snowfall',
             'promotion', 'holiday', 'lagged_promotion']
feat_list.extend([f'invoice_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'invoice_moving_avg_max{i}' for i in moving_avg_lag_list])
feat_list.extend([f'precip_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'invoice_velocity{i}' for i in velocity_lag_list])
feat_list.extend([f'invoice_lag{i}' for i in range(1,feat_invoice_lag_max)])
preproc_config = PreProcConfig(store_id=cur_store, lag=6, max_lag=24, moving_avg_lag_list=moving_avg_lag_list,
                               velocity_lag_list=velocity_lag_list,
                               feature_list=feat_list, is_single_store=True, output_var=output_var,
                               is_forecast_week=is_forecast_week,feat_invoice_lag_max=feat_invoice_lag_max)

# get the raw data
raw_data_obj = RawData(preproc_config)


raw_data_obj = make_output_data(raw_data_obj, is_seasonally_adjust)

feat_obj = SingleStoreFeatures(preproc_config, raw_data_obj=raw_data_obj)

# raw_invoice = raw_data_obj.seasonal_adjust_invoice[preproc_config.store_id]
# min_date_invoice, max_date_invoice = raw_invoice.index.min(), raw_invoice.index.max()
# raw_invoice = raw_invoice.reindex(
#     pd.date_range(raw_invoice.first_valid_index(), raw_invoice.last_valid_index()), fill_value=np.nan)
# raw_invoice = raw_invoice.fillna(method='ffill')
# raw_invoice = raw_invoice.reindex(pd.date_range(min_date_invoice, max_date_invoice))
#
# lagged_datesraw_dates = raw_invoice.index
#
#
#
#
feat_obj.feature_df.iloc[10]
feat_obj.feature_df.iloc[11]



