from preprocessing.preproc_read_data import RawData
from exploratory_analysis.preproc_seasonality import make_output_data
from preprocessing.preproc_config import PreProcConfig
from exploratory_analysis.run_model_pipeline import RunStorePipeline, run_individual_models_all_stores
from evaluation.eval_output import Output

# TODO: put this in an object
# seasonally adjust data
is_seasonally_adjust = False
is_fit_single_method = True
is_run_single_store = False
model_abbrev = 'gb'
output_var = 'phone_calls'

# set the current store
# cur_store = 'WIG 01'
# cur_store = 'OHC 02'
# cur_store = 'AZP 14'
# cur_store = 'COD 12'
# cur_store = 'COD 10'
# cur_store = 'NVL 04'
# cur_store = 'CAL 01'
cur_store = 'TXA 05'

# set the config
moving_avg_lag_list = [6]
velocity_lag_list = [6]
feat_phone_lag_max = 6
feat_list = ['month', 'dayofweek', 'prev_week_invoice', 'prev_week_precip', 'prev_week_snowfall',
             'promotion', 'holiday', 'lagged_promotion']
feat_list.extend([f'phone_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'invoice_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'phone_velocity{i}' for i in velocity_lag_list])
feat_list.extend([f'phone_lag{i}' for i in range(1,feat_phone_lag_max)])

preproc_config = PreProcConfig(store_id=cur_store, lag=6, max_lag=24, moving_avg_lag_list=moving_avg_lag_list,
                               velocity_lag_list=velocity_lag_list,
                               feature_list=feat_list, is_single_store=True, output_var=output_var,feat_phone_lag_max=feat_phone_lag_max)

# get the raw data
raw_data_obj = RawData(preproc_config)
raw_data_obj = make_output_data(raw_data_obj, is_seasonally_adjust)

if is_run_single_store:
    pipeline_obj = RunStorePipeline(raw_data_obj, preproc_config, is_fit_single_method, model_abbrev=model_abbrev)

    if is_fit_single_method:
        # make the plots
        plot_obj = Output(pipeline_obj.nonlinear_reg_obj, pipeline_obj.model_data, pipeline_obj.preproc_config,
                          in_samp_metrics=pipeline_obj.in_samp_metrics,
                          out_samp_metrics=pipeline_obj.out_samp_metrics)

        plot_obj.make_forecast_plots()
        plot_obj.output_results()
        plot_obj.output_metrics()

else:
    run_individual_models_all_stores(raw_data_obj, preproc_config, is_fit_single_method, model_abbrev,total_num_stores=10,
                                     is_output_plots=False)

