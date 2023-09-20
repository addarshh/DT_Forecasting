from preprocessing.preproc_read_data import RawData
from exploratory_analysis.preproc_seasonality import make_output_data
from preprocessing.preproc_config import PreProcConfig
from exploratory_analysis.run_model_pipeline import RunStorePipeline
from scipy.stats import lognorm

#TODO: put this in an object
is_seasonally_adjust = False
is_fit_single_method = False
is_run_single_store = True
is_forecast_week = False
model_abbrev = 'exp_arima'
output_var = 'invoices'

# set the current store
cur_store = 'WIG 01'
# cur_store = 'OHC 02'
# cur_store = 'AZP 14'
# cur_store = 'COD 12'
# cur_store = 'COD 10'
# cur_store = 'NVL 04'
# cur_store = 'CAL 25'

# set the config
moving_avg_lag_list = [6]
velocity_lag_list = [6]
feat_list = ['month', 'dayofweek', 'prev_week_invoice', 'prev_week_phone', 'prev_week_precip', 'prev_week_snowfall',
             'promotion', 'holiday', 'lagged_promotion']
feat_list.extend([f'invoice_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'invoice_moving_avg_max{i}' for i in moving_avg_lag_list])
feat_list.extend([f'precip_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'invoice_velocity{i}' for i in velocity_lag_list])

# get the raw data
raw_data_obj = RawData()


preproc_config = PreProcConfig(store_id=cur_store, lag=6, max_lag=24, moving_avg_lag_list=moving_avg_lag_list,
                               velocity_lag_list=velocity_lag_list,
                               feature_list=feat_list, is_single_store=True,output_var=output_var,is_forecast_week=is_forecast_week)


raw_data_obj = make_output_data(raw_data_obj,is_seasonally_adjust)


pipeline_obj = RunStorePipeline(raw_data_obj, preproc_config, is_fit_single_method,
                                                           model_abbrev=model_abbrev)




#####################################
##### BMA-EM ########################
#####################################
import numpy as np

num_methods = 6
samp_size = len(pipeline_obj.model_data.y_train)
y_in_samp_preds = np.log(pipeline_obj.ensem_obj.main_methods_df_in.values)

# set the latent variable 
cur_z_mat = np.full((num_methods,samp_size),np.nan)

# set the parameters 
cur_weights = np.array([1/num_methods]*num_methods)
in_samp_diffs_squared = [(y_in_samp_preds[:,indez_f]-np.log(pipeline_obj.model_data.y_train))**2 for indez_f in range(num_methods)]
# cur_sigma_2 = np.mean(in_samp_diffs_squared)
cur_sigma_2 = 1



for j in range(15):
    # E-Step
    for i in range(samp_size):
        cur_weighted_sum_array = cur_weights*lognorm.pdf(np.log(pipeline_obj.model_data.y_train[i]),y_in_samp_preds[i, :], np.sqrt(cur_sigma_2))
        cur_denom = np.sum(cur_weighted_sum_array)
        cur_z_mat[:,i] = [index_f/cur_denom for index_f in cur_weighted_sum_array]

    # M-Step
    cur_weights = np.mean(cur_z_mat,axis=1)
    cur_sigma_2 = np.mean([cur_z_mat[index_f,:]*np.array(in_samp_diffs_squared)[index_f,:] for index_f in range(num_methods)])


    # calc the log-likelihood
    log_sum = 0
    # for i in range(samp_size):
    #     log_sum+=np.log(np.sum(cur_weights*norm.pdf(pipeline_obj.model_data.y_train[i],y_in_samp_preds[i, :], np.sqrt(cur_sigma_2))))


    print(f'j {j}.......log sum {log_sum}.....sigma_2 {cur_sigma_2}.....weights {cur_weights}')




# pipeline_obj.model_data.y_data
# pipeline_obj.ensem_obj.main_methods_df_in
# pipeline_obj.ensem_obj.main_methods_df_out


num_mc_samples = 1000
y_out_samp_preds = np.log(pipeline_obj.ensem_obj.main_methods_df_out.values)

num_out_samp_periods = len(pipeline_obj.model_data.y_test)
out_samp_pred_samples = np.full((num_mc_samples,num_out_samp_periods),np.nan)

for i in range(num_mc_samples):
    print(f'iteration {i}')
    for j in range(num_out_samp_periods):
        out_samp_pred_samples[i,j] = np.sum(cur_weights*lognorm.rvs(y_out_samp_preds[j,:],np.sqrt(cur_sigma_2),num_methods))


# summaries
out_samp_preds = np.median(out_samp_pred_samples,axis=0)
out_sample_lw_95 = np.quantile(out_samp_pred_samples,.10,axis=0)
out_sample_up_95= np.quantile(out_samp_pred_samples,.90,axis=0)

# plot
import matplotlib.pyplot as plt

plt.plot(pipeline_obj.model_data.test_dates, pipeline_obj.model_data.y_test, color='lightblue', linewidth=3,
         label='Out-sample Truth')
plt.plot(pipeline_obj.model_data.test_dates, out_samp_preds, color='lightcoral', linewidth=2,
         label='Out-sample Predictions', linestyle='--')

plt.fill_between(pipeline_obj.model_data.test_dates, out_sample_lw_95, out_sample_up_95,
                 color='saddlebrown', alpha=.30, label='Uncertainty Envelope')

print(f'MSE {np.mean((out_samp_preds-pipeline_obj.model_data.y_test)**2)}')