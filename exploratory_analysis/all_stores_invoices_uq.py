from preprocessing.preproc_read_data import RawData
from exploratory_analysis.preproc_seasonality import make_output_data
from preprocessing.preproc_config import PreProcConfig
from preprocessing.preproc_features import AllStoreFeatures
from models.run_model_pipeline import RunEnsem

# seasonally adjust data
is_fit_single_method = False
model_abbrev = 'gb'
preproc_config = PreProcConfig( is_single_store=True)

# get the raw data
raw_data_obj = RawData(preproc_config)

# make the raw data
raw_data_obj = make_output_data(raw_data_obj)

model_data = AllStoreFeatures(raw_data_obj, preproc_config, total_num_stores=50,shuffle_stores=True)


ensem_obj = RunEnsem(model_data, preproc_config, is_single_model=True)



###############################################
##### Pure Ensemble CI ########################
###############################################
import numpy as np
store_id = 'OHS 05'
# store_id = 'NCC 37'
store_test_indexes = np.where(np.array(model_data.test_store_index) == store_id)[0]


pure_ensem_vals = ensem_obj.main_methods_df_out.values[store_test_indexes,:]
pure_ensem_mean = np.mean(pure_ensem_vals,axis=1)
pur_ensem_out_sample_lw_95 = np.quantile(pure_ensem_vals,.025,axis=1)
pur_ensem_out_sample_up_95= np.quantile(pure_ensem_vals,.975,axis=1)

# plot
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 12))
plt.plot(np.array(model_data.all_test_dates)[store_test_indexes], model_data.y_test[store_test_indexes], color='lightblue', linewidth=3,
         label='Out-sample Truth')
plt.plot(np.array(model_data.all_test_dates)[store_test_indexes], pure_ensem_mean, color='lightcoral', linewidth=2,
         label='Out-sample Predictions', linestyle='--')

plt.fill_between(np.array(model_data.all_test_dates)[store_test_indexes], pur_ensem_out_sample_lw_95, pur_ensem_out_sample_up_95,
                 color='saddlebrown', alpha=.30, label='Uncertainty Envelope')

plt.title(
    f'Ensemble UQ; Invoice Forecast for Store: {store_id} ',
    fontsize=20)
plt.xlabel('Date (day)', fontsize=18)
plt.ylabel(f'Number of Invoices', fontsize=18)

plt.legend(fontsize=16)

#####################################
##### BMA-EM ########################
#####################################
import numpy as np
from scipy.stats import norm

num_methods = 6
samp_size = len(model_data.y_train)


y_in_samp_preds = ensem_obj.main_methods_df_in.values

# set the latent variable
cur_z_mat = np.full((num_methods,samp_size),np.nan)

# set the parameters
cur_weights = np.array([1/num_methods]*num_methods)
in_samp_diffs_squared = [(y_in_samp_preds[:,indez_f]-model_data.y_train)**2 for indez_f in range(num_methods)]
cur_sigma_2 = np.mean(in_samp_diffs_squared)



for j in range(7):
    # E-Step
    for i in range(samp_size):
        cur_weighted_sum_array = cur_weights*norm.pdf(model_data.y_train[i],y_in_samp_preds[i, :], np.sqrt(cur_sigma_2))
        cur_denom = np.sum(cur_weighted_sum_array)
        cur_z_mat[:,i] = [index_f/cur_denom for index_f in cur_weighted_sum_array]

    # M-Step
    cur_weights = np.mean(cur_z_mat,axis=1)
    cur_sigma_2 = np.mean([cur_z_mat[index_f,:]*np.array(in_samp_diffs_squared)[index_f,:] for index_f in range(num_methods)])


    # calc the log-likelihood
    log_sum = 0
    for i in range(samp_size):
        log_sum+=np.log(np.sum(cur_weights*norm.pdf(model_data.y_train[i],y_in_samp_preds[i, :], np.sqrt(cur_sigma_2))))


    print(f'j {j}.......log sum {log_sum}.....sigma_2 {cur_sigma_2}.....weights {cur_weights}')




store_id = 'WIG 01'
store_test_indexes = np.where(np.array(model_data.test_store_index) == store_id)[0]

num_mc_samples = 1500
y_out_samp_preds = ensem_obj.main_methods_df_out.values[store_test_indexes,:]

num_out_samp_periods = len(y_out_samp_preds)
out_samp_pred_samples = np.full((num_mc_samples,num_out_samp_periods),np.nan)

for i in range(num_mc_samples):
    print(f'iteration {i}')
    for j in range(num_out_samp_periods):
        out_samp_pred_samples[i,j] = np.sum(cur_weights*norm.rvs(y_out_samp_preds[j,:],np.sqrt(cur_sigma_2),num_methods))


# summaries
out_samp_preds = np.mean(out_samp_pred_samples,axis=0)
out_sample_lw_95 = np.quantile(out_samp_pred_samples,.025,axis=0)
out_sample_up_95= np.quantile(out_samp_pred_samples,.975,axis=0)

mad = np.round(np.mean(np.abs(out_samp_preds - model_data.y_test[store_test_indexes])), 3)
perc_error = mad / np.mean(model_data.y_test[store_test_indexes])
mse = np.round(np.mean((out_samp_preds-model_data.y_test[store_test_indexes])**2),2)
corr = np.round(np.corrcoef((out_samp_preds,model_data.y_test[store_test_indexes]))[0,1],2)
print(f'MSE {mse}')
print(f'Corr Coeff {corr}')
print(f'MAD {mad}')
print(f'perc_error {perc_error}')


# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 12))
plt.plot(np.array(model_data.all_test_dates)[store_test_indexes], model_data.y_test[store_test_indexes], color='lightblue', linewidth=3,
         label='Out-sample Truth')
plt.plot(np.array(model_data.all_test_dates)[store_test_indexes], out_samp_preds, color='lightcoral', linewidth=2,
         label='Out-sample Predictions', linestyle='--')

plt.fill_between(np.array(model_data.all_test_dates)[store_test_indexes], out_sample_lw_95, out_sample_up_95,
                 color='saddlebrown', alpha=.30, label='Uncertainty Envelope')

plt.title(
    f'BMA All Store Model; Week Ahead Forecast for Store: {store_id} \n Out-of-sample MSE: {mse} Corr: {corr}',
    fontsize=20)
plt.xlabel('Date (day)', fontsize=18)
plt.ylabel(f'Number of Invoices', fontsize=18)

plt.legend(fontsize=16)








num_mc_samples = 100
y_out_samp_preds = ensem_obj.main_methods_df_out.values

num_out_samp_periods = len(y_out_samp_preds)
out_samp_pred_samples = np.full((num_mc_samples,num_out_samp_periods),np.nan)

for i in range(num_mc_samples):
    print(f'iteration {i}')
    for j in range(num_out_samp_periods):
        out_samp_pred_samples[i,j] = np.sum(cur_weights*norm.rvs(y_out_samp_preds[j,:],np.sqrt(cur_sigma_2),num_methods))


# summaries
out_samp_preds = np.mean(out_samp_pred_samples,axis=0)
out_sample_lw_95 = np.quantile(out_samp_pred_samples,.025,axis=0)
out_sample_up_95= np.quantile(out_samp_pred_samples,.975,axis=0)

mad = np.round(np.mean(np.abs(out_samp_preds - model_data.y_test)), 3)
perc_error = mad / np.mean(model_data.y_test)

print(f'MSE {np.mean((out_samp_preds-model_data.y_test)**2)}')
print(f'Corr Coeff {np.corrcoef((out_samp_preds,model_data.y_test))}')
print(f'MAD {mad}')
print(f'perc_error {perc_error}')
