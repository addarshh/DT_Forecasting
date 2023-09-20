from preprocessing.preproc_read_data import RawData
from exploratory_analysis.preproc_seasonality import make_output_data
from preprocessing.preproc_config import PreProcConfig

#TODO: put this in an object
is_seasonally_adjust = False
is_fit_single_method = True
is_run_single_store = True
is_forecast_week = True
model_abbrev = 'rf'
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
feat_list.extend([f'phone_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'precip_moving_avg_mean{i}' for i in moving_avg_lag_list])
feat_list.extend([f'invoice_velocity{i}' for i in velocity_lag_list])




# get the raw data
raw_data_obj = RawData()


preproc_config = PreProcConfig(store_id=cur_store, lag=6, max_lag=24, moving_avg_lag_list=moving_avg_lag_list,
                               velocity_lag_list=velocity_lag_list,
                               feature_list=feat_list, is_single_store=True,output_var=output_var,is_forecast_week=is_forecast_week)


raw_data_obj = make_output_data(raw_data_obj,is_seasonally_adjust)

from preprocessing.preproc_split_data import ModelData
from preprocessing.preproc_features import SingleStoreFeatures
from sklearn.ensemble import RandomForestRegressor

raw_data_obj = raw_data_obj
preproc_config = preproc_config
is_fit_single_method = is_fit_single_method

model_abbrev = model_abbrev

feat_obj = SingleStoreFeatures(preproc_config, raw_data_obj=raw_data_obj)


feat_obj.feature_df.iloc[10]
feat_obj.feature_df.iloc[11]
# get test and training set
model_data = ModelData(x_data=feat_obj.feature_df, y_data=feat_obj.y_data,
                       preproc_config=preproc_config,
                       train_perc=.80)

rf_model = RandomForestRegressor(random_state=0, max_depth=5).fit(
    model_data.x_train_vector_model, model_data.y_train_vector_model)
in_samp_preds = rf_model.predict(model_data.x_train_vector_model).flatten()
out_samp_preds = rf_model.predict(model_data.x_test_vector_model).flatten()

import numpy as np
print(f'mse {np.mean((out_samp_preds-model_data.y_test_vector_model.flatten())**2)}')
print(f'corr {np.corrcoef((out_samp_preds,model_data.y_test_vector_model.flatten()))[0,1]}')
print(f'mad {np.median(np.abs(out_samp_preds-model_data.y_test_vector_model.flatten()))}')
print(f'% error {np.median(np.abs(out_samp_preds-model_data.y_test_vector_model.flatten()))/np.mean(model_data.y_test)}')



def create_vector_data(cur_dates, y_data, x_data):
    train_dayofweek = [index_f.dayofweek for index_f in cur_dates]
    monday_indexes = np.where(np.array(train_dayofweek) == 0)[0]

    full_week_indexes = []
    for cur_monday in monday_indexes:
        if train_dayofweek[cur_monday:(cur_monday + 6)] == [0, 1, 2, 3, 4, 5]:
            full_week_indexes.append(cur_monday)

    y_array = np.full((len(full_week_indexes), 6), np.nan)
    x_indexes  = []
    for count, cur_monday in enumerate(full_week_indexes):
        y_array[count, :] = y_data[cur_monday:(cur_monday + 6)]
        x_indexes.extend(np.arange(cur_monday,(cur_monday + 6),1))

    x_array = x_data.values[full_week_indexes, :]
    x_alt_array = x_data.values[x_indexes, :]

    return y_array,x_alt_array


_,model_data.x_train = create_vector_data(model_data.train_dates, model_data.y_train,model_data.x_train)
_,model_data.x_test = create_vector_data(model_data.test_dates, model_data.y_test,model_data.x_test)

rf_model = RandomForestRegressor(random_state=0, max_depth=5).fit(
    model_data.x_train, model_data.y_train_vector_model.flatten())
in_samp_preds = rf_model.predict(model_data.x_train)
out_samp_preds = rf_model.predict(model_data.x_test)

import numpy as np
print(f'mse {np.mean((out_samp_preds-model_data.y_test_vector_model.flatten())**2)}')
print(f'corr {np.corrcoef((out_samp_preds,model_data.y_test_vector_model.flatten()))[0,1]}')
print(f'mad {np.median(np.abs(out_samp_preds-model_data.y_test_vector_model.flatten()))}')
print(f'% error {np.median(np.abs(out_samp_preds-model_data.y_test_vector_model.flatten()))/np.mean(model_data.y_test)}')





