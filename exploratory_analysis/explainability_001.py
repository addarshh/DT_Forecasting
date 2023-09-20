from preprocessing.preproc_read_data import RawData
from exploratory_analysis.preproc_seasonality import make_output_data
from preprocessing.preproc_config import PreProcConfig
from preprocessing.preproc_features import AllStoreFeatures
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

# seasonally adjust data
is_fit_single_method = False
model_abbrev = 'gb'
preproc_config = PreProcConfig(main_dir='/Users/mcdermop/Desktop/Projects/discount_tire/', is_single_store=True, output_var='invoices')

# get the raw data
raw_data_obj = RawData(preproc_config)

# make the raw data
raw_data_obj = make_output_data(raw_data_obj)

# get all features
cur_store_code = 'COD'
indexes = np.array([idx for idx, s in enumerate(raw_data_obj.invoice_df_by_store.columns) if cur_store_code in s])
raw_store_list = np.array(raw_data_obj.invoice_df_by_store.columns)[indexes]
# raw_store_list = raw_store_list[:2]

num_stores = len(raw_store_list)
# model_data = AllStoreFeatures(raw_data_obj, preproc_config, total_num_stores=num_stores,raw_store_list=raw_store_list, is_use_master_df=False)

model_data = AllStoreFeatures(raw_data_obj, preproc_config=preproc_config)



raw_store_list = np.unique( model_data.store_id_list)

preproc_config.store_id = 'all_stores'
x_test_df =pd.DataFrame(data=model_data.x_test,   # 1st column as index
            columns=model_data.feat_list)

gb_model = GradientBoostingRegressor(random_state=0,verbose=True).fit(
    model_data.x_train, model_data.y_train)
out_samp_preds = gb_model.predict(model_data.x_test)

explainerXGB = shap.TreeExplainer(gb_model)
shap_values_XGB_test = explainerXGB.shap_values(model_data.x_test)


store_id_list = []
outpput_date_list = []
for store in raw_store_list:
    cur_date_list = np.array(model_data.all_test_dates)[np.where(np.array(model_data.test_store_index)==store)[0]]
    outpput_date_list.extend(cur_date_list)
    store_id_list.extend([store]*len(cur_date_list))


output_df = pd.DataFrame(data=shap_values_XGB_test,columns=model_data.feat_list)
output_df['store_id'] = store_id_list
output_df['date'] = outpput_date_list

output_df.to_csv(f'/Users/mcdermop/Desktop/invoices_08_20_2020_all_stores_stores_shap_values.csv')

# output_df.to_csv(f'/Users/mcdermop/Desktop/stores_shap_values_{cur_store_code}.csv')
#output_df.to_csv(f'/Users/mcdermop/Desktop/phone_08_05_2020_all_stores_stores_shap_values.csv')

shap.force_plot(explainerXGB.expected_value, shap_values_XGB_test[229], x_test_df.iloc[229],show=False,matplotlib=True)
plt.savefig('/Users/mcdermop/Desktop/temp.png')
shap.decision_plot(explainerXGB.expected_value, shap_values_XGB_test, x_test_df)

shap.save_html('/Users/mcdermop/Desktop/explainer.html',shap.force_plot(explainerXGB.expected_value, shap_values_XGB_test[182], x_test_df.iloc[182]))
# shap.save_html('/Users/mcdermop/Desktop/explainer.html',shap.force_plot(explainerXGB.expected_value, shap_values_XGB_test,x_test_df))
# plt.savefig('/Users/mcdermop/Desktop/explainer.html')
#
# shap.summary_plot(shap_values_XGB_test, x_test_df, plot_type="bar")
# shap.summary_plot(shap_values_XGB_test, x_test_df)
# np.where(np.array([str(val).split(' ') for val in model_data.all_test_dates])=='2019-10-05')[0]