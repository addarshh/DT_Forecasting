import pandas as pd
import numpy as np
model_name_list = ['gam','rf', 'ert', 'lr', 'gb', 'nn']
main_methods_df_out = pd.DataFrame(columns=model_name_list)
main_methods_df_out['gam'] = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_17_16_15_68157_invoices_all_stores_gam/out_samp_preds.csv')['out_samp_preds_GAM']
main_methods_df_out['rf'] = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_17_16_30_68157_invoices_all_stores_rf/out_samp_preds.csv')['out_samp_preds_Random Forest']
main_methods_df_out['ert'] = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_17_16_41_68157_invoices_all_stores_ert/out_samp_preds.csv')['out_samp_preds_Extremely Random Trees']
main_methods_df_out['lr'] = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_17_16_55_68157_invoices_all_stores_lr/out_samp_preds.csv')['out_samp_preds_Linear Regression']
main_methods_df_out['gb'] = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_1_17_04_68157_invoices_all_stores_gb/out_samp_preds.csv')['out_samp_preds_XG-Boost']
main_methods_df_out['nn'] = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_17_17_16_68157_invoices_all_stores_nn/out_samp_preds.csv')['out_samp_preds_Neural Network']

ensem_count = 0
out_samp_ensem = np.zeros((len(main_methods_df_out['gam'])))
for cur_method in main_methods_df_out.columns:
    out_samp_ensem += main_methods_df_out[cur_method]
    ensem_count += 1

out_samp_preds = out_samp_ensem / ensem_count

raw_data = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_17_16_15_68157_invoices_all_stores_gam/out_samp_preds.csv')
raw_data.rename(columns={'out_samp_preds_GAM':'out_samp_preds_Ensem'}, inplace=True)
raw_data['out_samp_preds_Ensem'] = np.array(out_samp_preds)
raw_data = raw_data.sort_values(by=['date'])
raw_data.index = raw_data['date']

cur_df = raw_data.loc[:'2020-03-31']
forecast_name = 'out_samp_preds_Ensem'

mape = np.mean(np.abs(cur_df['out_samp_data']-cur_df[forecast_name]))/np.mean(cur_df['out_samp_data'])
corr = np.corrcoef(cur_df['out_samp_data'],cur_df[forecast_name])[0,1]
MSE = np.mean((cur_df['out_samp_data']-cur_df[forecast_name])**2)
mad = np.mean(np.abs(cur_df['out_samp_data']-cur_df[forecast_name]))
print(f'MSE {MSE}')
print(f'mad {mad}')
print(np.mean(np.abs(cur_df['out_samp_data'] - cur_df[forecast_name]) / cur_df['out_samp_data']))
print(np.median(np.abs(cur_df['out_samp_data'] - cur_df[forecast_name]) / cur_df['out_samp_data']))


###############################################
##### Pure Ensemble CI ########################
###############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ensem_data = pd.read_csv('/Users/mcdermop/Desktop/invoices/out_samp_preds.csv')
#ensem_data = raw_data

# store_id = 'IAD 02'
# store_id = 'ILC 11'
# store_id = 'OKO 10'
# store_id = 'COD 30'


# store_id = 'AZP 36'
store_id = 'COD 12'
store_test_indexes = np.where(np.array(ensem_data['store_id']) == store_id)[0]



plt.figure(figsize=(14, 12))
plt.plot(np.array(ensem_data['date'])[store_test_indexes], ensem_data['out_samp_data'][store_test_indexes], color='lightblue', linewidth=3,
         label='Out-sample Truth')
plt.plot(np.array(ensem_data['date'])[store_test_indexes], ensem_data['out_samp_preds_XG-Boost'][store_test_indexes], color='lightcoral', linewidth=2,
         label='Out-sample Predictions', linestyle='--')

plt.fill_between(np.array(ensem_data['date'])[store_test_indexes], ensem_data['lower_95_ci'][store_test_indexes], ensem_data['upper_95_ci'][store_test_indexes],
                 color='saddlebrown', alpha=.30, label='Uncertainty Envelope')

plt.plot(np.array(ensem_data['date'])[store_test_indexes], ensem_data['upper_95_ci'][store_test_indexes]-ensem_data['lower_95_ci'][store_test_indexes], color='blue', linewidth=4,
         label='Interval Width')

plt.title(
    f'XG-Boost Quantile Regression; Invoice Forecast for Store: {store_id} ',
    fontsize=20)

plt.xticks(np.array(ensem_data['date'])[store_test_indexes][np.arange(0,len(np.array(ensem_data['date'])[store_test_indexes]),35)])
plt.xlabel('Date (day)', fontsize=18)
plt.ylabel(f'Number of Invoices', fontsize=18)

plt.legend(fontsize=16)



# check coverage:
raw_data = ensem_data
raw_data['date'] = pd.to_datetime(raw_data['date'] )

raw_data = raw_data.sort_values(by=['date'])
raw_data.index = raw_data['date']
cur_df = raw_data.loc[:'2020-02-29']


in_ci_counter =0
for count,value in enumerate(cur_df['out_samp_data']):
    if (value>cur_df['lower_95_ci'][count]) and (value<cur_df['upper_95_ci'][count]):
        in_ci_counter+=1

print(in_ci_counter/len(cur_df))


len(np.where(cur_df['lower_95_ci'] -cur_df['predictions']>0)[0])
100
len(np.where(cur_df['predictions']-cur_df['upper_95_ci'] >0)[0])
17

# Enesemble
raw_data = pd.read_csv('/Users/mcdermop/Desktop/08_07_2020/2020-08-07_17_16_15_68157_invoices_all_stores_gam/out_samp_preds.csv')
raw_data.rename(columns={'out_samp_preds_GAM':'out_samp_preds_Ensem'}, inplace=True)
raw_data['out_samp_preds_Ensem'] = np.array(out_samp_preds)
raw_data['lower_95_ci'] = np.quantile(main_methods_df_out.values, .05, axis=1)
raw_data['upper_95_ci']= np.quantile(main_methods_df_out.values, .95, axis=1)

raw_data = raw_data.sort_values(by=['date'])
raw_data.index = raw_data['date']
cur_df = raw_data.loc[:'2020-02-29']

in_ci_counter =0
for count,value in enumerate(cur_df['out_samp_data']):
    if (value>cur_df['lower_95_ci'][count]) and (value<cur_df['upper_95_ci'][count]):
        in_ci_counter+=1

print(in_ci_counter/len(cur_df))
