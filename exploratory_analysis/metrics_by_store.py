import pandas as pd
import numpy as np

pred_df = pd.read_csv('/Users/mcdermop/Desktop/out_samp_preds.csv')
all_stores = np.array(pred_df['store_id'].unique())
mse_list = []
map_list = []
corr_list = []
mad_list = []

for cur_store in all_stores:
    print(cur_store)
    cur_df = pred_df[pred_df['store_id']==cur_store]
    mse_list.append(np.mean((cur_df['out_samp_data']-cur_df['out_samp_preds_Ensemble Forecast'])**2))
    corr_list.append(np.corrcoef(cur_df['out_samp_data'],cur_df['out_samp_preds_Ensemble Forecast'])[0,1])
    map_list.append((np.mean(np.abs(cur_df['out_samp_data']-cur_df['out_samp_preds_Ensemble Forecast'])))/np.mean(cur_df['out_samp_data']))
    mad_list.append((np.mean(np.abs(cur_df['out_samp_data']-cur_df['out_samp_preds_Ensemble Forecast']))))


output_df = pd.DataFrame(columns=['store_id','mse','mape','corr','mad'])

output_df['store_id'] = all_stores
output_df['mse'] = mse_list
output_df['mape'] = map_list
output_df['corr'] = corr_list
output_df['mad'] = mad_list
output_df.to_csv('/Users/mcdermop/Desktop/phone_metrics_by_store.csv')
