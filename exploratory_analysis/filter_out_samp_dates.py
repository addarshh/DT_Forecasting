import pandas as pd
import datetime
import numpy as np
raw_data = pd.read_csv('/Users/mcdermop/Desktop/out_samp_preds.csv')
raw_data['date'] = pd.to_datetime(raw_data['date'] )

raw_data = raw_data.sort_values(by=['date'])
raw_data.index = raw_data['date']
cur_df = raw_data.loc[:'2020-02-29']
forecast_name = 'out_samp_preds_XG-Boost'

mape = np.mean(np.abs(cur_df['out_samp_data']-cur_df[forecast_name]))/np.mean(cur_df['out_samp_data'])
corr = np.corrcoef(cur_df['out_samp_data'],cur_df[forecast_name])[0,1]
MSE = np.mean((cur_df['out_samp_data']-cur_df[forecast_name])**2)
mad = np.mean(np.abs(cur_df['out_samp_data']-cur_df[forecast_name]))

print(np.mean(np.abs(cur_df['out_samp_data'] - cur_df[forecast_name]) / cur_df['out_samp_data']))
print(np.median(np.abs(cur_df['out_samp_data'] - cur_df[forecast_name]) / cur_df['out_samp_data']))

#0.12878355687255938
#0.10604994400117611