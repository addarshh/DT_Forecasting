import pandas as pd
import numpy as np


scored_data = pd.read_csv('/Users/mcdermop/Desktop/Projects/discount_tire/scored_data/invoices_forecast_7_6_2020/out_samp_preds.csv')
scored_data = scored_data.rename(columns={'store_id':'store_code'})
true_data = pd.read_csv('/Users/mcdermop/Desktop/updated_dt_data/invoice_daily_actuals.csv')
true_data['prediction_date'] = pd.to_datetime(true_data['effective_date'], format='%Y%m%d')
true_data['prediction_date'] = true_data['prediction_date'].astype(str)
merge_data = scored_data.merge(true_data, how='left', on=['prediction_date','store_code'])

print(f"Median MAPE {np.nanmedian(np.abs(merge_data['predictions'] - merge_data['actual'])/ merge_data['actual'])}")
print(f"Mean MAPE {np.mean(np.abs(merge_data['predictions'] - merge_data['actual'])/ merge_data['actual'])}")
print(f"MSE {np.mean(np.abs(merge_data['predictions'] - merge_data['actual'])**2)}")