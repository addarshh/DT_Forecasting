import  pandas as pd
import numpy as np

input_dir = '/Users/mcdermop/Desktop/updated_dt_data/'
output_dir = '/Users/mcdermop/Desktop/Projects/discount_tire/field_test_1/'

current_car = 'weather'
raw_phone_df_1 = pd.read_csv(f'{output_dir}{current_car}_daily_actuals.csv')
raw_phone_df_2 = pd.read_csv(f'{input_dir}{current_car}_daily_actuals.csv')
raw_phone_df = pd.concat([raw_phone_df_1,raw_phone_df_2])
raw_phone_df = raw_phone_df.drop_duplicates(subset= ['store_code','date'])
raw_phone_df = raw_phone_df.loc[:, ~raw_phone_df.columns.str.contains('^Unnamed')]

raw_phone_df.to_csv(f'{output_dir}{current_car}_daily_actuals.csv',index=False)




current_car = 'phone'
raw_phone_df_1 = pd.read_csv(f'{output_dir}{current_car}_daily_actuals.csv')
raw_phone_df_2 = pd.read_csv(f'{input_dir}{current_car}_daily_actuals.csv')
# raw_phone_df_1['date'] = pd.to_datetime(raw_phone_df_1['effective_date'])
# raw_phone_df_2['date'] = pd.to_datetime(raw_phone_df_2['effective_date'])
raw_phone_df = pd.concat([raw_phone_df_1,raw_phone_df_2])
raw_phone_df = raw_phone_df.drop_duplicates(subset= ['store_code','effective_date'])

raw_phone_df.to_csv(f'{output_dir}{current_car}_daily_actuals.csv',index=False)





current_car = 'invoice'
raw_phone_df_1 = pd.read_csv(f'{output_dir}{current_car}_daily_actuals.csv')
raw_phone_df_2 = pd.read_csv(f'{input_dir}{current_car}_daily_actuals.csv')
# raw_phone_df_1['date'] = pd.to_datetime(raw_phone_df_1['effective_date'])
# raw_phone_df_2['date'] = pd.to_datetime(raw_phone_df_2['effective_date'])
raw_phone_df = pd.concat([raw_phone_df_1,raw_phone_df_2])
raw_phone_df = raw_phone_df.drop_duplicates(subset= ['store_code','effective_date'])

raw_phone_df.to_csv(f'{output_dir}{current_car}_daily_actuals.csv',index=False)

