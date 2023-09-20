# import pandas as pd
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#      print(x_data.isna().sum())
# # import numpy as np
#
# print(np.mean(model_data.y_test ** 2))
#
#
# for val in model_data.y_test:
#     print(f'val {val}...{val.index}')
# import pandas as pd
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(model_data.y_test)
#
# import matplotlib.pyplot as plt
# plt.plot(np.array(model_data.y_test))
# plt.plot(np.array(model_data.x_test['invoice_velocity2']))
# plt.plot(np.array(model_data.x_test['prev_week_phone']))
# 'prev_week_invoice'
#
#
#
# plt.figure(figsize=(14, 12))
#
# plt.plot(model_data.y_test.index.values, model_data.y_test.values, color='lightblue', linewidth=3,
#          label='Out-sample Truth')
#
# plt.plot(model_data.y_test.index.values, nonlinear_reg_obj.out_samp_preds, color='lightcoral', linewidth=2,
#          label='Out-sample Predictions', linestyle='--')
# plt.plot(model_data.y_test.index.values, model_data.x_test.prev_week_phone, color='green', linewidth=.5,
#          label='Previous Day Phone Calls', linestyle='--')
# plt.title(
#     f'{1}-Day Ahead Forecast with {nonlinear_reg_obj.model_name} for Store: {preproc_config.store_id} \n Out-of-sample MSE: {out_samp_metrics.mse} Corr: {out_samp_metrics.corr}',
#     fontsize=17)
#
# plt.ylim(0,150)
#
# plt.xlabel('Date (in days)')
# plt.ylabel(f'Number of Invoices')
#
# plt.legend(fontsize=16)
#
#
#
#
# plt.plot(model_data.y_test.index.values, model_data.y_test.values-nonlinear_reg_obj.out_samp_preds, color='lightblue', linewidth=3,
#          label='Out-sample Truth')
#
# plt.hist(model_data.y_test.values-nonlinear_reg_obj.out_samp_preds)
# x_data[list(x_data.columns)] = scaler.fit_transform(x_data[list(x_data.columns)])


# def make_first_snow_day_indicator(self):
#      first_snow_day_indicator = np.full((len(self.y_data)), np.nan)
#
#      first_snow_day_indicator[0:(266 - 10)] = 10
#      first_snow_day_indicator[(266 - 10):(266)] = np.flip(np.arange(1, 11, 1))
#      first_snow_day_indicator[266] = 0
#      first_snow_day_indicator[266 + 1:266 + 25] = -1 * np.arange(1, 25, 1)
#      first_snow_day_indicator[266 + 25:(630 - 10)] = 10
#
#      first_snow_day_indicator[(630 - 10):(630)] = np.flip(np.arange(1, 11, 1))
#      first_snow_day_indicator[630] = 0
#      first_snow_day_indicator[630 + 1:630 + 25] = -1 * np.arange(1, 25, 1)
#      first_snow_day_indicator[630 + 25:(997 - 10)] = 10
#
#      first_snow_day_indicator[(997 - 10):(997)] = np.flip(np.arange(1, 11, 1))
#      first_snow_day_indicator[997] = 0
#      first_snow_day_indicator[997 + 1:997 + 25] = -1 * np.arange(1, 25, 1)
#      first_snow_day_indicator[997 + 25:] = 10
#
#      scales = 18
#      scales = np.arange(1, scales)
#      coeffs, freqs = pywt.cwt(list(first_snow_day_indicator), scales, wavelet='mexh')
#      # create scalogram
#      cwt = pd.DataFrame(np.transpose(coeffs))
#
#      return list(-cwt[3])




# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# def get_weather_forecast_data(self):
#     self.raw_weather_station_data = pd.read_csv('weather_station_data.csv')
#     self.raw_weather_forecast_data = pd.read_csv('world_wide_technologies_2020-07-29t184021.csv')
#     self.raw_weather_forecast_data['f_forecast_date'] = pd.to_datetime(self.raw_weather_forecast_data['f_forecast_date'])
#
#     weather_station_coords = np.column_stack(
#         (self.raw_weather_station_data['lat'], self.raw_weather_station_data['lon']))
#     neigh = NearestNeighbors(n_neighbors=2)
#     neigh.fit(weather_station_coords)
#     raw_nn_indexes = neigh.kneighbors(np.column_stack(
#         (self.all_store_attibutes_df['store_latitude_degree'],
#          self.all_store_attibutes_df['store_longitude_degree'])), 1,
#         return_distance=False)
#
#     self.store_station_lookup = pd.DataFrame(columns=['store_code', 'station_id'])
#     self.store_station_lookup['store_code'] = self.all_store_attibutes_df['store_code']
#     self.store_station_lookup['station_id'] = [self.raw_weather_station_data['station_id'][index_f[0]] for index_f
#                                                in
#                                                raw_nn_indexes]