import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import LinearRegression


class Comparison:
    def __init__(self, model_data, preproc_config):
        self.model_data = model_data
        self.preproc_config = preproc_config

        if self.preproc_config.output_var == 'invoices':
            self.main_var_name = 'invoice_lag0'
            self.ma_var_name = 'invoice_moving_avg_mean'

        elif self.preproc_config.output_var == 'phone_calls':
            self.main_var_name = 'phone_lag0'
            self.ma_var_name = 'phone_moving_avg_mean'

        self.prev_week_invoice_index = np.where(np.array(self.model_data.feat_list) == self.main_var_name)[0][0]

        if isinstance(model_data.y_train, pd.Series):
            self.model_data.y_train = copy.deepcopy(model_data.y_train.values)
            self.model_data.y_test = copy.deepcopy(model_data.y_test.values)

        if isinstance(model_data.x_train, pd.DataFrame):
            self.model_data.x_train = copy.deepcopy(model_data.x_train.values)
            self.model_data.x_test = copy.deepcopy(model_data.x_test.values)

    def persistence_forecast(self):
        self.model_name = 'Persistance'
        self.in_samp_preds = self.model_data.x_train[:, self.prev_week_invoice_index]
        self.out_samp_preds = self.model_data.x_test[:, self.prev_week_invoice_index]

    def moving_average_forecast(self):
        self.model_name = 'Moving Average'
        if any([self.ma_var_name in index_f for index_f in self.preproc_config.feature_list]):

            ma_name = np.array(self.preproc_config.feature_list)[
                np.where([self.ma_var_name in index_f for index_f in self.preproc_config.feature_list])[0][0]]

            invoice_moving_avg_mean_index = np.where(np.array(self.model_data.feat_list) == ma_name)[
                0][0]
            self.model_name = 'Moving Average'
            self.in_samp_preds = self.model_data.x_train[:, invoice_moving_avg_mean_index]
            self.out_samp_preds = self.model_data.x_test[:, invoice_moving_avg_mean_index]
        else:
            print(f'ERROR! moving avg data is not available')

    def ar_forecast(self):

        self.model_name = 'AR 1'

        lr_model = LinearRegression().fit(
            self.model_data.x_train[:, self.prev_week_invoice_index].reshape(
                len(self.model_data.x_train[:, self.prev_week_invoice_index]), 1), self.model_data.y_train)
        self.in_samp_preds = lr_model.predict(self.model_data.x_train[:, self.prev_week_invoice_index].reshape(
            len(self.model_data.x_train[:, self.prev_week_invoice_index]), 1))
        self.out_samp_preds = lr_model.predict(self.model_data.x_test[:, self.prev_week_invoice_index].reshape(
            len(self.model_data.x_test[:, self.prev_week_invoice_index]), 1))
