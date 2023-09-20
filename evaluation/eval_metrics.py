import numpy as np
import pandas as pd

class Metrics:

    def __init__(self, raw_output_csv, preproc_config):

        self.metric_names = ['MSE', 'MAD', 'Corr', 'Mean MAPE', 'Median MAPE']

        df_list = []

        for cur_date in preproc_config.metric_end_dates:
            output_csv = raw_output_csv.sort_values(by=['prediction_date'])
            output_csv.index = output_csv['prediction_date']
            output_csv = output_csv.loc[:cur_date]

            self.y_pred = output_csv['predictions']
            self.y_truth = output_csv['out_samp_data']
            self.output_csv = output_csv

            self.calc_all_metrics()

            self.metric_list = [self.mse, self.mad, self.corr, self.mean_mape, self.median_mape]

            cur_df = pd.DataFrame([self.metric_list],columns=self.metric_names)
            df_list.append(cur_df)

        self.final_metric_df = pd.concat(df_list)
        self.final_metric_df['OOS_End_Date'] = preproc_config.metric_end_dates

    def calc_mse(self):
        self.mse = np.round(np.mean((self.y_truth - self.y_pred) ** 2), 3)

    def calc_mad(self):
        self.mad = np.round(np.mean(np.abs(self.y_truth - self.y_pred)), 3)

    def calc_corr(self):
        self.corr = np.round(np.corrcoef(self.y_truth, self.y_pred)[0, 1], 3)

    def calc_old_mape(self):
        self.old_mape = np.round(self.mad / np.mean(self.y_truth), 3)

    def calc_mape(self):

        all_stores_mape_list = np.abs(self.y_truth - self.y_pred) / self.y_truth

        self.mean_mape = np.round(np.mean(all_stores_mape_list), 3)
        self.median_mape = np.round(np.median(all_stores_mape_list), 3)

    def calc_all_metrics(self):
        # calculate metrics
        self.calc_mse()
        self.calc_mad()
        self.calc_corr()
        self.calc_mape()
        self.calc_old_mape()