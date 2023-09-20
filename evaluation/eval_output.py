import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import numpy as np
from pathlib import Path
import sys
from joblib import load
sys.path.append('/DTFORECASTING/')
from evaluation.eval_metrics import Metrics

class OutputConfig():
    def __init__(self,preproc_config,model_abbrev):
        self.working_dir = f'{preproc_config.main_dir}results/'
        self.output_id = str(np.random.choice(100000))
        self.master_dir = f"{self.working_dir}{datetime.now().strftime('%m-%d-%Y').replace('-', '_')}/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{self.output_id}_{preproc_config.output_var}_{preproc_config.store_id}_{model_abbrev}/".replace(' ','_')
        Path(self.master_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(self.master_dir)


class Output():

    def __init__(self, pred_obj, model_data, preproc_config, output_config_obj,
                 out_samp_metric_df=None):
        self.master_dir = output_config_obj.master_dir
        if len(model_data.y_test.shape) > 1:
            model_data.y_train = model_data.y_train.flatten()
            model_data.y_test = model_data.y_test.flatten()

        self.preproc_config = preproc_config
        self.pred_obj = pred_obj
        self.model_data = model_data

        self.out_samp_metric_df = out_samp_metric_df

    def all_stores_output_results(self):

        # make the in-sample output
        if self.preproc_config.run_type == 'full_pipeline':
            in_samp_csv = pd.DataFrame(
                columns=['current_date','prediction_date', 'store_id'])

            in_samp_csv['current_date'] = [datetime.now().strftime('%Y-%m-%d')] * len(self.model_data.all_train_dates)
            in_samp_csv['prediction_date'] = self.model_data.all_train_dates
            in_samp_csv['store_id'] = np.array(self.model_data.train_store_index)[self.model_data.train_mask_indexes]

            if self.preproc_config.run_type=='full_pipeline':
                in_samp_csv['in_samp_data'] = self.model_data.y_train[self.model_data.train_mask_indexes]
            in_samp_csv[f'predictions'] = list(self.pred_obj.in_samp_preds[self.model_data.train_mask_indexes])

            if self.preproc_config.is_output_uq:
                in_samp_csv['lower_95_ci'] = list(self.pred_obj.in_sample_lw_95[self.model_data.train_mask_indexes])
                in_samp_csv['upper_95_ci'] = list(self.pred_obj.in_sample_up_95[self.model_data.train_mask_indexes])
            in_samp_csv['output_type'] = [self.model_data.preproc_config.output_var]*len(self.model_data.all_train_dates)
            in_samp_csv.to_csv(f'in_samp_preds.csv', index=False)


        # make the out-sample output
        out_samp_csv = pd.DataFrame(
            columns=['current_date','prediction_date', 'store_id'])

        out_samp_csv['current_date'] = [datetime.now().strftime('%Y-%m-%d')]*len(self.model_data.all_test_dates)
        out_samp_csv['prediction_date'] = self.model_data.all_test_dates
        out_samp_csv['store_id'] = np.array(self.model_data.test_store_index)[self.model_data.test_mask_indexes]

        if self.preproc_config.run_type == 'full_pipeline':
            out_samp_csv['out_samp_data'] = self.model_data.y_test[self.model_data.test_mask_indexes]
        out_samp_csv[f'predictions'] = list(self.pred_obj.out_samp_preds[self.model_data.test_mask_indexes])

        # thrshold out the negative values
        out_samp_csv['predictions'][out_samp_csv['predictions'] < 0] = 0

        if self.preproc_config.is_output_uq:
            out_samp_csv['lower_95_ci'] = list(self.pred_obj.out_sample_lw_95[self.model_data.test_mask_indexes])
            out_samp_csv['upper_95_ci'] = list(self.pred_obj.out_sample_up_95[self.model_data.test_mask_indexes])
        out_samp_csv['output_type'] = [self.model_data.preproc_config.output_var]*len(self.model_data.all_test_dates)
        out_samp_csv.to_csv(f'out_samp_preds.csv', index=False)

        # can only output the metrics if we are running the full pipeline
        if self.preproc_config.run_type=='full_pipeline':
            if self.model_data.all_test_dates[-1].strftime('%Y-%m-%d') not in self.preproc_config.metric_end_dates:
                self.preproc_config.metric_end_dates.append(self.model_data.all_test_dates[-1].strftime('%Y-%m-%d'))

                # remove dates if they ae not the test set of dates
                string_test_dates = [str(index_f)[:10] for index_f in self.model_data.all_test_dates]
                dates_test_data = []
                for cur_end_date in self.preproc_config.metric_end_dates:
                    if cur_end_date in string_test_dates:
                        dates_test_data.append(cur_end_date)

            self.preproc_config.metric_end_dates = dates_test_data

            print(f'self.preproc_config.metric_end_dates {self.preproc_config.metric_end_dates}')
            self.out_samp_metrics = Metrics(out_samp_csv,self.preproc_config)

            self.output_metrics()

    def make_variable_importance(self):
        features = self.model_data.feat_list
        importances = load(self.pred_obj.main_model_fn).feature_importances_

        indices = np.argsort(importances)
        plt.figure(figsize=(22, 12))
        plt.title(
            f'Feature Importances; Store ID {self.preproc_config.store_id} for {self.preproc_config.output_plot_var}')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(f'variable_importance_{self.preproc_config.store_id}.png')
        plt.close()

        output_csv = pd.DataFrame(columns=['feature_name', 'feature_importance'])
        output_csv['feature_name'] = np.array(features)[np.flip(np.argsort(importances))]
        output_csv['feature_importance'] = np.array(importances)[np.flip(np.argsort(importances))]

        output_csv.to_csv(f'variable_importance_{self.preproc_config.store_id}.csv', index=False)

    def output_all_models_metrics(self):
        # self.out_samp_metric_df.index = self.out_samp_metrics.metric_names
        self.out_samp_metric_df.to_csv(f'all_model_metrics.csv')

    def output_feat_list_csv(self):
        feat_name_df = pd.DataFrame(columns=['Feature_Name'])
        feat_name_df['Feature_Name'] = self.model_data.feat_list
        feat_name_df.to_csv('feat_list.csv', index=False)

    def output_metrics(self):
        self.out_samp_metrics.final_metric_df.to_csv(f'out_samp_metrics.csv', index=False)

    def make_forecast_plots(self, is_add_ci=False):
        # make all data plot
        plt.figure(figsize=(14, 12))
        self.make_in_samp_plots()
        self.make_out_samp_plots()
        self.make_forecast_title()
        plt.xlabel('Date (in days)', fontsize=18)
        plt.ylabel(f'Number of {self.preproc_config.output_plot_var}', fontsize=18)
        plt.legend(fontsize=16)
        plt.savefig(f'all_preds.png')
        plt.close()

        # make out of sample plot
        plt.figure(figsize=(14, 12))
        if is_add_ci:
            self.make_ci_plot()
        self.make_out_samp_plots()
        self.make_forecast_title()
        plt.xlabel('Date (day)', fontsize=18)
        plt.ylabel(f'Number of {self.preproc_config.output_plot_var}', fontsize=18)

        plt.legend(fontsize=16)
        plt.savefig(f'out_samp_preds.png')
        plt.close()

        if self.pred_obj.model_name == 'Random Forest' or self.pred_obj.model_name == 'XG-Boost':
            self.make_variable_importance()

    def make_forecast_scatter_plot(self):
        plt.figure(figsize=(14, 12))
        plt.scatter()

    def make_out_samp_plots(self):
        plt.plot(self.model_data.test_dates, self.model_data.y_test[self.model_data.test_mask_indexes], color='lightblue', linewidth=3,
                 label='Out-sample Truth')
        plt.plot(self.model_data.test_dates, self.pred_obj.out_samp_preds[self.model_data.test_mask_indexes], color='lightcoral', linewidth=2,
                 label='Out-sample Predictions', linestyle='--')

    def make_in_samp_plots(self):
        plt.plot(self.model_data.train_dates, self.model_data.y_train[self.model_data.train_mask_indexes], color='blue', linewidth=3,
                 label='In-sample Truth')
        plt.plot(self.model_data.train_dates, self.pred_obj.in_samp_preds[self.model_data.train_mask_indexes], color='red', linewidth=2,
                 label='In-sample Predictions', linestyle='--')

    def make_forecast_title(self):
        plt.title(
            f'{self.preproc_config.lag + 1}-Day Ahead Forecast with {self.pred_obj.model_name} for Store: {self.preproc_config.store_id} \n Out-of-sample MSE: {self.out_samp_metrics.mse} Corr: {self.out_samp_metrics.corr}',
            fontsize=20)

    def make_ci_plot(self):
        plt.fill_between(self.model_data.test_dates, self.pred_obj.out_sample_lw_95[self.model_data.test_mask_indexes], self.pred_obj.out_sample_up_95,
                         color='saddlebrown', alpha=.30, label='Uncertainty Envelope')



