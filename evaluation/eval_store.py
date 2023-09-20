import pandas as pd
import numpy as np
import copy

from evaluation.eval_output import Output
from evaluation.eval_metrics import Metrics


class EvalStore:
    def __init__(self, pred_obj, model_data, preproc_config, store_id):
        # make the copy
        self.pred_obj = copy.deepcopy(pred_obj)
        self.model_data = copy.deepcopy(model_data)

        # get the indexes
        store_train_indexes = np.where(np.array(np.array(model_data.train_store_index)[model_data.train_mask_indexes] )== store_id)[0]
        store_test_indexes = np.where(np.array(np.array(model_data.test_store_index)[model_data.test_mask_indexes]) == store_id)[0]

        # reset data
        self.model_data.y_train = model_data.y_train[model_data.train_mask_indexes][store_train_indexes]
        self.model_data.y_test = model_data.y_test[model_data.test_mask_indexes][store_test_indexes]

        # rest dates
        self.model_data.train_dates = np.array(np.array(model_data.raw_all_train_dates)[model_data.train_mask_indexes])[store_train_indexes]
        self.model_data.test_dates = np.array(np.array(model_data.raw_all_test_dates)[model_data.test_mask_indexes])[store_test_indexes]


        # reset preds
        self.pred_obj.in_samp_preds = pred_obj.in_samp_preds[model_data.train_mask_indexes][store_train_indexes]
        self.pred_obj.out_samp_preds = pred_obj.out_samp_preds[model_data.test_mask_indexes][store_test_indexes]

        # a hack since everything is already filtered
        self.model_data.train_mask_indexes = np.arange(0,len(self.pred_obj.in_samp_preds),1)
        self.model_data.test_mask_indexes = np.arange(0, len(self.pred_obj.out_samp_preds), 1)

        # get the metrics
        in_samp_metrics = Metrics(self.model_data, self.pred_obj,is_out_samp=False)
        out_samp_metrics = Metrics(self.model_data, self.pred_obj)
        # make the plots
        plot_obj = Output(self.pred_obj, self.model_data, preproc_config, in_samp_metrics=in_samp_metrics,
                          out_samp_metrics=out_samp_metrics)

        plot_obj.make_forecast_plots()
        plot_obj.output_results()
        plot_obj.output_metrics()
