import sys
import numpy as np
sys.path.append('/DTFORECASTING/')
from preprocessing.preproc_split_data import ModelData
from models.run_model import NonlinearRegression
from evaluation.eval_metrics import Metrics
from preprocessing.preproc_features import SingleStoreFeatures
from models.run_model_pipeline import RunEnsem
from evaluation.eval_output import Output, OutputConfig


class RunStorePipeline:

    def __init__(self, raw_data_obj, preproc_config, is_fit_single_method, is_output_plots=False, model_abbrev=None):
        self.raw_data_obj = raw_data_obj
        self.preproc_config = preproc_config
        self.is_fit_single_method = is_fit_single_method
        self.is_output_plots = is_output_plots
        self.model_abbrev = model_abbrev

        self.run_store_pipeline()

    def run_store_pipeline(self):
        # make features
        feat_obj = SingleStoreFeatures(self.preproc_config, raw_data_obj=self.raw_data_obj)

        # get test and training set
        self.model_data = ModelData(x_data=feat_obj.feature_df, y_data=feat_obj.y_data,
                                    preproc_config=self.preproc_config,
                                    train_perc=0.8)

        if self.is_fit_single_method:

            print(f'self.model_abbrev {self.model_abbrev}')
            # fit the model
            self.nonlinear_reg_obj = NonlinearRegression(self.model_data)
            self.nonlinear_reg_obj.fit_model(self.model_abbrev)

            # calc metrics
            self.in_samp_metrics = Metrics(self.model_data.y_train, self.nonlinear_reg_obj.in_samp_preds,self.model_data.train_mask_indexes)
            self.out_samp_metrics = Metrics(self.model_data.y_test, self.nonlinear_reg_obj.out_samp_preds,self.model_data.test_mask_indexes)

            return

        else:

            self.ensem_obj = RunEnsem(self.model_data, self.preproc_config, is_single_model=False)




def run_individual_models_all_stores(raw_data_obj, preproc_config, is_fit_single_method, model_abbrev,total_num_stores,
                                     is_output_plots=False):
    # store the data
    all_y_train = []
    all_y_test = []

    # store the preds
    all_in_samp_preds = []
    all_out_samp_preds = []

    train_store_index = []
    test_store_index = []

    all_train_dates = []
    all_test_dates = []
    raw_all_train_dates = []
    raw_all_test_dates = []
    count = 0

    for cur_store in raw_data_obj.invoice_df_by_store.columns[1:total_num_stores]:

        print(f'Fitting model for store {cur_store}......')

        try:
            preproc_config.store_id = cur_store
            pipeline_obj = RunStorePipeline(raw_data_obj, preproc_config, is_fit_single_method,model_abbrev=model_abbrev)

            all_y_train.extend(pipeline_obj.model_data.y_train)
            all_y_test.extend(pipeline_obj.model_data.y_test)

            all_in_samp_preds.extend(pipeline_obj.nonlinear_reg_obj.in_samp_preds)
            all_out_samp_preds.extend(pipeline_obj.nonlinear_reg_obj.out_samp_preds)

            train_store_index.extend([cur_store] * len(pipeline_obj.model_data.y_train))
            test_store_index.extend([cur_store] * len(pipeline_obj.model_data.y_test))

            all_train_dates.extend(pipeline_obj.model_data.train_dates)
            all_test_dates.extend(pipeline_obj.model_data.test_dates)
            raw_all_train_dates.extend(pipeline_obj.model_data.raw_train_dates)
            raw_all_test_dates.extend(pipeline_obj.model_data.raw_test_dates)

        except:
            print(f'Cant run for store {cur_store}')

        print(count)
        count += 1


    pipeline_obj.preproc_config.store_id = 'All_Stores'

    # reset data and preds
    pipeline_obj.model_data.y_train = np.array(all_y_train)
    pipeline_obj.model_data.y_test = np.array(all_y_test)

    pipeline_obj.nonlinear_reg_obj.in_samp_preds = np.array(all_in_samp_preds)
    pipeline_obj.nonlinear_reg_obj.out_samp_preds = np.array(all_out_samp_preds)

    pipeline_obj.model_data.all_train_dates = all_train_dates
    pipeline_obj.model_data.all_test_dates = all_test_dates

    pipeline_obj.model_data.train_store_index = train_store_index
    pipeline_obj.model_data.test_store_index = test_store_index

    pipeline_obj.model_data.train_mask_indexes = np.where(np.array([index_f.dayofweek for index_f in raw_all_train_dates]) != 6)[0]
    pipeline_obj.model_data.test_mask_indexes = np.where(np.array([index_f.dayofweek for index_f in raw_all_test_dates]) != 6)[0]

    in_samp_metrics = Metrics(all_y_train, all_in_samp_preds,pipeline_obj.model_data.train_mask_indexes)
    out_samp_metrics = Metrics(all_y_test, all_out_samp_preds,pipeline_obj.model_data.test_mask_indexes)

    # make the plots
    plot_obj = Output(pipeline_obj.nonlinear_reg_obj, pipeline_obj.model_data, pipeline_obj.preproc_config,
                      in_samp_metrics=in_samp_metrics,
                      out_samp_metrics=out_samp_metrics)

    plot_obj.output_metrics()
    plot_obj.all_stores_output_results()


