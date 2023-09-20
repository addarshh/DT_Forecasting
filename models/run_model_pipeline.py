import sys
import pandas as pd
from multiprocessing import Pool
from functools import partial

sys.path.append('/DTFORECASTING/')
from models.run_model import NonlinearRegression
from evaluation.eval_output import Output, OutputConfig


class RunEnsem:

    def __init__(self, model_data, preproc_config, is_single_model=False):

        self.is_single_model = is_single_model
        self.run_all_models(model_data, preproc_config)

    def run_all_models(self, model_data, preproc_config):

        model_name_list = ['gam', 'rf', 'ert', 'lr', 'gb', 'nn']
        # model_name_list = [ 'lr', 'gb', 'nn', 'gam']
        self.main_methods_df_in = pd.DataFrame(columns=model_name_list)
        self.main_methods_df_out = pd.DataFrame(columns=model_name_list)

        # model_name_list.extend(['persistence', 'moving_average', 'ar_1', 'ensemble'])
        out_samp_metric_df = pd.DataFrame(columns=model_name_list)

        if preproc_config.is_run_parallel:

            pool = Pool(processes=preproc_config.num_cpus)
            func = partial(run_single_model, model_data, preproc_config)
            all_model_objs = pool.map(func, model_name_list)
            pool.close()
            pool.join()


        else:
            all_model_objs = [run_single_model(model_data, preproc_config, cur_model_name) for cur_model_name in
                              model_name_list]

        for count, cur_obj in enumerate(all_model_objs):
            self.metric_names = cur_obj[1].out_samp_metrics.metric_names

            out_samp_metric_df[model_name_list[count]] = cur_obj[1].out_samp_metrics.metric_list
            self.main_methods_df_in[model_name_list[count]] = cur_obj[0].in_samp_preds
            self.main_methods_df_out[model_name_list[count]] = cur_obj[0].out_samp_preds

        # ensemble
        nonlinear_reg_obj = NonlinearRegression(model_data)
        nonlinear_reg_obj.make_esemble(self.main_methods_df_in, self.main_methods_df_out)

        # make the plots
        output_config = OutputConfig(preproc_config, 'ensemble')
        plot_obj = Output(nonlinear_reg_obj, model_data, preproc_config, output_config)
        plot_obj.output_feat_list_csv()
        plot_obj.all_stores_output_results()
        # get the metrics names
        self.metric_names = plot_obj.out_samp_metrics.metric_names
        out_samp_metric_df['ensemble'] = plot_obj.out_samp_metrics.metric_list

        out_samp_metric_df.index = self.metric_names
        nonlinear_reg_obj.model_name = 'All_Methods'
        output_config = OutputConfig(preproc_config, 'All_Methods')
        plot_obj = Output(nonlinear_reg_obj, model_data, preproc_config, output_config,
                          out_samp_metric_df=out_samp_metric_df)

        plot_obj.output_all_models_metrics()


def run_single_model(model_data, preproc_config, model_abbrev):
    output_config = OutputConfig(preproc_config, model_abbrev)

    if preproc_config.run_type == 'full_pipeline':

        print(f'Running the full model pipeline.....')
        nonlinear_reg_obj = train_model(model_data, model_abbrev)
        nonlinear_reg_obj = score_model(nonlinear_reg_obj, model_data, preproc_config)
        _ = make_output(preproc_config, nonlinear_reg_obj, model_data, model_abbrev, output_config)

    elif preproc_config.run_type == 'train':

        print(f'Training the model.....')
        _ = train_model(model_data, model_abbrev)

    elif preproc_config.run_type == 'score':

        print(f'Scoring the model.....')
        nonlinear_reg_obj = NonlinearRegression(model_data)
        nonlinear_reg_obj = score_model(nonlinear_reg_obj, model_data, preproc_config)
        _ = make_output(preproc_config, nonlinear_reg_obj, model_data, model_abbrev, output_config)


def train_model(model_data, model_abbrev):
    # fit the model
    nonlinear_reg_obj = NonlinearRegression(model_data)
    nonlinear_reg_obj.fit_model(model_abbrev)
    return nonlinear_reg_obj


def score_model(nonlinear_reg_obj, model_data, preproc_config):
    # check if we need to get UQ
    if nonlinear_reg_obj.model_name in preproc_config.uq_models or preproc_config.is_output_uq:
        preproc_config.is_output_uq = True

    # get the model files if available
    if preproc_config.model_fn_dict:
        nonlinear_reg_obj.main_model_fn = preproc_config.model_fn_dict['main_model_fn']
        if preproc_config.is_output_uq:
            nonlinear_reg_obj.lower_ci_model_fn = preproc_config.model_fn_dict['lower_ci_model_fn']
            nonlinear_reg_obj.upper_ci_model_fn = preproc_config.model_fn_dict['upper_ci_model_fn']

    # in-sample scores
    if preproc_config.run_type == 'full_pipeline':
        nonlinear_reg_obj.in_samp_preds = nonlinear_reg_obj.score_model(nonlinear_reg_obj.main_model_fn,
                                                                        model_data.x_train)

        # score the UQ models if available
        if preproc_config.is_output_uq:
            nonlinear_reg_obj.in_sample_lw_95 = nonlinear_reg_obj.score_model(nonlinear_reg_obj.lower_ci_model_fn,
                                                                              model_data.x_train)
            nonlinear_reg_obj.in_sample_up_95 = nonlinear_reg_obj.score_model(nonlinear_reg_obj.upper_ci_model_fn,
                                                                              model_data.x_train)

    # out-of-sample scores
    nonlinear_reg_obj.out_samp_preds = nonlinear_reg_obj.score_model(nonlinear_reg_obj.main_model_fn, model_data.x_test)

    # score the UQ models if available
    if preproc_config.is_output_uq:
        nonlinear_reg_obj.out_sample_lw_95 = nonlinear_reg_obj.score_model(nonlinear_reg_obj.lower_ci_model_fn,
                                                                           model_data.x_test)
        nonlinear_reg_obj.out_sample_up_95 = nonlinear_reg_obj.score_model(nonlinear_reg_obj.upper_ci_model_fn,
                                                                           model_data.x_test)

    return nonlinear_reg_obj


def make_output(preproc_config, nonlinear_reg_obj, model_data, model_abbrev, output_config):
    # check if we need to get UQ
    if nonlinear_reg_obj.model_name in preproc_config.uq_models or preproc_config.is_output_uq:
        preproc_config.is_output_uq = True

    # all store plots
    preproc_config.store_id = 'All_Stores'
    output_obj = Output(nonlinear_reg_obj, model_data, preproc_config, output_config)
    output_obj.all_stores_output_results()
    output_obj.output_feat_list_csv()

    if model_abbrev == 'rf' or model_abbrev == 'gb':
        output_obj.make_variable_importance()

    return output_obj
