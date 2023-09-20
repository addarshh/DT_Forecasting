import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM
# from xgboost.sklearn import XGBRegressor

from joblib import dump, load


class NonlinearRegression:
    def __init__(self, model_data):
        self.model_data = model_data
        self.model_name = None

        if isinstance(model_data.y_train, pd.Series):
            self.model_data.y_train = copy.deepcopy(model_data.y_train.values)
            self.model_data.y_test = copy.deepcopy(model_data.y_test.values)

        if isinstance(model_data.x_train, pd.DataFrame):
            self.model_data.x_train = copy.deepcopy(model_data.x_train.values)
            self.model_data.x_test = copy.deepcopy(model_data.x_test.values)

    def fit_model(self, model_abbrev):

        self.model_abbrev = model_abbrev

        print(f'Fitting {self.model_abbrev} model......')

        if self.model_abbrev == 'rf':
            self.fit_rf()
        elif self.model_abbrev == 'gb':
            self.fit_gb()
        elif self.model_abbrev == 'nn':
            self.fit_nn()
        elif self.model_abbrev == 'lr':
            self.fit_lr()
        elif self.model_abbrev == 'ert':
            self.fit_ert()
        elif self.model_abbrev == 'lasso':
            self.fit_lasso()
        elif self.model_abbrev == 'gam':
            self.fit_gam()

    def make_esemble(self, main_methods_df_in, main_methods_df_out):
        self.model_name = 'Ensemble Forecast'

        in_samp_ensem = np.zeros((len(self.model_data.y_train)))
        out_samp_ensem = np.zeros((len(self.model_data.y_test)))

        ensem_count = 0
        for cur_method in main_methods_df_out.columns:
            in_samp_ensem += main_methods_df_in[cur_method]
            out_samp_ensem += main_methods_df_out[cur_method]
            ensem_count += 1

        self.in_samp_preds = in_samp_ensem / ensem_count
        self.out_samp_preds = out_samp_ensem / ensem_count

        self.out_sample_lw_95 = np.quantile(main_methods_df_out.values, .025, axis=1)
        self.out_sample_up_95 = np.quantile(main_methods_df_out.values, .975, axis=1)

    def fit_gb(self):
        self.model_name = 'XG-Boost'

        # fit all xg-boost models
        model_type_list = ['main', 'lower_ci', 'upper_ci']

        # run the different models
        model_obj_list = [self.fit_gb_by_type(cur_model_type) for cur_model_type in model_type_list]

        # put into objects
        self.main_model_fn = model_obj_list[0]
        self.lower_ci_model_fn = model_obj_list[1]
        self.upper_ci_model_fn = model_obj_list[2]

    def fit_gb_by_type(self, model_type):

        if model_type == 'main':
            print(f'Fitting the main XG-Boost model........')
            # fit the main model
            return self.fit_main_xg_boost()

        elif model_type == 'lower_ci':
            # get the confidence intervals
            print(f'Fitting the lower {(1 - self.model_data.preproc_config.ci_perc) / 2} CI XG-Boost model.......')
            return self.fit_quantile_xg_boost(alpha=(1 - self.model_data.preproc_config.ci_perc) / 2,
                                              prefix='lower_')
        elif model_type == 'upper_ci':
            print(f'Fitting the upper {1 - (1 - self.model_data.preproc_config.ci_perc) / 2} CI XG-Boost model.......')
            return self.fit_quantile_xg_boost(
                alpha=1 - (1 - self.model_data.preproc_config.ci_perc) / 2, prefix='upper_')

    def fit_main_xg_boost(self):

        self.gb_model = GradientBoostingRegressor(random_state=0, n_estimators=250, max_depth=5, verbose=True).fit(
            self.model_data.x_train, self.model_data.y_train)

        return self.save_model(self.gb_model)

    def fit_quantile_xg_boost(self, alpha, prefix):

        self.ci_gb_model = GradientBoostingRegressor(random_state=0, loss="quantile",
                                                     alpha=alpha, verbose=True, n_estimators=100).fit(
            self.model_data.x_train, self.model_data.y_train)

        return self.save_model(self.ci_gb_model, prefix=prefix)

    def fit_rf(self, ):
        self.model_name = 'Random Forest'

        self.rf_model = RandomForestRegressor(random_state=0).fit(
            self.model_data.x_train, self.model_data.y_train)

        self.main_model_fn = self.save_model(self.rf_model)

    def fit_nn(self, max_iter=100):
        self.model_name = 'Neural Network'

        if len(self.model_data.y_train) > 200:
            batch_size = 200
        else:
            batch_size = 1000

        self.nn_model = MLPRegressor(random_state=0, batch_size=batch_size, max_iter=max_iter, verbose=True).fit(
            self.model_data.x_train, self.model_data.y_train)

        self.main_model_fn = self.save_model(self.nn_model)

    def fit_ert(self):
        self.model_name = 'Extremely Random Trees'

        self.ert_model = ExtraTreesRegressor(random_state=0,
                                             verbose=True).fit(
            self.model_data.x_train, self.model_data.y_train)

        self.main_model_fn = self.save_model(self.ert_model)

    def fit_lasso(self):
        self.model_name = "Lasso"
        self.lasso_model = linear_model.Lasso(alpha=0.1).fit(
            self.model_data.x_train, self.model_data.y_train)

        self.main_model_fn = self.save_model(self.nn_model)

    def fit_gam(self):
        self.model_name = "GAM"
        self.gam_model = LinearGAM().fit(
            self.model_data.x_train, self.model_data.y_train)

        self.main_model_fn = self.save_model(self.nn_model)

    def fit_lr(self):
        self.model_name = 'Linear Regression'
        self.lr_model = LinearRegression().fit(
            self.model_data.x_train, self.model_data.y_train)

        self.main_model_fn = self.save_model(self.lr_model)

    def save_model(self, model, prefix=''):
        model_fn = f'{prefix}{self.model_abbrev}.p'
        dump(model, model_fn)
        return model_fn

    def score_model(self, model_loc, score_data):
        model_obj = load(model_loc)
        return model_obj.predict(score_data)
