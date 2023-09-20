import six
import sys
sys.modules['sklearn.externals.six'] = six
from skgarden.quantile import ExtraTreesQuantileRegressor
quantile_ert_model = ExtraTreesQuantileRegressor(random_state=0, n_estimators=10, verbose=True, n_jobs=-1).fit(
    self.model_data.x_train, self.model_data.y_train)

self.out_sample_lw_95 = quantile_ert_model.predict(self.model_data.x_test, quantile=2.5)
self.out_sample_up_95 = quantile_ert_model.predict(self.model_data.x_test, quantile=97.5)

# docker file
# scikit-garden needs these packages to be installed first
# RUN pip install setuptools numpy scipy scikit-learn cython