import numpy as np


class ModelData:
    def __init__(self, x_data, y_data, preproc_config,train_perc):
        # set the initial variables
        self.x_data = x_data
        self.y_data = y_data
        self.preproc_config = preproc_config
        self.train_perc = train_perc
        self.feat_list = list(self.x_data.columns)

        if self.preproc_config.run_type == 'full_pipeline':
            self.make_train_test_data()

        elif self.preproc_config.run_type == 'train':
            self.make_data_for_training()

        elif self.preproc_config.run_type == 'score':
            self.make_data_for_scoring()

    def make_data_for_scoring(self):

        self.x_train, self.y_train, self.raw_train_dates, self.train_dates = \
            np.array([None]), np.array([None]), np.array([None]), np.array([None])
        self.y_test = self.y_data[-6:]
        if self.y_test.index[-1].dayofweek != self.preproc_config.predict_day:
            raise ValueError('Forecast days not aligned with specified forecast window')
        self.x_test = self.x_data[-6:]
        if self.x_test.isna().sum().sum() > 0:
            raise ValueError('Processed data contains null values')
        self.raw_test_dates, self.test_dates = self.get_dates(self.y_test)


    def make_data_for_training(self):
        self.x_test, self.y_test, self.raw_test_dates, self.test_dates = \
            np.array([None]), np.array([None]), np.array([None]), np.array([None])
        self.x_train_raw = self.x_data
        self.y_train_raw = self.y_data
        self.x_train, self.y_train = self.remove_missing_values(self.x_train_raw, self.y_train_raw)
        self.raw_train_dates, self.train_dates = self.get_dates(self.y_train)

    def make_train_test_data(self):
        self.total_num_periods = len(self.x_data)

        self.in_sample_indexes = np.arange(0, np.round(self.total_num_periods * self.train_perc), 1)
        self.out_sample_indexes = np.arange(
            np.round(self.total_num_periods * self.train_perc) + self.preproc_config.max_lag, self.total_num_periods, 1)

        # split the x data
        self.x_train_raw = self.x_data.iloc[self.in_sample_indexes]
        self.x_test_raw = self.x_data.iloc[self.out_sample_indexes]

        # split the y data
        self.y_train_raw = self.y_data.iloc[self.in_sample_indexes]
        self.y_test_raw = self.y_data.iloc[self.out_sample_indexes]

        # remove the nan data
        self.x_train, self.y_train = self.remove_missing_values(self.x_train_raw, self.y_train_raw)
        self.x_test, self.y_test = self.remove_missing_values(self.x_test_raw, self.y_test_raw)

        # get all the dates
        self.raw_train_dates, self.train_dates = self.get_dates(self.y_train)
        self.raw_test_dates, self.test_dates = self.get_dates(self.y_test)

    def remove_missing_values(self, x_data, y_data):
        nan_indices = np.unique(
            np.concatenate((np.where(np.asanyarray(np.isnan(x_data)))[0].flatten(),
                            np.where(np.asanyarray(np.isnan(y_data)))[0].flatten())))
        if len(nan_indices) > 0:
            x_cleaned = x_data.drop(x_data.index[nan_indices])
            y_cleaned = y_data.drop(y_data.index[nan_indices])
        else:
            x_cleaned = x_data
            y_cleaned = y_data
        return x_cleaned, y_cleaned

    def get_dates(self, y_data):
        raw_dates = y_data.index
        mask_indices = self.filter_out_sundays(raw_dates)
        filtered_dates = raw_dates[mask_indices]
        return raw_dates, filtered_dates

    def filter_out_sundays(self, date_list):
        return np.where(np.array([index_f.dayofweek for index_f in date_list]) != 6)[0]
