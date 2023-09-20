
def seasonally_detrend(raw_df, detrend_type='mean_diff'):
    seasonal_adjusted_df = raw_df

    if detrend_type == 'mean_diff':

        for cur_store in raw_df.columns:
            seasonal_adjusted_df[cur_store] = mean_diffs(raw_df[cur_store])


    elif detrend_type == 'weekly diff':
        print(f'Have not implimented this method yet.....')

    return seasonal_adjusted_df


def mean_diffs(cur_df):
    day_of_week_means = cur_df.groupby(cur_df.index.dayofweek).mean()

    for i in range(0, 6):
        cur_df[cur_df.index.dayofweek == i] = cur_df[cur_df.index.dayofweek == i] - \
                                              day_of_week_means[day_of_week_means.index == i].values[0]

    return cur_df


def filter_out_sundays(date_list):
    non_sunday_list = []
    for index_f in date_list:
        if index_f.dayofweek != 6:
            non_sunday_list.append(index_f)
    return non_sunday_list


def make_output_data(raw_data_obj, is_seasonally_adjust=False):
    if is_seasonally_adjust:
        raw_data_obj.seasonal_adjust_invoice = seasonally_detrend(raw_data_obj.invoice_df_by_store)
        raw_data_obj.seasonal_adjust_phone_calls = seasonally_detrend(raw_data_obj.phone_df_by_store)
    else:
        raw_data_obj.seasonal_adjust_invoice = raw_data_obj.invoice_df_by_store
        raw_data_obj.seasonal_adjust_phone_calls = raw_data_obj.phone_df_by_store

    raw_data_obj.seasonal_adjust_invoice = raw_data_obj.seasonal_adjust_invoice.loc[
        filter_out_sundays(raw_data_obj.seasonal_adjust_invoice.index)]
    raw_data_obj.seasonal_adjust_phone_calls = raw_data_obj.seasonal_adjust_phone_calls.loc[
        filter_out_sundays(raw_data_obj.seasonal_adjust_phone_calls.index)]
    return raw_data_obj
