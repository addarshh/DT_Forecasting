import pandas as pd


def get_max_divergence_index(macd, macd_max_thresh):
    if macd.max() > macd_max_thresh:
        return macd.idxmax()
    else:
        return None


def get_threshold_cross_index(macd, max_div_idx, macd_break_thresh):
    rev_series = macd[:max_div_idx][::-1]
    break_idx = rev_series[rev_series.lt(macd_break_thresh)].index[0] + 1
    if break_idx:
        return break_idx
    else:
        return 0


def get_stable_index(ma_resid, max_div_idx):
    resid_from_max_div = ma_resid[max_div_idx:]
    try:
        return resid_from_max_div[resid_from_max_div.lt(0)].index[0]
    except:
        return ma_resid.index.max()


def make_flags(macd, ma_resid, macd_max_thresh, macd_break_thresh):
    trigger_flag = [0] * len(macd)
    recovery_flag = [0] * len(macd)
    max_div_idx = get_max_divergence_index(macd, macd_max_thresh)
    if max_div_idx:
        break_idx = get_threshold_cross_index(macd, max_div_idx, macd_break_thresh)
        stable_idx = get_stable_index(ma_resid, max_div_idx)
        trigger_flag = [1 if break_idx <= i <= max_div_idx else j for i, j in enumerate(trigger_flag)]
        recovery_flag = [1 if max_div_idx < i <= stable_idx else j for i, j in enumerate(recovery_flag)]
    return trigger_flag, recovery_flag


def flag_black_swans(source_df, macd_max_thresh, macd_break_thresh):
    sdf_list = []
    actuals = source_df.pivot(index='date', columns='store_id', values='y_data')
    predicted = source_df.pivot(index='date', columns='store_id', values='predicted')
    stores = pd.unique(source_df['store_id'])
    for store in stores:
        sdf = pd.concat([actuals[store], predicted[store]], axis=1).reset_index()
        sdf.columns = ['date', 'actual', 'predicted']
        sdf['resid_frac'] = (sdf['predicted'] - sdf['actual']) / sdf['actual']
        sdf['resid_ma_14d'] = sdf['resid_frac'].rolling(window=14, min_periods=1).mean()
        sdf['resid_ma_120d'] = sdf['resid_frac'].rolling(window=120, min_periods=1).mean()
        sdf['macd'] = sdf['resid_ma_14d'] - sdf['resid_ma_120d']
        sdf['depletion_flag'], sdf['depletion_recovery_flag'] = make_flags(sdf['macd'], sdf['resid_ma_14d'], 
                                                                           macd_max_thresh, macd_break_thresh)
        sdf['store_code'] = store
        sdf['effective_date'] = pd.to_datetime(sdf['date']).dt.strftime('%Y%m%d')
        sdf_list.append(sdf)
    out_df = pd.concat(sdf_list, axis=0)
    return out_df