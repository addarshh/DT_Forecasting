import pandas as pd


def get_max_divergence_index(ma_resid, resid_max_thresh):
    if ma_resid.min() < resid_max_thresh:
        return ma_resid.idxmin()
    else:
        return None

        
def get_threshold_cross_index(ma_resid, resid_max_thresh, resid_break_thresh):
    max_div_idx = get_max_divergence_index(ma_resid, resid_max_thresh)
    if max_div_idx:
        try:
            rev_ma_resid = ma_resid[:max_div_idx][::-1]
            break_idx = rev_ma_resid[rev_ma_resid.gt(resid_break_thresh)].index[0] + 1
            return break_idx
        except:
            return None
    else:
        return None

    
def get_reversal_index(ma_long, break_idx):
    rev_idx = ma_long[break_idx:].idxmax()
    return rev_idx


def get_stable_index(ma_short, ma_long, rev_idx):
    ma_diff = ma_short[rev_idx:] - ma_long[rev_idx:]
    ma_cross_flag = ma_diff.gt(0)
    try:
        if any(ma_cross_flag):
            # Use 30-day buffer after reversal to look for MA crossover
            stab_idx = ma_diff[30:][ma_cross_flag[30:]].index[0]
            return stab_idx
        else:
            return ma_diff.index.max()
    except:
        return ma_diff.index.max()

    
def make_flags(df, resid_max_thresh, resid_break_thresh):
    trigger_flag = [0] * len(df)
    reversal_flag = [0] * len(df)
    break_idx = get_threshold_cross_index(df['resid_ma_14d'], resid_max_thresh, resid_break_thresh)
    if break_idx:
        rev_idx = get_reversal_index(df['volume_ma_28d'], break_idx)
        trigger_flag = [1 if break_idx <= i <= rev_idx else j for i, j in enumerate(trigger_flag)]
        if rev_idx < df.index.max():
            stab_idx = get_stable_index(df['volume_ma_14d'], df['volume_ma_28d'], rev_idx)
            reversal_flag = [1 if rev_idx < i <= stab_idx else j for i, j in enumerate(reversal_flag)]
    return trigger_flag, reversal_flag


def flag_black_swans(source_df, resid_max_thresh, resid_break_thresh):
    sdf_list = []
    actuals = source_df.pivot(index='date', columns='store_id', values='y_data')
    predicted = source_df.pivot(index='date', columns='store_id', values='predicted')
    stores = pd.unique(source_df['store_id'])
    for store in stores:
        sdf = pd.concat([actuals[store], predicted[store]], axis=1).reset_index()
        sdf.columns = ['date', 'actual', 'predicted']
        sdf['resid_frac'] = (sdf['predicted'] - sdf['actual']) / sdf['actual']
        sdf['resid_ma_14d'] = sdf['resid_frac'].rolling(window=14).mean()
        sdf['volume_ma_28d'] = sdf['actual'].rolling(window=28).mean()
        sdf['volume_ma_14d'] = sdf['actual'].rolling(window=14).mean()
        sdf['surge_flag'], sdf['surge_recovery_flag'] = make_flags(sdf, resid_max_thresh, resid_break_thresh)
        sdf['store_code'] = store
        sdf['effective_date'] = pd.to_datetime(sdf['date']).dt.strftime('%Y%m%d')
        sdf_list.append(sdf)
    out_df = pd.concat(sdf_list, axis=0)
    return out_df