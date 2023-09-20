# warp a mexican hat wavelet around a particular day determined by the flags in calender
# the range of the warping is determined by pts (days), 30 is the default value
# the scale of the mexican hat wavelet is determined by a, 7 is the default value
import pandas as pd
import pywt
import numpy as np

# get the wavelet ave for promos (just based on if there is promo or not)
def mexh_wavelet(x, colname, scales=18, wavelet='mexh'):
    scales = np.arange(1, scales)
    coeffs, freqs = pywt.cwt(list(x), scales, wavelet=wavelet)

    # create scalogram
    cwt = pd.DataFrame(np.transpose(coeffs)).set_index(x.index)

    if colname == 'promotion':
        cwt['ave_cwt'] = (cwt[3] + cwt[7] + cwt[15]) / 3
    else:
        cwt['ave_cwt'] = -cwt[3]
    return cwt['ave_cwt']


def wavelet_for_df(raw_df, col):
    df = raw_df
    cnt = 0
    for store_code in df['store_code'].unique():
        x = df.loc[df['store_code'] == store_code]
        if cnt == 0:
            cwt = mexh_wavelet(x, colname=col)
        else:
            cwt = cwt.append(mexh_wavelet(x, colname=col))
        cnt += 1

    ave_cwt = cwt[['date', 'store_code', 'ave_cwt']]

    df = pd.merge(df, ave_cwt[['store_code', 'date', 'ave_cwt']], how='left',
                  left_on=['store_code', 'date'], right_on=['store_code', 'date'])
    return df
