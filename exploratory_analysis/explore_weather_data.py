import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
raw_data = pd.read_csv('/Users/mcdermop/Desktop/Projects/discount_tire/data/world_wide_technologies_2020-07-29t184021.csv')

comb_df = raw_data[raw_data['st_icao']=='KAPA'].groupby(['f_forecast_date']).mean()



plt.plot(comb_df.index,comb_df['sfi_snow_chance_id'])
plt.xticks(comb_df.index[np.arange(0,len(comb_df.index),50)],rotation=90)


for index,val in enumerate(comb_df['sft_snow_chance_id']):
    print(f'Index........ {index} Value {val}........date {comb_df.index[index]}')


# 2019-10-10
# 2018-10-08
# 2018-10-14
# 2017-10-09