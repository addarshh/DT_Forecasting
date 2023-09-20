from netCDF4 import Dataset, num2date
import pandas as pd
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cloud_dest', '-cd', type=str, help='Cloud storage destination for CSV files')
parser.add_argument('--data_dir', '-d', type=str, default='./', help='Local data directory for temporary CSV files')
parser.add_argument('--names', '-n', nargs='+', type=str, help='NARR variables to process')
parser.add_argument('--start_year', '-s', type=int, help='First year in year range')
parser.add_argument('--end_year', '-e', type=int, help='Last year in year range')
parser.add_argument('--download', '-dl', type=bool, default=False, help='download .nc files True/False')
parser.add_argument('--store_cloud', '-sc', type=bool, default=False, help='copy files to cloud True/False')
args = parser.parse_args()

CLOUD_DEST = args.cloud_dest
DATA_DIR = args.data_dir
NAMES = args.names
START_YEAR = args.start_year
END_YEAR = args.end_year
DOWNLOAD = args.download
TO_CLOUD = args.store_cloud


if None in list(vars(args).values()):
    print('Specify required arguments. See -h for help.')
    SystemExit

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def downloadVarNCs(varnames, year):
    """
    Retrieve NetCDF data files from NOAA for specified years and weather variables

    """
    for v in varnames:
        cmd = f'curl -L ftp://ftp.cdc.noaa.gov/Datasets/NARR/Dailies/monolevel/{v}.{year}.nc -o {DATA_DIR}/{v}.{year}.nc'
        os.system(cmd)


def writeCSVLocal(varnames, year):
    """
    Convert NetCDF data into CSV format
    Store temporarily in local directory

    """
    var_data = {v: [] for v in varnames}

    for i, v in enumerate(varnames):

        filename = os.path.join(DATA_DIR, '{}.{}.nc'.format(v, year))
        dat = Dataset(filename)

        if i == 0:
            lat = dat.variables['lat'][:].flatten()
            lon = dat.variables['lon'][:].flatten()
            grid_keys = [k for k in range(len(lat))]
            time_var = dat.variables['time']
            dtime = num2date(time_var[:], time_var.units)

        var_keys = list(dat.variables.keys())
        dt_index = len(dtime)
        var_data[v] = {dt: dat.variables[var_keys[6]][dt].flatten() for dt in range(dt_index)}

    out_file = os.path.join(DATA_DIR, 'NARR_{}.csv'.format(year))

    with open(out_file, 'a') as f:

        f.write('grid_key,latitude,longitude,time,{}\n'.format(','.join([v for v in varnames])))

        for i in tqdm(range(dt_index)):
            # print('Processing chunk %s of %s for %s' % (i, len(dtime), year))
            df = pd.DataFrame({'grid_key': grid_keys,
                               'latitude': lat,
                               'longitude': lon,
                               'time': dtime[i],
                               **{v: var_data[v][i] for v in varnames}})
            df.to_csv(f, index=False, header=False, line_terminator='\n',
                      encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')

    f.close()


def toStorage(year):
    """
    Copy local CSV to cloud storage
    Delete local file

    """
    csv_file = os.path.join(DATA_DIR, 'NARR_{}.csv'.format(year))
    copy = 'gsutil cp {} {}'.format(csv_file, CLOUD_DEST)
    os.system(copy)

    remove_csv = 'rm {}'.format(csv_file)
    os.system(remove_csv)

    nc_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.nc')]
    remove_ncs = 'rm {}'.format(' '.join([os.path.join(DATA_DIR, ncf) for ncf in nc_files]))
    os.system(remove_ncs)


def main(start, end, varnames):
    """
    Download and process yearly NetCDF data files

    :start      -   first year of analysis time frame
    :end        -   last year of analysis time frame
    :varnames   -   names of NARR variables to be included

    """
    yrs = [y for y in range(start, end + 1)]

    for y in yrs:

        # Download data files
        if DOWNLOAD:
            print('Downloading {} NARR files'.format(y))
            downloadVarNCs(varnames, y)

        else:
            print('Copying {} NARR files to local directory'.format(y))
            cmd = 'gsutil cp {} {}'.format(os.path.join(CLOUD_DEST, '*%s.nc' % y), DATA_DIR)
            os.system(cmd)

        # Create local CSV
        print('Writing {} data to CSV'.format(y))
        writeCSVLocal(varnames, y)

        # Push to cloud storage
        if TO_CLOUD:
            print('Copying {} data to cloud storage. Removing temporary files on local'.format(y))
            toStorage(y)


if __name__ == "__main__":

    main(START_YEAR, END_YEAR, NAMES)

    if TO_CLOUD:
        os.system('gcloud compute instances stop 2779320979208069934')


