import sys
import argparse

sys.path.append('/DTFORECASTING/')
from preprocessing.preproc_read_data import RawData
from preprocessing.preproc_config import PreProcConfig
from preprocessing.preproc_features import AllStoreFeatures


def run_preprocessing(args):
    # set config
    preproc_config = PreProcConfig(main_dir=args.main_dir, output_var=args.output_var, run_type=args.run_type,
                                   is_create_master_df=True, data_folder=args.data_folder)

    # get the raw data
    raw_data_obj = RawData(preproc_config, start_date=args.start_date, end_date=args.end_date,
                           is_make_derived_feats=args.is_make_derived_feats)

    # get all features
    _ = AllStoreFeatures(raw_data_obj=raw_data_obj, preproc_config=preproc_config, is_make_master_df=True)


if __name__ == "__main__":
    # set the args
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", help="Main directory", type=str)
    parser.add_argument("--output_var", help="invoices or phone_calls", type=str)
    parser.add_argument("--run_type", help="How to run the model (e.g., train, score, full_pipeline)", type=str)
    parser.add_argument("--data_folder", help="Location to read and write data files", type=str)
    parser.add_argument("--start_date", help="Date to start data pre-processing", type=str)
    parser.add_argument("--end_date", help="Date to end data pre-processing", type=str)
    parser.add_argument("--is_make_derived_feats", help="Flag to make the derived features", action='store_true')

    parser.set_defaults(main_dir='/scratch/')
    parser.set_defaults(output_var='invoices')
    parser.set_defaults(run_type='full_pipeline')
    parser.set_defaults(data_folder='data')
    parser.set_defaults(start_date='2017-01-01')
    parser.set_defaults(end_date='2020-06-30')
    parser.set_defaults(is_make_derived_feats=False)
    args = parser.parse_known_args()[0]

    # run the preprocessing
    run_preprocessing(args)
