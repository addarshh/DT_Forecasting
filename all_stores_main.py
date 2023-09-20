import sys
import argparse

sys.path.append('/DTFORECASTING/')
from preprocessing.preproc_read_data import RawData
from preprocessing.preproc_config import PreProcConfig
from preprocessing.preproc_features import AllStoreFeatures
from models.run_model_pipeline import RunEnsem, run_single_model


def run_main(args):
    # set config
    model_abbrev = 'gb'
    preproc_config = PreProcConfig(main_dir=args.main_dir, output_var=args.output_var, run_type=args.run_type,
                                   model_fn_dict=args.model_fn_dict, is_create_master_df=False,
                                   master_df_fn=args.master_df_fn,data_folder=args.data_folder)
    # get the raw data
    if args.master_df_fn == None:
        print(f'Getting the raw data......')
        raw_data_obj = RawData(preproc_config,is_make_derived_feats=False)
    else:
        print(f'Using the master df and skipping data retrieval......')
        raw_data_obj = None

    # get all features
    model_data = AllStoreFeatures(raw_data_obj=raw_data_obj, preproc_config=preproc_config)
    preproc_config.store_id = 'all_stores'
    run_single_model(model_data, preproc_config, model_abbrev)


if __name__ == "__main__":

    # set the args
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", help="Main directory", type=str)
    parser.add_argument("--output_var", help="invoices or phone_calls", type=str)
    parser.add_argument("--run_type", help="How to run the model (e.g., train, score, full_pipeline)", type=str)
    parser.add_argument("--main_model_fn", help="Location of the main model file", type=str)
    parser.add_argument("--lower_ci_model_fn", help="Location of the lower CI model file", type=str)
    parser.add_argument("--upper_ci_model_fn", help="Location of the upper CI model file", type=str)
    parser.add_argument("--master_df_fn", help="Location of the master_df", type=str)
    parser.add_argument("--data_folder", help="Location to read and write data files", type=str)

    parser.set_defaults(main_dir='/scratch/')
    parser.set_defaults(output_var='invoices')
    parser.set_defaults(run_type='full_pipeline')
    parser.set_defaults(main_model_fn=None)
    parser.set_defaults(lower_ci_model_fn=None)
    parser.set_defaults(upper_ci_model_fn=None)
    parser.set_defaults(master_df_fn=None)
    parser.set_defaults(data_folder='data')
    args = parser.parse_known_args()[0]

    if args.main_model_fn == None or args.main_model_fn == 'None':

        if args.run_type == 'score':
            print(f'ERROR! Can not score model without a main file')
            sys.exit()

        args.model_fn_dict = {}

    else:

        args.model_fn_dict = {'main_model_fn': args.main_model_fn, 'lower_ci_model_fn': args.lower_ci_model_fn,
                              'upper_ci_model_fn': args.upper_ci_model_fn}
    # run the main command
    run_main(args)
