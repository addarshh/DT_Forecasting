# dt-forecasting

<<<<<<< Updated upstream
=======
## Files that need to be created
These are a set of files that can be created using the code within this code repository. Further, these files need to be updated at varying different lengths (see below)


## CWT promos/holidays 

## CWT invoices/phones


## Files that need used to run locally



>>>>>>> Stashed changes
## Running locally

### Running preprocessing and main model separately

Run the preprocessing:
```bash
python  dt-forecasting/all_stores_preprocessing.py --main_dir /discount_tire/

```

Run the main model training:
```bash
python  dt-forecasting/all_stores_main.py --main_dir /discount_tire/ --master_df_fn /discount_tire/data/master_df.csv --is_run_parallel --run_type full_pipeline

```

### Running preprocessing and main Model together

Run the main model training:
```bash
python  dt-forecasting/all_stores_main.py --main_dir /discount_tire/  --is_run_parallel  --run_type full_pipeline

```
### Train the model
```bash
python  dt-forecasting/all_stores_main.py --main_dir /discount_tire/ --run_type train

```
### Score the model
```bash
python  dt-forecasting/all_stores_main.py --main_dir /discount_tire/ --run_type score --main_model_fn /discount_tire/results/08_18_2020/models_to_score/gb.p --lower_ci_model_fn /discount_tire/results/08_18_2020/models_to_score/lower_gb.p --upper_ci_model_fn /discount_tire/results/08_18_2020/models_to_score/upper_gb.p

```

## Run on the container

Make sure you have docker installed on your local machine or a remote instance. Docker can be installed from here: https://docs.docker.com/get-docker/

```bash
cd <location_of_code>
docker build  -t  dt_forecasting . && docker run dt_forecasting 
docker run -it -v <working_directory>:/scratch/  --rm dt_forecasting   all_stores_main.py  
```

Note, the container has a scratch directory, various files can be stored here. This is also helpful when writing files to the cloud, hey can first be written to the scratch directory before being upload to the cloud.