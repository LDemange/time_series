import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from tensorflow.keras import Sequential, layers
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", False)
from sklearn.metrics import mean_absolute_percentage_error
from datetime import date, timedelta, datetime
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam
######################################################
from gluonts.dataset.util import to_pandas
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
import matplotlib.pyplot as plt

def replace_val(df,date1,date2): #on remplace les NaN dans l intervalle de date
    for col in df.columns:
        imputer = KNNImputer(n_neighbors=7) # Instantiate a SimpleImputer object with your strategy of choice
        imputer.fit(df[[col]]) # Call the "fit" method on the object
        df[col] = imputer.transform(df[[col]]) # Call the "transform" method on the object

df = pd.read_csv('../data/data_preparation.csv', index_col=[0])

df['Date']=pd.to_datetime(df['Date'], errors='coerce')
df = df.set_index('Date').asfreq('D')
replace_val(df,'2022-04-30','2022-05-06')

df.drop(columns=['t - 1', 't - 2', 't - 3', 't - 4', 't - 5',
       't - 6', 't - 7', 't - 8', 't - 9', 't - 10', 't - 11', 't - 12',
       't - 13', 't - 14', 't - 15', 't - 16', 't - 17', 't - 18', 't - 19',
       't - 20', 't - 21', 't - 22', 't - 23', 't - 24', 't - 25', 't - 26',
       't - 27', 't - 28', 't - 29', 't - 30','sin365_1', 'cos365_1', 'sin365_2', 'cos365_2',
       'sin365_3', 'cos365_3','bank_holiday', 'school_holidays',
       'season','Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)',
       'Hydraulique (MW)', 'Pompage (MW)', 'Bioénergies (MW)',
       'Ech. physiques (MW)', 'Stockage batterie', 'Déstockage batterie',
       'Eolien terrestre', 'Eolien offshore', 'TCO Thermique (%)',
       'TCH Thermique (%)', 'TCO Nucléaire (%)', 'TCH Nucléaire (%)',
       'TCO Eolien (%)', 'TCH Eolien (%)', 'TCO Solaire (%)',
       'TCH Solaire (%)', 'Column 30','Code INSEE région','sin_month', 'cos_month',
        'num_day',
        'YEAR', 'MONTH', 'DAY',
        'T2MDEW', 'T2MWET', 'TS', 'T2M_RANGE',
       'T2M_MAX', 'T2M_MIN','PS', 'QV2M','WS10M','WS50M','RH2M','week_day' #temoin
        #'PRECTOTCORR' # PRECTOTCORR is bias-corrected total precipitation
        #'sin_day','cos_day'
                ],inplace=True)

df['id']='A'

N_test = 100

df_train=df.iloc[:-N_test]

df_test=df.iloc[-N_test:]

ds = PandasDataset(df_train, target="Consommation (MW)",feat_dynamic_real=["T2M"])

estimator = DeepAREstimator(
    freq=ds.freq, prediction_length=N_test, trainer=Trainer(epochs=3),use_feat_dynamic_real=True)

predictor = estimator.train(ds)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=ds,#dataset.test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation, is useful for confidence interval
)

forecasts = list(forecast_it)
tss = list(ts_it)

# first entry of the time series list
ts_entry = tss[0]

# first 5 values of the time series (convert from pandas to numpy)
np.array(ts_entry[:5]).reshape(
    -1,
)

# first entry of dataset.test
dataset_test_entry = next(iter(ds))

# first 5 values
dataset_test_entry["target"][:5]

forecast_entry = forecasts[0]

prediction_intervals = (50.0, 90.0)
forecast_entry.plot(prediction_intervals=prediction_intervals, color="g")

