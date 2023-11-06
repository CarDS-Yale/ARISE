# %%
# This file is for data cleaning for CARDIA.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

path = "/Cardia/Y05/"

calip = pd.read_csv(path + "calip.csv")
caf = pd.read_csv(path + "caf08.csv")
caf11 = pd.read_csv(path + "caf11.csv")
calpa = pd.read_csv(path + "calpa.csv")
caref = pd.read_csv(path + "caref.csv")
caf20 = pd.read_csv(path + "caf20.csv")
cardia = calip.merge(caf, on='PID', how='outer').merge(calpa, on='PID', how='outer').merge(caref, on='PID', how='outer').merge(caf20, on='PID', how='outer').merge(caf11, on='PID', how='outer')
cardia.rename(columns={'PID': 'id', 'EX3_AGE': 'age', 'SEX': 'sex', 'RACE': 'ethnicity', 'C20BMI': 'bmi', 'C08HBP': 'htn', 'C08DIAB': 'dm', 'C08HBNOW': 'anti_htn', 'C08CHNOW': 'statin', 'CL6LPA': 'lp_a_value'}, inplace=True)

cardia['cholesterol'] = cardia['CL1CHOL']*0.02586
cardia['ldl'] = cardia['CL1LDL']*0.02586
cardia['hdl'] = cardia['CL1HDL']*0.02586
cardia['triglycerides'] = cardia['CL1NTRIG']*0.01129
cardia['lp_a_value'] = cardia['lp_a_value']*2.15
cardia['sex'].replace({2:0}, inplace=True)
cardia['ethnicity'].replace({4: 'African-American', 5: 'White'}, inplace=True)
cardia['htn'].replace({1:0, 2:1, 8: np.nan}, inplace=True)
cardia['anti_htn'].replace({1:0, 2:1, 8: np.nan}, inplace=True)
cardia['statin'].replace({1:0, 2:1, 8: np.nan}, inplace=True)
cardia['dm'].replace({1:0, 2:1, 8: np.nan}, inplace=True)
cardia['C08HEART'].replace({8: np.nan}, inplace=True)
cardia['ihd'] = np.where((cardia['C08HEART']==2)|(cardia['C08HRTAK']==2)|(cardia['C08ANGIN']==2), 1, np.where((cardia['C08HEART'].isna())&(cardia['C08HRTAK'].isna())&(cardia['C08ANGIN'].isna()), np.nan, 0))
cardia['ascvd'] = cardia['ihd'].copy()
cardia['hormone_est_prog'] = np.where((cardia['C08BCNOW']==2)|(cardia['C08HMNOW']==2), 1, np.where((cardia['C08BCNOW']==1)&(cardia['C08HMNOW']==1), 0, np.nan))
cardia['cvd_family_history'] = np.where((cardia['C11MSTRK']==2)|(cardia['C11MATCK']==2)|(cardia['C11FSTRK']==2)|(cardia['C11FATCK']==2), 1, np.where(
    (cardia['C11MSTRK']==1)&(cardia['C11MATCK']==1)&(cardia['C11FSTRK']==1)&(cardia['C11FATCK']==1), 0, np.nan))
cardia['premature_chd_family_history'] = np.where((cardia['C11MHAGE']<=65)|(cardia['C11FHAGE']<=55), 1, np.where((cardia['C11MATCK']==1)&(cardia['C11FATCK']==1), 0, np.nan))

cardia.dropna(subset=['lp_a_value'], inplace=True)
cardia = cardia[['id', 'age', 'sex', 'ethnicity', 'bmi', 'htn', 'dm', 'ihd', 'ascvd', 'cvd_family_history', 'premature_chd_family_history',
                'statin', 'anti_htn', 'hormone_est_prog', 'cholesterol', 'ldl', 'hdl', 'triglycerides', 'lp_a_value']]
cardia.to_csv("/Cardia/cardia.csv", index=False)
cardia.to_csv("/Lp(a)/External_Validation/cardia.csv", index=False)
cardia

# %%
import matplotlib.pyplot as plt
cardia.hist(bins=50, figsize=(20, 15))
plt.show()


