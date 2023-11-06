# %%
# This file is for data cleaning for MESA.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

path = "/MESA/Exam1/"

dres = pd.read_csv(path + "mesae1dres06192012.csv")
lpa = pd.read_csv("/MESA/mesaas081_drepos_20151230.csv")
mesa = dres.merge(lpa, on='MESAID', how='outer')

mesa['ldl'] = mesa['ldl1']*0.02586
mesa['hdl'] = mesa['hdl1']*0.02586
mesa['triglycerides'] = mesa['trig1']*0.01129
mesa['cholesterol'] = mesa['chol1']*0.02586
mesa.rename(columns={'age1c': 'age', 'gender1': 'sex', 'race1c': 'ethnicity', 'bmi1c': 'bmi', 'htn1c': 'htn', 'dm031c': 'dm',
                     'htnmed1c': 'anti_htn', 'sttn1c': 'statin', 'asacat1c': 'aspirin', 'insln1c': 'insulin', 'lipid1c': 'llt',
                     'pnlpad1': 'lp_a_value', 'MESAID':'id'}, inplace=True)
mesa['ethnicity'].replace({1: 'Caucasian', 2: 'Chinese', 3: 'African-American', 4: 'Hispanic'}, inplace=True)
mesa['dm'].replace({1:0, 2:1, 3:1}, inplace=True)
mesa['t1dm'] = np.where(mesa['dmtype1c']==1, 1, np.where(mesa['dm'].isna(), np.nan, 0))
mesa['t2dm'] = np.where(mesa['dmtype1c']==2, 1, np.where(mesa['dm'].isna(), np.nan, 0))
mesa['cvd_family_history'] = np.where((mesa['pmi1']==1)|(mesa['pstk1']==1)|(mesa['shrtatt1']==1)|(mesa['sstk1']==1)|(mesa['chrtatt1']==1)|(mesa['cstk1']==1), 1,
                                      np.where((mesa['pmi1'].isin([0, 8]))&(mesa['pstk1'].isin([0, 8]))&(mesa['shrtatt1'].isin([0, 8]))&
                                      (mesa['sstk1'].isin([0, 8]))&(mesa['chrtatt1'].isin([0, 8]))&(mesa['cstk1'].isin([0, 8])), 0, np.nan))
mesa['hormone_est_prog'] = np.where((mesa['hrmrepc1']==1)|(mesa['estrgn1c']==1)|(mesa['prgstn1c']==1), 1,
                                    np.where((mesa['hrmrepc1']==0)&(mesa['estrgn1c']==0)&(mesa['prgstn1c']==0), 0, np.nan))
mesa['ascvd'] = float(0)
mesa = mesa[['id', 'age', 'sex', 'ethnicity', 'bmi', 'htn', 'dm', 't1dm', 't2dm', 'cvd_family_history', 'ascvd', 'statin', 'llt',
             'anti_htn', 'aspirin', 'insulin', 'hormone_est_prog', 'cholesterol', 'ldl', 'hdl', 'triglycerides', 'lp_a_value']]
mesa.dropna(subset='lp_a_value', inplace=True)
mesa.to_csv("/MESA/mesa.csv", index=False)
mesa.to_csv("/Lp(a)/External_Validation/mesa.csv", index=False)
mesa

# %%
import matplotlib.pyplot as plt
mesa.hist(bins=50, figsize=(20,15))
plt.show()


