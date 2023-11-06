# %%
# This file is for data cleaning for ARIC.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

path_1 = "/ARIC/visit1/"
path_2 = "/ARIC/visit2/"

derive13 = pd.read_csv(path_1 + 'derive13.csv')
lipa = pd.read_csv(path_1 + 'lipa.csv', encoding= 'unicode_escape')
stroke01 = pd.read_csv(path_1 + 'stroke01.csv')
v2_labs = pd.read_csv(path_2 + 'v2_labs.csv')
phea = pd.read_csv(path_1 + 'phea.csv')

aric = derive13.merge(lipa, on=['ID_C', 'FORPROFIT'], how='outer').merge(stroke01, on=['ID_C', 'FORPROFIT'], how='outer').merge(
    v2_labs, on=['ID_C', 'FORPROFIT'], how='outer').merge(phea, on=['ID_C', 'FORPROFIT'], how='outer')

#Definition of V2ALTDETECT: Indicator for ALT values outside LOD, 1=detectable, 0= <4 U/L
#ALT was used from Visit 2 rather than Visit 1
#aric['V2ALT'] = np.where((aric['V2ALT'].isna())&(aric['V2ALTDETECT']==0), 4, aric['V2ALT'])

#Anti_HTN
aric['anti_htn'] = np.where((aric['HYPTMDCODE01']=='1')|(aric['HYPTMD01']=='1'), 1, np.where(
    (aric['HYPTMDCODE01']=='0')&(aric['HYPTMD01']=='0'), 0, np.nan))

#LLT
aric['llt'] = np.where(aric['CHOLMDCODE01']=='1', 1, np.where(aric['CHOLMDCODE01']=='0', 0, np.nan))

#Hormone
aric['hormone_est_prog'] = np.where(aric['HORMON02'].isin([1, 2]), 1, np.where(aric['HORMON02'].isin([3, 4]), 0, np.nan))

#Dictionary for PREVMI05, MDDXMI02: T Missing value 
for i in ['PREVMI05', 'MDDXMI02', 'HXOFMI02']:
    aric[i] = np.where(aric[i]=='T', np.nan, aric[i])
    aric[i] = aric[i].astype(str).astype(float)

#Dictionary for STIA01: Y, N, M
aric['STIA01'] = np.where(aric['STIA01']=='Y', 1, np.where(aric['STIA01']=='N', 0, np.nan))

#Defining ASCVD
aric['ascvd'] = np.where((aric['PREVMI05']==1)|
                         (aric['PRVCHD05']==1)|
                         (aric['MDDXMI02']==1)|
                         (aric['HXOFMI02']==1)|
                         (aric['SYMCHD03']==1)|
                         (aric['PAD02']==1)|
                         (aric['STIA01']==1), 1, np.where((aric['PREVMI05']==0)|
                                                            (aric['PRVCHD05']==0)|
                                                            (aric['MDDXMI02']==0)|
                                                            (aric['HXOFMI02']==0)|
                                                            (aric['SYMCHD03']==0)|
                                                            (aric['PAD02']==0)|
                                                            (aric['STIA01']==0), 0, np.nan))
aric['ihd'] = np.where((aric['PREVMI05']==1)|
                         (aric['MDDXMI02']==1)|
                         (aric['HXOFMI02']==1), 1, np.where((aric['PREVMI05']==0)|
                                                            (aric['MDDXMI02']==0)|
                                                            (aric['HXOFMI02']==0), 0, np.nan))
aric['cabg'] = np.where((aric['PHEA07A']=="Y"), 1, np.where((aric['PHEA06']=='N')|(aric['PHEA07A']=="N"), 0, np.nan))
aric['carotid_revasc'] = np.where((aric['PHEA07C']=="Y"), 1, np.where((aric['PHEA06']=='N')|(aric['PHEA07C']=="N"), 0, np.nan))
aric['pad_revasc'] = np.where((aric['PHEA07E']=="Y")|(aric['PHEA09B']=="Y"), 1, np.where((aric['PHEA06']=='N')|(aric['PHEA07E']=="N")
                                                                                         |(aric['PHEA09B']=='N'), 0, np.nan))
aric['pci'] = np.where((aric['PHEA09A']=="Y")|(aric['PHEA09C']=="Y"), 1, np.where((aric['PHEA08']=='N')|(aric['PHEA09A']=="N")|
                                                                                  (aric['PHEA09C']=="N"), 0, np.nan))
#PROBABLY Dictionary for LIPA08, Lp(a): A is missing!
aric['LIPA08'] = np.where(aric['LIPA08']=='A', np.nan, aric['LIPA08'])
aric['LIPA08'] = aric['LIPA08'].astype(float)
aric.rename(columns={'V1AGE01': 'age', 'GENDER': 'sex', 'RACEGRP': 'ethnicity', 'BMI01': 'bmi', 'HYPERT05': 'htn', 'DIABTS03': 'dm',
                     'PREVHF01': 'heart_failure', 'STIA01': 'stroke', 'PAD02': 'pad', 'STATINCODE01': 'statin', 'GLUSIU01': 'glucose',
                     'TCHSIU01': 'cholesterol', 'HDLSIU02': 'hdl', 'LDLSIU02': 'ldl', 'TRGSIU01': 'triglycerides', 'ID_C': 'id',
                     'LIPA08': 'lp_a_value',
                     #From Second Visit
                     'V2ALT': 'alt'}, inplace=True)
aric['sex'].replace({'F': 0, 'M': 1}, inplace=True)
aric['ethnicity'].replace({'W': 'White', 'B': 'African-American'}, inplace=True)

#CVD Family History
aric['premature_chd_family_history'] = np.where((aric['MOMPRECHD']=='Y')|(aric['DADPRECHD']=='Y'), 1, np.where(
    (aric['MOMPRECHD']=='N')|(aric['DADPRECHD']=='N'), 0 , np.nan))
aric['cvd_family_history'] = np.where((aric['premature_chd_family_history']==1)|(aric['MOMHISTORYSTR']==1)|
                                      (aric['DADHISTORYSTR']==1)|(aric['MOMHISTORYCHD']==1)|(aric['DADHISTORYCHD']==1), 1,
                                      np.where((aric['premature_chd_family_history']==0)|(aric['MOMHISTORYSTR']==0)|
                                      (aric['DADHISTORYSTR']==0)|(aric['MOMHISTORYCHD']==0)|(aric['DADHISTORYCHD']==0), 0, np.nan))

aric = aric[['id', 'age', 'sex', 'ethnicity', 'bmi', 'htn', 'dm', 'ihd', 'heart_failure', 'stroke', 'pad', 'cabg', 'pci',
            'carotid_revasc', 'pad_revasc', 'ascvd', 'premature_chd_family_history', 'cvd_family_history', 'statin', 'llt',
            'anti_htn', 'hormone_est_prog', 'hdl', 'ldl', 'cholesterol', 'triglycerides', 'lp_a_value']]
aric['dm'].replace({'T': np.nan}, inplace=True)
aric['dm'] = aric['dm'].astype('Int64')
aric = aric[~aric['lp_a_value'].isna()]
aric['lp_a_value'] = aric['lp_a_value']*0.215*3
aric.to_csv("/ARIC/aric.csv", index=False)
aric.to_csv("/Lp(a)/External_Validation/aric.csv", index=False)
aric

# %%
import matplotlib.pyplot as plt
aric.hist(bins=50, figsize=(20,15))
plt.show()


