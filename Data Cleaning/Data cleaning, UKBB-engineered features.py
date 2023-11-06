# %%
# This file is for data cleaning for UK Biobank with engineered features.

import pandas as pd
import numpy as np
import datatable as dt
import math
from datetime import date 
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

path = "/Lp(a)/"

ukb_data_dt = dt.fread("/home/rk725/ukbdata/ukb47034.csv")
column_names_recoding = pd.read_csv(path + "colnames_my_ukb_data.csv")
ukb_key = pd.read_excel(path + "my_ukb_key_11012022.xlsx")

# %%
ukb_data_dt.names = column_names_recoding['name'].tolist()
ukb_data_dt

# %%
ukb_data_dt = ukb_data_dt[:, ['eid','sex_f31_0_0','year_of_birth_f34_0_0','month_of_birth_f52_0_0',
                          'date_of_attending_assessment_centre_f53_0_0','date_of_attending_assessment_centre_f53_1_0',
                          'date_of_attending_assessment_centre_f53_2_0','date_of_attending_assessment_centre_f53_3_0',
                          'date_lost_to_followup_f191_0_0', 'ethnic_background_f21000_0_0','ethnic_background_f21000_1_0',
                          'ethnic_background_f21000_2_0', 'date_lost_to_followup_f191_0_0', 'date_of_death_f40000_0_0', 'date_of_death_f40000_1_0',
                          'date_of_myocardial_infarction_f42000_0_0', 'date_of_ischaemic_stroke_f42008_0_0'
                          ] + ukb_key[ukb_key['field.showcase'].isin([21001,48,49,20116,20003,
                          42000,42002,42004,42006,42008,42010,42012,42026,42014,42016,41270,41280,41271,41281,20002,20008,
                          20004,20010,41272,41282,41273,41283,30790,30796,30780,30760,30690,30870,30710,30740,30750,30670,
                          30700,30620,30610,30650,30660,30840,30600,30860,30020,30000,30080,30040,4079,4080,93,94,95,102,
                          6153,6177,
                          30626,30606,30616,30656,30716,30696,30706,30666,30746,30756,30766,30786,30846,30866,30876,30676,30896,30890,
                          30810,30680,30816,30686,
                          41202,41262,41203,41263,20107,20110,20111,
                          20116,
                          30791, 30795, 
                          40000, 40001, 40002
                            ])]['col.name'].tolist()]
ukb_data_dt.to_csv("/Lp(a)/ukb_lpa.csv")
ukb_data_dt

# %%
import pandas as pd
import numpy as np
import datatable as dt
import math
from datetime import date
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

path = "/Lp(a)/"

ukb_data = pd.read_csv(path + "ukb_lpa.csv")
ukb_key = pd.read_excel(path + "my_ukb_key_11012022.xlsx")
ukb_data

# %%
#Demographics
demo_ukb_data = ukb_data[['eid','sex_f31_0_0','year_of_birth_f34_0_0','month_of_birth_f52_0_0',
                          'date_of_attending_assessment_centre_f53_0_0','date_of_attending_assessment_centre_f53_1_0',
                          'ethnic_background_f21000_0_0','ethnic_background_f21000_1_0',
                          'ethnic_background_f21000_2_0', 'date_lost_to_followup_f191_0_0', 'date_of_death_f40000_0_0', 'date_of_death_f40000_1_0']]

demo_ukb_data['birth_date'] = pd.to_datetime((demo_ukb_data[demo_ukb_data['year_of_birth_f34_0_0'].notna()][
                                                'year_of_birth_f34_0_0'].astype(int).astype(str) + '_' + 
                                             demo_ukb_data[demo_ukb_data['month_of_birth_f52_0_0'].notna()][
                                                'month_of_birth_f52_0_0'].astype(int).astype(str) + '_' + '15'), format='%Y_%m_%d')
demo_ukb_data['ethnicity'] = np.where(demo_ukb_data['ethnic_background_f21000_0_0'].notna(), 
                                      demo_ukb_data['ethnic_background_f21000_0_0'], 
                                      np.where(demo_ukb_data['ethnic_background_f21000_1_0'].notna(), 
                                      demo_ukb_data['ethnic_background_f21000_1_0'], 
                                      np.where(demo_ukb_data['ethnic_background_f21000_2_0'].notna(),
                                      demo_ukb_data['ethnic_background_f21000_2_0'], np.nan)))

#Check ethnicity coding
#demo_ukb_data[['ethnicity','ethnic_background_f21000_0_0','ethnic_background_f21000_1_0',
#                          'ethnic_background_f21000_2_0']][demo_ukb_data['ethnicity'].isna()]
demo_ukb_data = demo_ukb_data.rename(columns={'sex_f31_0_0':'sex', 'date_of_attending_assessment_centre_f53_0_0':'visit_0',
                                              'date_of_attending_assessment_centre_f53_1_0':'visit_1',
                                              'date_lost_to_followup_f191_0_0':'lost_fu_date'}).drop(columns = 
                                              ['year_of_birth_f34_0_0','month_of_birth_f52_0_0','ethnic_background_f21000_0_0',
                                              'ethnic_background_f21000_1_0','ethnic_background_f21000_2_0'])

demo_ukb_data['death_date'] = np.where(demo_ukb_data['date_of_death_f40000_0_0'].notna(), demo_ukb_data['date_of_death_f40000_0_0'],
                                            np.where(demo_ukb_data['date_of_death_f40000_1_0'].notna(), demo_ukb_data['date_of_death_f40000_1_0'], np.nan))

demo_ukb_data = demo_ukb_data[['eid', 'birth_date', 'sex', 'ethnicity', 'visit_0', 'visit_1', 'lost_fu_date', 'death_date']]

for i in ['birth_date', 'visit_0', 'visit_1', 'lost_fu_date', 'death_date']:
    demo_ukb_data[i] = pd.to_datetime(demo_ukb_data[i])

demo_ukb_data['age_0'] = (demo_ukb_data['visit_0'] - demo_ukb_data['birth_date']
                        ).astype(str).str.split(" ", expand=True)[0].replace('NaT', np.nan).astype(float).mul(1/365.25)
demo_ukb_data['age_1'] = pd.to_numeric((demo_ukb_data['visit_1'] - demo_ukb_data['birth_date']
                          ).astype(str).str.split(" ", expand=True)[0], errors='coerce').astype(float).mul(1/365.25)


for i in ['_0', '_1']:
    demo_ukb_data['Death' + i] = np.where((demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']>=demo_ukb_data['visit' + i]), 1, np.where(
        (demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']<demo_ukb_data['visit' + i]), np.nan, 0))
    demo_ukb_data['Time2Death' + i] = np.where(demo_ukb_data['Death' + i].isna(), np.nan, np.where(
                                demo_ukb_data['Death' + i]==1, (demo_ukb_data["death_date"] - demo_ukb_data['visit' + i]).dt.days,
                                np.where(demo_ukb_data["lost_fu_date"].notna(), (demo_ukb_data["lost_fu_date"] - demo_ukb_data['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - demo_ukb_data['visit' + i]).dt.days)))
for i in ['Death_1', 'Time2Death_1']:
    demo_ukb_data[i] = np.where(demo_ukb_data['visit_1'].isna(), np.nan, demo_ukb_data[i])
    
#sex_coding = dict({0:'Female',1:'Male'})
#demo_ukb_data = demo_ukb_data.replace({'sex':sex_coding})

ethnicity = dict({1:1,1001:1,1002:1,1003:1,
                  2:2,2001:2,2002:2,2003:2,2004:2,
                  3:3,3001:3,3002:3,3003:3,3004:3,
                  4:4,4001:4,4002:4,4003:4,
                  5:5,
                  6:6,
                  -1:np.nan, -3:np.nan})
demo_ukb_data = demo_ukb_data.replace({'ethnicity': ethnicity})

#ethnicity_cat = dict({1:'White',2:"Mixed",3:'South Asian',4:'Black',5:'Chinese',6:"Other"})
#demo_ukb_data = demo_ukb_data.replace({'ethnicity':ethnicity_cat})

demo_ukb_data

# %%
#Cause of death
death_cause = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([40001, 40002])]['col.name'].tolist()]
primary_death_stubname = ukb_key[ukb_key['field.showcase'] == 40001]['col.name'].iloc[0].rsplit('_', 2)[0]
secondary_death_stubname = ukb_key[ukb_key['field.showcase'] == 40002]['col.name'].iloc[0].rsplit('_', 2)[0]
death_cause = pd.wide_to_long(df=death_cause, stubnames=[primary_death_stubname, secondary_death_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
death_cause[['instance','array']] = death_cause['instance_array'].str.split("_", expand=True)
death_cause.rename(columns={'underlying_primary_cause_of_death_icd10_f40001':'primary_cause_death',
                            'contributory_secondary_ca_f40002':'secondary_cause_death'},inplace=True)

#Defining cause of death

cv_death_list = [
#heart_failure_list
            'I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509',
#ihd_list                     
            'I20', 'I200', 'I208', 'I209', 'I21', 'I210', 'I211', 'I212', 'I213',
            'I214', 'I219', 'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I23', 'I230',
            'I231', 'I232', 'I233', 'I234', 'I235', 'I236', 'I238', 'I24', 'I240', 'I241',
            'I248', 'I249', 'I25', 'I250', 'I251', 'I252', 'I255', 'I256', 'I258', 'I259',
            'Z951', 'Z955',

#pad_list
            'I702', 'I7020', 'I7021', 'I742', 'I743', 'I744'
            
#stroke_list
            'G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459', 'I63', 
             'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638', 'I639', 'I64',
             'I65', 'I650', 'I651', 'I652', 'I653', 'I658', 'I659', 'I66', 'I660',
             'I661', 'I662', 'I663', 'I664', 'I668', 'I669', 'I672', 'I693', 'I694']

death_cause['primary_cv_mortality'] = np.where(death_cause['primary_cause_death'].isna(), np.nan,
                                               np.where(death_cause['primary_cause_death'].isin(cv_death_list), 1, 0))
death_cause['secondary_cv_mortality'] = np.where(death_cause['secondary_cause_death'].isna(), np.nan,
                                               np.where(death_cause['secondary_cause_death'].isin(cv_death_list), 1, 0))

death_cause['primary_cv_mortality'] = death_cause.groupby('eid')['primary_cv_mortality'].transform(max)
death_cause['secondary_cv_mortality'] = death_cause.groupby('eid')['secondary_cv_mortality'].transform(max)
death_cause.drop_duplicates(subset='eid', inplace=True)
death_cause = death_cause[['eid', 'primary_cv_mortality', 'secondary_cv_mortality']]
death_cause['primary_secondary_cv_mortality'] = np.where((death_cause['primary_cv_mortality']==1)|(death_cause['secondary_cv_mortality']==1), 1,
                                                         np.where((death_cause['primary_cv_mortality']==0)|(death_cause['secondary_cv_mortality']==0), 0, np.nan))
demo_ukb_data = demo_ukb_data.merge(death_cause, on='eid', how='left')
demo_ukb_data 

# %%
# This will define death outcomes after Lp(a) assay date.

lpa_lab = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([30791, 30795])]['col.name'].tolist()]
lpa_missing_stubname = ukb_key[ukb_key['field.showcase'] == 30795]['col.name'].iloc[0].rsplit('_', 2)[0]
lpa_date_stubname = ukb_key[ukb_key['field.showcase'] == 30791]['col.name'].iloc[0].rsplit('_', 2)[0]
lpa_lab = pd.wide_to_long(df=lpa_lab, stubnames=[lpa_missing_stubname, lpa_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
lpa_lab[['instance','array']] = lpa_lab['instance_array'].str.split("_", expand=True)
lpa_lab.rename(columns={'lipoprotein_a_missing_reason_f30795':'lpa_missing_reason','lipoprotein_a_assay_date_f30791':'lpa_assay_date'},inplace=True)
lpa_lab['lpa_missing_reason_0'] = np.where(lpa_lab['instance']==str(0), lpa_lab['lpa_missing_reason'],np.nan)
lpa_lab['lpa_missing_reason_1'] = np.where(lpa_lab['instance']==str(1), lpa_lab['lpa_missing_reason'],np.nan)
lpa_lab['lpa_assay_date_0'] = pd.to_datetime(np.where(lpa_lab['instance']==str(0), lpa_lab['lpa_assay_date'],np.nan))
lpa_lab['lpa_assay_date_1'] = pd.to_datetime(np.where(lpa_lab['instance']==str(1), lpa_lab['lpa_assay_date'],np.nan))
lpa_lab['lpa_missing_reason_0'] = lpa_lab.groupby('eid')['lpa_missing_reason_0'].transform(max)
lpa_lab['lpa_missing_reason_1'] = lpa_lab.groupby('eid')['lpa_missing_reason_1'].transform(max)
lpa_lab['lpa_assay_date_0'] = lpa_lab.groupby('eid')['lpa_assay_date_0'].transform(max)
lpa_lab['lpa_assay_date_1'] = lpa_lab.groupby('eid')['lpa_assay_date_1'].transform(max)
lpa_lab.drop_duplicates(subset='eid', inplace=True)
lpa_lab = lpa_lab[['eid', 'lpa_missing_reason_0', 'lpa_missing_reason_1', 'lpa_assay_date_0', 'lpa_assay_date_1']]
#Replacing missing lpa_assay_dates with median. This is to ensure we have a time-to-event variable for every one
lpa_lab['lpa_assay_date_0'].fillna(lpa_lab['lpa_assay_date_0'].describe().loc['50%'], inplace=True)
lpa_lab['lpa_assay_date_1'].fillna(lpa_lab['lpa_assay_date_1'].describe().loc['50%'], inplace=True)

demo_ukb_data = demo_ukb_data.merge(lpa_lab, on='eid', how='left')
#If the patient had not visit 1, they shouldn't have lpa_assay_date_1
demo_ukb_data['lpa_assay_date_1'] = pd.to_datetime(np.where(demo_ukb_data['visit_1'].isna(), pd.NaT, demo_ukb_data['lpa_assay_date_1']))

for i in ['_0', '_1']:
    demo_ukb_data['Death_lpa' + i] = np.where((demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']>=demo_ukb_data['lpa_assay_date' + i]), 1, np.where(
        (demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']<demo_ukb_data['lpa_assay_date' + i]), np.nan, 0))
    demo_ukb_data['Time2Death_lpa' + i] = np.where(demo_ukb_data['Death_lpa' + i].isna(), np.nan, np.where(
                                demo_ukb_data['Death_lpa' + i]==1, (demo_ukb_data["death_date"] - demo_ukb_data['lpa_assay_date' + i]).dt.days,
                                np.where((demo_ukb_data["lost_fu_date"].notna()) & (demo_ukb_data['lost_fu_date']>=demo_ukb_data['lpa_assay_date' + i]),
                                            (demo_ukb_data["lost_fu_date"] - demo_ukb_data['lpa_assay_date' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - demo_ukb_data['lpa_assay_date' + i]).dt.days)))

for i in ['Death_lpa', 'Time2Death_lpa']:
    demo_ukb_data[i + '_0'] = np.where(demo_ukb_data['lpa_assay_date_0'].isna(), np.nan, demo_ukb_data[i + '_0'])
    demo_ukb_data[i + '_1'] = np.where(demo_ukb_data['lpa_assay_date_1'].isna(), np.nan, demo_ukb_data[i + '_1'])

for i in ['_0', '_1']:
    demo_ukb_data['primary_cv_death_lpa' + i] = np.where((demo_ukb_data['Death_lpa' + i]==1)&(demo_ukb_data['primary_cv_mortality']==1), 1,
                                            np.where((demo_ukb_data['Death_lpa' + i]==1)&(demo_ukb_data['primary_cv_mortality']==0), 0, demo_ukb_data['Death_lpa' + i]))
    demo_ukb_data['primary_secondary_cv_death_lpa' + i] = np.where((demo_ukb_data['Death_lpa' + i]==1)&(demo_ukb_data['primary_secondary_cv_mortality']==1), 1,
                                    np.where((demo_ukb_data['Death_lpa' + i]==1)&(demo_ukb_data['primary_secondary_cv_mortality']==0), 0, demo_ukb_data['Death_lpa' + i]))
    
demo_ukb_data

# %%
# This will define MACE outcomes after Lp(a) assay date.

#ICD10
pmh_ukb_maindx_icd10 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41202,41262])]['col.name'].tolist()]
icd10_maindx_stubname = ukb_key[ukb_key['field.showcase'] == 41202]['col.name'].iloc[0].rsplit('_', 2)[0]
icd10_maindate_stubname = ukb_key[ukb_key['field.showcase'] == 41262]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_maindx_icd10 = pd.wide_to_long(df=pmh_ukb_maindx_icd10, stubnames=[icd10_maindx_stubname, icd10_maindate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_maindx_icd10 = long_pmh_ukb_maindx_icd10.drop(columns='instance_array').rename(
    columns={'diagnoses_main_icd10_f41202':'icd', 'date_of_first_inpatient_diagnosis_main_icd10_f41262':'main_icd_date'})
final_pmh_ukb_maindx_icd10['icd_type'] = 'icd10'

#ICD9
pmh_ukb_maindx_icd9 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41203,41263])]['col.name'].tolist()]
icd9_maindx_stubname = ukb_key[ukb_key['field.showcase'] == 41203]['col.name'].iloc[0].rsplit('_', 2)[0]
icd9_maindate_stubname = ukb_key[ukb_key['field.showcase'] == 41263]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_maindx_icd9 = pd.wide_to_long(df=pmh_ukb_maindx_icd9, stubnames=[icd9_maindx_stubname, icd9_maindate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_maindx_icd9 = long_pmh_ukb_maindx_icd9.drop(columns='instance_array').rename(columns={
    'diagnoses_main_icd9_f41203':'icd', 'date_of_first_inpatient_diagnosis_main_icd9_f41263':'main_icd_date'})
final_pmh_ukb_maindx_icd9['icd_type'] = 'icd9'

#Merging ICD9 and ICD10
pmh_ukb_maindx = pd.concat([final_pmh_ukb_maindx_icd10, final_pmh_ukb_maindx_icd9], ignore_index=True)
pmh_ukb_maindx = pmh_ukb_maindx.merge(demo_ukb_data, on='eid', how='left')
pmh_ukb_maindx['main_icd_date'] = pd.to_datetime(pmh_ukb_maindx['main_icd_date'])

#Creating variables for episodes occurred at or before visit_0 and visit_1
#Defining Each Disease Using ICD-9/ICD-10 Codes
ihd_hosp = ['I200', 'I21', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219',
            'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I24', 'I248',
            'I249',

            '410', '4109']

stroke_hosp=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459',
             'I63', 'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638',
             'I639', 'I64',
             
             
             '433', '4330', '4331', '4332', '4333', '4338', '4339', '434',
             '4340', '4341', '4349', '435', '4359']

pmh_hosp = {'ihd': ihd_hosp, 'stroke': stroke_hosp}
hosp_list = ['ihd', 'stroke'] 

pmh_ukb_maindx['ihd'] = np.where(pmh_ukb_maindx['icd'].isin(ihd_hosp), 1, 0)
pmh_ukb_maindx['ihd_date'] = np.where(pmh_ukb_maindx['ihd']==1, pmh_ukb_maindx['main_icd_date'], pd.NaT)
pmh_ukb_maindx['stroke'] = np.where(pmh_ukb_maindx['icd'].isin(stroke_hosp), 1, 0)
pmh_ukb_maindx['stroke_date'] = np.where(pmh_ukb_maindx['stroke']==1, pmh_ukb_maindx['main_icd_date'], pd.NaT)
for i in ['ihd_date', 'stroke_date']:
    pmh_ukb_maindx[i] = pd.to_datetime(pmh_ukb_maindx[i])

for i in ['_0', '_1']:
    pmh_ukb_maindx['IHD_lpa' + i] = np.where((pmh_ukb_maindx['ihd_date'].notna()) & (pmh_ukb_maindx['ihd_date']>=pmh_ukb_maindx['lpa_assay_date' + i]), 1, np.where(
        (pmh_ukb_maindx['ihd_date'].notna()) & (pmh_ukb_maindx['ihd_date']<pmh_ukb_maindx['lpa_assay_date' + i]), np.nan, 0))
    pmh_ukb_maindx['Time2IHD_lpa' + i] = np.where(pmh_ukb_maindx['IHD_lpa' + i].isna(), np.nan, np.where(
                                pmh_ukb_maindx['IHD_lpa' + i]==1, (pmh_ukb_maindx["ihd_date"] - pmh_ukb_maindx['lpa_assay_date' + i]).dt.days,
                                np.where((pmh_ukb_maindx["lost_fu_date"].notna()) & (pmh_ukb_maindx['lost_fu_date']>=pmh_ukb_maindx['lpa_assay_date' + i]),
                                            (pmh_ukb_maindx["lost_fu_date"] - pmh_ukb_maindx['lpa_assay_date' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_maindx['lpa_assay_date' + i]).dt.days)))
    
    pmh_ukb_maindx['Stroke_lpa' + i] = np.where((pmh_ukb_maindx['stroke_date'].notna()) & (pmh_ukb_maindx['stroke_date']>=pmh_ukb_maindx['lpa_assay_date' + i]), 1, np.where(
        (pmh_ukb_maindx['stroke_date'].notna()) & (pmh_ukb_maindx['stroke_date']<pmh_ukb_maindx['lpa_assay_date' + i]), np.nan, 0))
    pmh_ukb_maindx['Time2Stroke_lpa' + i] = np.where(pmh_ukb_maindx['Stroke_lpa' + i].isna(), np.nan, np.where(
                                pmh_ukb_maindx['Stroke_lpa' + i]==1, (pmh_ukb_maindx["stroke_date"] - pmh_ukb_maindx['lpa_assay_date' + i]).dt.days,
                                np.where((pmh_ukb_maindx["lost_fu_date"].notna()) & (pmh_ukb_maindx['lost_fu_date']<pmh_ukb_maindx['lpa_assay_date' + i]),
                                            (pmh_ukb_maindx["lost_fu_date"] - pmh_ukb_maindx['lpa_assay_date' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_maindx['lpa_assay_date' + i]).dt.days)))
    
for i in ['IHD_lpa', 'Time2IHD_lpa', 'Stroke_lpa', 'Time2Stroke_lpa']:
    pmh_ukb_maindx[i + '_0'] = np.where((pmh_ukb_maindx['lost_fu_date']<pmh_ukb_maindx['lpa_assay_date_0']) | (pmh_ukb_maindx['lpa_assay_date_0'].isna()), 
                                        np.nan, pmh_ukb_maindx[i + '_0'])
    pmh_ukb_maindx[i + '_1'] = np.where((pmh_ukb_maindx['lost_fu_date']<pmh_ukb_maindx['lpa_assay_date_1']) | (pmh_ukb_maindx['lpa_assay_date_1'].isna()), 
                                        np.nan, pmh_ukb_maindx[i + '_1'])


for i in ['IHD_lpa_0', 'IHD_lpa_1', 'Stroke_lpa_0', 'Stroke_lpa_1']:
    pmh_ukb_maindx[i] = pmh_ukb_maindx.groupby('eid')[i].transform(max)

for i in ['IHD_lpa_0', 'IHD_lpa_1', 'Stroke_lpa_0', 'Stroke_lpa_1']:
    pmh_ukb_maindx['Time2' + i] = pmh_ukb_maindx.groupby('eid')['Time2' + i].transform(min)

pmh_ukb_maindx.drop_duplicates(subset='eid', inplace=True)

demo_ukb_data = pmh_ukb_maindx

for i in ['_0', '_1']:
    demo_ukb_data['MACE' + i] = np.where((demo_ukb_data['IHD_lpa' + i]==1) | (demo_ukb_data['Stroke_lpa' + i]==1) | (demo_ukb_data['Death_lpa' + i]==1), 1,
                            np.where((demo_ukb_data['IHD_lpa' + i]==0) & (demo_ukb_data['Stroke_lpa' + i]==0) & (demo_ukb_data['Death_lpa' + i]==0), 0, np.nan))
    demo_ukb_data['Time2MACE' + i] = np.where(demo_ukb_data['MACE' + i]==1, demo_ukb_data[['Time2IHD_lpa' + i, 'Time2Stroke_lpa' + i, 'Time2Death_lpa' + i]].min(axis=1),
                        np.where(demo_ukb_data['MACE' + i]==0, demo_ukb_data[['Time2IHD_lpa' + i, 'Time2Stroke_lpa' + i, 'Time2Death_lpa' + i]].max(axis=1), np.nan))
    
demo_ukb_data.drop(columns=['icd', 'main_icd_date', 'icd_type', 'ihd', 'ihd_date', 'stroke', 'stroke_date'], inplace=True)
demo_ukb_data

# %%
#Lifestyle
lifestyle_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([21001,48,49,20116])]['col.name'].tolist()]
bmi_stubname = ukb_key[ukb_key['field.showcase'] == 21001]['col.name'].iloc[0].rsplit('_', 2)[0]
wc_stubname = ukb_key[ukb_key['field.showcase'] == 48]['col.name'].iloc[0].rsplit('_', 2)[0]
hc_stubname = ukb_key[ukb_key['field.showcase'] == 49]['col.name'].iloc[0].rsplit('_', 2)[0]
smoke_stubname = ukb_key[ukb_key['field.showcase'] == 20116]['col.name'].iloc[0].rsplit('_', 2)[0]

lifestyle_ukb_data = pd.wide_to_long(df=lifestyle_ukb_data, stubnames=[bmi_stubname,wc_stubname,hc_stubname,smoke_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
lifestyle_ukb_data[['instance','array']] = lifestyle_ukb_data['instance_array'].str.split("_", expand=True)

lifestyle_ukb_data.rename(columns={'body_mass_index_bmi_f21001':'bmi','waist_circumference_f48':'waist_circumference',
                                   'hip_circumference_f49':'hip_circumference','smoking_status_f20116':'smoking'},inplace=True)
lifestyle_ukb_data['smoking'].replace({-3:np.nan},inplace=True)
#smoking_coding = dict({0:'Never',1:'Previous',2:'Current'})
#lifestyle_ukb_data = lifestyle_ukb_data.replace({'smoking':smoking_coding})

lifestyle_ukb_data['bmi_0'] = np.where(lifestyle_ukb_data['instance']==str(0),lifestyle_ukb_data['bmi'],np.nan)
lifestyle_ukb_data['bmi_1'] = np.where(lifestyle_ukb_data['instance']==str(1),lifestyle_ukb_data['bmi'],np.nan)
lifestyle_ukb_data['waist_circumference_0'] = np.where(
                    lifestyle_ukb_data['instance']==str(0),lifestyle_ukb_data['waist_circumference'],np.nan)
lifestyle_ukb_data['waist_circumference_1'] = np.where(
                    lifestyle_ukb_data['instance']==str(1),lifestyle_ukb_data['waist_circumference'],np.nan)
lifestyle_ukb_data['hip_circumference_0'] = np.where(lifestyle_ukb_data['instance']==str(0),
                                                    lifestyle_ukb_data['hip_circumference'],np.nan)
lifestyle_ukb_data['hip_circumference_1'] = np.where(lifestyle_ukb_data['instance']==str(1),
                                                    lifestyle_ukb_data['hip_circumference'],np.nan)
lifestyle_ukb_data['smoking_0'] = np.where(lifestyle_ukb_data['instance']==str(0),lifestyle_ukb_data['smoking'],np.nan)
lifestyle_ukb_data['smoking_1'] = np.where(lifestyle_ukb_data['instance']==str(1),lifestyle_ukb_data['smoking'],np.nan)

lifestyle_ukb_data['bmi_0'] = lifestyle_ukb_data.groupby('eid')['bmi_0'].transform(max)
lifestyle_ukb_data['bmi_1'] = lifestyle_ukb_data.groupby('eid')['bmi_1'].transform(max)
lifestyle_ukb_data['waist_circumference_0'] = lifestyle_ukb_data.groupby('eid')['waist_circumference_0'].transform(max)
lifestyle_ukb_data['waist_circumference_1'] = lifestyle_ukb_data.groupby('eid')['waist_circumference_1'].transform(max)
lifestyle_ukb_data['hip_circumference_0'] = lifestyle_ukb_data.groupby('eid')['hip_circumference_0'].transform(max)
lifestyle_ukb_data['hip_circumference_1'] = lifestyle_ukb_data.groupby('eid')['hip_circumference_1'].transform(max)
lifestyle_ukb_data['smoking_0'] = lifestyle_ukb_data.groupby('eid')['smoking_0'].transform(max)
lifestyle_ukb_data['smoking_1'] = lifestyle_ukb_data.groupby('eid')['smoking_1'].transform(max)

lifestyle_ukb_data = lifestyle_ukb_data[['eid','bmi_0','bmi_1','waist_circumference_0','waist_circumference_1',
                                          'hip_circumference_0','hip_circumference_1','smoking_0','smoking_1']]

lifestyle_ukb_data.drop_duplicates(subset='eid', inplace=True)
lifestyle_ukb_data

# %%
#Inpatient Diagnoses
#ICD10
pmh_ukb_indx_icd10 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41270,41280])]['col.name'].tolist()]
icd10_dx_stubname = ukb_key[ukb_key['field.showcase'] == 41270]['col.name'].iloc[0].rsplit('_', 2)[0]
icd10_date_stubname = ukb_key[ukb_key['field.showcase'] == 41280]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_indx_icd10 = pd.wide_to_long(df=pmh_ukb_indx_icd10, stubnames=[icd10_dx_stubname, icd10_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_indx_icd10 = long_pmh_ukb_indx_icd10.drop(columns='instance_array').rename(columns={'diagnoses_icd10_f41270':'icd',
                                        'date_of_first_inpatient_diagnosis_icd10_f41280':'icd_date'})
final_pmh_ukb_indx_icd10['icd_type'] = 'icd10'

#ICD9
pmh_ukb_indx_icd9 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41271,41281])]['col.name'].tolist()]
icd9_dx_stubname = ukb_key[ukb_key['field.showcase'] == 41271]['col.name'].iloc[0].rsplit('_', 2)[0]
icd9_date_stubname = ukb_key[ukb_key['field.showcase'] == 41281]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_indx_icd9 = pd.wide_to_long(df=pmh_ukb_indx_icd9, stubnames=[icd9_dx_stubname, icd9_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_indx_icd9 = long_pmh_ukb_indx_icd9.drop(columns='instance_array').rename(columns={'diagnoses_icd9_f41271':'icd',
                                        'date_of_first_inpatient_diagnosis_icd9_f41281':'icd_date'})
final_pmh_ukb_indx_icd9['icd_type'] = 'icd9'

#Merging ICD9 and ICD10
pmh_ukb_indx = pd.concat([final_pmh_ukb_indx_icd10, final_pmh_ukb_indx_icd9], ignore_index=True)
pmh_ukb_indx = pmh_ukb_indx.merge(demo_ukb_data[['eid', 'sex', 'birth_date', 'visit_0', 'visit_1']], on='eid', how='left')
pmh_ukb_indx['icd_date'] = pd.to_datetime(pmh_ukb_indx['icd_date'])

#Defining Each Disease Using ICD-9/ICD-10 Codes
htn_list = ['I10', 'I11', 'I110', 'I119', 'I12', 'I120', 'I129', 'I13', 'I130', 'I131',
            'I132', 'I139', 'I674', 'O10', 'O100', 'O101', 'O102', 'O103', 'O109', 'O11',
            
            '401', '4010', '4011', '4019', '402', '4020', '4021', '4029', '403', '4030',
            '4031', '4039', '404', '4040', '4041', '4049', '6420', '6422', '6427', '6429']
dm_list = ['E10', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108',
           'E109', 'E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117',
           'E118', 'E119', 'E12', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126',
           'E127', 'E128', 'E129', 'E13', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135',
           'E136', 'E137', 'E138', 'E139', 'E14', 'E140', 'E141', 'E142', 'E143', 'E144',
           'E145', 'E146', 'E147', 'E148', 'E149', 'O240', 'O241', 'O242', 'O243', 'O249',
    
           '250', '2500', '25000', '25001', '25009', '2501', '25010', '25011', '25019', '2502',
           '25020', '25021', '25029', '2503', '2504', '2505', '2506', '2507', '2509', '25090',
           '25091', '25099', '6480']
t1dm_list = ['E10', 'E100', 'E101', 'E102','E103', 'E104', 'E105', 'E106', 'E107', 'E108',
             'E109', 'O240',
             
             '25001', '25011', '25021', '25091']
t2dm_list = ['E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118',
             'E119', 'O241',
             
             '25000', '25010', '25020', '25090']
ckd_list = ['I12', 'I120', 'I13', 'I130', 'I131', 'I132', 'I139', 'N18', 'N180', 'N181',
            'N182', 'N183', 'N184', 'N185', 'N188', 'N189', 'Z49', 'Z490', 'Z491', 'Z492',
            
            '403', '4030', '4031', '4039', '404', '4040', '4041', '4049', '585', '5859',
            '6421', '6462']
heart_failure_list = ['I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509',
                     
                      '428','4280','4281','4289']
ihd_list = ['I20', 'I200', 'I208', 'I209', 'I21', 'I210', 'I211', 'I212', 'I213',
            'I214', 'I219', 'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I23', 'I230',
            'I231', 'I232', 'I233', 'I234', 'I235', 'I236', 'I238', 'I24', 'I240', 'I241',
            'I248', 'I249', 'I25', 'I250', 'I251', 'I252', 'I255', 'I256', 'I258', 'I259',
            'Z951', 'Z955',

            '410', '4109', '411', '4119', '412', '4129', '413', '4139', '414', '4140',
            '4148', '4149']
pad_list = ['I702', 'I7020', 'I7021', 'I742', 'I743', 'I744'
            
            '4402', '4442']
stroke_list=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459', 'I63', 
             'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638', 'I639', 'I64',
             'I65', 'I650', 'I651', 'I652', 'I653', 'I658', 'I659', 'I66', 'I660',
             'I661', 'I662', 'I663', 'I664', 'I668', 'I669', 'I672', 'I693', 'I694',
             
             '433', '4330', '4331', '4332', '4333', '4338', '4339', '434', '4340',
             '4341', '4349', '435', '4359', '437', '4370', '4371']

pmh_dict = {'htn': htn_list, 'dm': dm_list, 't1dm': t1dm_list, 't2dm': t2dm_list,
            'ckd': ckd_list, 'heart_failure': heart_failure_list, 
            'ihd': ihd_list, 'pad': pad_list, 'stroke': stroke_list}

for i in pmh_dict.keys():
    #Defining PMH_0 and PMH_1
    pmh_ukb_indx[i + '_0'] = np.where((pmh_ukb_indx['icd'].isin(pmh_dict[i])) & 
                                      (pmh_ukb_indx['icd_date']<=pmh_ukb_indx['visit_0']), 1, 0)
    pmh_ukb_indx[i + '_1'] = np.where(pmh_ukb_indx['visit_1'].isna(), np.nan, np.where(
                        (pmh_ukb_indx['icd'].isin(pmh_dict[i])) & (pmh_ukb_indx['icd_date']<=pmh_ukb_indx['visit_1']), 1, 0))

ascvd_list = ['ihd', 'pad', 'stroke']
for i in ascvd_list:
    #Finding the age at which each ASCVD first occurred for visit_0
    pmh_ukb_indx[i + '_first_date_0'] = pd.to_datetime(np.where(pmh_ukb_indx[i + '_0']==1,
                                        pmh_ukb_indx.groupby(by=['eid', i + '_0'])['icd_date'].transform(min), pd.NaT))
    pmh_ukb_indx[i + '_first_age_0'] = (pmh_ukb_indx[i + '_first_date_0'] - pmh_ukb_indx['birth_date']).astype(str).str.split(
                                       ' ', expand=True)[0].replace('NaT', np.nan).astype(float)/365.25
    pmh_ukb_indx[i + '_first_age_0'] = pmh_ukb_indx.groupby('eid')[i + '_first_age_0'].transform(min)

    #Finding the age at which each ASCVD first occurred for visit_1
    pmh_ukb_indx[i + '_first_1'] = np.where((pmh_ukb_indx['icd'].isin(pmh_dict[i])) &
                                            (pmh_ukb_indx['icd_date']<=pmh_ukb_indx['visit_1']), 1, 0)
    pmh_ukb_indx[i + '_first_date_1'] = pd.to_datetime(np.where(pmh_ukb_indx[i + '_1']==1,
                                        pmh_ukb_indx.groupby(by=['eid', i + '_1'])['icd_date'].transform(min), pd.NaT))
    pmh_ukb_indx[i + '_first_age_1'] = (pmh_ukb_indx[i + '_first_date_1'] - pmh_ukb_indx['birth_date']).astype(str).str.split(
                                       ' ', expand=True)[0].replace('NaT', np.nan).astype(float)/365.25
    pmh_ukb_indx[i + '_first_age_1'] = pmh_ukb_indx.groupby('eid')[i + '_first_age_1'].transform(min)

for i in pmh_dict.keys():    
    #Transforming PMH_0 and PMH_1 to avoid loss of data when removing duplicates
    pmh_ukb_indx[i + '_0'] = pmh_ukb_indx.groupby('eid')[i + '_0'].transform(max)
    pmh_ukb_indx[i + '_1'] = pmh_ukb_indx.groupby('eid')[i + '_1'].transform(max)

#Defining premature ASCVD at visit_0
pmh_ukb_indx['age_at_first_ascvd_0'] = pmh_ukb_indx[['ihd_first_age_0', 'pad_first_age_0', 'stroke_first_age_0']].min(axis=1)
pmh_ukb_indx['premature_ascvd_0'] = np.where(
    ((pmh_ukb_indx['sex']==0)&((pmh_ukb_indx['age_at_first_ascvd_0']<65)))|
    ((pmh_ukb_indx['sex']==1)&((pmh_ukb_indx['age_at_first_ascvd_0']<55))), 1, 0)

#Defining premature ASCVD at visit_1
pmh_ukb_indx['age_at_first_ascvd_1'] = pmh_ukb_indx[['ihd_first_age_1', 'pad_first_age_1', 'stroke_first_age_1']].min(axis=1)
pmh_ukb_indx['premature_ascvd_1'] = np.where(
    ((pmh_ukb_indx['sex']==0)&((pmh_ukb_indx['age_at_first_ascvd_1']<65)))|
    ((pmh_ukb_indx['sex']==1)&((pmh_ukb_indx['age_at_first_ascvd_1']<55))), 1, 0)

pmh_ukb_indx.drop_duplicates(subset='eid', inplace=True)
pmh_ukb_indx.drop(columns=pmh_ukb_indx.columns[pmh_ukb_indx.columns.str.contains('icd|sex|date|visit|first_age|first_1')], inplace=True)
pmh_ukb_indx

# %%
#PMH-Number of Hospitalization Episodes with a Main Diagnosis of ASCVD
#ICD10
pmh_ukb_maindx_icd10 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41202,41262])]['col.name'].tolist()]
icd10_maindx_stubname = ukb_key[ukb_key['field.showcase'] == 41202]['col.name'].iloc[0].rsplit('_', 2)[0]
icd10_maindate_stubname = ukb_key[ukb_key['field.showcase'] == 41262]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_maindx_icd10 = pd.wide_to_long(df=pmh_ukb_maindx_icd10, stubnames=[icd10_maindx_stubname, icd10_maindate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_maindx_icd10 = long_pmh_ukb_maindx_icd10.drop(columns='instance_array').rename(
    columns={'diagnoses_main_icd10_f41202':'icd', 'date_of_first_inpatient_diagnosis_main_icd10_f41262':'main_icd_date'})
final_pmh_ukb_maindx_icd10['icd_type'] = 'icd10'

#ICD9
pmh_ukb_maindx_icd9 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41203,41263])]['col.name'].tolist()]
icd9_maindx_stubname = ukb_key[ukb_key['field.showcase'] == 41203]['col.name'].iloc[0].rsplit('_', 2)[0]
icd9_maindate_stubname = ukb_key[ukb_key['field.showcase'] == 41263]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_maindx_icd9 = pd.wide_to_long(df=pmh_ukb_maindx_icd9, stubnames=[icd9_maindx_stubname, icd9_maindate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_maindx_icd9 = long_pmh_ukb_maindx_icd9.drop(columns='instance_array').rename(columns={
    'diagnoses_main_icd9_f41203':'icd', 'date_of_first_inpatient_diagnosis_main_icd9_f41263':'main_icd_date'})
final_pmh_ukb_maindx_icd9['icd_type'] = 'icd9'

#Merging ICD9 and ICD10
pmh_ukb_maindx = pd.concat([final_pmh_ukb_maindx_icd10, final_pmh_ukb_maindx_icd9], ignore_index=True)
pmh_ukb_maindx = pmh_ukb_maindx.merge(demo_ukb_data[['eid', 'visit_0', 'visit_1']], on='eid', how='left')
pmh_ukb_maindx['main_icd_date'] = pd.to_datetime(pmh_ukb_maindx['main_icd_date'])

#Creating variables for episodes occurred at or before visit_0 and visit_1
#Defining Each Disease Using ICD-9/ICD-10 Codes
heart_failure_hosp = ['I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509',
                     
                      '428','4280','4281','4289']
ihd_hosp = ['I200', 'I21', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219',
            'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I24', 'I248',
            'I249',

            '410', '4109']
pad_hosp = ['I742', 'I743', 'I744'
            
            '4442']
stroke_hosp=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459',
             'I63', 'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638',
             'I639', 'I64',
             
             
             '433', '4330', '4331', '4332', '4333', '4338', '4339', '434',
             '4340', '4341', '4349', '435', '4359']

pmh_hosp = {'heart_failure': heart_failure_hosp, 'ihd': ihd_hosp,
            'pad': pad_hosp, 'stroke': stroke_hosp}
hosp_list = ['heart_failure', 'ihd', 'pad', 'stroke'] 
for i in hosp_list:
    pmh_ukb_maindx[i + '_0'] = np.where((pmh_ukb_maindx['icd'].isin(pmh_hosp[i])) &
                                        (pmh_ukb_maindx['main_icd_date']<=pmh_ukb_maindx['visit_0']), 1, 0)
    pmh_ukb_maindx[i + '_1'] = np.where(pmh_ukb_maindx['visit_1'].isna(), np.nan, np.where(
                (pmh_ukb_maindx['icd'].isin(pmh_hosp[i])) & (pmh_ukb_maindx['main_icd_date']<=pmh_ukb_maindx['visit_1']), 1, 0))

#Calculating the time difference between two episodes of the same disease and removing those <30 days apart to avoid
#double counting 
for i in hosp_list:
    #Number of Dx before visit_0
    pmh_ukb_maindx = pmh_ukb_maindx.sort_values(by=['eid', i + '_0', 'main_icd_date'], ascending=False)
    pmh_ukb_maindx[i + '_lag_0'] = pmh_ukb_maindx.groupby(by=['eid', i + '_0'])['main_icd_date'].shift(1)
    pmh_ukb_maindx[i + '_lag_diff_0'] = np.where(pmh_ukb_maindx[i + '_0']==1,
                                            ((pmh_ukb_maindx[i + '_lag_0'] - pmh_ukb_maindx['main_icd_date']).astype(
                            str).str.split(' ', expand=True)[0]).replace('NaT', np.nan).astype(float), np.nan)
    #Removing Dx within 30 days of each other
    pmh_ukb_maindx = pmh_ukb_maindx.drop(pmh_ukb_maindx[(pmh_ukb_maindx[i + '_lag_diff_0']<30)&(pmh_ukb_maindx[i + '_0']==1)].index)
    
    #Number of Dx before visit_1
    pmh_ukb_maindx = pmh_ukb_maindx.sort_values(by=['eid', i + '_1', 'main_icd_date'], ascending=False)
    pmh_ukb_maindx[i + '_lag_1'] = pmh_ukb_maindx.groupby(by=['eid', i + '_1'])['main_icd_date'].shift(1)
    pmh_ukb_maindx[i + '_lag_diff_1'] = np.where(pmh_ukb_maindx[i + '_1']==1, 
                            ((pmh_ukb_maindx[i + '_lag_1'] - pmh_ukb_maindx['main_icd_date']).astype(
                            str).str.split(' ', expand=True)[0]).replace('NaT', np.nan).astype(float), np.nan)
    #Removing Dx within 30 days of each other
    pmh_ukb_maindx = pmh_ukb_maindx.drop(pmh_ukb_maindx[(pmh_ukb_maindx[i + '_lag_diff_1']<30)&(pmh_ukb_maindx[i + '_1']==1)].index)

#Creating variables for number of Dx
for i in hosp_list:
    pmh_ukb_maindx[i + '_0_num'] = pmh_ukb_maindx.groupby('eid')[i + '_0'].transform(sum)
    pmh_ukb_maindx[i + '_1_num'] = np.where(pmh_ukb_maindx['visit_1'].notna(),
                                            pmh_ukb_maindx.groupby('eid')[i + '_1'].transform(sum), np.nan)

pmh_ukb_maindx.drop_duplicates(subset='eid', inplace=True)
pmh_ukb_maindx = pmh_ukb_maindx[pmh_ukb_maindx.columns[pmh_ukb_maindx.columns.str.contains('eid|num')]]
pmh_ukb_maindx

# %%
#OPCS4 
opcs4_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41272,41282])]['col.name'].tolist()]

opcs4_proc_stubname = ukb_key[ukb_key['field.showcase'] == 41272]['col.name'].iloc[0].rsplit('_', 2)[0]
opcs4_date_stubname = ukb_key[ukb_key['field.showcase'] == 41282]['col.name'].iloc[0].rsplit('_', 2)[0]

long_opcs4_ukb_data = pd.wide_to_long(df=opcs4_ukb_data, stubnames=[opcs4_proc_stubname, opcs4_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_opcs4_data = long_opcs4_ukb_data.drop(columns='instance_array').rename(columns={'operative_procedures_opcs4_f41272':'opcs',
                                        'date_of_first_operative_procedure_opcs4_f41282':'opcs_date'})
final_opcs4_data['opcs_type'] = 'OPCS4'

#OPCS3
opcs3_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41273,41283])]['col.name'].tolist()]

opcs3_proc_stubname = ukb_key[ukb_key['field.showcase'] == 41273]['col.name'].iloc[0].rsplit('_', 2)[0]
opcs3_date_stubname = ukb_key[ukb_key['field.showcase'] == 41283]['col.name'].iloc[0].rsplit('_', 2)[0]

long_opcs3_ukb_data = pd.wide_to_long(df=opcs3_ukb_data, stubnames=[opcs3_proc_stubname, opcs3_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_opcs3_data = long_opcs3_ukb_data.drop(columns='instance_array').rename(columns={'operative_procedures_opcs3_f41273':'opcs',
                                        'date_of_first_operative_procedure_opcs3_f41283':'opcs_date'})
final_opcs3_data['opcs_type'] = 'OPCS3'

#Merging OPCS3 and OPCS4
opcs_data = pd.concat([final_opcs4_data, final_opcs3_data], ignore_index=True)
opcs_data['opcs_date'] = pd.to_datetime(opcs_data['opcs_date'])
opcs_data = opcs_data.merge(demo_ukb_data[['eid', 'sex', 'birth_date', 'visit_0', 'visit_1']], on='eid', how='left')

cabg_list = ['K40','K401','K402','K403','K404','K408','K409','K41','K411','K412',
             'K413','K414','K418','K419','K42','K421','K422','K423','K424','K428',
             'K429','K43','K431','K432','K433','K434','K438','K439','K44','K441',
             'K442','K448','K449','K45','K451','K452','K453','K454','K455','K456',
             'K458','K459','K46','K461','K462','K463','K464','K465','K468','K469',
             3043]
pci_list = ['K49','K491','K492','K493','K494','K498','K499','K50','K501','K502',
            'K503','K504','K508','K509','K75','K751','K752','K753','K754','K758',
            'K759']
carotid_revasc_list = ['L29','L291','L292','L293','L294','L295','L296','L297',
                       'L298','L299','L303','L31','L311','L313','L314','L318',
                       'L319']
pad_revasc_list = ['L50','L501','L502','L503','L504','L505','L506','L508','L509',
                   'L51','L511','L512','L513','L514','L515','L516','L518','L519',
                   'L52','L521','L522','L528','L529','L532','L54','L541','L542',
                   'L544','L548','L549','L58','L581','L582','L583','L584','L585',
                   'L586','L587','L588','L589','L59','L591','L592','L593','L594',
                   'L595','L596','L597','L598','L599','L60','L601','L602','L603',
                   'L604','L608','L609','L622','L63','L631','L632','L633','L635',
                   'L638','L639','L66','L661','L662','L665','L667','L681','L682',
                   'L701','L71','L711','L712','L713','L714','L715','L716','L717',
                   'L718','L719',
                   881,8811]
opcs_dict = {'cabg': cabg_list, 'pci': pci_list, 'carotid_revasc': carotid_revasc_list,
             'pad_revasc': pad_revasc_list}
for i in opcs_dict.keys():
    #Defining procedure_0 and procedure_1
    opcs_data[i + '_0'] = np.where((opcs_data['opcs'].isin(opcs_dict[i])) & 
                                   (opcs_data['opcs_date']<=opcs_data['visit_0']), 1, 0)
    opcs_data[i + '_1'] = np.where(opcs_data['visit_1'].isna(), np.nan, np.where(
                                   (opcs_data['opcs'].isin(opcs_dict[i])) & 
                                   (opcs_data['opcs_date']<=opcs_data['visit_1']), 1, 0))
    
    #Finding the age at which each procedure first occurred for visit_0
    opcs_data[i + '_first_date_0'] = pd.to_datetime(np.where(opcs_data[i + '_0']==1, 
                                    opcs_data.groupby(by=['eid', i + '_0'])['opcs_date'].transform(min), pd.NaT))
    opcs_data[i + '_first_age_0'] = ((opcs_data[i + '_first_date_0']-opcs_data['birth_date']).astype(str).str.split(
                                                    ' ', expand=True)[0]).replace('NaT', np.nan).astype(float)/365.25
    opcs_data[i + '_first_age_0'] = opcs_data.groupby('eid')[i + '_first_age_0'].transform(min)

    #Finding the age at which each procedure first occurred for visit_1
    #We define 'i + _first_1' varibale specifically for defining 'i + _first_age_1',
    #since the 'i + _1' variable was defined differently compared with 'i + _0' variable
    opcs_data[i + '_first_1'] = np.where((opcs_data['opcs'].isin(opcs_dict[i])) &
                                         (opcs_data['opcs_date']<=opcs_data['visit_1']), 1, 0)
    opcs_data[i + '_first_date_1'] = pd.to_datetime(np.where(opcs_data[i + '_first_1']==1, 
                                    opcs_data.groupby(by=['eid', i + '_first_1'])['opcs_date'].transform(min), pd.NaT))
    opcs_data[i + '_first_age_1'] = ((opcs_data[i + '_first_date_1']-opcs_data['birth_date']).astype(str).str.split(
                                                    ' ', expand=True)[0]).replace('NaT', np.nan).astype(float)/365.25
    opcs_data[i + '_first_age_1'] = opcs_data.groupby('eid')[i + '_first_age_1'].transform(min)

for i in opcs_dict.keys():
    #Number of Px before visit_0
    opcs_data = opcs_data.sort_values(by=['eid', i + '_0', 'opcs_date'], ascending=False)
    opcs_data[i + '_lag_0'] = opcs_data.groupby(by=['eid', i + '_0'])['opcs_date'].shift(1)
    opcs_data[i + '_lag_diff_0'] = np.where(opcs_data[i + '_0']==1,
                                            ((opcs_data[i + '_lag_0'] - opcs_data['opcs_date']).astype(
                            str).str.split(' ', expand=True)[0]).replace('NaT', np.nan).astype(float), np.nan)
    #Removing Px within 30 days of each other
    opcs_data = opcs_data.drop(opcs_data[(opcs_data[i + '_lag_diff_0']<30)&(opcs_data[i + '_0']==1)].index)

    #Number of Px before visit_1
    opcs_data = opcs_data.sort_values(by=['eid', i + '_1', 'opcs_date'], ascending=False)
    opcs_data[i + '_lag_1'] = opcs_data.groupby(by=['eid', i + '_1'])['opcs_date'].shift(1)
    opcs_data[i + '_lag_diff_1'] = np.where(opcs_data[i + '_1']==1,
                                            ((opcs_data[i + '_lag_1'] - opcs_data['opcs_date']).astype(
                            str).str.split(' ', expand=True)[0]).replace('NaT', np.nan).astype(float), np.nan)
    #Removing Px within 30 days of each other
    opcs_data = opcs_data.drop(opcs_data[(opcs_data[i + '_lag_diff_1']<30)&(opcs_data[i + '_1']==1)].index)

#Creating variables for number of Px
for i in opcs_dict.keys():
    opcs_data[i + '_0_num'] = opcs_data.groupby('eid')[i + '_0'].transform(sum)
    opcs_data[i + '_1_num'] = np.where(opcs_data['visit_1'].notna(),
                                            opcs_data.groupby('eid')[i + '_1'].transform(sum), np.nan)

#Transforming procedure_0 and procedure_1 to avoid loss of data when removing duplicates
#Doing this after defining '_num' variables
for i in opcs_dict.keys():
    opcs_data[i + '_0'] = opcs_data.groupby('eid')[i + '_0'].transform(max)
    opcs_data[i + '_1'] = opcs_data.groupby('eid')[i + '_1'].transform(max)

#Defining premature CV Px at visit_0
opcs_data['age_at_first_cv_procedure_0'] = opcs_data[['cabg_first_age_0', 'pci_first_age_0', 
                                                    'carotid_revasc_first_age_0', 'pad_revasc_first_age_0']].min(axis=1)
opcs_data['premature_cv_procedure_0'] = np.where(
    ((opcs_data['sex']==0)&((opcs_data['age_at_first_cv_procedure_0']<65)))|
    ((opcs_data['sex']==1)&((opcs_data['age_at_first_cv_procedure_0']<55))), 1, 0)

#Defining premature CV Px at visit_1
opcs_data['age_at_first_cv_procedure_1'] = opcs_data[['cabg_first_age_1', 'pci_first_age_1', 
                                                    'carotid_revasc_first_age_1', 'pad_revasc_first_age_1']].min(axis=1)
opcs_data['premature_cv_procedure_1'] = np.where(
    ((opcs_data['sex']==0)&((opcs_data['age_at_first_cv_procedure_1']<65)))|
    ((opcs_data['sex']==1)&((opcs_data['age_at_first_cv_procedure_1']<55))), 1, 0)

opcs_data.drop_duplicates(subset='eid', inplace=True)
opcs_data.drop(columns= opcs_data.columns[opcs_data.columns.str.contains('visit|sex|first_age|opcs|date|first_1|lag')], inplace=True)
opcs_data

# %%
#Medications

#Medications-Individual
drug_ukb_ind = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase']==20003]['col.name'].tolist()]
drug_stubname = ukb_key[ukb_key['field.showcase'] == 20003]['col.name'].iloc[0].rsplit('_', 2)[0]
drug_ukb_ind = pd.wide_to_long(df=drug_ukb_ind, stubnames=drug_stubname,
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
drug_ukb_ind[['instance','array']] = drug_ukb_ind['instance_array'].str.split("_", expand=True)

#Statin
drug_ukb_ind['statin'] = np.where(
    drug_ukb_ind['treatmentmedication_code_f20003'].isin([
        1140861958,1140910654,1140881748,1141188146,1140910652, #Simvastatin, Velastatin, Zocor, Simvador, Synvinolin
        1140888594,1140864592, #Fluvastatin, Lescol
        1140888648,1140910632,1140861970, #Pravastatin, Eptastatin, Lipostat
        1141146234,1141146138, #Atorvastatin, Lipitor
        1141192410,1141192414 #Rosuvastatin, Crestor
        ]),1,0)
drug_ukb_ind['statin_0'] = np.where((drug_ukb_ind['instance']==str(0)) & (drug_ukb_ind['statin']==1),1,0)
drug_ukb_ind['statin_1'] = np.where((drug_ukb_ind['instance']==str(1)) & (drug_ukb_ind['statin']==1),1,0)
drug_ukb_ind['statin_0'] = drug_ukb_ind.groupby('eid')['statin_0'].transform(max)
drug_ukb_ind['statin_1'] = drug_ukb_ind.groupby('eid')['statin_1'].transform(max)

#ASA
drug_ukb_ind['aspirin'] = np.where(
    drug_ukb_ind['treatmentmedication_code_f20003'].isin([
        1140861806,1140864860,1140868226, #Aspirin 75mg, Nu-Seals 75mg, Aspirin
        1141164044,1141167844,1140925942,1141177826, #Isosorbide mononitrate+Aspirin, Dipyridamole+Aspirin, Caprin 75mg, Micropirin 75mg
        1140861808,1141151924,1140917408,1140861804 #Disprin CV 100mg, Enprin 75mg, Postmi 75mg, Angettes 75mg
        ]),1,0)
drug_ukb_ind['aspirin_0'] = np.where((drug_ukb_ind['instance']==str(0)) & (drug_ukb_ind['aspirin']==1),1,0)
drug_ukb_ind['aspirin_1'] = np.where((drug_ukb_ind['instance']==str(1)) & (drug_ukb_ind['aspirin']==1),1,0)
drug_ukb_ind['aspirin_0'] = drug_ukb_ind.groupby('eid')['aspirin_0'].transform(max)
drug_ukb_ind['aspirin_1'] = drug_ukb_ind.groupby('eid')['aspirin_1'].transform(max)
drug_ukb_ind.drop_duplicates(subset='eid', inplace=True)
drug_ukb_ind = drug_ukb_ind[['eid','statin_0','statin_1','aspirin_0','aspirin_1']]

#Medications-Categories
drug_ukb_agg = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([6153, 6177])]['col.name'].tolist()]
drug_stubname_a = ukb_key[ukb_key['field.showcase'] == 6177]['col.name'].iloc[0].rsplit('_', 2)[0] #3Cat medications
drug_stubname_b = ukb_key[ukb_key['field.showcase'] == 6153]['col.name'].iloc[0].rsplit('_', 2)[0] #5Cat medications
drug_ukb_agg = pd.wide_to_long(df=drug_ukb_agg, stubnames=[drug_stubname_a, drug_stubname_b],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
drug_ukb_agg[['instance','array']] = drug_ukb_agg['instance_array'].str.split("_", expand=True)
drug_a_coding = dict({-1:np.nan,-3:np.nan,-7:0,1:'llt',2:'anti_htn',3:'insulin'})
drug_ukb_agg.replace({'medication_for_cholesterol_blood_pressure_or_diabetes_f6177':drug_a_coding},inplace=True)
drug_b_coding = dict({-1:np.nan,-3:np.nan,-7:0,1:'llt',2:'anti_htn',3:'insulin',4:'hrt',5:'ocp'})
drug_ukb_agg.replace({'medication_for_cholesterol_blood_pressure_diabetes_or_take_exogenous_hormones_f6153':drug_b_coding},
                        inplace=True)

def llt(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='llt')|(row[drug_stubname_b]=='llt'):
        return 1
    else:
        return 0
def anti_htn(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='anti_htn')|(row[drug_stubname_b]=='anti_htn'):
        return 1
    else:
        return 0
def insulin(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='insulin')|(row[drug_stubname_b]=='insulin'):
        return 1
    else:
        return 0
def hrt(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='hrt')|(row[drug_stubname_b]=='hrt'):
        return 1
    else:
        return 0
def ocp(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='ocp')|(row[drug_stubname_b]=='ocp'):
        return 1
    else:
        return 0
def hormone_est_prog(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='ocp')|(row[drug_stubname_b]=='ocp')|(row[drug_stubname_a]=='hrt')|(row[drug_stubname_b]=='hrt'):
        return 1
    else:
        return 0

drug_ukb_agg['llt_0'] = drug_ukb_agg.apply(lambda row: llt(row) if row['instance']==str(0) else None, axis=1)
drug_ukb_agg['llt_1'] = drug_ukb_agg.apply(lambda row: llt(row) if row['instance']==str(1) else None, axis=1)
drug_ukb_agg['anti_htn_0'] = drug_ukb_agg.apply(lambda row: anti_htn(row) if row['instance']==str(0) else None, axis=1)
drug_ukb_agg['anti_htn_1'] = drug_ukb_agg.apply(lambda row: anti_htn(row) if row['instance']==str(1) else None, axis=1)
drug_ukb_agg['insulin_0'] = drug_ukb_agg.apply(lambda row: insulin(row) if row['instance']==str(0) else None, axis=1)
drug_ukb_agg['insulin_1'] = drug_ukb_agg.apply(lambda row: insulin(row) if row['instance']==str(1) else None, axis=1)
drug_ukb_agg['hrt_0'] = drug_ukb_agg.apply(lambda row: hrt(row) if row['instance']==str(0) else None, axis=1)
drug_ukb_agg['hrt_1'] = drug_ukb_agg.apply(lambda row: hrt(row) if row['instance']==str(1) else None, axis=1)
drug_ukb_agg['ocp_0'] = drug_ukb_agg.apply(lambda row: ocp(row) if row['instance']==str(0) else None, axis=1)
drug_ukb_agg['ocp_1'] = drug_ukb_agg.apply(lambda row: ocp(row) if row['instance']==str(1) else None, axis=1)
drug_ukb_agg['hormone_est_prog_0'] = drug_ukb_agg.apply(lambda row: hormone_est_prog(row) if row['instance']==str(0) else None, axis=1)
drug_ukb_agg['hormone_est_prog_1'] = drug_ukb_agg.apply(lambda row: hormone_est_prog(row) if row['instance']==str(1) else None, axis=1)

for i in ['llt', 'anti_htn', 'insulin', 'hrt', 'ocp', 'hormone_est_prog']:
    drug_ukb_agg[i + '_0'] = drug_ukb_agg.groupby('eid')[i + '_0'].transform(max)
    drug_ukb_agg[i + '_1'] = drug_ukb_agg.groupby('eid')[i + '_1'].transform(max)

drug_ukb_agg.drop_duplicates(subset='eid', inplace=True)
drug_ukb_agg.drop(columns=['instance_array', drug_stubname_a, drug_stubname_b, 'instance', 'array'], inplace=True)

drug_ukb_data = drug_ukb_ind.merge(drug_ukb_agg, on='eid', how='outer')
drug_ukb_data

# %%
#Family History
familyhis_ukb_data = ukb_data[['eid'] + 
                    ukb_key[ukb_key['field.showcase'].isin([20107,20110,20111])]['col.name'].tolist()]
father_stubname = ukb_key[ukb_key['field.showcase'] == 20107]['col.name'].iloc[0].rsplit('_', 2)[0]
mother_stubname = ukb_key[ukb_key['field.showcase'] == 20110]['col.name'].iloc[0].rsplit('_', 2)[0]
sibling_stubname = ukb_key[ukb_key['field.showcase'] == 20111]['col.name'].iloc[0].rsplit('_', 2)[0]

long_familyhis_ukb_data = pd.wide_to_long(df=familyhis_ukb_data, stubnames=[father_stubname, mother_stubname, sibling_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
familyhis_data = long_familyhis_ukb_data.drop(columns='instance_array')
for i in [father_stubname, mother_stubname, sibling_stubname]:
    familyhis_data[i].replace({-11:np.nan, -13:np.nan, -21:np.nan, -23:np.nan}, inplace=True)
def cvd_family_history(row):
    if (pd.isna(row[father_stubname]))&(pd.isna(row[mother_stubname]))&(pd.isna(row[sibling_stubname])):
        return np.nan
    elif (row[father_stubname]==1)|(row[father_stubname]==2)|(row[mother_stubname]==1)|(row[
        mother_stubname]==2)|(row[sibling_stubname]==1)|(row[sibling_stubname]==2):
        return 1
    else:
        return 0
familyhis_data['cvd_family_history'] = familyhis_data.apply(lambda row: cvd_family_history(row), axis=1)
familyhis_data['cvd_family_history'] = familyhis_data.groupby('eid')['cvd_family_history'].transform(max)
familyhis_data.drop(columns=[father_stubname, mother_stubname, sibling_stubname], inplace=True)
familyhis_data.drop_duplicates(subset='eid', inplace=True)
familyhis_data

# %%
#Lab
lab_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([30790,30796,#Lp(a)
                                                                          30780,30760,30690,30870,#Lipid
                                                                          30710, #CRP
                                                                          30740,30750, #DM
                                                                          30670,30700, #Renal
                                                                          30620,30610,30650,30660,30840, #Liver
                                                                          30600,30860, #Albumin, Total Protein
                                                                          30020,30000,30080,30040, #CBC
                                                                          30626,30606,30616,30656,30716,30696,30706,30666,
                                                                          30746,30756,30766,30786,30846,30866,30876,30676,
                                                                          30896,30890,30810,30680,30816,30686
                                                                          ])]['col.name'].tolist()]

data_coding_4917 = dict({1:'Reportable',2:'low',3:'high',4:'low',5:'high'})
lab_ukb_data.replace({'lipoprotein_a_reportability_f30796_0_0':data_coding_4917,
        'lipoprotein_a_reportability_f30796_1_0':data_coding_4917,
        'albumin_reportability_f30606_0_0':data_coding_4917,
       'albumin_reportability_f30606_1_0':data_coding_4917,
       'alkaline_phosphatase_reportability_f30616_0_0':data_coding_4917,
       'alkaline_phosphatase_reportability_f30616_1_0':data_coding_4917,
       'alanine_aminotransferase_reportability_f30626_0_0':data_coding_4917,
       'alanine_aminotransferase_reportability_f30626_1_0':data_coding_4917,
       'aspartate_aminotransferase_reportability_f30656_0_0':data_coding_4917,
       'aspartate_aminotransferase_reportability_f30656_1_0':data_coding_4917,
       'direct_bilirubin_reportability_f30666_0_0':data_coding_4917,
       'direct_bilirubin_reportability_f30666_1_0':data_coding_4917,
       'urea_reportability_f30676_0_0':data_coding_4917, 'urea_reportability_f30676_1_0':data_coding_4917,
       'cholesterol_reportability_f30696_0_0':data_coding_4917,
       'cholesterol_reportability_f30696_1_0':data_coding_4917,
       'creatinine_reportability_f30706_0_0':data_coding_4917,
       'creatinine_reportability_f30706_1_0':data_coding_4917,
       'creactive_protein_reportability_f30716_0_0':data_coding_4917,
       'creactive_protein_reportability_f30716_1_0':data_coding_4917,
       'glucose_reportability_f30746_0_0':data_coding_4917, 'glucose_reportability_f30746_1_0':data_coding_4917,
       'glycated_haemoglobin_hba1c_reportability_f30756_0_0':data_coding_4917,
       'glycated_haemoglobin_hba1c_reportability_f30756_1_0':data_coding_4917,
       'hdl_cholesterol_reportability_f30766_0_0':data_coding_4917,
       'hdl_cholesterol_reportability_f30766_1_0':data_coding_4917,
       'ldl_direct_reportability_f30786_0_0':data_coding_4917,
       'ldl_direct_reportability_f30786_1_0':data_coding_4917, 'total_bilirubin_reportability_f30846_0_0':data_coding_4917,
       'total_bilirubin_reportability_f30846_1_0':data_coding_4917,
       'total_protein_reportability_f30866_0_0':data_coding_4917,
       'total_protein_reportability_f30866_1_0':data_coding_4917,
       'triglycerides_reportability_f30876_0_0':data_coding_4917,
       'triglycerides_reportability_f30876_1_0':data_coding_4917,'vitamin_d_reportability_f30896_0_0':data_coding_4917,
       'vitamin_d_reportability_f30896_1_0':data_coding_4917}, inplace=True)

lab_ukb_data.rename(columns={'lipoprotein_a_f30790_0_0':'lipoprotein_a_0','lipoprotein_a_f30790_1_0':'lipoprotein_a_1',
                             'cholesterol_f30690_0_0':'cholesterol_0','cholesterol_f30690_1_0':'cholesterol_1',
                             'hdl_cholesterol_f30760_0_0':'hdl_0','hdl_cholesterol_f30760_1_0':'hdl_1',
                             'ldl_direct_f30780_0_0':'ldl_0','ldl_direct_f30780_1_0':'ldl_1',
                             'triglycerides_f30870_0_0':'triglycerides_0','triglycerides_f30870_1_0':'triglycerides_1',
                             'glucose_f30740_0_0':'glucose_0','glucose_f30740_1_0':'glucose_1', 
                             'glycated_haemoglobin_hba1c_f30750_0_0':'hba1c_0',
                             'glycated_haemoglobin_hba1c_f30750_1_0':'hba1c_1',
                             'urea_f30670_0_0':'urea_0','urea_f30670_1_0':'urea_1',
                             'creatinine_f30700_0_0':'creatinine_0','creatinine_f30700_1_0':'creatinine_1',
                             'alkaline_phosphatase_f30610_0_0':'alp_0','alkaline_phosphatase_f30610_1_0':'alp_1',
                             'alanine_aminotransferase_f30620_0_0':'alt_0','alanine_aminotransferase_f30620_1_0':'alt_1',
                             'aspartate_aminotransferase_f30650_0_0':'ast_0','aspartate_aminotransferase_f30650_1_0':'ast_1',
                             'direct_bilirubin_f30660_0_0':'bil_d_0','direct_bilirubin_f30660_1_0':'bil_d_1',
                             'total_bilirubin_f30840_0_0':'bil_t_0','total_bilirubin_f30840_1_0':'bil_t_1',
                             'creactive_protein_f30710_0_0':'creactive_protein_0','creactive_protein_f30710_1_0':'creactive_protein_1',
                             'albumin_f30600_0_0':'albumin_0', 'albumin_f30600_1_0':'albumin_1',
                             'total_protein_f30860_0_0':'total_protein_0','total_protein_f30860_1_0':'total_protein_1',
                             'vitamin_d_f30890_0_0':'vitamin_d_0', 'vitamin_d_f30890_1_0':'vitamin_d_1',
                             'calcium_f30680_0_0':'calcium_0','calcium_f30680_1_0':'calcium_1',
                             'phosphate_f30810_0_0':'phosphate_0', 'phosphate_f30810_1_0':'phosphate_1',
                             'white_blood_cell_leukocyte_count_f30000_0_0':'wbc_0',
                             'white_blood_cell_leukocyte_count_f30000_1_0':'wbc_1',
                             'white_blood_cell_leukocyte_count_f30000_2_0':'wbc_2',
                             'haemoglobin_concentration_f30020_0_0':'hb_0','haemoglobin_concentration_f30020_1_0':'hb_1',
                             'haemoglobin_concentration_f30020_2_0':'hb_2','mean_corpuscular_volume_f30040_0_0':'mcv_0',
                             'mean_corpuscular_volume_f30040_1_0':'mcv_1','mean_corpuscular_volume_f30040_2_0':'mcv_2',
                             'platelet_count_f30080_0_0':'plt_0','platelet_count_f30080_1_0':'plt_1', 'platelet_count_f30080_2_0':'plt_2'},
                             inplace=True)


#Reducing missingness by reportability variables
lab_ukb_data['lp_a_0_value'] = np.where((lab_ukb_data['lipoprotein_a_0'].isna()) & (lab_ukb_data['lipoprotein_a_reportability_f30796_0_0']=='low'), 
                    3.8, np.where((lab_ukb_data['lipoprotein_a_0'].isna()) & (lab_ukb_data['lipoprotein_a_reportability_f30796_0_0']=='high'),
                    189, lab_ukb_data['lipoprotein_a_0']))
lab_ukb_data['lp_a_1_value'] = np.where((lab_ukb_data['lipoprotein_a_1'].isna()) & (lab_ukb_data['lipoprotein_a_reportability_f30796_1_0']=='low'), 
                    3.8, np.where((lab_ukb_data['lipoprotein_a_1'].isna()) & (lab_ukb_data['lipoprotein_a_reportability_f30796_1_0']=='high'),
                    189, lab_ukb_data['lipoprotein_a_1']))
lab_ukb_data['albumin_0'] = np.where((lab_ukb_data['albumin_0'].isna()) & (lab_ukb_data['albumin_reportability_f30606_0_0']=='low'), 
                    17.38, np.where((lab_ukb_data['albumin_0'].isna()) & (lab_ukb_data['albumin_reportability_f30606_0_0']=='high'),
                    59.8, lab_ukb_data['albumin_0']))
lab_ukb_data['albumin_1'] = np.where((lab_ukb_data['albumin_1'].isna()) & (lab_ukb_data['albumin_reportability_f30606_1_0']=='low'), 
                    17.38, np.where((lab_ukb_data['albumin_1'].isna()) & (lab_ukb_data['albumin_reportability_f30606_1_0']=='high'),
                    59.8, lab_ukb_data['albumin_1']))
lab_ukb_data['alp_0'] = np.where((lab_ukb_data['alp_0'].isna()) & (lab_ukb_data['alkaline_phosphatase_reportability_f30616_0_0']=='low'), 
                    8, np.where((lab_ukb_data['alp_0'].isna()) & (lab_ukb_data['alkaline_phosphatase_reportability_f30616_0_0']=='high'),
                    1416.7, lab_ukb_data['alp_0']))
lab_ukb_data['alp_1'] = np.where((lab_ukb_data['alp_1'].isna()) & (lab_ukb_data['alkaline_phosphatase_reportability_f30616_1_0']=='low'), 
                    8, np.where((lab_ukb_data['alp_1'].isna()) & (lab_ukb_data['alkaline_phosphatase_reportability_f30616_1_0']=='high'),
                    1416.7, lab_ukb_data['alp_1']))
lab_ukb_data['alt_0'] = np.where((lab_ukb_data['alt_0'].isna()) & (lab_ukb_data['alanine_aminotransferase_reportability_f30626_0_0']=='low'), 
                    3.01, np.where((lab_ukb_data['alt_0'].isna()) & (lab_ukb_data['alanine_aminotransferase_reportability_f30626_0_0']=='high'),
                    495.19, lab_ukb_data['alt_0']))
lab_ukb_data['alt_1'] = np.where((lab_ukb_data['alt_1'].isna()) & (lab_ukb_data['alanine_aminotransferase_reportability_f30626_1_0']=='low'), 
                    3.01, np.where((lab_ukb_data['alt_1'].isna()) & (lab_ukb_data['alanine_aminotransferase_reportability_f30626_1_0']=='high'),
                    495.19, lab_ukb_data['alt_1']))
lab_ukb_data['ast_0'] = np.where((lab_ukb_data['ast_0'].isna()) & (lab_ukb_data['aspartate_aminotransferase_reportability_f30656_0_0']=='low'), 
                    3.3, np.where((lab_ukb_data['ast_0'].isna()) & (lab_ukb_data['aspartate_aminotransferase_reportability_f30656_0_0']=='high'),
                    947.2, lab_ukb_data['ast_0']))
lab_ukb_data['ast_1'] = np.where((lab_ukb_data['ast_1'].isna()) & (lab_ukb_data['aspartate_aminotransferase_reportability_f30656_1_0']=='low'), 
                    3.3, np.where((lab_ukb_data['ast_1'].isna()) & (lab_ukb_data['aspartate_aminotransferase_reportability_f30656_1_0']=='high'),
                    947.2, lab_ukb_data['ast_1']))
lab_ukb_data['bil_d_0'] = np.where((lab_ukb_data['bil_d_0'].isna()) & (lab_ukb_data['direct_bilirubin_reportability_f30666_0_0']=='low'), 
                    1, np.where((lab_ukb_data['bil_d_0'].isna()) & (lab_ukb_data['direct_bilirubin_reportability_f30666_0_0']=='high'),
                    70.06, lab_ukb_data['bil_d_0']))
lab_ukb_data['bil_d_1'] = np.where((lab_ukb_data['bil_d_1'].isna()) & (lab_ukb_data['direct_bilirubin_reportability_f30666_1_0']=='low'), 
                    1, np.where((lab_ukb_data['bil_d_1'].isna()) & (lab_ukb_data['direct_bilirubin_reportability_f30666_1_0']=='high'),
                    70.06, lab_ukb_data['bil_d_1']))
lab_ukb_data['urea_0'] = np.where((lab_ukb_data['urea_0'].isna()) & (lab_ukb_data['urea_reportability_f30676_0_0']=='low'), 
                    0.81, np.where((lab_ukb_data['urea_0'].isna()) & (lab_ukb_data['urea_reportability_f30676_0_0']=='high'),
                    41.83, lab_ukb_data['urea_0']))
lab_ukb_data['urea_1'] = np.where((lab_ukb_data['urea_1'].isna()) & (lab_ukb_data['urea_reportability_f30676_1_0']=='low'), 
                    0.81, np.where((lab_ukb_data['urea_1'].isna()) & (lab_ukb_data['urea_reportability_f30676_1_0']=='high'),
                    41.83, lab_ukb_data['urea_1']))
lab_ukb_data['cholesterol_0'] = np.where((lab_ukb_data['cholesterol_0'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_0_0']=='low'), 
                    0.601, np.where((lab_ukb_data['cholesterol_0'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_0_0']=='high'),
                    15.46, lab_ukb_data['cholesterol_0']))
lab_ukb_data['cholesterol_1'] = np.where((lab_ukb_data['cholesterol_1'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_1_0']=='low'), 
                    0.601, np.where((lab_ukb_data['cholesterol_1'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_1_0']=='high'),
                    15.46, lab_ukb_data['cholesterol_1']))
lab_ukb_data['creatinine_0'] = np.where((lab_ukb_data['creatinine_0'].isna()) & (lab_ukb_data['creatinine_reportability_f30706_0_0']=='low'), 
                    10.7, np.where((lab_ukb_data['creatinine_0'].isna()) & (lab_ukb_data['creatinine_reportability_f30706_0_0']=='high'),
                    1499.3, lab_ukb_data['creatinine_0']))
lab_ukb_data['creatinine_1'] = np.where((lab_ukb_data['creatinine_1'].isna()) & (lab_ukb_data['creatinine_reportability_f30706_1_0']=='low'), 
                    10.7, np.where((lab_ukb_data['creatinine_1'].isna()) & (lab_ukb_data['creatinine_reportability_f30706_1_0']=='high'),
                    1499.3, lab_ukb_data['creatinine_1']))
lab_ukb_data['creactive_protein_0'] = np.where((lab_ukb_data['creactive_protein_0'].isna()) & (lab_ukb_data['creactive_protein_reportability_f30716_0_0']=='low'), 
                    0.08, np.where((lab_ukb_data['creactive_protein_0'].isna()) & (lab_ukb_data['creactive_protein_reportability_f30716_0_0']=='high'),
                    79.96, lab_ukb_data['creactive_protein_0']))
lab_ukb_data['creactive_protein_1'] = np.where((lab_ukb_data['creactive_protein_1'].isna()) & (lab_ukb_data['creactive_protein_reportability_f30716_1_0']=='low'), 
                    0.08, np.where((lab_ukb_data['creactive_protein_1'].isna()) & (lab_ukb_data['creactive_protein_reportability_f30716_1_0']=='high'),
                    79.96, lab_ukb_data['creactive_protein_1']))
lab_ukb_data['glucose_0'] = np.where((lab_ukb_data['glucose_0'].isna()) & (lab_ukb_data['glucose_reportability_f30746_0_0']=='low'), 
                    0.995, np.where((lab_ukb_data['glucose_0'].isna()) & (lab_ukb_data['glucose_reportability_f30746_0_0']=='high'),
                    36.813, lab_ukb_data['glucose_0']))
lab_ukb_data['glucose_1'] = np.where((lab_ukb_data['glucose_1'].isna()) & (lab_ukb_data['glucose_reportability_f30746_1_0']=='low'), 
                    0.995, np.where((lab_ukb_data['glucose_1'].isna()) & (lab_ukb_data['glucose_reportability_f30746_1_0']=='high'),
                    36.813, lab_ukb_data['glucose_1']))
lab_ukb_data['hba1c_0'] = np.where((lab_ukb_data['hba1c_0'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_0_0']=='low'), 
                    15, np.where((lab_ukb_data['hba1c_0'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_0_0']=='high'),
                    515.2, lab_ukb_data['hba1c_0']))
lab_ukb_data['hba1c_1'] = np.where((lab_ukb_data['hba1c_1'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_1_0']=='low'), 
                    15, np.where((lab_ukb_data['hba1c_1'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_1_0']=='high'),
                    515.2, lab_ukb_data['hba1c_1']))
lab_ukb_data['hdl_0'] = np.where((lab_ukb_data['hdl_0'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_0_0']=='low'), 
                    0.219, np.where((lab_ukb_data['hdl_0'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_0_0']=='high'),
                    4.401, lab_ukb_data['hdl_0']))
lab_ukb_data['hdl_1'] = np.where((lab_ukb_data['hdl_1'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_1_0']=='low'), 
                    0.219, np.where((lab_ukb_data['hdl_1'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_1_0']=='high'),
                    4.401, lab_ukb_data['hdl_1']))
lab_ukb_data['ldl_0'] = np.where((lab_ukb_data['ldl_0'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_0_0']=='low'), 
                    0.219, np.where((lab_ukb_data['ldl_0'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_0_0']=='high'),
                    4.401, lab_ukb_data['ldl_0']))
lab_ukb_data['ldl_1'] = np.where((lab_ukb_data['ldl_1'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_1_0']=='low'), 
                    0.219, np.where((lab_ukb_data['ldl_1'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_1_0']=='high'),
                    4.401, lab_ukb_data['ldl_1']))
lab_ukb_data['bil_t_0'] = np.where((lab_ukb_data['bil_t_0'].isna()) & (lab_ukb_data['total_bilirubin_reportability_f30846_0_0']=='low'), 
                    1.08, np.where((lab_ukb_data['bil_t_0'].isna()) & (lab_ukb_data['total_bilirubin_reportability_f30846_0_0']=='high'),
                    144.52, lab_ukb_data['bil_t_0']))
lab_ukb_data['bil_t_1'] = np.where((lab_ukb_data['bil_t_1'].isna()) & (lab_ukb_data['total_bilirubin_reportability_f30846_1_0']=='low'), 
                    1.08, np.where((lab_ukb_data['bil_t_1'].isna()) & (lab_ukb_data['total_bilirubin_reportability_f30846_1_0']=='high'),
                    144.52, lab_ukb_data['bil_t_1']))
lab_ukb_data['total_protein_0'] = np.where((lab_ukb_data['total_protein_0'].isna()) & (lab_ukb_data['total_protein_reportability_f30866_0_0']=='low'), 
                    36.27, np.where((lab_ukb_data['total_protein_0'].isna()) & (lab_ukb_data['total_protein_reportability_f30866_0_0']=='high'),
                    117.36, lab_ukb_data['total_protein_0']))
lab_ukb_data['total_protein_1'] = np.where((lab_ukb_data['total_protein_1'].isna()) & (lab_ukb_data['total_protein_reportability_f30866_1_0']=='low'), 
                    36.27, np.where((lab_ukb_data['total_protein_1'].isna()) & (lab_ukb_data['total_protein_reportability_f30866_1_0']=='high'),
                    117.36, lab_ukb_data['total_protein_1']))
lab_ukb_data['triglycerides_0'] = np.where((lab_ukb_data['triglycerides_0'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_0_0']=='low'), 
                    0.231, np.where((lab_ukb_data['triglycerides_0'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_0_0']=='high'),
                    11.278, lab_ukb_data['triglycerides_0']))
lab_ukb_data['triglycerides_1'] = np.where((lab_ukb_data['triglycerides_1'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_1_0']=='low'), 
                    0.231, np.where((lab_ukb_data['triglycerides_1'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_1_0']=='high'),
                    11.278, lab_ukb_data['triglycerides_1']))
lab_ukb_data['vitamin_d_0'] = np.where((lab_ukb_data['vitamin_d_0'].isna()) & (lab_ukb_data['vitamin_d_reportability_f30896_0_0']=='low'), 
                    10, np.where((lab_ukb_data['vitamin_d_0'].isna()) & (lab_ukb_data['vitamin_d_reportability_f30896_0_0']=='high'),
                    362, lab_ukb_data['vitamin_d_0']))
lab_ukb_data['vitamin_d_1'] = np.where((lab_ukb_data['vitamin_d_1'].isna()) & (lab_ukb_data['vitamin_d_reportability_f30896_1_0']=='low'), 
                    10, np.where((lab_ukb_data['vitamin_d_1'].isna()) & (lab_ukb_data['vitamin_d_reportability_f30896_1_0']=='high'),
                    362, lab_ukb_data['vitamin_d_1']))
lab_ukb_data['calcium_0'] = np.where((lab_ukb_data['calcium_0'].isna()) & (lab_ukb_data['calcium_reportability_f30686_0_0']=='low'), 
                    1.05, np.where((lab_ukb_data['calcium_0'].isna()) & (lab_ukb_data['calcium_reportability_f30686_0_0']=='high'),
                    3.611, lab_ukb_data['calcium_0']))
lab_ukb_data['calcium_1'] = np.where((lab_ukb_data['calcium_1'].isna()) & (lab_ukb_data['calcium_reportability_f30686_1_0']=='low'), 
                    1.05, np.where((lab_ukb_data['calcium_1'].isna()) & (lab_ukb_data['calcium_reportability_f30686_1_0']=='high'),
                    3.611, lab_ukb_data['calcium_1']))
lab_ukb_data['phosphate_0'] = np.where((lab_ukb_data['phosphate_0'].isna()) & (lab_ukb_data['phosphate_reportability_f30816_0_0']=='low'), 
                    0.369, np.where((lab_ukb_data['phosphate_0'].isna()) & (lab_ukb_data['phosphate_reportability_f30816_0_0']=='high'),
                    4.702, lab_ukb_data['phosphate_0']))
lab_ukb_data['phosphate_1'] = np.where((lab_ukb_data['phosphate_1'].isna()) & (lab_ukb_data['phosphate_reportability_f30816_1_0']=='low'), 
                    0.369, np.where((lab_ukb_data['phosphate_1'].isna()) & (lab_ukb_data['phosphate_reportability_f30816_1_0']=='high'),
                    4.702, lab_ukb_data['phosphate_1']))
                    
lab_ukb_data.drop(columns=['lipoprotein_a_reportability_f30796_0_0', 'lipoprotein_a_reportability_f30796_1_0',
       'albumin_reportability_f30606_0_0', 'albumin_reportability_f30606_1_0',
       'alkaline_phosphatase_reportability_f30616_0_0', 'alkaline_phosphatase_reportability_f30616_1_0',
       'alanine_aminotransferase_reportability_f30626_0_0', 'alanine_aminotransferase_reportability_f30626_1_0',
       'aspartate_aminotransferase_reportability_f30656_0_0', 'aspartate_aminotransferase_reportability_f30656_1_0',
       'direct_bilirubin_reportability_f30666_0_0', 'direct_bilirubin_reportability_f30666_1_0',
       'urea_reportability_f30676_0_0', 'urea_reportability_f30676_1_0',
       'cholesterol_reportability_f30696_0_0', 'cholesterol_reportability_f30696_1_0',
       'creatinine_reportability_f30706_0_0', 'creatinine_reportability_f30706_1_0',
       'creactive_protein_reportability_f30716_0_0', 'creactive_protein_reportability_f30716_1_0',
       'glucose_reportability_f30746_0_0', 'glucose_reportability_f30746_1_0',
       'glycated_haemoglobin_hba1c_reportability_f30756_0_0', 'glycated_haemoglobin_hba1c_reportability_f30756_1_0',
       'hdl_cholesterol_reportability_f30766_0_0', 'hdl_cholesterol_reportability_f30766_1_0',
       'ldl_direct_reportability_f30786_0_0', 'ldl_direct_reportability_f30786_1_0',
       'total_bilirubin_reportability_f30846_0_0', 'total_bilirubin_reportability_f30846_1_0',
       'total_protein_reportability_f30866_0_0', 'total_protein_reportability_f30866_1_0',
       'triglycerides_reportability_f30876_0_0', 'triglycerides_reportability_f30876_1_0', 
       'vitamin_d_reportability_f30896_0_0', 'vitamin_d_reportability_f30896_1_0',
       'calcium_reportability_f30686_0_0', 'calcium_reportability_f30686_1_0',
       'phosphate_reportability_f30816_0_0', 'phosphate_reportability_f30816_1_0'], inplace=True)
lab_ukb_data

# %%
#Vitals
vital_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([4079,4080,93,94,95,102
                                                                           ])]['col.name'].tolist()]

sbp_auto_stubname = ukb_key[ukb_key['field.showcase'] == 4080]['col.name'].iloc[0].rsplit('_', 2)[0]
sbp_manual_stubname = ukb_key[ukb_key['field.showcase'] == 93]['col.name'].iloc[0].rsplit('_', 2)[0]
dbp_auto_stubname = ukb_key[ukb_key['field.showcase'] == 4079]['col.name'].iloc[0].rsplit('_', 2)[0]
dbp_manual_stubname = ukb_key[ukb_key['field.showcase'] == 94]['col.name'].iloc[0].rsplit('_', 2)[0]
hr_auto_stubname = ukb_key[ukb_key['field.showcase'] == 102]['col.name'].iloc[0].rsplit('_', 2)[0]
hr_manual_stubname = ukb_key[ukb_key['field.showcase'] == 95]['col.name'].iloc[0].rsplit('_', 2)[0]
vital_ukb_data = pd.wide_to_long(df=vital_ukb_data, stubnames=[sbp_auto_stubname,sbp_manual_stubname,dbp_auto_stubname,
                                                               dbp_manual_stubname,hr_auto_stubname,hr_manual_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
vital_ukb_data[['instance','array']] = vital_ukb_data['instance_array'].str.split("_", expand=True)

vital_ukb_data['sbp'] = np.where(vital_ukb_data['systolic_blood_pressure_automated_reading_f4080'].notna(),
    vital_ukb_data['systolic_blood_pressure_automated_reading_f4080'], vital_ukb_data['systolic_blood_pressure_manual_reading_f93'])
vital_ukb_data['dbp'] = np.where(vital_ukb_data['diastolic_blood_pressure_automated_reading_f4079'].notna(),
    vital_ukb_data['diastolic_blood_pressure_automated_reading_f4079'], vital_ukb_data['diastolic_blood_pressure_manual_reading_f94'])
vital_ukb_data['hr'] = np.where(vital_ukb_data['pulse_rate_automated_reading_f102'].notna(),
    vital_ukb_data['pulse_rate_automated_reading_f102'], vital_ukb_data['pulse_rate_during_bloodpressure_measurement_f95'])

vital_ukb_data.drop(columns=['instance_array','systolic_blood_pressure_automated_reading_f4080',
                             'systolic_blood_pressure_manual_reading_f93','diastolic_blood_pressure_automated_reading_f4079',
                             'diastolic_blood_pressure_manual_reading_f94','pulse_rate_automated_reading_f102',
                             'pulse_rate_during_bloodpressure_measurement_f95'], inplace=True)

vital_ukb_data = vital_ukb_data[vital_ukb_data['instance'].isin([str(0),str(1)])]

vital_ukb_data['sbp_0'] = vital_ukb_data[vital_ukb_data['instance']==str(0)].groupby('eid')['sbp'].transform('mean')
vital_ukb_data['sbp_1'] = vital_ukb_data[vital_ukb_data['instance']==str(1)].groupby('eid')['sbp'].transform('mean')
vital_ukb_data['dbp_0'] = vital_ukb_data[vital_ukb_data['instance']==str(0)].groupby('eid')['dbp'].transform('mean')
vital_ukb_data['dbp_1'] = vital_ukb_data[vital_ukb_data['instance']==str(1)].groupby('eid')['dbp'].transform('mean')
vital_ukb_data['hr_0'] = vital_ukb_data[vital_ukb_data['instance']==str(0)].groupby('eid')['hr'].transform('mean')
vital_ukb_data['hr_1'] = vital_ukb_data[vital_ukb_data['instance']==str(1)].groupby('eid')['hr'].transform('mean')

vital_ukb_data['sbp_0'] = vital_ukb_data.groupby('eid')['sbp_0'].transform('mean')
vital_ukb_data['sbp_1'] = vital_ukb_data.groupby('eid')['sbp_1'].transform('mean')
vital_ukb_data['dbp_0'] = vital_ukb_data.groupby('eid')['dbp_0'].transform('mean')
vital_ukb_data['dbp_1'] = vital_ukb_data.groupby('eid')['dbp_1'].transform('mean')
vital_ukb_data['hr_0'] = vital_ukb_data.groupby('eid')['hr_0'].transform('mean')
vital_ukb_data['hr_1'] = vital_ukb_data.groupby('eid')['hr_1'].transform('mean')

vital_ukb_data.drop_duplicates(subset='eid', inplace=True)
vital_ukb_data.drop(columns=['instance','array','sbp','dbp','hr'], inplace=True)

vital_ukb_data

# %%
#Merging datasets
ukb = demo_ukb_data.merge(lifestyle_ukb_data, how='left')
ukb = ukb.merge(pmh_ukb_indx, how='left', on='eid')
ukb = ukb.merge(pmh_ukb_maindx, how='left', on='eid')
ukb = ukb.merge(opcs_data, how='left', on='eid')
ukb = ukb.merge(familyhis_data, how='left', on='eid')
ukb = ukb.merge(drug_ukb_data, how='left', on='eid')
ukb = ukb.merge(lab_ukb_data, how='left', on='eid')
ukb = ukb.merge(vital_ukb_data, how='left', on='eid')
ukb['ascvd_0'] = np.where((ukb['ihd_0']==1)|(ukb['pad_0']==1)|(ukb['stroke_0']==1)|(ukb['cabg_0']==1)|
                          (ukb['pci_0']==1)|(ukb['carotid_revasc_0']==1)|(ukb['pad_revasc_0']==1), 1, 0)
ukb['ascvd_1'] = np.where((ukb['ihd_1']==1)|(ukb['pad_1']==1)|(ukb['stroke_1']==1)|(ukb['cabg_1']==1)|
                          (ukb['pci_1']==1)|(ukb['carotid_revasc_1']==1)|(ukb['pad_revasc_1']==1), 1, 0)

ukb['premature_ascvd_cvpx_0'] = np.where((ukb['premature_ascvd_0']==1)|(ukb['premature_cv_procedure_0']==1), 1, 0)
ukb['premature_ascvd_cvpx_1'] = np.where((ukb['premature_ascvd_1']==1)|(ukb['premature_cv_procedure_1']==1), 1, 0)

ukb['age_at_first_ascvd_cvpx_0'] = ukb[['age_at_first_ascvd_0', 'age_at_first_cv_procedure_0']].min(axis=1).fillna(0)
ukb['age_at_first_ascvd_cvpx_1'] = ukb[['age_at_first_ascvd_1', 'age_at_first_cv_procedure_1']].min(axis=1).fillna(0)

ukb.drop(columns=['visit_0', 'visit_1'], inplace=True)
pd.set_option('display.max_columns', None)
ukb

# %%
#Creating separate datasets for visit_0 and visit_1 Removing records with NA values for Lp(a)
ukb_0 = ukb[['eid', 'birth_date', 'sex', 'ethnicity', 'cvd_family_history', 'lost_fu_date', 'death_date', 
                  'primary_cv_mortality', 'secondary_cv_mortality', 'primary_secondary_cv_mortality'] + ukb.columns[ukb.columns.str.contains('_0')].tolist()]
ukb_1 = ukb[['eid', 'birth_date', 'sex', 'ethnicity', 'cvd_family_history', 'lost_fu_date', 'death_date', 
                  'primary_cv_mortality', 'secondary_cv_mortality', 'primary_secondary_cv_mortality'] + ukb.columns[ukb.columns.str.contains('_1')].tolist()]
#Removing records with NA values for Lp(a)
ukb_0.dropna(subset='lp_a_0_value', inplace=True)
ukb_1.dropna(subset='lp_a_1_value', inplace=True)
#Setting a column to distinct lp_a_value is from visit_0 or visit_1
ukb_0['visit'] = 0
ukb_1['visit'] = 1
#Removing _0 and _1 from column names
ukb_0.columns = ukb_0.columns.str.replace('_0', '')
ukb_1.columns = ukb_1.columns.str.replace('_1', '')
#Retaining the second visit (visit_1) information for those with Lp(a) measurement at both visit_1 and visit_0
ukb_0 = ukb_0[~ukb_0['eid'].isin(ukb_1['eid'].tolist())]
print(ukb_0.columns.tolist()==ukb_1.columns.tolist())
print('Number of patients at visit_0: ', len(ukb_0))
print('Number of patients at visit_1: ', len(ukb_1))
ukb_wf = pd.concat([ukb_0, ukb_1], ignore_index=True)
ukb_wf

# %%
ukb_wf.to_csv("/Lp(a)/ukb_wf_parsimonious.csv", index=False)

# %%
ukb_no_lpa = ukb[['eid', 'birth_date', 'sex', 'ethnicity', 'cvd_family_history', 'lost_fu_date', 'death_date', 
                  'primary_cv_mortality', 'secondary_cv_mortality', 'primary_secondary_cv_mortality'] + ukb.columns[ukb.columns.str.contains('_0')].tolist()]
ukb_no_lpa.columns = ukb_no_lpa.columns.str.replace('_0', '')
ukb_no_lpa = ukb_no_lpa[~ukb_no_lpa['eid'].isin(ukb_wf['eid'].tolist())]
ukb_comp = pd.concat([ukb_wf, ukb_no_lpa])
ukb_comp.to_csv("/Lp(a)/ukb_complete_parsimonious.csv", index=False)
ukb_comp


