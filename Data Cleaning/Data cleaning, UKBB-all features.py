# %%
# This file is for data cleaning for UK Biobank with all available features.

import pandas as pd
import numpy as np
import datatable as dt
import math
from datetime import date
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

path = "/Lp(a)/"

ukb_data_dt = dt.fread("/home/rk725/ukbdata/ukb47034.csv")
column_names_recoding = pd.read_csv(path + "colnames_my_ukb_data.csv")
ukb_key = pd.read_excel(path + "my_ukb_key_11012022.xlsx")
ukb_data_dt

# %%
ukb_data_dt.names = column_names_recoding['name'].tolist()
ukb_data_dt

# %%
ukb_data_dt = ukb_data_dt[:, ['eid','sex_f31_0_0','year_of_birth_f34_0_0','month_of_birth_f52_0_0',
                          'date_of_attending_assessment_centre_f53_0_0','date_of_attending_assessment_centre_f53_1_0',
                          'date_of_attending_assessment_centre_f53_2_0','date_of_attending_assessment_centre_f53_3_0',
                          'date_lost_to_followup_f191_0_0', 'ethnic_background_f21000_0_0','ethnic_background_f21000_1_0',
                          'ethnic_background_f21000_2_0'] + ukb_key[ukb_key['field.showcase'].isin([21001,48,49,20116,20003,
                          42000,42002,42004,42006,42008,42010,42012,42026,42014,42016,41270,41280,41271,41281,20002,20008,
                          20004,20010,41272,41282,41273,41283,30790,30796,30780,30760,30690,30870,30710,30740,30750,30670,
                          30700,30620,30610,30650,30660,30840,30600,30860,30020,30000,30080,30040,4079,4080,93,94,95,102,
                          6153,6177,
                          30626,30606,30616,30656,30716,30696,30706,30666,30746,30756,30766,30786,30846,30866,30876,30676,30896,30890,
                          30810,30680,30816,30686
                            ])]['col.name'].tolist()]
ukb_data_dt

# %%
ukb_data_dt.to_csv("/Lp(a)/ukb_lpa.csv")
ukb_data = pd.read_csv(path + "ukb_lpa.csv")
ukb_key = pd.read_excel(path + "my_ukb_key_11012022.xlsx")
ukb_data

# %%
#Demographics
demo_ukb_data = ukb_data[['eid','sex_f31_0_0','year_of_birth_f34_0_0','month_of_birth_f52_0_0',
                          'date_of_attending_assessment_centre_f53_0_0','date_of_attending_assessment_centre_f53_1_0',
                          'date_of_attending_assessment_centre_f53_2_0','date_of_attending_assessment_centre_f53_3_0',
                          'date_lost_to_followup_f191_0_0', 'ethnic_background_f21000_0_0','ethnic_background_f21000_1_0',
                          'ethnic_background_f21000_2_0']]

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
                                              'date_of_attending_assessment_centre_f53_2_0':'visit_2',
                                              'date_of_attending_assessment_centre_f53_3_0':'visit_3',
                                              'date_lost_to_followup_f191_0_0':'date_lost_to_followup'}).drop(columns = 
                                              ['year_of_birth_f34_0_0','month_of_birth_f52_0_0','ethnic_background_f21000_0_0',
                                              'ethnic_background_f21000_1_0','ethnic_background_f21000_2_0'])
demo_ukb_data = demo_ukb_data[['eid','birth_date','sex','ethnicity','visit_0','visit_1','visit_2','visit_3','date_lost_to_followup']]

for i in ['visit_0','visit_1','visit_2','visit_3']:
    demo_ukb_data[i] = pd.to_datetime(demo_ukb_data[i])

demo_ukb_data['age_0'] = (demo_ukb_data['visit_0'] - demo_ukb_data['birth_date']).dt.days/365.25

demo_ukb_data['age_1'] = (demo_ukb_data['visit_1'] - demo_ukb_data['birth_date']).dt.days/365.25


#sex_coding = dict({0:'Female',1:'Male'})
#demo_ukb_data = demo_ukb_data.replace({'sex':sex_coding})

ethnicity = dict({1:1,1001:1,1002:1,1003:1,
                  2:2,2001:2,2002:2,2003:2,2004:2,
                  3:3,3001:3,3002:3,3003:3,3004:3,
                  4:4,4001:4,4002:4,4003:4,
                  5:5,
                  6:6,
                  -1:np.nan, -3:np.nan})
demo_ukb_data = demo_ukb_data.replace({'ethnicity':ethnicity})

#ethnicity_cat = dict({1:'White',2:"Mixed",3:'South Asian',4:'Black',5:'Chinese',6:"Other"})
#demo_ukb_data = demo_ukb_data.replace({'ethnicity':ethnicity_cat})


demo_ukb_data.drop(columns=['birth_date','visit_2','visit_3','date_lost_to_followup'], inplace=True)
demo_ukb_data = demo_ukb_data[['eid','age_0','age_1','sex','ethnicity','visit_0','visit_1']]
demo_ukb_data.drop(columns=['age_1', 'visit_0', 'visit_1'],inplace=True)

# %%
#Lifestyle
lifestyle_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([21001,48,49])]['col.name'].tolist()]
bmi_stubname = ukb_key[ukb_key['field.showcase'] == 21001]['col.name'].iloc[0].rsplit('_', 2)[0]
wc_stubname = ukb_key[ukb_key['field.showcase'] == 48]['col.name'].iloc[0].rsplit('_', 2)[0]
hc_stubname = ukb_key[ukb_key['field.showcase'] == 49]['col.name'].iloc[0].rsplit('_', 2)[0]

lifestyle_ukb_data = pd.wide_to_long(df=lifestyle_ukb_data, stubnames=[bmi_stubname,wc_stubname,hc_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
lifestyle_ukb_data[['instance','array']] = lifestyle_ukb_data['instance_array'].str.split("_", expand=True)

lifestyle_ukb_data.rename(columns={'body_mass_index_bmi_f21001':'bmi','waist_circumference_f48':'waist_circumference',
                                   'hip_circumference_f49':'hip_circumference'},inplace=True)

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

lifestyle_ukb_data['bmi_0'] = lifestyle_ukb_data.groupby('eid')['bmi_0'].transform(max)
lifestyle_ukb_data['bmi_1'] = lifestyle_ukb_data.groupby('eid')['bmi_1'].transform(max)
lifestyle_ukb_data['waist_circumference_0'] = lifestyle_ukb_data.groupby('eid')['waist_circumference_0'].transform(max)
lifestyle_ukb_data['waist_circumference_1'] = lifestyle_ukb_data.groupby('eid')['waist_circumference_1'].transform(max)
lifestyle_ukb_data['hip_circumference_0'] = lifestyle_ukb_data.groupby('eid')['hip_circumference_0'].transform(max)
lifestyle_ukb_data['hip_circumference_1'] = lifestyle_ukb_data.groupby('eid')['hip_circumference_1'].transform(max)

lifestyle_ukb_data = lifestyle_ukb_data[['eid','bmi_0','bmi_1','waist_circumference_0','waist_circumference_1',
                                          'hip_circumference_0','hip_circumference_1']]

lifestyle_ukb_data.drop_duplicates(subset='eid', inplace=True)

lifestyle_ukb_data = lifestyle_ukb_data[['eid','bmi_0','waist_circumference_0','hip_circumference_0']]

# %%
#PMH-Inpatient Diagnoses
#ICD10
pmh_ukb_indx_icd10 = ukb_data[['eid', 'date_of_attending_assessment_centre_f53_0_0', 'date_of_attending_assessment_centre_f53_1_0'] + 
                        ukb_key[ukb_key['field.showcase'].isin([41270,41280])]['col.name'].tolist()]
icd10_dx_stubname = ukb_key[ukb_key['field.showcase'] == 41270]['col.name'].iloc[0].rsplit('_', 2)[0]
icd10_date_stubname = ukb_key[ukb_key['field.showcase'] == 41280]['col.name'].iloc[0].rsplit('_', 2)[0]
pmh_ukb_indx_icd10.rename(columns={'date_of_attending_assessment_centre_f53_0_0': 'visit_0',
                                   'date_of_attending_assessment_centre_f53_1_0': 'visit_1'}, inplace=True)

long_pmh_ukb_indx_icd10 = pd.wide_to_long(df=pmh_ukb_indx_icd10, stubnames=[icd10_dx_stubname, icd10_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_indx_icd10 = long_pmh_ukb_indx_icd10.drop(columns='instance_array').rename(columns={'diagnoses_icd10_f41270':'icd',
                                        'date_of_first_inpatient_diagnosis_icd10_f41280':'icd_date'})
final_pmh_ukb_indx_icd10['icd_type'] = 'icd10'
for i in ['icd_date', 'visit_0', 'visit_1']:
    final_pmh_ukb_indx_icd10[i] = pd.to_datetime(final_pmh_ukb_indx_icd10[i])

#ICD9
pmh_ukb_indx_icd9 = ukb_data[['eid', 'date_of_attending_assessment_centre_f53_0_0', 'date_of_attending_assessment_centre_f53_1_0'] + 
                        ukb_key[ukb_key['field.showcase'].isin([41271,41281])]['col.name'].tolist()]
icd9_dx_stubname = ukb_key[ukb_key['field.showcase'] == 41271]['col.name'].iloc[0].rsplit('_', 2)[0]
icd9_date_stubname = ukb_key[ukb_key['field.showcase'] == 41281]['col.name'].iloc[0].rsplit('_', 2)[0]
pmh_ukb_indx_icd9.rename(columns={'date_of_attending_assessment_centre_f53_0_0': 'visit_0',
                                   'date_of_attending_assessment_centre_f53_1_0': 'visit_1'}, inplace=True)

long_pmh_ukb_indx_icd9 = pd.wide_to_long(df=pmh_ukb_indx_icd9, stubnames=[icd9_dx_stubname, icd9_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_indx_icd9 = long_pmh_ukb_indx_icd9.drop(columns='instance_array').rename(columns={'diagnoses_icd9_f41271':'icd',
                                        'date_of_first_inpatient_diagnosis_icd9_f41281':'icd_date'})
final_pmh_ukb_indx_icd9['icd_type'] = 'icd9'
for i in ['icd_date', 'visit_0', 'visit_1']:
    final_pmh_ukb_indx_icd9[i] = pd.to_datetime(final_pmh_ukb_indx_icd9[i])

#Merging ICD9 and ICD10
pmh_ukb_indx = pd.concat([final_pmh_ukb_indx_icd10, final_pmh_ukb_indx_icd9])
pmh_ukb_indx['icd'] =pmh_ukb_indx['icd'].astype(str)

for i in pmh_ukb_indx['icd'].unique().tolist():
    pmh_ukb_indx[i + '_0'] = np.where((pmh_ukb_indx['icd']==i) & (pmh_ukb_indx['icd_date']<pmh_ukb_indx['visit_0']), 1, 0)
    pmh_ukb_indx[i + '_0'] = pmh_ukb_indx.groupby('eid')[i + '_0'].transform(max)

    pmh_ukb_indx[i + '_1'] = np.where((pmh_ukb_indx['icd']==i) & (pmh_ukb_indx['icd_date']<pmh_ukb_indx['visit_1']), 1, 0)
    pmh_ukb_indx[i + '_1'] = pmh_ukb_indx.groupby('eid')[i + '_1'].transform(max)

pmh_ukb_indx.drop_duplicates(subset=['eid'], inplace=True)
pmh_ukb_indx.drop(columns=['visit_0', 'visit_1', 'icd', 'icd_date', 'icd_type'], inplace=True)
pmh_ukb_indx = pmh_ukb_indx.fillna(0)
pmh_ukb_indx

# %%
#OPCS4 
opcs4_ukb_data = ukb_data[['eid', 'date_of_attending_assessment_centre_f53_0_0', 'date_of_attending_assessment_centre_f53_1_0'] + 
                    ukb_key[ukb_key['field.showcase'].isin([41272,41282])]['col.name'].tolist()]

opcs4_proc_stubname = ukb_key[ukb_key['field.showcase'] == 41272]['col.name'].iloc[0].rsplit('_', 2)[0]
opcs4_date_stubname = ukb_key[ukb_key['field.showcase'] == 41282]['col.name'].iloc[0].rsplit('_', 2)[0]
opcs4_ukb_data.rename(columns={'date_of_attending_assessment_centre_f53_0_0': 'visit_0',
                                   'date_of_attending_assessment_centre_f53_1_0': 'visit_1'}, inplace=True)

long_opcs4_ukb_data = pd.wide_to_long(df=opcs4_ukb_data, stubnames=[opcs4_proc_stubname, opcs4_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_opcs4_data = long_opcs4_ukb_data.drop(columns='instance_array').rename(columns={'operative_procedures_opcs4_f41272':'opcs',
                                        'date_of_first_operative_procedure_opcs4_f41282':'opcs_date'})
final_opcs4_data['opcs_type'] = 'OPCS4'
for i in ['opcs_date', 'visit_0', 'visit_1']:
    final_opcs4_data[i] = pd.to_datetime(final_opcs4_data[i])

#OPCS3
opcs3_ukb_data = ukb_data[['eid', 'date_of_attending_assessment_centre_f53_0_0', 'date_of_attending_assessment_centre_f53_1_0'] + 
                ukb_key[ukb_key['field.showcase'].isin([41273,41283])]['col.name'].tolist()]
                
opcs3_proc_stubname = ukb_key[ukb_key['field.showcase'] == 41273]['col.name'].iloc[0].rsplit('_', 2)[0]
opcs3_date_stubname = ukb_key[ukb_key['field.showcase'] == 41283]['col.name'].iloc[0].rsplit('_', 2)[0]
opcs3_ukb_data.rename(columns={'date_of_attending_assessment_centre_f53_0_0': 'visit_0',
                                   'date_of_attending_assessment_centre_f53_1_0': 'visit_1'}, inplace=True)

long_opcs3_ukb_data = pd.wide_to_long(df=opcs3_ukb_data, stubnames=[opcs3_proc_stubname, opcs3_date_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_opcs3_data = long_opcs3_ukb_data.drop(columns='instance_array').rename(columns={'operative_procedures_opcs3_f41273':'opcs',
                                        'date_of_first_operative_procedure_opcs3_f41283':'opcs_date'})
final_opcs3_data['opcs_type'] = 'OPCS3'
for i in ['opcs_date', 'visit_0', 'visit_1']:
    final_opcs3_data[i] = pd.to_datetime(final_opcs3_data[i])

#Merging OPCS3 and OPCS4
opcs_ukb = pd.concat([final_opcs4_data, final_opcs3_data])
opcs_ukb['opcs'] = opcs_ukb['opcs'].astype(str)

for i in opcs_ukb['opcs'].unique().tolist():
    opcs_ukb[i + '_0'] = np.where((opcs_ukb['opcs']==i) & (opcs_ukb['opcs_date']<opcs_ukb['visit_0']), 1, 0)
    opcs_ukb[i + '_0'] = opcs_ukb.groupby('eid')[i + '_0'].transform(max)

    opcs_ukb[i + '_1'] = np.where((opcs_ukb['opcs']==i) & (opcs_ukb['opcs_date']<opcs_ukb['visit_1']), 1, 0)
    opcs_ukb[i + '_1'] = opcs_ukb.groupby('eid')[i + '_1'].transform(max)

opcs_ukb.drop_duplicates(subset=['eid'], inplace=True)
opcs_ukb.drop(columns=['visit_0', 'visit_1', 'opcs', 'opcs_date', 'opcs_type'], inplace=True)
opcs_ukb = opcs_ukb.fillna(0)
opcs_ukb

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

lab_ukb_data['lp_a_0_cat'] = np.where((lab_ukb_data['lp_a_0_value'].notna()) & (lab_ukb_data['lp_a_0_value']>=150), 
                    1, np.where((lab_ukb_data['lp_a_0_value'].notna()) & (lab_ukb_data['lp_a_0_value']<150),
                    0, lab_ukb_data['lp_a_0_value']))
lab_ukb_data['lp_a_0_cat'] = np.where((lab_ukb_data['lp_a_0_value'].notna()) & (lab_ukb_data['lp_a_0_value']>=150), 
                    1, np.where((lab_ukb_data['lp_a_0_value'].notna()) & (lab_ukb_data['lp_a_0_value']<150),
                    0, lab_ukb_data['lp_a_0_value']))
                    
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

lab_ukb_data = lab_ukb_data[['eid','urea_0','creatinine_0','creactive_protein_0','glucose_0','hba1c_0','hdl_0',
                            'ldl_0','cholesterol_0','triglycerides_0','wbc_0','hb_0','mcv_0','plt_0',
                            'albumin_0','total_protein_0','alp_0','alt_0','ast_0','bil_d_0','bil_t_0',
                            'calcium_0','phosphate_0','vitamin_d_0','lp_a_0_cat','lp_a_0_value']]

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

vital_ukb_data.set_index('eid', inplace=True)
vital_ukb_data = vital_ukb_data[~vital_ukb_data.index.duplicated(keep='first')]
vital_ukb_data.sort_values(by='eid', inplace=True)
vital_ukb_data.reset_index(inplace=True)
vital_ukb_data.drop(columns=['instance','array','sbp','dbp','hr'], inplace=True)

vital_ukb_data = vital_ukb_data[['eid','sbp_0','dbp_0','hr_0']]

# %%
ukb_wf = demo_ukb_data.merge(lifestyle_ukb_data, how='left')
ukb_wf = ukb_wf.merge(pmh_ukb_indx, how='left', on='eid')
ukb_wf = ukb_wf.merge(opcs_ukb, how='left', on='eid')
ukb_wf = ukb_wf.merge(lab_ukb_data, how='left', on='eid')
ukb_wf = ukb_wf.merge(vital_ukb_data, how='left', on='eid')
ukb_wf.dropna(subset='lp_a_0_value', inplace=True)
ukb_wf.to_csv("/Lp(a)/Archive Data/ukb_wf_all_codes.csv", index=False)
ukb_wf


