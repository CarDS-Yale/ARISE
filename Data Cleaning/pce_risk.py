#%%
# This file calculates PCE risk score for UK Biobank.

import pandas as pd
import numpy as np
import math

df = pd.read_csv("/Lp(a)/ukb_wf_parsimonious.csv")

def black_women_no_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = 17.114 * math.log(age) + 0.94 * math.log(TC) - 18.92 * math.log(HDL_C) + \
            4.475 * math.log(age) * math.log(HDL_C) + 27.82 * math.log(SBP) - \
            6.087 * math.log(age) * math.log(SBP)

    if smoker==2:
        score += 0.691

    if diabetes==1:
        score += 0.874

    risk = 1 - math.pow(0.9533, math.exp(score - 86.61))
    return risk

def white_women_no_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = -29.799 * math.log(age) + 4.884 * math.pow(math.log(age), 2) + \
            13.54 * math.log(TC) - 3.114 * math.log(age) * math.log(TC) - \
            13.578 * math.log(HDL_C) + 3.149 * math.log(age) * math.log(HDL_C) + \
            1.957 * math.log(SBP)

    if smoker==2:
        score += 7.574 - 1.665 * math.log(age)

    if diabetes==1:
        score += 0.661

    risk = 1 - math.pow(0.9665, math.exp(score + 29.18))
    return risk


def black_men_no_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = 2.469 * math.log(age) + 0.302 * math.log(TC) - 0.307 * math.log(HDL_C) + \
            1.809 * math.log(SBP)

    if smoker==2:
        score += 0.549

    if diabetes==1:
        score += 0.645

    risk = 1 - math.pow(0.8954, math.exp(score - 19.54))
    return risk

def white_men_no_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = 12.344 * math.log(age) + 11.853 * math.log(TC) - 2.664 * math.log(age) * math.log(TC) - \
            7.99 * math.log(HDL_C) + 1.769 * math.log(age) * math.log(HDL_C) + 1.764 * math.log(SBP)

    if smoker==2:
        score += 7.837 - 1.795 * math.log(age)

    if diabetes==1:
        score += 0.658

    risk = 1 - math.pow(0.9144, math.exp(score - 61.18))
    return risk

def black_women_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = 17.114 * math.log(age) + 0.94 * math.log(TC) - 18.92 * math.log(HDL_C) + \
            4.475 * math.log(age) * math.log(HDL_C) + 29.291 * math.log(SBP) - \
            6.432 * math.log(age) * math.log(SBP)

    if smoker==2:
        score += 0.691

    if diabetes==1:
        score += 0.874

    risk = 1 - math.pow(0.9533, math.exp(score - 86.61))
    return risk

def white_women_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = -29.799 * math.log(age) + 4.884 * math.log(age)**2 + 13.54 * math.log(TC) - \
            3.114 * math.log(age) * math.log(TC) - 13.578 * math.log(HDL_C) + \
            3.149 * math.log(age) * math.log(HDL_C) + 2.019 * math.log(SBP)

    if smoker==2:
        score += 7.574 - 1.665 * math.log(age)

    if diabetes==1:
        score += 0.661

    risk = 1 - math.pow(0.9665, math.exp(score + 29.18))
    return risk

def black_men_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = 2.469 * math.log(age) + 0.302 * math.log(TC) - 0.307 * math.log(HDL_C) + \
            1.916 * math.log(SBP)

    if smoker==2:
        score += 0.549

    if diabetes==1:
        score += 0.645

    risk = 1 - math.pow(0.8954, math.exp(score - 19.54))
    return risk

def white_men_antihtn_risk(row):
    age = row['age']
    TC = row['cholesterol']*38.67
    HDL_C = row['hdl']*38.67
    SBP = row['sbp']
    smoker = row['smoking']
    diabetes = row['dm']

    score = 12.344 * math.log(age) + 11.853 * math.log(TC) - 2.664 * math.log(age) * math.log(TC) - \
            7.99 * math.log(HDL_C) + 1.769 * math.log(age) * math.log(HDL_C) + 1.797 * math.log(SBP)

    if smoker==2:
        score += 7.837 - 1.795 * math.log(age)

    if diabetes==1:
        score += 0.658

    risk = 1 - math.pow(0.9144, math.exp(score - 61.18))
    return risk


df['black_women_no_antihtn_risk'] = np.where((df['ethnicity']==4)&(df['sex']==0)&(df['anti_htn']==0), 1, 0)
df['white_women_no_antihtn_risk'] = np.where((df['ethnicity'].isin([1, 2, 3, 5, 6]))&(df['sex']==0)&(df['anti_htn']==0), 1, 0)
df['black_men_no_antihtn_risk'] = np.where((df['ethnicity']==4)&(df['sex']==1)&(df['anti_htn']==0), 1, 0)
df['white_men_no_antihtn_risk'] = np.where((df['ethnicity'].isin([1, 2, 3, 5, 6]))&(df['sex']==1)&(df['anti_htn']==0), 1, 0)

df['black_women_antihtn_risk'] = np.where((df['ethnicity']==4)&(df['sex']==0)&(df['anti_htn']==1), 1, 0)
df['white_women_antihtn_risk'] = np.where((df['ethnicity'].isin([1, 2, 3, 5, 6]))&(df['sex']==0)&(df['anti_htn']==1), 1, 0)
df['black_men_antihtn_risk'] = np.where((df['ethnicity']==4)&(df['sex']==1)&(df['anti_htn']==1), 1, 0)
df['white_men_antihtn_risk'] = np.where((df['ethnicity'].isin([1, 2, 3, 5, 6]))&(df['sex']==1)&(df['anti_htn']==1), 1, 0)

df['pce_risk'] = np.where(df['black_women_no_antihtn_risk']==1, df.apply(lambda row: black_women_no_antihtn_risk(row), axis=1), 
    np.where(df['white_women_no_antihtn_risk']==1, df.apply(lambda row: white_women_no_antihtn_risk(row), axis=1),
        np.where(df['black_men_no_antihtn_risk']==1, df.apply(lambda row: black_men_no_antihtn_risk(row), axis=1),
            np.where(df['white_men_no_antihtn_risk']==1, df.apply(lambda row: white_men_no_antihtn_risk(row), axis=1),
                np.where(df['black_women_antihtn_risk']==1, df.apply(lambda row: black_women_antihtn_risk(row), axis=1),
                    np.where(df['white_women_antihtn_risk']==1, df.apply(lambda row: white_women_antihtn_risk(row), axis=1),
                        np.where(df['black_men_antihtn_risk']==1, df.apply(lambda row: black_men_antihtn_risk(row), axis=1),
                            np.where(df['white_men_antihtn_risk']==1, df.apply(lambda row: white_men_antihtn_risk(row), axis=1), np.nan))))))))


df.to_csv("/Lp(a)/ukb_wf_parsimonious_pce_risk.csv", index=False)
# %%
