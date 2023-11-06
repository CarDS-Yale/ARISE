# %%
# This file is for model development and validation.

import pandas as pd 
import numpy as np  
import scipy.stats as stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import torch 
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import EarlyStopping
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.font_manager import FontProperties
import shap
shap.initjs()
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

path = "/Lp(a)/"

ukb_data = pd.read_csv(path + "ukb_wf_parsimonious_pce_risk.csv", low_memory = False)
ukb_data = ukb_data[~ukb_data['lp_a_value'].isna()]
aric = pd.read_csv("/Lp(a)/External_Validation/aric.csv")
cardia = pd.read_csv("/Lp(a)/External_Validation/cardia.csv")
mesa = pd.read_csv("/Lp(a)/External_Validation/mesa.csv")

# %%
#Finding and removing highly correlated features
corr_matrix = ukb_data.corr()

# %%
#Model Selection
#All Features

ukb_data = pd.read_csv('/Lp(a)/Archive Data/ukb_wf_all_codes.csv')

cat_non_ordinal_attribs = ['ethnicity']

num_attribs = ['age_0', 'bmi_0', 'waist_circumference_0',
 'hip_circumference_0', 'wbc_0', 'hb_0',
 'mcv_0', 'plt_0', 'albumin_0', 'alp_0', 'alt_0', 'ast_0', 'bil_d_0', 'urea_0',
 'calcium_0', 'cholesterol_0', 'creatinine_0', 'creactive_protein_0',
 'glucose_0', 'hba1c_0', 'hdl_0', 'ldl_0', 'phosphate_0', 'bil_t_0',
 'total_protein_0', 'triglycerides_0', 'vitamin_d_0', 'sbp_0', 'dbp_0', 'hr_0']

# This will include all diagnoses and procedure codes as 0 and 1, defined in the Data cleaning, UKBB-all features.ipynb file.
cat_ordinal_attribs = ukb_data.drop(columns=cat_non_ordinal_attribs + num_attribs + ['eid', 'lp_a_0_value', 'lp_a_0_cat']).columns.tolist()


ukb_train, ukb_test = train_test_split(ukb_data, test_size=0.2, random_state=1)
X_train = ukb_train[num_attribs + cat_ordinal_attribs + cat_non_ordinal_attribs]
y_train = ukb_train['lp_a_0_value'].copy()
y_train_class = y_train>=150

#Standard Scaling
cat_non_ordinal_pipeline = Pipeline([('imputer_cat_non_ordinal', SimpleImputer(strategy='most_frequent')),
                                     ('cat_encoder', OneHotEncoder())])

num_pipeline = Pipeline([('imputer_num', SimpleImputer(strategy='mean')),
                         ('std_scaler', StandardScaler())])

cat_ordinal_pipeline = Pipeline([('imputer_cat_ordinal', SimpleImputer(strategy='most_frequent'))])


full_pipline = ColumnTransformer([('cat_non_ordinal', cat_non_ordinal_pipeline, cat_non_ordinal_attribs),
                                   ('num', num_pipeline, num_attribs),
                                   ('cat_ordinal', cat_ordinal_pipeline, cat_ordinal_attribs)])

X_train = full_pipline.fit_transform(X_train)


print('Logistic Regression')
lr_clf_final = LogisticRegression(C=0.01, max_iter=10000, solver='liblinear', penalty='l1', random_state=22)
lr_clf_final.fit(X_train, y_train_class) 
print('Train, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_train_class, lr_clf_final.predict_proba(X_train)[:, 1]), 3))
print('CV, post-tuning')
scores_lr_final = cross_val_score(lr_clf_final, X_train, y_train_class, scoring='roc_auc', cv=5) 
print("Cross-Validation AUROC:", scores_lr_final.mean())
print('')

print('XGBoost')
xgb_clf_final = XGBClassifier(n_estimators=50, max_depth=3, colsample_bytree=0.6, min_child_weight=5, gamma=5, 
                                enable_categorical=False, random_state=22)
xgb_clf_final.fit(X_train, y_train_class) 
print('Train, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1]), 3))
print('CV, post-tuning')
scores_xgb_final = cross_val_score(xgb_clf_final, X_train, y_train_class, scoring='roc_auc', cv=5) 
print("Cross-Validation AUROC:", scores_xgb_final.mean())
print('')

print('TabNet')

tabnet_clf = TabNetClassifier(
    n_d=8,           # Dimension of the model
    n_a=8,           # Attention dimension
    n_steps=5,       # Number of steps in the Gated Attention Mechanism
    gamma=1.5,       # Regularization coefficient for the sparsity
    n_independent=2, # Number of independent GLU units in each feature column
    n_shared=2,      # Number of shared GLU units in each feature column
    cat_dims=[],     # List of categorical feature indices
    cat_emb_dim=1,   # Dimension of the categorical embeddings
    lambda_sparse=0.001, # Sparsity loss coefficient
    optimizer_fn=torch.optim.Adam, # Optimizer
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax', # Attention mask type
    seed=42,
    verbose=10
)

scores_tabnet = cross_val_score(
    tabnet_clf, X_train, y_train_class,
    scoring='roc_auc', cv=5,
    fit_params={"eval_set": [(X_train, y_train_class)], "patience": 10, "max_epochs": 100}
)

print("Cross-Validation AUROC:", np.mean(scores_tabnet))

# %%
#Model Selection
#eTable 2 Features

ukb_data = pd.read_csv(path + "ukb_wf_parsimonious_pce_risk.csv", low_memory = False)
ukb_data = ukb_data[~ukb_data['lp_a_value'].isna()]

cat_non_ordinal_attribs = ['ethnicity', 'smoking']

num_attribs = ['age', 'bmi', 'waist_circumference',
 'hip_circumference', 'age_at_first_ascvd', 'heart_failure_num', 'ihd_num',
 'pad_num', 'stroke_num', 'cabg_num', 'pci_num',
 'carotid_revasc_num', 'pad_revasc_num', 'age_at_first_cv_procedure', 'wbc', 'hb',
 'mcv', 'plt', 'albumin', 'alp', 'alt', 'ast', 'bil_t', 'urea',
 'calcium', 'cholesterol', 'creatinine', 'creactive_protein',
 'glucose', 'hba1c', 'hdl', 'ldl', 'phosphate',
 'total_protein', 'triglycerides', 'vitamin_d', 'sbp', 'dbp', 'hr', 'age_at_first_ascvd_cvpx']

cat_ordinal_attribs = ['sex', 'cvd_family_history', 'htn',
 'dm', 't1dm', 't2dm', 'ckd', 'heart_failure', 'ihd', 'pad',
 'stroke', 'premature_ascvd', 'cabg', 'pci', 'carotid_revasc', 'pad_revasc',
 'premature_cv_procedure', 'statin', 'aspirin', 'anti_htn',
 'insulin', 'hrt', 'ocp', 'hormone_est_prog', 'ascvd', 'premature_ascvd_cvpx']


ukb_train, ukb_test = train_test_split(ukb_data, test_size=0.2, random_state=1)
X_train = ukb_train[num_attribs + cat_ordinal_attribs + cat_non_ordinal_attribs]
y_train = ukb_train['lp_a_value'].copy()
y_train_class = y_train>=150

#Standard Scaling
cat_non_ordinal_pipeline = Pipeline([('imputer_cat_non_ordinal', SimpleImputer(strategy='most_frequent')),
                                     ('cat_encoder', OneHotEncoder())])

num_pipeline = Pipeline([('imputer_num', SimpleImputer(strategy='mean')),
                         ('std_scaler', StandardScaler())])

cat_ordinal_pipeline = Pipeline([('imputer_cat_ordinal', SimpleImputer(strategy='most_frequent'))])


full_pipline = ColumnTransformer([('cat_non_ordinal', cat_non_ordinal_pipeline, cat_non_ordinal_attribs),
                                   ('num', num_pipeline, num_attribs),
                                   ('cat_ordinal', cat_ordinal_pipeline, cat_ordinal_attribs)])

X_train = full_pipline.fit_transform(X_train)


print('Logistic Regression')
lr_clf_final = LogisticRegression(C=0.1, max_iter=1000, solver='liblinear', penalty='l1', random_state=22)
lr_clf_final.fit(X_train, y_train_class) 
print('Train, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_train_class, lr_clf_final.predict_proba(X_train)[:, 1]), 3))
print('CV, post-tuning')
scores_lr_final = cross_val_score(lr_clf_final, X_train, y_train_class, scoring='roc_auc', cv=5) 
print("Cross-Validation AUROC:", scores_lr_final.mean())
print('')

print('XGBoost')
xgb_clf_final = XGBClassifier(n_estimators=100, max_depth=3, colsample_bytree=0.6, min_child_weight=5, gamma=5, 
                                enable_categorical=False, random_state=22)
xgb_clf_final.fit(X_train, y_train_class) 
print('Train, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1]), 3))
print('CV, post-tuning')
scores_xgb_final = cross_val_score(xgb_clf_final, X_train, y_train_class, scoring='roc_auc', cv=5) 
print("Cross-Validation AUROC:", scores_xgb_final.mean())
print('')

print('TabNet')

tabnet_clf = TabNetClassifier(
    n_d=8,           # Dimension of the model
    n_a=8,           # Attention dimension
    n_steps=5,       # Number of steps in the Gated Attention Mechanism
    gamma=1.5,       # Regularization coefficient for the sparsity
    n_independent=2, # Number of independent GLU units in each feature column
    n_shared=2,      # Number of shared GLU units in each feature column
    cat_dims=[],     # List of categorical feature indices
    cat_emb_dim=1,   # Dimension of the categorical embeddings
    lambda_sparse=0.001, # Sparsity loss coefficient
    optimizer_fn=torch.optim.Adam, # Optimizer
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax', # Attention mask type
    seed=42,
    verbose=10
)

scores_tabnet = cross_val_score(
    tabnet_clf, X_train, y_train_class,
    scoring='roc_auc', cv=5,
    fit_params={"eval_set": [(X_train, y_train_class)], "patience": 10, "max_epochs": 100}
)

print("Cross-Validation AUROC:", np.mean(scores_tabnet))

# %%
#Model Selection
#ARISE Features

print('Logistic Regression')
lr_clf_final = LogisticRegression(C=0.1, max_iter=1000, solver='liblinear', penalty='l1', random_state=22)
lr_clf_final.fit(X_train, y_train_class) 
print('Train, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_train_class, lr_clf_final.predict_proba(X_train)[:, 1]), 3))
print('CV, post-tuning')
scores_lr_final = cross_val_score(lr_clf_final, X_train, y_train_class, scoring='roc_auc', cv=5) 
print("Cross-Validation AUROC:", scores_lr_final.mean())
print('')

print('XGBoost')
xgb_clf_final = XGBClassifier(n_estimators=100, max_depth=3, colsample_bytree=0.6, min_child_weight=5, gamma=5, 
                                enable_categorical=False, random_state=22)
xgb_clf_final.fit(X_train, y_train_class) 
print('Train, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1]), 3))
print('CV, post-tuning')
scores_xgb_final = cross_val_score(xgb_clf_final, X_train, y_train_class, scoring='roc_auc', cv=5) 
print("Cross-Validation AUROC:", scores_xgb_final.mean())
print('')

print('TabNet')

tabnet_clf = TabNetClassifier(
    n_d=8,           # Dimension of the model
    n_a=8,           # Attention dimension
    n_steps=5,       # Number of steps in the Gated Attention Mechanism
    gamma=1.5,       # Regularization coefficient for the sparsity
    n_independent=2, # Number of independent GLU units in each feature column
    n_shared=2,      # Number of shared GLU units in each feature column
    cat_dims=[],     # List of categorical feature indices
    cat_emb_dim=1,   # Dimension of the categorical embeddings
    lambda_sparse=0.001, # Sparsity loss coefficient
    optimizer_fn=torch.optim.Adam, # Optimizer
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax', # Attention mask type
    seed=42,
    verbose=10
)

scores_tabnet = cross_val_score(
    tabnet_clf, X_train, y_train_class,
    scoring='roc_auc', cv=5,
    fit_params={"eval_set": [(X_train, y_train_class)], "patience": 10, "max_epochs": 100}
)

print("Cross-Validation AUROC:", np.mean(scores_tabnet))

# %%
#Development of ARISE
#Spliting the dataset
ukb_train, ukb_test = train_test_split(ukb_data, test_size=0.2, random_state=1)
num_attribs = ['hdl', 'ldl', 'triglycerides']
cat_ordinal_attribs = ['statin', 'anti_htn', 'ascvd', 'cvd_family_history']

X_train = ukb_train[num_attribs + cat_ordinal_attribs]
y_train = ukb_train['lp_a_value'].copy()
X_test = ukb_test[num_attribs + cat_ordinal_attribs]
y_test = ukb_test['lp_a_value'].copy()

X_aric = aric[num_attribs + cat_ordinal_attribs]
y_aric = aric['lp_a_value'].copy()

X_cardia = cardia[num_attribs + cat_ordinal_attribs]
y_cardia = cardia['lp_a_value'].copy()

X_mesa = mesa[num_attribs + cat_ordinal_attribs]
y_mesa = mesa['lp_a_value'].copy()


#Transforming data before fitting the model
num_pipeline = Pipeline([('imputer_num', KNNImputer()),
                        ('scaler', StandardScaler())])

cat_ordinal_pipeline = Pipeline([('imputer_cat_ordinal', KNNImputer())])


full_pipline = ColumnTransformer([('num', num_pipeline, num_attribs),
                                  ('cat_ordinal', cat_ordinal_pipeline, cat_ordinal_attribs)])

X_train = full_pipline.fit_transform(X_train)

X_train = full_pipline.transform(X_train)
X_test = full_pipline.transform(X_test)
X_aric = full_pipline.transform(X_aric)
X_cardia = full_pipline.transform(X_cardia) 
X_mesa = full_pipline.transform(X_mesa)

lp_a_cutoff = 150
y_train_class = y_train>=lp_a_cutoff
y_test_class = y_test>=lp_a_cutoff
y_aric_class = y_aric>=lp_a_cutoff
y_cardia_class = y_cardia>=lp_a_cutoff
y_mesa_class = y_mesa>=lp_a_cutoff


#XGBoost-Final
#XGBoost hyperparameters were derived using GridSearchCV
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train_class) 
#Fine Tuning
param_grid_xgb = {'min_child_weight': [1, 5, 10], 'gamma': [0.5, 1, 1.5, 2, 5], 'subsample': [0.6, 0.8, 1.0],
                  'colsample_bytree': [0.6, 0.8, 1.0], 'max_depth': [3, 4, 5]}
grid_search_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=5, scoring='roc_auc', return_train_score=True)
grid_search_xgb.fit(X_train, y_train_class)
grid_search_xgb.best_estimator_ 

#Using the hyperparameters from GridSearch
xgb_clf_final = XGBClassifier(n_estimators=100, max_depth=3, colsample_bytree=0.6, min_child_weight=5, gamma=5, 
                                enable_categorical=False, random_state=22)
xgb_clf_final.fit(X_train, y_train_class) 

#Train
print('Train, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1]), 3))
print("AUPRC:", round(metrics.average_precision_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1]), 3))


#Test
print('Test, post-tuning')
print("AUROC:", round(metrics.roc_auc_score(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1]), 3))
print("AUPRC:", round(metrics.average_precision_score(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1]), 3))

#ARIC
print('ARIC')
print("AUROC:", round(metrics.roc_auc_score(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1]), 3))
print("AUPRC:", round(metrics.average_precision_score(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1]), 3))

#CARDIA
print('CARDIA')
print("AUROC:", round(metrics.roc_auc_score(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1]), 3))
print("AUPRC:", round(metrics.average_precision_score(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1]), 3))

#MESA
print('MESA')
print("AUROC:", round(metrics.roc_auc_score(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1]), 3))
print("AUPRC:", round(metrics.average_precision_score(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1]), 3))


#Adding Predictions
#ukb_train['high_lp_a_probability'] = xgb_clf_final.predict_proba(X_train)[:, 1].copy()
ukb_test['high_lp_a_probability'] = xgb_clf_final.predict_proba(X_test)[:, 1].copy()
aric['high_lp_a_probability'] = xgb_clf_final.predict_proba(X_aric)[:, 1].copy()
cardia['high_lp_a_probability'] = xgb_clf_final.predict_proba(X_cardia)[:, 1].copy()
mesa['high_lp_a_probability'] = xgb_clf_final.predict_proba(X_mesa)[:, 1].copy()

#Saving Datasets
#ukb_train.to_csv("/Lp(a)/External_Validation/ukb_train.csv", index=False)
ukb_test.to_csv("/Lp(a)/External_Validation/ukb_test.csv", index=False)
aric.to_csv("/Lp(a)/External_Validation/aric.csv", index=False)
cardia.to_csv("/Lp(a)/External_Validation/cardia.csv", index=False)
mesa.to_csv("/Lp(a)/External_Validation/mesa.csv", index=False)

# %%
#95% CI for AUROC and AUPRC
def bootstrap_auroc(y_true, y_pred, n_iterations=1000, confidence_level=0.95):
    y_true = y_true.reset_index(drop=True, inplace=False)
    y_pred = pd.Series(y_pred).reset_index(drop=True, inplace=False)
    
    # Initialize an array to store the AUROC values from each bootstrap iteration
    auroc_values = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Create a bootstrap sample by resampling with replacement
        sample_indices = resample(range(len(y_true)))
        y_true_bootstrap = y_true[sample_indices]
        y_pred_bootstrap = y_pred[sample_indices]

        # Calculate AUROC for the bootstrap sample
        auroc_values[i] = metrics.roc_auc_score(y_true_bootstrap, y_pred_bootstrap)

    # Calculate the confidence interval
    lower_bound = np.percentile(auroc_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auroc_values, (1 + confidence_level) / 2 * 100)

    # Print the results
    return print(f"AUROC: {metrics.roc_auc_score(y_true, y_pred):.3f} ({lower_bound:.3f}-{upper_bound:.3f})")

def bootstrap_auprc(y_true, y_pred, n_iterations=1000, confidence_level=0.95):
    y_true = y_true.reset_index(drop=True, inplace=False)
    y_pred = pd.Series(y_pred).reset_index(drop=True, inplace=False)
    
    # Initialize an array to store the AUROC values from each bootstrap iteration
    auprc_values = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Create a bootstrap sample by resampling with replacement
        sample_indices = resample(range(len(y_true)))
        y_true_bootstrap = y_true[sample_indices]
        y_pred_bootstrap = y_pred[sample_indices]

        # Calculate AUROC for the bootstrap sample
        auprc_values[i] = metrics.average_precision_score(y_true_bootstrap, y_pred_bootstrap)

    # Calculate the confidence interval
    lower_bound = np.percentile(auprc_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auprc_values, (1 + confidence_level) / 2 * 100)

    # Print the results
    return print(f"AUPRC: {metrics.average_precision_score(y_true, y_pred):.3f} ({lower_bound:.3f}-{upper_bound:.3f})")

# %%
print('UKBB Train Set')
bootstrap_auroc(y_train_class.astype(int), xgb_clf_final.predict_proba(X_train)[:, 1], n_iterations=1000, confidence_level=0.95)
print('UKBB Held-out Test Set')
bootstrap_auroc(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1], n_iterations=1000, confidence_level=0.95)
print('ARIC')
bootstrap_auroc(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1], n_iterations=1000, confidence_level=0.95)
print('CARDIA')
bootstrap_auroc(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1], n_iterations=1000, confidence_level=0.95)
print('MESA')
bootstrap_auroc(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1], n_iterations=1000, confidence_level=0.95)
print('PCE')
pce_ukb_test = ukb_test.dropna(subset='pce_risk')
pce_ukb_test = pce_ukb_test[pce_ukb_test['ascvd']==0]
bootstrap_auroc((pce_ukb_test['lp_a_value']>=150).astype(int), pce_ukb_test['pce_risk'], n_iterations=1000, confidence_level=0.95)

# %%
print('UKBB Train Set')
#bootstrap_auprc(y_train_class.astype(int), xgb_clf_final.predict_proba(X_train)[:, 1], n_iterations=1000, confidence_level=0.95)
print('UKBB Held-out Test Set')
bootstrap_auprc(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1], n_iterations=1000, confidence_level=0.95)
print('ARIC')
bootstrap_auprc(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1], n_iterations=1000, confidence_level=0.95)
print('CARDIA')
bootstrap_auprc(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1], n_iterations=1000, confidence_level=0.95)
print('MESA')
bootstrap_auprc(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1], n_iterations=1000, confidence_level=0.95)
print('PCE')
pce_ukb_test = ukb_test.dropna(subset='pce_risk')
pce_ukb_test = pce_ukb_test[pce_ukb_test['ascvd']==0]
bootstrap_auprc((pce_ukb_test['lp_a_value']>=150).astype(int), pce_ukb_test['pce_risk'], n_iterations=1000, confidence_level=0.95)

# %%
#AUROC 
#set up plotting area
plt.figure(0).clf()
lw = 1.5
plt.style.use('seaborn-talk')

#UKBB-Train
fpr, tpr, _ = metrics.roc_curve(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1])
plt.plot(fpr, tpr, label='UKB Train, 0.660')

#UKBB-Test
fpr, tpr, _ = metrics.roc_curve(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='UKB Test, 0.655')


#ARIC
fpr, tpr, _ = metrics.roc_curve(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1])
plt.plot(fpr, tpr, label='ARIC, 0.656')

#CARDIA
fpr, tpr, _ = metrics.roc_curve(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1])
plt.plot(fpr, tpr, label='CARDIA, 0.675')

#MESA
fpr, tpr, _ = metrics.roc_curve(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1])
plt.plot(fpr, tpr, label='MESA, 0.686')

#PCE
pce_ukb_test = ukb_test.dropna(subset='pce_risk')
pce_ukb_test = pce_ukb_test[pce_ukb_test['ascvd']==0]
fpr, tpr, _ = metrics.roc_curve(pce_ukb_test['lp_a_value']>=150, pce_ukb_test['pce_risk'])
plt.plot(fpr, tpr, label='PCE in UKB Test, 0.517')

#Custom
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18, weight = 'bold')
plt.ylabel('True Positive Rate', fontsize=18, weight = 'bold')
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
legend = plt.legend(loc="best", title='Dataset, AUROC', title_fontsize=17, fontsize=16.7)
title = legend.get_title()
title.set_weight("bold")
fig = plt.gcf()
fig.savefig('/Lp_a_UKBB/roc.svg', format='svg')

# %%
#Statistical comparison of ARISE vs PCE

#Calculating AUROC among those without ASCVD in the held-out test set and with a non-NA PCE value
pce_ukb_test = ukb_test.dropna(subset='pce_risk')
pce_ukb_test = pce_ukb_test[pce_ukb_test['ascvd']==0]

# Calculate AUROC values
auroc1 = metrics.roc_auc_score(pce_ukb_test['lp_a_value']>=150, pce_ukb_test['high_lp_a_probability'])
auroc2 = metrics.roc_auc_score(pce_ukb_test['lp_a_value']>=150, pce_ukb_test['pce_risk'])

# Calculate variances of the AUROC estimates
n1 = len(y_test_class)
n2 = len(pce_ukb_test['lp_a_value']>=150)
var_auroc1 = (auroc1 * (1 - auroc1) + (n1 - 1) / (2 * n1)) / n1
var_auroc2 = (auroc2 * (1 - auroc2) + (n2 - 1) / (2 * n2)) / n2

# Calculate z-score
z = (auroc1 - auroc2) / np.sqrt(var_auroc1 + var_auroc2)

# Calculate p-value using the normal distribution
p_value = 2 * (1 - norm.cdf(abs(z)))

print("AUROC Model 1:", auroc1)
print("AUROC Model 2:", auroc2)
print("p-value:", p_value)


# %%
#SHAP value-UKBB

explainer = shap.TreeExplainer(xgb_clf_final)
shap_values = explainer(X_test)
feature_names = ['HDL', 'LDL', 'Triglycerides', 'Statin', 'Anti-Hypertensive', 'ASCVD']
#shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title("UK Biobank Held-out Test Set", weight = 'bold', fontsize=18)
plt.yticks(fontsize=14)
fig = plt.gcf()
fig.savefig(path + 'shap_ukb.svg', format='svg')
plt.show()

# %%
#SHAP value-ARIC
explainer = shap.TreeExplainer(xgb_clf_final)
shap_values = explainer(X_aric)
feature_names = ['HDL', 'LDL', 'Triglycerides', 'Statin', 'Anti-Hypertensive', 'ASCVD']
#shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, X_aric, feature_names=feature_names, show=False)
plt.title("ARIC", weight = 'bold', fontsize=18)
plt.yticks(fontsize=14)
fig = plt.gcf()
fig.savefig(path + 'shap_aric.svg', format='svg')
plt.show()

# %%
#SHAP value-CARDIA
explainer = shap.TreeExplainer(xgb_clf_final)
shap_values = explainer(X_cardia)
feature_names = ['HDL', 'LDL', 'Triglycerides', 'Statin', 'Anti-Hypertensive', 'ASCVD']
#shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, X_cardia, feature_names=feature_names, show=False)
plt.title("CARDIA", weight = 'bold', fontsize=18)
plt.yticks(fontsize=14)
fig = plt.gcf()
fig.savefig(path + 'shap_cardia.svg', format='svg')
plt.show()

# %%
#SHAP value-MESA
explainer = shap.TreeExplainer(xgb_clf_final)
shap_values = explainer(X_mesa)
feature_names = ['HDL', 'LDL', 'Triglycerides', 'Statin', 'Anti-Hypertensive', 'ASCVD']
#shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, X_mesa, feature_names=feature_names, show=False)
plt.title("MESA", weight = 'bold', fontsize=18)
plt.yticks(fontsize=14)
fig = plt.gcf()
fig.savefig(path + 'shap_mesa.svg', format='svg')
plt.show()

# %%
#Finding optimal thresholds
fpr, tpr, thresholds = metrics.roc_curve(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1])
i = np.arange(len(tpr)) # index for df
xgb_perf_df = pd.DataFrame({#'fpr' : pd.Series(fpr, index=i),
                            'Sensitivity' : pd.Series(tpr, index = i), 
                            'Specificity' : pd.Series(1-fpr, index = i),
                            #'tf' : pd.Series(tpr - (1-fpr), index = i), 
                            'thresholds' : pd.Series(thresholds, index = i)})
#roc.iloc[(roc.tf-0).abs().argsort()[:1]]
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auroc = thresholds[optimal_idx]

xgb_perf_df['Prevalence'] = (y_train_class==1).sum()/len(y_train_class)
xgb_perf_df['AUROC'] = metrics.roc_auc_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1])
xgb_perf_df['AUPRC'] = metrics.average_precision_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1])
#xgb_perf_df['F1_Score'] = metrics.f1_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1])
#xgb_perf_df['Weighted_F1_Score'] = metrics.f1_score(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1], average='weighted')
xgb_perf_df['PPV'] = xgb_perf_df['Sensitivity']*xgb_perf_df['Prevalence']/((xgb_perf_df['Sensitivity']*xgb_perf_df['Prevalence'])
                                + ((1-xgb_perf_df['Specificity'])*(1-xgb_perf_df['Prevalence'])))
xgb_perf_df['NPV'] = xgb_perf_df['Specificity']*(1-xgb_perf_df['Prevalence'])/(
    (xgb_perf_df['Specificity']*(1-xgb_perf_df['Prevalence'])) + ((1-xgb_perf_df['Sensitivity'])*xgb_perf_df['Prevalence']))

def calculate_f1_score(row):
    precision = row['PPV']
    recall = row['Sensitivity']
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
xgb_perf_df['F1_Score'] = xgb_perf_df.apply(lambda row: calculate_f1_score(row), axis=1)
optimal_threshold_f1 = xgb_perf_df[xgb_perf_df['F1_Score']==xgb_perf_df['F1_Score'].max()]['thresholds'].iloc[0]

optimal_threshold_spec = xgb_perf_df[xgb_perf_df['Specificity']>0.9]['thresholds'].iloc[-1]

xgb_perf_df.to_csv("/Lp(a)/External_Validation/performance_threshold.csv", index=False)

print(f'Optimal Threshold for AUROC = {optimal_threshold_auroc}')
print(f'Optimal Threshold for F1 Score = {optimal_threshold_f1}')
print(f'Optimal Threshold for Optimal Specificity = {optimal_threshold_spec}')
xgb_perf_df

# %%
#Performance of ARISE across probability thresholds
data = xgb_perf_df[xgb_perf_df['thresholds']<=1]
data['smooth'] = savgol_filter(data['PPV'], window_length=1200, polyorder=2)
plt.plot('thresholds', 'Sensitivity', label='Sensitivity', data=data)
plt.plot('thresholds', 'smooth', label='PPV', data=data)
plt.plot('thresholds', 'F1_Score', label='F1 Score', data=data)
plt.plot('thresholds', 'Specificity', label='Specificity', data=data)
plt.plot('thresholds', 'NPV', label='NPV', data=data)
plt.xlabel('Threshold', fontsize=18, weight = 'bold')
plt.ylabel('Perfromance Metric', fontsize=18, weight = 'bold')
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
legend = plt.legend(bbox_to_anchor=(0.97, 0.82), loc="best", fontsize=18)
title = legend.get_title()
fig = plt.gcf()
fig.savefig(path + 'threshold.svg', format='svg')
plt.show()

# %%
#Performance Measures-Point estimates
def calculate_metrics(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = (TP / (TP + FN))
    specificity = (TN / (TN + FP))
    npv = (TN / (TN + FN))
    ppv = (TP / (TP + FP))
    prevalence = ((TP + FN) / (TP + TN + FP + FN))
    nnt_without_model = (1/prevalence)
    nnt_with_model = (1/ppv)
    nnt_relative_reduction = (nnt_without_model - nnt_with_model)*100/nnt_without_model
    
    return print(f'Sensitivity={round(sensitivity, 3)}, Specificity={round(specificity, 3)}, NPV={round(npv, 3)}, PPV={round(ppv, 3)}, Prevalence={round(prevalence, 3)}, Overall NNT={round(nnt_without_model, 0)}, Model NNT={round(nnt_with_model, 0)}, Relative Reduction of NNT={round(nnt_relative_reduction, 1)}%')

threshold = optimal_threshold_spec
print('Specific Threshold')
print('Train')
calculate_metrics(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1], threshold)
print('')
print('Test:')
calculate_metrics(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1], threshold)
print('')
print('ARIC:')
calculate_metrics(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1], threshold)
print('')
print('CARDIA:')
calculate_metrics(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1], threshold)
print('')
print('MESA:')
calculate_metrics(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1], threshold)
print('')
print('')
print('')

threshold = optimal_threshold_auroc
print('Threshold for Optimal AUROC')
print('Train')
calculate_metrics(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1], threshold)
print('')
print('Test:')
calculate_metrics(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1], threshold)
print('')
print('ARIC:')
calculate_metrics(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1], threshold)
print('')
print('CARDIA:')
calculate_metrics(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1], threshold)
print('')
print('MESA:')
calculate_metrics(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1], threshold)
print('')
print('')
print('')

threshold = optimal_threshold_f1
print('Threshold for Optimal F1 Score')
print('Train')
calculate_metrics(y_train_class, xgb_clf_final.predict_proba(X_train)[:, 1], threshold)
print('')
print('Test:')
calculate_metrics(y_test_class, xgb_clf_final.predict_proba(X_test)[:, 1], threshold)
print('')
print('ARIC:')
calculate_metrics(y_aric_class, xgb_clf_final.predict_proba(X_aric)[:, 1], threshold)
print('')
print('CARDIA:')
calculate_metrics(y_cardia_class, xgb_clf_final.predict_proba(X_cardia)[:, 1], threshold)
print('')
print('MESA:')
calculate_metrics(y_mesa_class, xgb_clf_final.predict_proba(X_mesa)[:, 1], threshold)

# %%
#Performance Measures-95% CI
def ppv_pe_ci(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    ppv = (TP / (TP + FP))
    se = np.sqrt((ppv * (1 - ppv)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = ppv - margin_of_error
    upper_bound = ppv + margin_of_error

    return f'{ppv:.3f} ({lower_bound:.3f}-{upper_bound:.3f})'


def npv_pe_ci(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    npv = (TN / (TN + FN))
    se = np.sqrt((npv * (1 - npv)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = npv - margin_of_error
    upper_bound = npv + margin_of_error

    return f'{npv:.3f} ({lower_bound:.3f}-{upper_bound:.3f})'

def sen_pe_ci(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = (TP / (TP + FN))
    se = np.sqrt((sensitivity * (1 - sensitivity)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = sensitivity - margin_of_error
    upper_bound = sensitivity + margin_of_error

    return f'{sensitivity:.3f} ({lower_bound:.3f}-{upper_bound:.3f})'

def spe_pe_ci(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    specificity = (TN / (TN + FP))
    se = np.sqrt((specificity * (1 - specificity)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = specificity - margin_of_error
    upper_bound = specificity + margin_of_error

    return f'{specificity:.3f} ({lower_bound:.3f}-{upper_bound:.3f})'

def prev_pe_ci(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    prevalence = ((TP + FN) / (TP + TN + FP + FN))
    se = np.sqrt((prevalence * (1 - prevalence)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = prevalence - margin_of_error
    upper_bound = prevalence + margin_of_error

    return f'{prevalence:.3f} ({lower_bound:.3f}-{upper_bound:.3f})'

def overall_nnt(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    prevalence = ((TP + FN) / (TP + TN + FP + FN))
    nnt_without_model = (1/prevalence)
    se = np.sqrt((nnt_without_model * (1 - nnt_without_model)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = nnt_without_model - (1/margin_of_error)
    upper_bound = nnt_without_model + (1/margin_of_error)
    
    return f'{nnt_without_model:.1f}'

def model_nnt(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    ppv = (TP / (TP + FP))
    nnt_with_model = (1/ppv)
    se = np.sqrt((nnt_with_model * (1 - nnt_with_model)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = nnt_with_model - margin_of_error
    upper_bound = nnt_with_model + margin_of_error
    
    return f'{nnt_with_model:.1f}'

def relative_reduction_nnt(y_true, y_prob, threshold):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate True Positive (TP), False Positive (FP), and Positive Predictive Value (PPV)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    ppv = (TP / (TP + FP))
    prevalence = ((TP + FN) / (TP + TN + FP + FN))
    nnt_without_model = (1/prevalence)
    nnt_with_model = (1/ppv)
    nnt_relative_reduction = (nnt_without_model - nnt_with_model)*100/nnt_without_model
    se = np.sqrt((nnt_relative_reduction * (1 - nnt_relative_reduction)) / (TP + TN + FP + FN))
    z_score = stats.norm.ppf(1 - (1 - 0.95) / 2)
    margin_of_error = z_score * se
    lower_bound = nnt_relative_reduction - margin_of_error
    upper_bound = nnt_relative_reduction + margin_of_error
    
    return f'{nnt_relative_reduction:.1f}%'

# %%
#Performance across probability thresholds in the study cohorts
length=7
#Specificity
performance_metrics_spe = pd.DataFrame({'Performance Metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Overall NNT', 'Model NNT', 'NNT Relative Reduction'],
                                    'UK Biobank Held-out Test Set': np.zeros(length),
                                    'ARIC': np.zeros(length),
                                    'CARDIA': np.zeros(length),
                                    'MESA': np.zeros(length),
                                    'Thresholds': ['Optimal Specificity', 'Optimal Specificity', 'Optimal Specificity', 'Optimal Specificity', 'Optimal Specificity',
                                                   'Optimal Specificity', 'Optimal Specificity']})
threshold = optimal_threshold_spec
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Sensitivity', 'UK Biobank Held-out Test Set'] = sen_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Specificity', 'UK Biobank Held-out Test Set'] = spe_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='PPV', 'UK Biobank Held-out Test Set'] = ppv_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NPV', 'UK Biobank Held-out Test Set'] = npv_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Overall NNT', 'UK Biobank Held-out Test Set'] = overall_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Model NNT', 'UK Biobank Held-out Test Set'] = model_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NNT Relative Reduction', 'UK Biobank Held-out Test Set'] = relative_reduction_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)

performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Sensitivity', 'ARIC'] = sen_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Specificity', 'ARIC'] = spe_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='PPV', 'ARIC'] = ppv_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NPV', 'ARIC'] = npv_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Overall NNT', 'ARIC'] = overall_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Model NNT', 'ARIC'] = model_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NNT Relative Reduction', 'ARIC'] = relative_reduction_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)

performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Sensitivity', 'CARDIA'] = sen_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Specificity', 'CARDIA'] = spe_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='PPV', 'CARDIA'] = ppv_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NPV', 'CARDIA'] = npv_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Overall NNT', 'CARDIA'] = overall_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Model NNT', 'CARDIA'] = model_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NNT Relative Reduction', 'CARDIA'] = relative_reduction_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)

performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Sensitivity', 'MESA'] = sen_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Specificity', 'MESA'] = spe_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='PPV', 'MESA'] = ppv_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NPV', 'MESA'] = npv_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Overall NNT', 'MESA'] = overall_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='Model NNT', 'MESA'] = model_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_spe.loc[performance_metrics_spe['Performance Metric']=='NNT Relative Reduction', 'MESA'] = relative_reduction_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)

#F1 Score
performance_metrics_f1 = pd.DataFrame({'Performance Metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Overall NNT', 'Model NNT', 'NNT Relative Reduction'],
                                    'UK Biobank Held-out Test Set': np.zeros(length),
                                    'ARIC': np.zeros(length),
                                    'CARDIA': np.zeros(length),
                                    'MESA': np.zeros(length),
                                    'Thresholds': ['Optimal F1 Score', 'Optimal F1 Score', 'Optimal F1 Score', 'Optimal F1 Score', 'Optimal F1 Score',
                                                   'Optimal F1 Score', 'Optimal F1 Score']})
threshold = optimal_threshold_f1
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Sensitivity', 'UK Biobank Held-out Test Set'] = sen_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Specificity', 'UK Biobank Held-out Test Set'] = spe_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='PPV', 'UK Biobank Held-out Test Set'] = ppv_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NPV', 'UK Biobank Held-out Test Set'] = npv_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Overall NNT', 'UK Biobank Held-out Test Set'] = overall_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Model NNT', 'UK Biobank Held-out Test Set'] = model_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NNT Relative Reduction', 'UK Biobank Held-out Test Set'] = relative_reduction_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)

performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Sensitivity', 'ARIC'] = sen_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Specificity', 'ARIC'] = spe_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='PPV', 'ARIC'] = ppv_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NPV', 'ARIC'] = npv_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Overall NNT', 'ARIC'] = overall_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Model NNT', 'ARIC'] = model_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NNT Relative Reduction', 'ARIC'] = relative_reduction_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)

performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Sensitivity', 'CARDIA'] = sen_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Specificity', 'CARDIA'] = spe_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='PPV', 'CARDIA'] = ppv_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NPV', 'CARDIA'] = npv_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Overall NNT', 'CARDIA'] = overall_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Model NNT', 'CARDIA'] = model_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NNT Relative Reduction', 'CARDIA'] = relative_reduction_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)

performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Sensitivity', 'MESA'] = sen_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Specificity', 'MESA'] = spe_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='PPV', 'MESA'] = ppv_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NPV', 'MESA'] = npv_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Overall NNT', 'MESA'] = overall_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='Model NNT', 'MESA'] = model_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_f1.loc[performance_metrics_f1['Performance Metric']=='NNT Relative Reduction', 'MESA'] = relative_reduction_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)

#Youdens Index
performance_metrics_you = pd.DataFrame({'Performance Metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Overall NNT', 'Model NNT', 'NNT Relative Reduction'],
                                    'UK Biobank Held-out Test Set': np.zeros(length),
                                    'ARIC': np.zeros(length),
                                    'CARDIA': np.zeros(length),
                                    'MESA': np.zeros(length),
                                    'Thresholds': ['Optimal Youdens Index', 'Optimal Youdens Index', 'Optimal Youdens Index', 'Optimal Youdens Index', 'Optimal Youdens Index',
                                                   'Optimal Youdens Index', 'Optimal Youdens Index']})
threshold = optimal_threshold_auroc
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Sensitivity', 'UK Biobank Held-out Test Set'] = sen_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Specificity', 'UK Biobank Held-out Test Set'] = spe_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='PPV', 'UK Biobank Held-out Test Set'] = ppv_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NPV', 'UK Biobank Held-out Test Set'] = npv_pe_ci((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Overall NNT', 'UK Biobank Held-out Test Set'] = overall_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Model NNT', 'UK Biobank Held-out Test Set'] = model_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NNT Relative Reduction', 'UK Biobank Held-out Test Set'] = relative_reduction_nnt((ukb_test['lp_a_value']>=150).astype(int), ukb_test['high_lp_a_probability'], threshold)

performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Sensitivity', 'ARIC'] = sen_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Specificity', 'ARIC'] = spe_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='PPV', 'ARIC'] = ppv_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NPV', 'ARIC'] = npv_pe_ci((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Overall NNT', 'ARIC'] = overall_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Model NNT', 'ARIC'] = model_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NNT Relative Reduction', 'ARIC'] = relative_reduction_nnt((aric['lp_a_value']>=150).astype(int), aric['high_lp_a_probability'], threshold)

performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Sensitivity', 'CARDIA'] = sen_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Specificity', 'CARDIA'] = spe_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='PPV', 'CARDIA'] = ppv_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NPV', 'CARDIA'] = npv_pe_ci((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Overall NNT', 'CARDIA'] = overall_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Model NNT', 'CARDIA'] = model_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NNT Relative Reduction', 'CARDIA'] = relative_reduction_nnt((cardia['lp_a_value']>=150).astype(int), cardia['high_lp_a_probability'], threshold)

performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Sensitivity', 'MESA'] = sen_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Specificity', 'MESA'] = spe_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='PPV', 'MESA'] = ppv_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NPV', 'MESA'] = npv_pe_ci((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Overall NNT', 'MESA'] = overall_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='Model NNT', 'MESA'] = model_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)
performance_metrics_you.loc[performance_metrics_you['Performance Metric']=='NNT Relative Reduction', 'MESA'] = relative_reduction_nnt((mesa['lp_a_value']>=150).astype(int), mesa['high_lp_a_probability'], threshold)

#Concatenation
performance_metrics = pd.concat([performance_metrics_spe, performance_metrics_f1, performance_metrics_you])
performance_metrics.to_csv(path + 'performance metrics.csv', index=False)

# %%
#95% CI
def bootstrap_auroc(y_true, y_pred, n_iterations=1000, confidence_level=0.95):
    y_true = y_true.reset_index(drop=True, inplace=False)
    y_pred = pd.Series(y_pred).reset_index(drop=True, inplace=False)
    
    # Initialize an array to store the AUROC values from each bootstrap iteration
    auroc_values = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Create a bootstrap sample by resampling with replacement
        sample_indices = resample(range(len(y_true)))
        y_true_bootstrap = y_true[sample_indices]
        y_pred_bootstrap = y_pred[sample_indices]

        # Calculate AUROC for the bootstrap sample
        auroc_values[i] = metrics.roc_auc_score(y_true_bootstrap, y_pred_bootstrap)

    # Calculate the confidence interval
    lower_bound = np.percentile(auroc_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auroc_values, (1 + confidence_level) / 2 * 100)

    # Print the results
    return f"{metrics.roc_auc_score(y_true, y_pred):.3f} ({lower_bound:.3f}-{upper_bound:.3f})"

def bootstrap_auprc(y_true, y_pred, n_iterations=1000, confidence_level=0.95):
    y_true = y_true.reset_index(drop=True, inplace=False)
    y_pred = pd.Series(y_pred).reset_index(drop=True, inplace=False)
    
    # Initialize an array to store the AUROC values from each bootstrap iteration
    auprc_values = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Create a bootstrap sample by resampling with replacement
        sample_indices = resample(range(len(y_true)))
        y_true_bootstrap = y_true[sample_indices]
        y_pred_bootstrap = y_pred[sample_indices]

        # Calculate AUROC for the bootstrap sample
        auprc_values[i] = metrics.average_precision_score(y_true_bootstrap, y_pred_bootstrap)

    # Calculate the confidence interval
    lower_bound = np.percentile(auprc_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auprc_values, (1 + confidence_level) / 2 * 100)

    # Print the results
    return f"{metrics.average_precision_score(y_true, y_pred):.3f} ({lower_bound:.3f}-{upper_bound:.3f})"

# %%
#Performance across subgroups in UKBB

#OneHotEncoding for UKBB
print(ukb_test['ethnicity'].unique())
ukb_test['ethnicity'] = ukb_test['ethnicity'].fillna(1)
print(ukb_test['ethnicity'].unique())
ukb_test['white_ethnicity'] = np.where(ukb_test['ethnicity']==1, 1, 0)
ukb_test['mixed_ethnicity'] = np.where(ukb_test['ethnicity']==2, 1, 0)
ukb_test['south_asian_ethnicity'] = np.where(ukb_test['ethnicity']==3, 1, 0)
ukb_test['black_ethnicity'] = np.where(ukb_test['ethnicity']==4, 1, 0)
ukb_test['chinese_ethnicity'] = np.where(ukb_test['ethnicity']==5, 1, 0)
ukb_test['other_ethnicity'] = np.where(ukb_test['ethnicity']==6, 1, 0)
ukb_test['female_sex'] = np.where(ukb_test['sex']==0, 1, 0)
ukb_test['male_sex'] = np.where(ukb_test['sex']==1, 1, 0)

#Dataframe for subgroup analysis
length=26
subgroup_ukbb = pd.DataFrame({'dataset' : np.zeros(length),
                            'Subpopulation' : ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 'south_asian_ethnicity', 'chinese_ethnicity',
                                            'htn', 'without_htn', 'dm', 'without_dm', 'ihd', 'without_ihd', 'heart_failure', 'without_heart_failure', 'ascvd', 'without_ascvd', 'premature_ascvd_cvpx', 'without_premature_ascvd_cvpx',
                                                'aspirin', 'without_aspirin', 'statin', 'without_statin', 'anti_htn', 'without_anti_htn',
                                            'cvd_family_history', 'without_cvd_family_history'], 
                            'AUROC' : np.zeros(length),
                            'AUPRC' : np.zeros(length),
                            'Prevalence' : np.zeros(length),
                            'PPV' : np.zeros(length),
                            'Relative Reduction of NNT' : np.zeros(length),
                            'Sensitivity' : np.zeros(length),
                            'Specificity' : np.zeros(length),
                            'NPV' : np.zeros(length)})
#UKBB Test
threshold = optimal_threshold_spec
for i in ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 'south_asian_ethnicity', 'chinese_ethnicity', 'mixed_ethnicity', 
          'mixed_ethnicity', 'statin', 'anti_htn', 'aspirin', 'ihd', 'ascvd', 'premature_ascvd_cvpx', 'dm', 'htn', 'heart_failure', 'cvd_family_history']:
    df_subset = ukb_test[ukb_test[i]==1]
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']==i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)

for i in ['statin',  'anti_htn', 'aspirin', 'ihd', 'ascvd', 'premature_ascvd_cvpx', 'dm', 'htn', 'heart_failure', 'cvd_family_history']:
    df_subset = ukb_test[ukb_test[i]==0]
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_ukbb.loc[subgroup_ukbb['Subpopulation']=='without_' + i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)

subgroup_ukbb['dataset'] = 'ukbb_test'
subgroup_ukbb

# %%
#Performance across subgroups in ARIC
#OneHotEncoding for ARIC
aric['white_ethnicity'] = np.where(aric['ethnicity']=='White', 1, 0)
aric['black_ethnicity'] = np.where(aric['ethnicity']=='African-American', 1, 0)
aric['female_sex'] = np.where(aric['sex']==0, 1, 0)
aric['male_sex'] = np.where(aric['sex']==1, 1, 0)

#Dataframe for subgroup analysis
length=20
subgroup_aric = pd.DataFrame({'dataset' : np.zeros(length),
                            'Subpopulation' : ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 
                                                'htn', 'without_htn', 'dm', 'without_dm', 'heart_failure', 'without_heart_failure', 'ascvd', 'without_ascvd',
                                               'cvd_family_history', 'without_cvd_family_history', 'premature_chd_family_history', 'without_premature_chd_family_history',
                                               'statin', 'without_statin', 'anti_htn', 'without_anti_htn'], 
                            'AUROC' : np.zeros(length),
                            'AUPRC' : np.zeros(length),
                            'Prevalence' : np.zeros(length),
                            'PPV' : np.zeros(length),
                            'Relative Reduction of NNT' : np.zeros(length),
                            'Sensitivity' : np.zeros(length),
                            'Specificity' : np.zeros(length),
                            'NPV' : np.zeros(length)})
#UKBB Test
threshold = optimal_threshold_spec
for i in ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 
            'ascvd', 'dm', 'htn', 'heart_failure', 
            'statin', 'anti_htn',
            'cvd_family_history', 'premature_chd_family_history']:
    df_subset = aric[aric[i]==1]
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']==i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)

for i in ['ascvd', 'dm', 'htn', 'heart_failure', 
            'statin', 'anti_htn',
            'cvd_family_history', 'premature_chd_family_history']:
    df_subset = aric[aric[i]==0]
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_aric.loc[subgroup_aric['Subpopulation']=='without_' + i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)

subgroup_aric['dataset'] = 'ARIC'

# %%
#Performance across subgroups in CARDIA
#OneHotEncoding for CARDIA
cardia['white_ethnicity'] = np.where(cardia['ethnicity']=='White', 1, 0)
cardia['black_ethnicity'] = np.where(cardia['ethnicity']=='African-American', 1, 0)
cardia['female_sex'] = np.where(cardia['sex']==0, 1, 0)
cardia['male_sex'] = np.where(cardia['sex']==1, 1, 0)

#Dataframe for subgroup analysis
length=20
subgroup_cardia = pd.DataFrame({'dataset' : np.zeros(length),
                            'Subpopulation' : ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 
                                              'htn', 'without_htn', 'dm', 'without_dm', 'ihd', 'without_ihd', 'ascvd', 'without_ascvd',
                                               'cvd_family_history', 'without_cvd_family_history', 'premature_chd_family_history', 'without_premature_chd_family_history',
                                               'statin', 'without_statin', 'anti_htn', 'without_anti_htn'], 
                            'AUROC' : np.zeros(length),
                            'AUPRC' : np.zeros(length),
                            'Prevalence' : np.zeros(length),
                            'PPV' : np.zeros(length),
                            'Relative Reduction of NNT' : np.zeros(length),
                            'Sensitivity' : np.zeros(length),
                            'Specificity' : np.zeros(length),
                            'NPV' : np.zeros(length)})

threshold = optimal_threshold_spec
for i in ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 
          'ascvd', 'ihd', 'dm', 'htn',
            'statin', 'anti_htn',
            'cvd_family_history', 'premature_chd_family_history']:
    df_subset = cardia[cardia[i]==1]
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'AUROC'] = metrics.roc_auc_score(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
  
for i in ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 
           'htn',
           'cvd_family_history', 'premature_chd_family_history']:
    df_subset = cardia[cardia[i]==1]
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']==i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])

for i in ['ascvd', 'ihd', 'dm', 'htn',
            'statin', 'anti_htn',
            'cvd_family_history', 'premature_chd_family_history']:
    df_subset = cardia[cardia[i]==0]
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_cardia.loc[subgroup_cardia['Subpopulation']=='without_' + i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)

subgroup_cardia['dataset'] = 'CARDIA'

# %%
#Performance across subgroups in MESA
#OneHotEncoding for MESA
mesa['white_ethnicity'] = np.where(mesa['ethnicity']=='Caucasian', 1, 0)
mesa['black_ethnicity'] = np.where(mesa['ethnicity']=='African-American', 1, 0)
mesa['hispanic_ethnicity'] = np.where(mesa['ethnicity']=='Hispanic', 1, 0)
mesa['chinese_ethnicity'] = np.where(mesa['ethnicity']=='Chinese', 1, 0)
mesa['female_sex'] = np.where(mesa['sex']==0, 1, 0)
mesa['male_sex'] = np.where(mesa['sex']==1, 1, 0)

#Dataframe for subgroup analysis
length=18
subgroup_mesa = pd.DataFrame({'dataset' : np.zeros(length),
                            'Subpopulation' : ['female_sex', 'male_sex', 'black_ethnicity', 'hispanic_ethnicity', 'white_ethnicity', 'chinese_ethnicity',
                                               'htn', 'without_htn', 'dm', 'without_dm',
                                               'cvd_family_history', 'without_cvd_family_history',
                                                'aspirin', 'without_aspirin', 'statin', 'without_statin', 'anti_htn', 'without_anti_htn'], 
                            'AUROC' : np.zeros(length),
                            'AUPRC' : np.zeros(length),
                            'Prevalence' : np.zeros(length),
                            'PPV' : np.zeros(length),
                            'Relative Reduction of NNT' : np.zeros(length),
                            'Sensitivity' : np.zeros(length),
                            'Specificity' : np.zeros(length),
                            'NPV' : np.zeros(length)})
#UKBB Test
threshold = optimal_threshold_spec
for i in ['female_sex', 'male_sex', 'black_ethnicity', 'white_ethnicity', 'hispanic_ethnicity', 'chinese_ethnicity',
            'dm', 'htn',
            'anti_htn', 'aspirin',
            'cvd_family_history']:
    df_subset = mesa[mesa[i]==1]
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']==i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)

for i in ['dm', 'htn',
            'anti_htn', 'aspirin',
            'cvd_family_history']:
    df_subset = mesa[mesa[i]==0]
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'AUROC'] = bootstrap_auroc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'PPV'] = ppv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'Prevalence'] =  prev_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'Relative Reduction of NNT'] = relative_reduction_nnt(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'AUPRC'] = bootstrap_auprc(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'])
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'Sensitivity'] = sen_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'NPV'] = npv_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)
    subgroup_mesa.loc[subgroup_mesa['Subpopulation']=='without_' + i, 'Specificity'] = spe_pe_ci(df_subset['lp_a_value']>=150, df_subset['high_lp_a_probability'], threshold)

subgroup_mesa['dataset'] = 'MESA'
subgroup_mesa

# %%
#Combining all subgroup dataframes
subgroup = pd.concat([subgroup_ukbb, subgroup_aric, subgroup_cardia, subgroup_mesa])
subgroup.to_csv('/Lp(a)/External_Validation/subgroup.csv', index=False)
subgroup


