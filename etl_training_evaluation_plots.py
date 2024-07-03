# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:53:36 2024

@author: garic
"""
#%% Libraries

# import sys
# import os

import numpy as np
import pandas as pd
# import random
from google.cloud import bigquery
# import scipy.stats.distributions as dist
from  matplotlib import pyplot


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from scipy.stats import randint

# import seaborn as sns
# from scipy import stats

#%% Functions

def query_bigquery_db(query):
    client = bigquery.Client()
    query_job = client.query(query)
    df = query_job.to_dataframe()
    return df

#%% Data

jobs = query_bigquery_db(query = """
WITH
number_of_required_skills AS
(
SELECT j.job_id, COUNT(DISTINCT js.skill_id) AS number_of_required_skills
FROM `andela-data-lake.jobs_system_production_new_public.job` j
  LEFT JOIN `andela-data-lake.jobs_system_production_new_public.job_skill` js ON j.job_id = js.job_id
WHERE 1=1
  AND js.is_deleted IS FALSE
  AND js.required IS TRUE
GROUP BY j.job_id
),
number_of_optional_skills AS
(
SELECT j.job_id, COUNT(DISTINCT js.skill_id) AS number_of_optional_skills
FROM `andela-data-lake.jobs_system_production_new_public.job` j
  LEFT JOIN `andela-data-lake.jobs_system_production_new_public.job_skill` js ON j.job_id = js.job_id
WHERE 1=1
  AND js.is_deleted IS FALSE
  AND js.required IS FALSE
GROUP BY j.job_id
)
SELECT
  j.job_id,
  j.role_type,
  r.name AS job_role,
  c.name,
  IF(j.language_requirements != '[]', 1,0) AS language_requirements,
  IF(j.matcher_id != '[]', 1,0) AS matcher_id,
  j.created_at,
  DATE_DIFF(CURRENT_DATE(), j.created_at, DAY) AS created_at_diff,
  j.claimed_date,
  DATE_DIFF(CURRENT_DATE(), j.claimed_date, DAY) AS claimed_date_diff,
  j.match_ready_date,
  DATE_DIFF(CURRENT_DATE(), j.match_ready_date, DAY) AS match_ready_date_diff,
  ARRAY_LENGTH(SPLIT(REGEXP_REPLACE(j.description, r'[^a-zA-Z0-9]+', ' '), ' ')) AS job_description_length,
  IF(j.requires_management_experience IS TRUE,1,0) AS requires_management_experience,
  IF(j.device_requirements != '',1,0) AS device_requirements,
  j.estimated_duration,
  IF(j.is_from_client_portal IS TRUE,1,0) AS is_from_client_portal,
  IF(j.is_replacement IS TRUE,1,0) AS is_replacement,
  IF(j.keywords != '[]',1,0) AS keywords,
  IF(nors.number_of_required_skills IS NULL, 0, nors.number_of_required_skills) AS number_of_required_skills,
  IF(noos.number_of_optional_skills IS NULL, 0, noos.number_of_optional_skills) AS number_of_optional_skills,
  IF(j.location != '',1,0) AS location,
  j.office_consideration_type,
  j.location_reason,
  j.country,
  j.max_monthly_budgeted,
  j.timezone,
  -- j.minimum_overlapping_hours,
  j.priority,
  IF(j.status='Closed Filled',1,0) AS status
FROM `andela-data-lake.jobs_system_production_new_public.job` j
  LEFT JOIN `andela-data-lake.andela_match_platform_public.myandela_matcher` m ON j.matcher_id = m.matcher_id
  LEFT JOIN `andela-data-lake.jobs_system_production_new_public.opportunity` o ON j.opportunity_id = o.opportunity_id
  LEFT JOIN `andela-data-lake.jobs_system_production_new_public.client` c ON o.client_id = c.client_id
  -- LEFT JOIN `andela-data-lake.jobs_system_production_new_public.job_skill` js ON j.job_id = js.job_id
  LEFT JOIN `andela-data-lake.jobs_system_production_new_public.role` r ON j.role_id = r.role_id
  LEFT JOIN number_of_required_skills nors ON j.job_id = nors.job_id
  LEFT JOIN number_of_optional_skills noos ON j.job_id = noos.job_id
WHERE 1=1
  AND j.match_ready_date BETWEEN '2023-01-01' AND '2024-04-30'
  AND (j.is_deleted IS FALSE OR j.is_deleted IS NULL)
  AND (m.testing_user IS FALSE OR m.testing_user IS NULL)
  AND j.payroll_passthrough IS FALSE
  AND j.monetized_talent_delivery_manager IS FALSE
  AND j.status IN ('Closed Filled', 'Closed Lost')
  AND LOWER(j.description) != 'test'
  -- AND (j.description IS NULL OR j.description = '')
;
  """
  )

#%% Dates

jobs['year_created_at'] = pd.DatetimeIndex(jobs.created_at).year
jobs['month_created_at'] = pd.DatetimeIndex(jobs.created_at).month
jobs['day_created_at'] = pd.DatetimeIndex(jobs.created_at).day

jobs['year_claimed_date'] = pd.DatetimeIndex(jobs.claimed_date).year
jobs['month_claimed_date'] = pd.DatetimeIndex(jobs.claimed_date).month
jobs['day_claimed_date'] = pd.DatetimeIndex(jobs.claimed_date).day

jobs['year_match_ready_date'] = pd.DatetimeIndex(jobs.match_ready_date).year
jobs['month_match_ready_date'] = pd.DatetimeIndex(jobs.match_ready_date).month
jobs['day_match_ready_date'] = pd.DatetimeIndex(jobs.match_ready_date).day

jobs = jobs.drop(columns=['created_at','claimed_date','match_ready_date'])

#%% Recommendable talent today

recommendable_talent_today = pd.read_csv(r'C:\Users\garic\OneDrive\Escritorio\maestria_2024\taller_de_tesis_2024\recommendable_talent_today.csv')
jobs = pd.merge(jobs, recommendable_talent_today, how='left', left_on='job_id', right_on='job_id')

#%% Multiple one-hot encoding

# Initialize the OneHotEncoder and ColumnTransformer
encoder = OneHotEncoder(sparse_output=False)
transformer = ColumnTransformer(
    transformers=[
        ('cat', encoder, ['role_type', 'job_role', 'office_consideration_type','location_reason','country','priority'])
    ],
    remainder='passthrough'  # This keeps the other columns (e.g., 'Value') in the DataFrame
)

# Fit and transform the data
transformed_data = transformer.fit_transform(jobs)

# Get the new column names after one-hot encoding
encoded_columns = transformer.named_transformers_['cat'].get_feature_names_out(['role_type', 'job_role','office_consideration_type','location_reason','country','priority'])
# all_columns = np.hstack([encoded_columns, jobs.columns.difference(['role_type', 'job_role'])])
all_columns = np.hstack([encoded_columns, jobs.drop(columns=['role_type','job_role','office_consideration_type','location_reason','country','priority']).columns])

# Create a new DataFrame with the transformed data
df_encoded = pd.DataFrame(transformed_data, columns=all_columns)
df_encoded = df_encoded.drop(columns=['job_id','name'])

# Format columns
df_encoded = df_encoded.apply(pd.to_numeric)

# Impute missing values with the column mean
df_imputed = df_encoded.apply(lambda col: col.fillna(col.mean()), axis=0)

# df_encoded.to_csv(r'C:\Users\garic\Downloads\encoding_test1.csv')

#%% Train-test split

X = df_imputed.drop(columns=['status'])
y = df_imputed[['status']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

#%% Decision tree - Randomized search cross validation

param_dist_tree = {
    'splitter': ['best','random'],
    'criterion': ['gini','entropy','log_loss'],
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1,20),
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
    }

# Create a base model
tree = DecisionTreeClassifier(random_state=0)

# Initialize RandomizedSearchCV
random_search_tree = RandomizedSearchCV(estimator=tree, param_distributions=param_dist_tree,
                                   n_iter=100, cv=5, scoring='accuracy', random_state=42)

# Fit the RandomizedSearchCV instance
random_search_tree.fit(X_train, y_train)

# Get the best parameters and the best score
print("Best Parameters:", random_search_tree.best_params_)
print("Best CV Score:", random_search_tree.best_score_)

# Predict with the best model
best_tree = random_search_tree.best_estimator_
y_pred_tree = best_tree.predict(X_test)
y_true = y_test
# accuracy_score(y_true, y_pred)
accuracy_score(y_true, y_pred_tree, normalize=True)

# Feature importance
feat_importance_tree = best_tree.tree_.compute_feature_importances(normalize=False)
# print("feat importance = " + str(feat_importance))
features_by_importance_tree = pd.DataFrame(df_imputed.drop(columns='status').columns.to_numpy(), feat_importance_tree).reset_index().rename(columns={'index':'importance', 0:'feature'}).sort_values(by='importance', ascending = False)[:10]

#%% Random Forest - Randomized search cross validation

# Define the parameter grid
param_dist_rf = {
    'n_estimators': randint(10, 1000),  # Number of trees in random forest
    'max_depth': randint(1, 20),        # Maximum depth of trees
    'min_samples_split': randint(2, 20), # Minimum samples required to split a node
    'min_samples_leaf': randint(1, 20)   # Minimum samples required at each leaf node
}

# Create a base model
rf = RandomForestClassifier()

# Initialize RandomizedSearchCV
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf,
                                   n_iter=100, cv=5, scoring='accuracy', random_state=42, verbose=0)

# Fit the RandomizedSearchCV instance
random_search_rf.fit(X_train, y_train)

# Get the best parameters and the best score
print("Best Parameters:", random_search_rf.best_params_)
print("Best CV Score:", random_search_rf.best_score_)

# Predict with the best model
best_rf = random_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_true = y_test
# accuracy_score(y_true, y_pred)
accuracy_score(y_true, y_pred_rf, normalize=True)

# Feature importance
feat_importance_rf = best_rf.feature_importances_
# print("feat importance = " + str(feat_importance))
features_by_importance_rf = pd.DataFrame(df_imputed.drop(columns='status').columns.to_numpy(), feat_importance_rf).reset_index().rename(columns={'index':'importance', 0:'feature'}).sort_values(by='importance', ascending = False)[:10]

#%% Getting the decision tree

pyplot.figure(figsize=(60, 30))  # Adjust the figure size as needed
# plot_tree(best_tree, filled=True)
plot_tree(best_tree, filled=True, feature_names=X.columns, fontsize=15)
pyplot.show()

#%% Ploting most important features

# Decision tree
pyplot.figure(figsize=(10, 6))
pyplot.barh(features_by_importance_tree['feature'], features_by_importance_tree['importance'], align='center')
pyplot.xlabel('Feature Importance')
pyplot.title('Feature Importance of Decision tree')
pyplot.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
pyplot.show()

# Random forest
pyplot.figure(figsize=(10, 6))
pyplot.barh(features_by_importance_rf['feature'], features_by_importance_rf['importance'], align='center')
pyplot.xlabel('Feature Importance')
pyplot.title('Feature Importance of Random Forest')
pyplot.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
pyplot.show()

#%% Investigating specific variables

#%% Country_none

df_country_analysis = df_imputed[df_imputed['country_None']==1][['country_None','status']].groupby(by='status').count().reset_index()

# df_country_analysis = df_imputed[df_imputed['country_None']==0][['country_None','status']].groupby(by='status').count().reset_index()

# Plotting the bar chart
pyplot.figure(figsize=(8, 6))  # Adjust the figure size as needed

pyplot.bar(df_country_analysis['status'], df_country_analysis['country_None'], color='skyblue')

# Add labels and title
pyplot.xlabel('Éxito del puesto de trabajo (1 = exitoso)')
pyplot.ylabel('Cantidad de puestos de trabajo')
pyplot.title('Relación entre País (valor nulo) y éxito del puesto de trabajo')

# Show the plot
pyplot.show()

#%% location

# df_location_analysis = df_imputed[df_imputed['location']==1][['location','status']].groupby(by='status').count().reset_index()

df_location_analysis = df_imputed[df_imputed['location']==0][['location','status']].groupby(by='status').count().reset_index()

# Plotting the bar chart
pyplot.figure(figsize=(8, 6))  # Adjust the figure size as needed

pyplot.bar(df_location_analysis['status'], df_location_analysis['location'], color='skyblue')

# Add labels and title
pyplot.xlabel('Éxito del puesto de trabajo (1 = exitoso)')
pyplot.ylabel('Cantidad de puestos de trabajo')
pyplot.title('Relación entre Locación y éxito del puesto de trabajo')

# Show the plot
pyplot.show()