# Capstone 2: Hospital Resource Management
Capstone-2 submission: A predictive analysis for Resource Management in hospitals

### Link to Jupyter notebook with analysis:
https://github.com/purnamididuddi/capstone1_hospital_resource_mgmt/blob/main/healthcare_analysis.ipynb


### Problem Description:
Resource management is very critical in any business/vertical. Often times, businesses lose money due to un-informed resource allocation. It is even more important in hospitals as it impacts how many patients the hospital can serve and thus the health/life of patients.
In this project, I have chosen the below dataset from Kaggle:
https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii

The goal is to help hospital management to predict the length-of-stay of a patient. This knowledge will help the management streamline necessary resources and better manage their overall resources

### Excerpts of the analysis from the Jupyter notebook:
##### Many models were built in an effort to give best results to our customer (i.e., hospital management). Below is a summary of each of their performance

| <span style='background:orange'>Model # | <span style='background:orange'>Model Description | <span style='background:orange'>Parameters | <span style='background:orange'># of Dimensions | <span style='background:orange'>Train/Dev/Test Accuracies | <span style='background:orange'>F1 Score on Testset |
| :-: | :-: | :-: | :-: | :-: | :-: |
|<span style='background:yellow'> Model 0 | Random guessing | For a baseline | --- | 9.09 | --- |
|<span style='background:yellow'> Model 1 | Logistic Regr | Initial logistic regr model | --- | 38.91/38.85/38.65 | 38.65 |
|<span style='background:yellow'> Model 2 | Logistic Regr with OneHot on all categ | Consolidated onehot encoding | 130 | 38.97/38.89/38.87 | 38.87 |
|<span style='background:yellow'> Model 3 | GridSearch Logistic Regr hyper params | {'lgr__C': 0.18957356524063793, 'lgr__max_iter... | 130 | 40.13/40.14/40.13 | 40.13 |
|<span style='background:yellow'> Model 4 | GridSearch kNN hyper params |{'knn__n_neighbors': 85} | 130 | 40.83/39.21/39.25 | 39.25|
|<span style='background:yellow'> Model 5 | Decision Tree 1 | Moderate depth of 10 | 130 | 42.14/41.22/41.22 | 41.22 |
|<span style='background:yellow'> Model 6 | Decision Tree 2 | Deeper tree (max-depth=100) | 130 | 99.96/30.07/30.04 | 30.04 |
|<span style='background:yellow'> Model 7 | Gridsearch on Decision tree | {'dt__max_depth': 10, 'dt__min_impurity_decrea... | 130 | 39.58/40.04/39.98 | 39.98 |
|<span style='background:yellow'> Model 8 | Gridsearch on Feature extraction alpha | {'selector__estimator__alpha': 0.01} | 30 | 41.7/40.65/40.42 | 40.42 |
|<span style='background:yellow'> Model 9 | Adaboost to compensate for high bias | {'boost__n_estimators': 25} | 30 | 36.96/37.32/37.38 | 37.38 |
|<span style='background:yellow'> Model 10 | Gradientboost to compensate for high bias | {'gboost__n_estimators': 200} | 30 | 41.58/41.3/41.19 | 41.19 |
|<span style='background:yellow'> Model 11 | RandomForests with deep trees to compensate for high variance | {'n_estimators': 50  'max_depth': 100} | 30 | 99.81/33.98/33.79 | 33.79 |

    
### Recommendation:
Model 5 and Model 10 in the above table, have the highest predictive capability of all the models. 


If I were to recommend using a model, I would recommend Model 10, because of the fact that it is an ensemble technique and a better diverse model compared to a single decision tree.


### Summary of the findings:
#### From the Data Understanding stage:
1. The Admission_deposit has a very nice normal distribution curve. No fine tuning needed for this column

2. A given patientid has multiple entries. This could be because the patient got admitted more than once. However, the case_ids of a patient are consecutive; this looks a bit weird and hinting that only for one admission, multiple entries got added (due to clerical error or different departments, etc). This might skew the results.

3. Very high percentage of patients were admitted into Gynaecology department. 

4. Another observation is that "anesthesia" is a department by itself. A patient wouldn't get admitted just for getting anesthesia. I strongly believe these patients were actually admitted in regular departments who happened to need anesthesia for the treatment. Need to see if this 'superficial' department will impact the predictions. Let's keep this in mind for any fine-tuning the results

5. Bulk of the Stays are between 11 days to 40days. 
     a. Of these Stays, there are more 'Moderate' Severity-of-illness cases than 'Extreme' Severity
     b. Of these Stays, the patients with Bed_grades 3 and 4 are more than Bed_grades 1 and 2
     c. Of these Stays, most patients are from Gynaecology Department. This is not new information as the total patients admitted are in that department anyway
    
#### Summary of the Models and Conclusion:
I have built several models, as shown in the table above, in the goal of improving the prediction performance while keeping the model complexity in check. Overall, the models have performed much better in comparison with someone doing random guessing of predicting the Stay of a patient. However, most models had tough time raising the prediction accuracy beyond 45%.

This is a challenging dataset: There are 11 classes in the target variable. In general with datasets like this, the odds of predicting the correct class slims down in comparison with datasets with fewer classes in the target variable. Also, the distribution of classes in target variable is very uneven.

Only, the very deep Decision trees have exhibited very high training accuracy. The rest of the models had 40-45% accuracy even with the training data, pointing that there was high bias. Even the deep decision trees had accuracy on the lower side with the dev and test data, again pointing that there was high bias.

The AUC in the ROC curves of some target classes were good while the other target classes weren't performing well. This is reflected in the overall F1 score.

The ensemble techniques did not improve the performance either. One potential reason for this could be because of (2) listed in the Data Understanding stage. The inherent requirement of Ensemble techiniques, which is: the crowd should have independent opinions. In this dataset, there are many Case_ids per Patient_id; this might be intuitively violating the above inherent requirement, as there are multiple records for a single patient (that is, the crowd is not independent).

The top 5 features that has most impact on predicting the target variable in our recommended model (Model 10) are: (i) Visitors (ii)Ward_type (iii)Bed_grade (iv) Admission_deposit (v) Severity_illness

One interesting observation from the Confusion matrix of Model 10 (and also other models) is that, The False-Positives and False-Negatives of a given target class lie mostly in the adjacent class; for ex: for True class = 2 (a.k.a. 21-30 days of Stay), Model 10 correctly predicted 5555 cases as Predicted class = 2. There are 2699 cases that Model is predicting as class = 1 whose True class = 2 (ie, False Negatives). This is observed in many target classes. This is useful intuition because, if the target classes were reduced from 11 to say 3 or say 4, the prediction performance would be lot higher. In other words, for a non-technical person, if the tool is predicting as class = X, be a little more generic and do the resource management for classes X-1, X and X+1 for that case. The odds of succeeded will improve with this scheme. Given the limited features in the data, this is probably a decent compromise for the user of the model.


    
### Summary of the project to non-technical audience:

The goal of the project is to provide a 'tool' to the Hospital management in estimating the length of Stay of a patient, given certain characteristics of the patient.

The tool takes these characteristics of the patient as input from the Hospital management, then does a statistical analysis based of historical information and provides the management with an estimated length of Stay. This helps the Hospital management to align the necessary resources for that amount of time for the given patient.

With the limited amount of historical data provided to the tool, this project work has found that the tool can estimate the length of Stay with a 42% correctness. This measure of correctness is found by testing the tool with a subset of samples from the historical data.

An interesting fact also observed from the testing of the tool is: of the cases that the tool estimated incorrectly, the correct estimation is a close neighbor. For example, for some of the test samples, the tool estimated a Stay of 21-30 days, while the correct Stay for most of those samples were either 11-20 days or 31-40 days.

Given that there are only limited number of characteristics of the patients, my recommendation to the Hospital management is to use this tool as a complimentary asset for resource estimation and be prepared to allocate resources for a slightly wider range of Stay to be more successful.
    

### Next steps and Recommendations:

From a technical standpoint, a few next steps I identified in this entire exercise:

1. Understand how to gather insights from permutation importance of a categorical variable (for ex: perm importance of 'Ward_type' is 0.063: What does this mean? which Ward_type is influencing the target variable?)

2. Use Explainable AI (XAI) techniques to understand why a model is behaving the way it is. Some of the XAI techniques I would like to explore is SHAP and LIME. This is part of my future work

3. Based on outcomes from (2), fine-tune the model(s) to reduce the false positives and false negatives