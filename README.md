# Predicting Visitor Purchases with a Classification Model with BigQuery ML

This project was executed using [this](https://www.cloudskillsboost.google/course_sessions/553964/labs/102262) Qwiklabs tutorial

## Overview

[BigQuery ML](https://cloud.google.com/bigquery-ml/docs/bigqueryml-web-ui-start) (BigQuery machine learning) is a feature in BigQuery where data analysts can create, train, evaluate, and predict with machine learning models with minimal coding.

The Google Analytics Sample [Ecommerce dataset](https://www.blog.google/products/marketingplatform/analytics/introducing-google-analytics-sample/) that has millions of Google Analytics records for the [Google Merchandise Store](https://shop.googlemerchandisestore.com/) loaded into BigQuery. In this lab, you will use this data to run some typical queries that businesses would want to know about their customers' purchasing habits.

## Objectives

- In this lab, you learn to perform the following tasks:

- Use BigQuery to find public datasets

- Query and explore the ecommerce dataset

- Create a training and evaluation dataset to be used for batch prediction

- Create a classification (logistic regression) model in BigQuery ML

- Evaluate the performance of your machine learning model

- Predict and rank the probability that a visitor will make a purchase

### Task 1. Explore ecommerce data

The querys used on this task can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/task1.sql) and the results can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/tree/master/resultados/task_1) as .CSV files. 

### Task 2. Select features and create your training dataset

For this task, the lab choose to use the following columns as features:

- totals.bounces (whether the visitor left the website immediately)
- totals.timeOnSite (how long the visitor was on our website)

The query used to create the training dataset can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/task2.sql) and the dataset generated can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/tree/master/resultados/task_2).

After a first look out of the top 10 time_on_site, only 1 customer is a recurrent buyer, which isn't very promising. So we can't expect that the model using only this two features to be much precise. But it's a good start.

### Task 3. Create a BigQuery dataset to store models

The dataset **eccomerce** was created using the instructions provided by the tutorial.

### Task 4. Select a BigQuery ML model type and specify options

The query used to create the model training can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/task4.sql), note that is a modified version of the query used on **task 2**. At the end of the process, we can see the following message: Consulta finalizada (tempo decorrido: 2 min 15 s, bytes processados: 41,8 MB (ML)), which shows that ML querys take more time to process then normal querys.

More details about the execution can be found on the following images

![Alt text](assets\img1.png)
![Alt text](assets\img2.png)
![Alt text](assets\img3.png)

And the following images show the details of the execution of the query without the parameters for the ML model creation, not that the time for processing the request is much lower

![Alt text](assets\img4.png)
![Alt text](assets\img5.png)
![Alt text](assets\img6.png)
![Alt text](assets\img7.png)

Note: You cannot feed all of your available data to the model during training since you need to save some unseen data points for model evaluation and testing. To accomplish this, add a WHERE clause condition is being used to filter and train on only the first 9 months of session data in your 12 month dataset.

### Task 5. Evaluate classification model performance

For classification problems in ML, you want to minimize the False Positive Rate (predict that the user will return and purchase and they don't) and maximize the True Positive Rate (predict that the user will return and purchase and they do).

This relationship is visualized with a ROC (Receiver Operating Characteristic) curve like the one shown here, where you try to maximize the area under the curve or AUC:

![Alt text](assets\img8.png)

In BigQuery ML, roc_auc is simply a queryable field when evaluating your trained ML model.

Once that training is complete, we evaluated how well the model performed by running [this](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/task5.sql) query using ML.EVALUATE, the result can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/tree/master/resultados/task_5) and at this image

![Alt text](assets\img9.png)

As we can see, after evaluating your model you get a **roc_auc** of 0.72, which shows that the model has not great predictive power. Since the goal is to get the area under the curve as close to 1.0 as possible, there is room for improvement.

### Task 6. Improve model performance with feature engineering

At this task we will be adding some new features and creating a second machine learning model called classification_model_2, once there are many more features in the dataset that may help the model better understand the relationship between a visitor's first session and the likelihood that they will purchase on a subsequent visit.

The new features that we will use are going to answer the following questions

- How far the visitor got in the checkout process on their first visit

- Where the visitor came from (traffic source: organic search, referring site etc.)

- Device category (mobile, tablet, desktop)

- Geographic information (country)

The second model was create by running the first query in [this](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/task6.sql) file;

   - **Note**: You are still training on the same first 9 months of data, even with this new model. It's important to have the same training dataset so you can be certain a better model output is attributable to better input features and not new or different training data.

A key new feature that was added to the training dataset query is the maximum checkout progress each visitor reached in their session, which is recorded in the field hits.eCommerceAction.action_type. If you search for that field in the field definitions you will see the field mapping of 6 = Completed Purchase.

As an aside, the web analytics dataset has nested and repeated fields like ARRAYS which need to be broken apart into separate rows in your dataset. This is accomplished by using the UNNEST() function, which you can see in the above query.

Now that we have a new model we can use the second query in [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/task6.sql) to evaluate the quality of the model 

You can see the result of the evaluation [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/tree/master/resultados/task_6) or in the image below

![Alt text](assets\img16.png)

You can see that with this new model you now get a **roc_auc** of 0.91 which is significantly better than the first model.

Now that you have a trained model, time to make some predictions.

### Task 7. Predict which new visitors will come back and purchase

The query used to make the predictions can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/task7.sql) and a .CSV file with a part of the results can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/tree/master/resultados/task_7).

The predictions are made in the last 1 month (out of 12 months) of the dataset.

Your model will now output the predictions it has for those July 2017 ecommerce sessions. You can see three newly added fields:

predicted_will_buy_on_return_visit: whether the model thinks the visitor will buy later (1 = yes)
predicted_will_buy_on_return_visit_probs.label: the binary classifier for yes / no
predicted_will_buy_on_return_visit_probs.prob: the confidence the model has in it's prediction (1 = 100%)

### Results

Of the top 6% of first-time visitors (sorted in decreasing order of predicted probability), more than 6% make a purchase in a later visit.

These users represent nearly 50% of all first-time visitors who make a purchase in a later visit.

Overall, only 0.7% of first-time visitors make a purchase in a later visit.

Targeting the top 6% of first-time increases marketing ROI by 9x vs targeting them all!

#### Additional information
roc_auc is just one of the performance metrics available during model evaluation. Also available are accuracy, precision, and recall. Knowing which performance metric to rely on is highly dependent on what your overall objective or goal is.

### Challenge

#### Summary

In the previous two tasks you saw the power of feature engineering at work in improving our models performance. However, we still may be able to improve our performance by exploring other model types. For classification problems, BigQuery ML also supports the following model types:

- Deep Neural Networks .
- Boosted Decision Trees (XGBoost) .
- AutoML Tables Models .
- Importing Custom TensorFlow Models .

#### Task
Though our linear classification (logistic regression) model performed well after feature engineering, it may be too simple of a model to fully capture the relationship between the features and the label. Using the same dataset and labels as you did in Task 6 to create the model ecommerce.classification_model_2, your challenge is to create a XGBoost Classifier.

**Hint** : Use following options for Boosted_Tree_Classifier:
1. L2_reg = 0.1
2. num_parallel_tree = 8
3. max_tree_depth = 10
You may need to look at the documentation linked above to see the exact syntax. The model will take around 7 minutes to train. The solution can be found in the solution section below if you need help writing the query.

The output now shows a classification model that can better predict the probability that a first-time visitor to the Google Merchandise Store will make a purchase in a later visit. By comparing the result above with the previous model shown in Task 7, you can see the confidence the model has in its predictions is more accurate when compared to the logistic_regression model type.

The querys with the solution to the **Challange** can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/blob/master/querys_sql/extra.sql), and the .csv file with part of the result can be found [here](https://github.com/lilacostaro/Predicting_Visitor_Purchases_with_a_Classification_Model_with_BigQuery_ML/tree/master/resultados/extra).

OBS.: Great part of the readme and all the querys ware taken from the Qwiklabs instructions.