# Project 13

This Project is one of the Machine Learning Competitions on Hackerearth.
The Bank Indessa has not done well in last 3 quarters. Their NPAs (Non Performing Assets) have reached all time high. It is starting to lose confidence of its investors. As a result, it’s stock has fallen by 20% in the previous quarter alone.

After careful analysis, it was found that the majority of NPA was contributed by loan defaulters. With the messy data collected over all the years, this bank has decided to use machine learning to figure out a way to find these defaulters and devise a plan to reduce them.

This bank uses a pool of investors to sanction their loans. For example: If any customer has applied for a loan of $20000, along with bank, the investors perform a due diligence on the requested loan application. Keep this in mind while understanding data.

In this challenge, you will help this bank by predicting the probability that a member will default.

Here is the link : https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-one/

These are my implementations :

1. Simple Feature Engineering: Replaced all Object types with their lenghts and trained a RF model.
2. Advanced Feature Engineering: Did feature Engineering & Data cleaning of whole data.
3. Different Model: Tried using LR model however got a lower accuracy.
4. LR model with Regularization and Parameter Tuning with all the engineered features
5. To Reduce Dimensionality of Data , Extracted Feature Importance using RF model.
6. Ensembling of LR ( With L1 Regularization ) and RF (300 estim) model with only selected features providing better information gain.
