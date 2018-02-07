# XGBoost-on-Rental-Listing-Inquiries
use XGBoost model to predict the interest level of renting house

Data: Two Sigma Connect: Rental Listing Inquiries(2017 kaggle competetion) 
Task: predict how popular an apartment rental listing is based on the listing content like text description, photos, number of bedrooms, price, etc. 
The target variable : interest_level. Solution: Logistic regression with L1 regularization.

The XGBoost solution works as:

1. Load & prepare data
  - Data size: 49352(train), 74659(test)
  - It has 15 features, with no null value
2. Feature Engineering
  - encode interest level with number
  - for price, bathrooms and bedrooms features, do some numerical calculating
  - create date: encode date to integer
  - remove building_id
  - latitude & longtitude: clustering
  - display_address is a high-cardinality category, use MeanEncoder method
  - street address: remove, contains same information with display_address
  - photos: remove fisrt (photos per house considered a useful feature later)
3. tuning parameter by gridsearchCV
  - tuning n_estimators, best at 33
  - tuning max_depth & min_child_weight
  - tuning subsample and colsample_bytree
  - tuning reg_alpha and reg_lambda
4. Train the model with specified parameters, and use the model to calculate accuracy score in train dataset
5. Decide model and predict
