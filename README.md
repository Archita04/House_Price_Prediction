# House_Price_Prediction
House_Price_Prediction/Regression Problem

# Overview

The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset.

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

## **Business Case** :Predicting the price of a house

Using Machine Learning and various data science libraries to build a model that can reliably predict the price of the house. We will use the following pipeline to create a reliable model:

- Problem Definition
- Data 
- Explorartory Data Analysis
- Data Preprocessing
- Feature Selection
- Model Selection
- HyperParameter Tunning
- Model Evaluation
- Feature Importance

## Problem Definition

Create a robust machine learning algorithm to accurately predict the price of the house given the various factors across the market. 

## Evaluation

If we can predict the house of the price.

## Create Data Dictionary

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Here's a brief version of what you'll find in the data description file.


- <span style="color:red">SalePrice: </span> - the property's sale price in dollars. This is the target variable that you're trying to predict.
- <span style="color:red">MSSubClass: </span> -  The building class
- <span style="color:red">MSZoning: </span> - The general zoning classification
- <span style="color:red">LotFrontage:</span> - Linear feet of street connected to property
- <span style="color:red">LotArea:</span> - Lot size in square feet
- <span style="color:red">Street:</span> - Type of road access
- <span style="color:red">Alley:</span> - Type of alley access
- <span style="color:red">LotShape:</span> - General shape of property
- <span style="color:red">LandContour:</span> - Flatness of the property
- <span style="color:red">Utilities:</span> - Type of utilities available
- <span style="color:red">LotConfig:</span> - Lot configuration
- <span style="color:red">LandSlope:</span> - Slope of property
- <span style="color:red">Neighborhood:</span> - Physical locations within Ames city limits
- <span style="color:red">Condition1:</span> - Proximity to main road or railroad
- <span style="color:red">Condition2:</span> - Proximity to main road or railroad (if a second is present)
- <span style="color:red">BldgType:</span> - Type of dwelling
- <span style="color:red">HouseStyle:</span> - Style of dwelling
- <span style="color:red">OverallQual:</span> - Overall material and finish quality
- <span style="color:red">OverallCond:</span> - Overall condition rating
- <span style="color:red">YearBuilt:</span> - Original construction date
- <span style="color:red">YearRemodAdd:</span> - Remodel date
- <span style="color:red">RoofStyle:</span> - Type of roof
- <span style="color:red">RoofMatl:</span> - Roof material
- <span style="color:red">Exterior1st:</span> - Exterior covering on house
- <span style="color:red">Exterior2nd:</span> - Exterior covering on house (if more than one material)
- <span style="color:red">MasVnrType:</span> - Masonry veneer type
- <span style="color:red">MasVnrArea:</span> - Masonry veneer area in square feet
- <span style="color:red">ExterQual:</span> - Exterior material quality
- <span style="color:red">ExterCond:</span> - Present condition of the material on the exterior
- <span style="color:red">Foundation:</span> - Type of foundation
- <span style="color:red">BsmtQual:</span> - Height of the basement
- <span style="color:red">BsmtCond:</span> - General condition of the basement
- <span style="color:red">BsmtExposure:</span> - Walkout or garden level basement walls
- <span style="color:red">BsmtFinType1:</span> - Quality of basement finished area
- <span style="color:red">BsmtFinSF1:</span> - Type 1 finished square feet
- <span style="color:red">BsmtFinType2:</span> - Quality of second finished area (if present)
- <span style="color:red">BsmtFinSF2:</span> - Type 2 finished square feet
- <span style="color:red">BsmtUnfSF:</span> - Unfinished square feet of basement area
- <span style="color:red">TotalBsmtSF:</span> - Total square feet of basement area
- <span style="color:red">Heating:</span> - Type of heating
- <span style="color:red">HeatingQC:</span> - Heating quality and condition
- <span style="color:red">CentralAir:</span> - Central air conditioning
- <span style="color:red">Electrical:</span> - Electrical system
- <span style="color:red">1stFlrSF:</span> - First Floor square feet
- <span style="color:red">2ndFlrSF:</span> - Second floor square feet
- <span style="color:red">LowQualFinSF:</span> - Low quality finished square feet (all floors)
- <span style="color:red">GrLivArea:</span> - Above grade (ground) living area square feet
- <span style="color:red">BsmtFullBath:</span> - Basement full bathrooms
- <span style="color:red">BsmtHalfBath:</span> - Basement half bathrooms
- <span style="color:red">FullBath:</span> - Full bathrooms above grade
- <span style="color:red">HalfBath:</span> - Half baths above grade
- <span style="color:red">Bedroom:</span> - Number of bedrooms above basement level
- <span style="color:red">Kitchen:</span> - Number of kitchens
- <span style="color:red">KitchenQual:</span> - Kitchen quality
- <span style="color:red">TotRmsAbvGrd:</span> - Total rooms above grade (does not include bathrooms)
- <span style="color:red">Functional:</span> - Home functionality rating
- <span style="color:red">Fireplaces:</span> - Number of fireplaces
- <span style="color:red">FireplaceQu:</span> - Fireplace quality
- <span style="color:red">GarageType:</span> - Garage location
- <span style="color:red">GarageYrBlt: </span> -Year garage was built
- <span style="color:red">GarageFinish: </span> -Interior finish of the garage
- <span style="color:red">GarageCars:</span> - Size of garage in car capacity
- <span style="color:red">GarageArea:</span> - Size of garage in square feet
- <span style="color:red">GarageQual:</span> - Garage quality
- <span style="color:red">GarageCond:</span> - Garage condition
- <span style="color:red">PavedDrive:</span> -  Paved driveway
- <span style="color:red">WoodDeckSF:</span> - Wood deck area in square feet
- <span style="color:red">OpenPorchSF:</span> - Open porch area in square feet
- <span style="color:red">EnclosedPorch:</span> - Enclosed porch area in square feet
- <span style="color:red">3SsnPorch: </span> -Three season porch area in square feet
- <span style="color:red">ScreenPorch:</span> - Screen porch area in square feet
- <span style="color:red">PoolArea:</span> - Pool area in square feet
- <span style="color:red">PoolQC:</span> - Pool quality
- <span style="color:red">Fence:</span> - Fence quality
- <span style="color:red">MiscFeature:</span> - Miscellaneous feature not covered in other categories
- <span style="color:red">MiscVal:</span> - $Value of miscellaneous feature
- <span style="color:red">MoSold:</span> - Month Sold
- <span style="color:red">YrSold: </span> -Year Sold
- <span style="color:red">SaleType:</span> - Type of sale
- <span style="color:red">SaleCondition:</span> - Condition of sale


<h1 style='color:green'> XGBoost Regressor With RandomizedSearch HyperTune Model is the best generalized model with least MSE/MAE to predict the price of the house given the various factors across the market.</h1>

##  <span style="color:red"> Here are some of the key outcomes of the project:
    
- The dataset was large totally around 1460 samples with 81 columns including one target variable and after preprocessing 5columns were dropped and missing value filled with mostly mean,median .
- Visualizing the distribution of data and thier relationships,helped us to get some insights on the relationship between the featureset.
- Feature Selection/Eliminination was carried out and appropriate feature were shortlisted.
- Testing multiple algorithms with fine-tuning hyperparamters gave us some understanding on the model performance for various algorithms on this specific dataset.
-The boosting algorithms perform the best on the current dataset.
