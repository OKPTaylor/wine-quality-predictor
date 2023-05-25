# Sommelier Sciencers

# Project Description
 
Clustering project built to predict the quality of wine.
 
# Project Goal
 
* Find the key drivers of wine quality.
* Using 3 clustering techniques to construct 4 machine learning models to predict wine quality from the wine quality (red and white) csv's from the Data World dataset.
* Display results using vizzes.
 
# Initial Thoughts
 
My initial hypothesis is that wine quality is affected by acidity and alcohol content.
 
# The Plan
 
* Acquire data:
    * get csv's from Data World website
    * merge red and white csv's into one column stating if the wine is red or white
 
* Prepare data:
   * Look at the data:
		* nulls
		* value counts
		* data types
		* numerical/categorical columns
		* names of columns
            * related columns
 
* Explore data:
   * Answer the following initial questions:
       1. Does sodium (chlorides) affect quality?
       2. Is free sulfur dioxide related to pH?
       3. Is alcohol content related to residual sugars?
       4. Does alcohol content affect quality of wine?
       
* Model data:
    * 3 different clustering combinations
    * 4 different models
        * Classification
        * Regression
    * 5 different vizzes

* Conclusions:
	* Identify drivers of quality
    * Define any appropriate clusters
    * Develop a model that beats baseline

# Data Dictionary

| Feature | Definition (measurement)|
|:--------|:-----------|
|Fixed Acidity| The fixed amount of tartaric acid. (g/L)|
|Volatile Acidity| A wine's acetic acid; (High Volatility = High Vinegar-like smell). (g/L)|
|Citric Acid| The amount of citric acid; (Raises acidity, Lowers shelf-life). (g/L)|
|Residual Sugar| Leftover sugars after fermentation. (g/L)|
|Chlorides| Increases sodium levels; (Affects color, clarity, flavor, aroma). (g/L)|
|Free Sulfur Dioxide| Related to pH. Determines how much SO2 is available. (Increases shelf-life, decreases palatability). (mg/L)|
|Total Sulfur Dioxide| Summation of free and bound SO2. (Limited to 350ppm: 0-150, low-processed, 150+ highly processed). (mg/L)|
|Density| Between 1.08 and 1.09. (Insight into fermentation process of yeast growth). (g/L)|
|pH| 2.5: more acidic - 4.5: less acidic (range)|
|Sulphates| Added to stop fermentation (Preservative) (g/L)|
|Alcohol| Related to Residual Sugars. By-product of fermentation process (vol%)|
|Quality| Score assigned between 0 and 10; 0=low, 10=best|
|Color| Red or White type of wine|
|Acid| Engineered Feature: Volatile Acidity + Fixed Acidity|
|Sweetness| Engineered Feature: Residual Sugar + Alcohol|
|Feel| Engineered Feature: pH + Density + Free Sulfur Dioxide|

# Steps to Reproduce
1) Clone this repo
2) Go to https://data.world/food/wine-quality and download red and white CSV's, save as original filename, and save to appropriate local directory
4) Run notebook
 
# Takeaways and Conclusions<br>

Clustering did not produce and *meaningful* clusters, but did highlight relationships:

* **Acidity negatively affects the quality of wine**
    * Higher the acid, lower the quality
    * Volatile Acidity results from the degradation of citric acid. As citric acid degrades, volatile acidity goes up
* **Alcohol positively affects the quality of wine**
    * 0.44 Correlation Score
    * Concerned about skewing
        * Quality of 3 has 30 wines
        * Quality of 9 has 5 wines
* **White wine results in a much wider range of residual sugars**
    * Clustering on Alcohol, Residual Sugars, and Density *attempted* to cluster to identify type of wine - got very close to being accurate
        
* **Modeling**

* The random forest performed well at predicting wines in the quality ranges of 4-7 even hitting over 80% true positive rates at 4 and 7. The model preformed worse at the end ranges mostly due to the fact that in those ranges there are limited number of wines.

* While the Polynomial Model beat baseline with a RMSE of .72 this means that predictions will still be off by nearly an entire quality level. This is refelcted in the graph by showing how that most of the predictions are clumped in the middle.

* Classification modeling (random forest) is comparativly better than any regression model for predicting specific catagories sprend across a greater range of predicted and actual datapoints.

# Recommendations
* For future researchers: This data set consisted of only variants of the Portuguese vinho verde and their associated grape types, in order to more fully explore and predict wine quality data on more grape types must be collected.
* For the data engineers: Either split the dataset into white and red and create separate models or gather more data on red types of vinho verde.
* For the data scientsists: Remove outlier and engineer "acid" feature and "feel" feature using appropriate columns.
* For the business: Do not recommend putting this model into production.