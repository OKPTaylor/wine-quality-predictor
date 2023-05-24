# Sommelier Sciencers

# Project Description
 
Clustering project built to predict the quality of wine.
 
# Project Goal
 
* Find the key drivers of wine quality.
* Using 3 clustering techniques to construct 4 machine learning models to predict wine quality from the wine quality (red and white) csv's from the Data World dataset.
* Display results using 5 vizzes.
 
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
       1. Do chlorides affect quality?
       2. Is free sulfur dioxide related to pH?
       3. Is alcohol content related to residual sugars?
       4. Does alcohol content affect quality of wine?
       
* Model data:
    * 3 different clustering combinations
    * 4 diffferent models
        * Classification
    * 5 different vizzes

* Conclusions:
	* Identify drivers of quality
    * Define appropriate clusters
    * Develop a classification model that beats baseline

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

# Steps to Reproduce
1) Clone this repo.
2) Go to https://data.world/food/wine-quality and download red and white CSV's, save as original filename, and save to appropriate local directory.
4) Run notebook.
 
# Takeaways and Conclusions
* TBD

# Recommendations
* For the data engineers: TBD
* For the data scientsists: TBD
* For the business: TBD