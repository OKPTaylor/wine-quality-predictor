import matplotlib.pyplot as plt
import seaborn as sns

### FUNCTIONS EXPLORATION:
# Explore Module

def sodium_quality_hist(df, col1, col2):
    """This function returns histograms of sodium and quality"""
    plt.figure(figsize=(10,6))

    plt.subplot(121)
    sns.histplot(df[col1], palette='mako_r')
    plt.title('Distribution of Sodium')

    plt.subplot(122)
    sns.histplot(df[col2], palette='mako_r')
    plt.title('Distribution of Quality')
    plt.show()

def free_ph_hist(df, col1, col2):
    """This function returns histograms of free SO2 and pH"""
    plt.figure(figsize=(12,6))
    
    # distribution of free_SO2
    plt.subplot(121)
    sns.histplot(df[col1], palette='mako_r')
    plt.title('Distribution of Free SO2 Shelf Life')

    # distribution of pH
    plt.subplot(122)
    sns.histplot(df[col2], palette='mako_r')
    plt.title('Distribution of pH')
    plt.show()
    
def alcohol_sugar_hist(df, col1, col2):
    """This function returns histograms of alochol and residual sugars"""
    plt.figure(figsize=(12,6))
    
    # distribution of alcohol
    plt.subplot(121)
    sns.histplot(df[col1], palette='mako_r')
    plt.title('Distribution of Alcohol Percentage')

    # distribution of residual sugars
    plt.subplot(122)
    sns.histplot(df[col2], palette='mako_r')
    plt.title('Distribution of Residual Sugars')
    plt.show()
    
def alcohol_quality_hist(df, col1, col2):
    """This function returns historgrams of alchohol and quality ratings"""
    plt.figure(figsize=(12,6))
    
    # distribution of alcohol
    plt.subplot(121)
    sns.histplot(df[col1], palette='mako_r')
    plt.title('Distribution of Alcohol Percentage')

    # distribution of quality ratings
    plt.subplot(122)
    sns.histplot(df[col2], palette='mako_r')
    plt.title('Distribution of Quality Rating')
    plt.show()
    
def summarize_sodium_quality(df, col1, col2):
    """This function shows a barplot of cluster set 1 and quality"""
    sns.barplot(y=df[col1], x=df[col2], ci=False, palette='mako_r')
    plt.show()    
    
def summarize_alcohol_quality(df, col1, col2):
    """This function displays a barplot of cluster set 2 and quality
    """
    sns.barplot(y=df[col1], x=df[col2], ci=False, palette='mako_r')
    plt.title("Correlation of Alcohol Percentage and Quality Rating")
    plt.show()
    
def summarize_alcohol_sugar_quality(df, col1, col2, col3):
    """This function shows a summary of alcohol content and residual sugars related to quality
    """
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=df[col1], y=df[col2],hue=df[col3], palette='mako_r')
    plt.title("Correlation of Alcohol and Residual Sugars")
    plt.show()
    
#     sns.barplot(data=train, y='alcohol', x='quality', ci=False, palette='flare')
# plt.title("Correlation of Alcohol Percentage and Quality Rating")
# plt.show()