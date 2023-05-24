# Explore Module

def sodium_quality_hist():
    """This function returns histograms of sodium and quality"""
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.hist(train.sodium)
    plt.title('Distribution of Sodium')

    plt.subplot(122)
    plt.hist(train.quality)
    plt.title('Distribution of Quality')
    plt.show()

def free_ph_hist():
    """This function returns histograms of free SO2 and pH"""
    plt.figure(figsize=(12,6))
    
    
    plt.figure(figsize=(12,6))
    
    # distribution of free_SO2
    plt.subplot(121)
    plt.hist(train.free_SO2_shelf_life)
    plt.title('Distribution of Free SO2 Shelf Life')

    # distribution of pH
    plt.subplot(122)
    plt.hist(train.pH)
    plt.title('Distribution of pH')
    plt.show()
    
def alcohol_sugar_hist():
    """This function returns histograms of alochol and residual sugars"""
    plt.figure(figsize=(12,6))
    
    # distribution of alcohol
    plt.subplot(121)
    plt.hist(train.alcohol)
    plt.title('Distribution of Alcohol Percentage')

    # distribution of residual sugars
    plt.subplot(122)
    plt.hist(train.residual_sugar)
    plt.title('Distribution of Residual Sugars')
    plt.show()
    
def alcohol_quality_hist():
    """This function returns historgrams of alchohol and quality ratings"""
    plt.figure(figsize=(12,6))
    
    # distribution of alcohol
    plt.subplot(121)
    plt.hist(train.alcohol)
    plt.title('Distribution of Alcohol Percentage')

    # distribution of quality ratings
    plt.subplot(122)
    plt.hist(train.quality)
    plt.title('Distribution of Quality Rating')
    plt.show()
    
