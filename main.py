#Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Reading the inputs
PATH = "D:/Learning/Resume_Projects/Mercari Price Suggestion Challenge/input/"
train = pd.read_csv(f'{PATH}train.tsv', sep='\t')
test = pd.read_csv(f'{PATH}test.tsv', sep='\t')

# size of training and dataset
print(train.shape)
print(test.shape)

#Checking data types. The files has numeric and string values
print(train.dtypes)

#checking for null values: There are null values in category_name, brand_name and item_description
train.isnull().sum()

#Checking the target value
print("The mean of the price is %d " % train.price.mean())
print("The median of the price is %d " % train.price.median())
print("The difference b/w them is %d " % (train.price.mean()- train.price.median()))
print("The maximum price is %d and the minimum price is %d" % (train.price.max(), train.price.min()))


# Below Plot shows the of price where its how heavy skewness as well as variation.
plt.subplot(2, 2, 1)
(train['price']).plot.hist(bins=50, figsize=(15,8), edgecolor='white',range=[0,300])
plt.xlabel('Price', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.tick_params(labelsize=8)
plt.title('Price Distribution', fontsize=8)

#plotting a log-transformation of price help us reduce the skewness as well as normalize the price distrution. Also adding +1 to deal with the 0 and negative values.
plt.subplot(2, 2, 2)
(np.log(train['price']+1)).plot.hist(bins=50, figsize=(15,8), edgecolor='white')
plt.xlabel('Log(Price+1)', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.tick_params(labelsize=8)
plt.title('Log(Price+1) Distribution', fontsize=8)
plt.show()

print("%d unique category and %d unique brands" % (train['category_name'].nunique(), train['brand_name'].nunique()))

 #Replace missing values with "No Label"
print(train['category_name'].fillna("No Label", inplace=True))