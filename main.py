#Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

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
train['category_name'].fillna("No Label", inplace=True)

#splitting the category data into 3 to get a column for each category
def split_categories(text):
    categories = text.split('/')
    if len(categories) >= 3:
        return pd.Series([categories[0], categories[1], categories[2]])
    elif len(categories) == 2:
        return pd.Series([categories[0], categories[1], None])
    else:
        return pd.Series([categories[0], None, None])

# Apply the function to create new columns
train[['First_cat', 'Second_cat', 'Third_cat']] = train['category_name'].apply(split_categories)

#checking for unique values 
print("%d unique 1st category, %d unique 2nd category and %d unique 3rd category" % (train['First_cat'].nunique(), train['Second_cat'].nunique(), train['Third_cat'].nunique()))

# Bar Plots for First Category
colors = sns.color_palette("colorblind", 10)
category_column = train['First_cat']
category_counts = category_column.value_counts()
plt.bar(category_counts.index, category_counts.values, color=colors) # type: ignore
plt.xlabel('First Category')
plt.ylabel('Count')
plt.title('Number of Items by First Category')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()

# Bar Plots for Top 10 Second Category
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
category_column1 = train['Second_cat']
category_counts1 = category_column1.value_counts()
top_10_categories1 = category_counts1.head(10)
plt.bar(top_10_categories1.index, top_10_categories1.values, color=colors) # type: ignore
plt.xlabel('Second Category')
plt.ylabel('Count')
plt.title('Top 10 by Second Category')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()

plt.subplot(1,2,2)
category_column2 = train['Third_cat']
category_counts2 = category_column2.value_counts()
top_10_categories2 = category_counts2.head(10)
plt.bar(top_10_categories2.index, top_10_categories2.values, color=colors) # type: ignore
plt.xlabel('Third Category')
plt.ylabel('Count')
plt.title('Top 10 by Third Category')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()

# Grouping by 'brand' and calculating the sum of prices and count of brands for each brand. We can see Apple has the highest per unit price and Forever 21 has the lowest.
brand_summary = train.groupby('brand_name').agg({'price': ['sum', 'count']})
brand_summary.columns = ['total_price', 'brand_count']
brand_summary['average_percentage'] = (brand_summary['total_price'] / brand_summary['brand_count'])
top_10_brands = brand_summary.nlargest(10, 'brand_count')

# Printing the top 10 brands with sum of prices, counts, and average percentage
print("Top 10 Brands by Count of Brands, with Total Price, Count, and Average Percentage:")
print(top_10_brands)



# Plotting the count of brands and total price on the same axis
fig, ax1 = plt.subplots(figsize=(10, 6))
top_10_brands['brand_count'].plot(kind='bar', ax=ax1, color='skyblue', position=0, width=0.4)

# Adding labels and title for the left y-axis
ax1.set_xlabel('Brand')
ax1.set_ylabel('Count of Brands', color='skyblue')
ax1.set_title('Top 10 Brands by Count of Brands and Total Price')

# Creating a second y-axis for total price
ax2 = ax1.twinx()
ax2.spines['right'].set_position(('outward', 60))  # Move the second y-axis to the right for better visibility

# Plotting the total price on the right y-axis
top_10_brands['total_price'].plot(kind='bar', ax=ax2, color='orange', position=1, width=0.4)

# Adding labels for the right y-axis
ax2.set_ylabel('Total Price ($)', color='orange')

# Adding legend
ax1.legend(['Count of Brands'], loc='upper left')
ax2.legend(['Total Price'], loc='upper right')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()

#Normalizing The price
brand_summary = train.groupby('brand_name').agg({'price': ['sum', 'count']})
brand_summary.columns = ['total_price', 'brand_count']

brand_summary['log_total_price'] = brand_summary['total_price'].apply(lambda x: max(0, x)).apply(np.log)
brand_summary['log_price'] = brand_summary['total_price'] / brand_summary['brand_count']
brand_summary['log_price'] = brand_summary['log_price'].apply(lambda x: max(0, x)).apply(np.log)
top_10_brands = brand_summary.nlargest(10, 'brand_count')

print("Top 10 Brands by Count of Brands, with Total Price, Count, and Logarithm of Price:")
print(top_10_brands[['brand_count', 'total_price', 'log_price']])

fig, ax1 = plt.subplots(figsize=(10, 6))
top_10_brands['brand_count'].plot(kind='bar', ax=ax1, color='skyblue', position=0, width=0.4)
ax1.set_xlabel('Brand')
ax1.set_ylabel('Count of Brands', color='skyblue')
ax1.set_title('Top 10 Brands by Count of Brands and Total Price (Logarithm)')
ax2 = ax1.twinx()
ax2.spines['right'].set_position(('outward', 60))


top_10_brands['log_total_price'] = top_10_brands['total_price'].apply(lambda x: max(0, x)).apply(np.log)
top_10_brands['log_total_price'].plot(kind='bar', ax=ax2, color='orange', position=1, width=0.4)
ax2.set_ylabel('Log Total Price', color='orange')
ax1.legend(['Count of Brands'], loc='upper left')
ax2.legend(['Log Total Price'], loc='upper right')

plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

# Calculate median price by brand
median_prices = train.groupby('brand_name')['price'].median().reset_index()

# Top 10 most expensive brands
top_10_brands = median_prices.nlargest(10, 'price')

plt.figure(figsize=(10, 8))
sns.barplot(data=top_10_brands, y='brand_name', x='price', color='cyan')
plt.xlabel('Median price')
plt.ylabel('')
plt.title('Top 10 most expensive brands')
plt.show()


#The distribution of price for each class is focused on 10^3.
colors = sns.color_palette("colorblind", n_colors=len(train['item_condition_id'].unique()))
my_colors = dict(zip(train['item_condition_id'].unique(), colors))

sns.boxplot(data=train, x='item_condition_id', y=np.log(train.price+1), palette=my_colors, hue='item_condition_id', legend=False)
plt.tight_layout()

#The shipping cost burden is decently splitted between sellers and buyers with more than half of the items' shipping fees are paid by the sellers (55%)
round(train["shipping"].value_counts()/len(train)*100,2)

#The graph shows if the product is more expensive the shipping fee is generally paid by the buyer.
seller = train[train.shipping==1]
buyer = train[train.shipping==0]

plt.subplots(figsize=(10,10))
plt.subplot(2,1,1)
seller.price.plot.hist(bins=50, range=[0,200])
buyer.price.plot.hist(alpha=0.7, bins=50,range=[0,200])
plt.legend(['Price when Seller pays Shipping', 'Price when Buyer pays Shipping'])
plt.title('Price Distribution by Shipping Type', fontsize=10)
plt.subplot(2,1,2)
np.log(seller.price+1).plot.hist(bins=50)
np.log(buyer.price+1).plot.hist(alpha=0.7, bins=50)
plt.legend(['Price when Seller pays Shipping', 'Price when Buyer pays Shipping'])
plt.title('Log of Price Distribution by Shipping Type', fontsize=10)
plt.tight_layout()

#"Brand new", "free shipping", "great condition", "good condition", "never worn", "never used", "Victoria Secret", "smoke free", "Size large", "Size medium", "Size small", "excellent condition" are some frequently appearing item description texts.
wordcloud = WordCloud(width = 2400, height = 1200).generate(" ".join(train.item_description.astype(str)))
plt.figure(figsize = (13, 10))
plt.imshow(wordcloud)
plt.show()


exp = train[train['price'] > 200]
"""
exp = train[train['price'] > 200]
exp.name = exp.name.str.upper()

wc = WordCloud(background_color="white", max_words=5000, 
               stopwords=STOPWORDS, max_font_size= 50)

wc.generate(" ".join(str(s) for s in exp.brand_name.values))

plt.figure(figsize=(20,12))
plt.title('What are the most expensive items', fontsize = 30)
plt.axis('off')
plt.imshow(wc, interpolation='bilinear')



# Lowercasing the brand_name
brand_names = [str(s).lower().replace(" inc", "").strip() for s in exp.brand_name.values if str(s).lower() != 'nan']

wc = WordCloud(background_color="white", max_words=5000, stopwords=STOPWORDS, max_font_size=50)
wc.generate(" ".join(brand_names))

plt.figure(figsize=(20, 12))
plt.title('What are the most expensive items', fontsize=30)
plt.axis('off')
plt.imshow(wc, interpolation='bilinear')
plt.show()
"""

brand_names = [str(s).lower() for s in exp.brand_name.values if str(s).lower() != 'nan']

# Define custom stop words to exclude specific terms
custom_stopwords = set(['apple bottoms', 'pineapple connection', 'green apple'])

# Exclude custom stop words from the brand names list
brand_names_filtered = [brand for brand in brand_names if brand not in custom_stopwords]

wc = WordCloud(background_color="white", max_words=5000, stopwords=STOPWORDS, max_font_size=50)
wc.generate(" ".join(brand_names_filtered))

plt.figure(figsize=(20, 12))
plt.title('What are the most expensive items', fontsize=30)
plt.axis('off')
plt.imshow(wc, interpolation='bilinear')
plt.show()