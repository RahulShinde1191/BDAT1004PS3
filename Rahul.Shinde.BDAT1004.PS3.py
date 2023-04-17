#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[35]:


import pandas as pd


# ### Importing the dataset

# In[36]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'


# In[37]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
users = pd.read_csv(url, sep='|')


# In[38]:


mean_age_by_occupation = users.groupby('occupation')['age'].mean()


# In[39]:


def gender_to_numeric(x):
    if x == 'M':
        return 1
    if x == 'F':
        return 0
    
users['gender_n'] = users['gender'].apply(gender_to_numeric)
male_ratio_by_occupation = users.groupby('occupation')['gender_n'].mean().sort_values(ascending=False)


# In[43]:


min_max_age_by_occupation = users.groupby('occupation')['age'].agg(['min', 'max'])


# In[44]:


mean_age_by_occ_and_gender = users.groupby(['occupation', 'gender'])['age'].mean()


# In[46]:


gender_counts_by_occupation = users.groupby(['occupation', 'gender'])['gender'].count()
total_counts_by_occupation = users.groupby('occupation')['gender'].count()
gender_percentage_by_occupation = (gender_counts_by_occupation / total_counts_by_occupation) * 100
gender_percentage_by_occupation = gender_percentage_by_occupation.unstack()


# In[54]:


print('Mean age by occupation:\n', mean_age_by_occupation)


# In[50]:


print('\nMale ratio by occupation (from highest to lowest):\n', male_ratio_by_occupation)


# In[51]:


print('\nMinimum and maximum age by occupation:\n', min_max_age_by_occupation)


# In[52]:


print('\nMean age by occupation and gender:\n', mean_age_by_occ_and_gender)


# In[53]:


print('\nGender percentage by occupation:\n', gender_percentage_by_occupation)


# # Question 2

# In[107]:


import pandas as pd


# In[108]:


url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'
euro12 = pd.read_csv(url)


# In[123]:


goals = euro12['Goals']
print(euro12.Goals)


# In[124]:


num_teams = euro12.shape[0]
print("Number of teams participated in Euro2012:", len(euro12))


# In[126]:


num_cols = euro12.shape[1]
print("The number of columns in the dataset is:", euro12.shape[1])


# In[127]:


discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
print(discipline)


# In[128]:


discipline_sorted = discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=[False, False])
print(discipline_sorted)


# In[129]:


mean_yellow_cards = discipline['Yellow Cards'].mean()
print(mean_yellow_cards)


# In[130]:


high_scoring_teams = euro12[euro12['Goals'] > 6]
print(high_scoring_teams)


# In[131]:


G_teams = euro12[euro12['Team'].str.startswith('G')]
print(G_teams)


# In[132]:


first_7_cols = euro12.iloc[:, :7]
print(first_7_cols)


# In[133]:


all_cols_except_last_3 = euro12.iloc[:, :-3]
print(all_cols_except_last_3)


# In[134]:


shooting_accuracy = euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]
print(shooting_accuracy)


# # Question 3

# In[157]:


import pandas as pd
import numpy as np


# In[158]:


series1 = pd.Series(np.random.randint(1, 5, 100))
series2 = pd.Series(np.random.randint(1, 4, 100))
series3 = pd.Series(np.random.randint(10000, 30001, 100))


# In[165]:


housing = pd.concat([series1, series2, series3], axis=1)
print(housing)


# In[166]:


housing.columns = ['bedrs', 'bathrs', 'price_sqr_meter']
print(housing.columns)


# In[167]:


bigcolumn = pd.concat([series1, series2, series3], axis=0)
print(bigcolumn)


# In[162]:


print(bigcolumn.index.max() == 99)


# In[164]:


df = pd.concat([series1, series2, series3], axis=1).reset_index(drop=True)
print(df)


# # Question 4

# In[51]:


import pandas as pd
import numpy as np


# In[52]:


data = pd.read_csv('wind.txt', sep='\s+', parse_dates=[[0,1,2]])


# In[53]:


data = pd.read_csv('wind.txt', sep='\s+', parse_dates=[[0,1,2]])
data.set_index('Yr_Mo_Dy', inplace=True)


# In[54]:


def fix_year(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return pd.to_datetime(year*10000+x.month*100+x.day, format='%Y%m%d')

data.index = data.index.map(fix_year)


# In[55]:


missing_values_count = data.isnull().sum()


# In[56]:


non_missing_values_count = data.notnull().sum().sum()


# In[57]:


mean_windspeed = data.mean().mean()


# In[58]:


loc_stats = pd.DataFrame()
loc_stats['min'] = data.min()
loc_stats['max'] = data.max()
loc_stats['mean'] = data.mean()
loc_stats['std'] = data.std()


# In[59]:


day_stats = pd.DataFrame()
day_stats['min'] = data.min(axis=1)
day_stats['max'] = data.max(axis=1)
day_stats['mean'] = data.mean(axis=1)
day_stats['std'] = data.std(axis=1)


# In[60]:


january_data = data[data.index.month == 1]
january_data.groupby(january_data.index.year).mean()


# In[61]:


data.groupby(data.index.year).mean()


# In[62]:


data.groupby([data.index.year, data.index.month]).mean()


# In[63]:


data.groupby(pd.Grouper(freq='W')).mean()


# In[64]:


# Min Max Mean and Std
weekly_data = data.loc['1961-01-02':'1961-12-31'].resample('W').mean()


weekly_stats = pd.DataFrame({
    'min': weekly_data.min(axis=1),
    'max': weekly_data.max(axis=1),
    'mean': weekly_data.mean(axis=1),
    'std': weekly_data.std(axis=1)
})


print(weekly_stats.head(52))


# # Question 5

# In[218]:


import pandas as pd
import numpy as np


# In[219]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, delimiter='\t')


# In[220]:


chipo = pd.read_csv(url, delimiter='\t')


# In[221]:


chipo.head(10)


# In[222]:


print(f"The number of observations in the dataset is {len(chipo)}")


# In[223]:


print(f"The number of columns in the dataset is {len(chipo.columns)}")


# In[224]:


print("Columns in the dataset:")
print(chipo.columns)


# In[225]:


print(chipo.index)


# In[226]:


most_ordered = chipo.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(1)
print(f"The most-ordered item is:\n{most_ordered}")


# In[227]:


print(f"The number of items ordered for the most-ordered item is {most_ordered.values[0]}")


# In[228]:


most_ordered_desc = chipo.groupby('choice_description')['quantity'].sum().sort_values(ascending=False).head(1)
print(f"The most ordered item in the choice_description column is:\n{most_ordered_desc}")


# In[229]:


total_items_ordered = chipo['quantity'].sum()
print(f"The total number of items ordered is {total_items_ordered}")


# In[230]:


chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:-1]))

print(chipo['item_price'].dtype)


# In[231]:


revenue = (chipo['quantity'] * chipo['item_price']).sum()
print(f"The revenue for the period in the dataset is ${revenue:.2f}")


# In[232]:


num_orders = len(chipo['order_id'].unique())
print(f"The number of orders made in the period is {num_orders}")


# In[236]:


total_revenue = (chipo['item_price'] * chipo['quantity']).sum()

total_orders = chipo['order_id'].nunique()

avg_revenue_per_order = total_revenue / total_orders

print("The average revenue amount per order is: $", round(avg_revenue_per_order, 2))


# # Question 6

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


marriages_divorces = pd.read_csv('us-marriages-divorces-1867-2014.csv')


# In[3]:


plt.plot(marriages_divorces.Year, marriages_divorces.Marriages_per_1000, label='Marriages')
plt.plot(marriages_divorces.Year, marriages_divorces.Divorces_per_1000, label='Divorces')
plt.xlabel('Year')
plt.ylabel('Number per capita')
plt.title('Number of marriages and divorces per capita in the U.S. between 1867 and 2014')
plt.legend()
plt.show()


# # Question 7

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the given source
data = pd.read_csv('us-marriages-divorces-1867-2014.csv') 


# In[14]:


years = [1900, 1950, 2000]
subset = data[data['Year'].isin(years)]


# In[18]:


subset = data[data['Year'].isin(years)].copy()
subset.loc[:, 'Marriages_per_capita'] = subset['Marriages'] / subset['Population']
subset.loc[:, 'Divorces_per_capita'] = subset['Divorces'] / subset['Population']


# In[23]:


ax = subset.plot(x='Year', y=['Marriages_per_capita', 'Divorces_per_capita'], kind='bar', figsize=(8,6), color=['blue','red'])
ax.set_xlabel('Year')
ax.set_ylabel('Number per capita')
ax.set_title('Marriages and Divorces per capita in the U.S. between 1900, 1950, and 2000')
ax.legend(['Marriages', 'Divorces'])
plt.show()


# # Question 8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt


# In[25]:


data = pd.read_csv('actor_kill_counts.csv')


# In[26]:


data = data.sort_values('Count', ascending=False)


# In[27]:


plt.barh(data['Actor'], data['Count'], color='red')
plt.xlabel('Kill Count')
plt.ylabel('Actor')
plt.title('Deadliest Actors in Hollywood')
plt.show()


# # Question 9

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt


# In[29]:


data = pd.read_csv('roman-emperor-reigns.csv')


# In[30]:


num_assassinated = len(data[data['Cause_of_Death'] == 'Assassinated'])


# In[31]:


num_not_assassinated = len(data) - num_assassinated


# In[32]:


labels = ['Assassinated', 'Not Assassinated']


# In[33]:


values = [num_assassinated, num_not_assassinated]


# In[34]:


plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Fraction of Roman Emperors who were Assassinated')
plt.show()


# # Question 10

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt


# In[36]:


data = pd.read_csv('arcade-revenue-vs-cs-doctorates.csv')


# In[38]:


by_year = data[['Year', 'Total Arcade Revenue (billions)', 'Computer Science Doctorates Awarded (US)']].groupby('Year')


# In[41]:


fig, ax = plt.subplots()
for year, group in by_year:
    ax.scatter(group['Total Arcade Revenue (billions)'], group['Computer Science Doctorates Awarded (US)'], label=year)
ax.set_xlabel('Total Arcade Revenue (billions)')
ax.set_ylabel('Computer Science Doctorates Awarded (US)')
ax.set_title('Arcade Revenue vs. Computer Science PhDs awarded in the U.S. between 2000 and 2009')
ax.legend()
plt.show()


# In[ ]:





# In[ ]:




