import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('Bike_Buyer_Data_edited.txt')

df = df.drop(['Unnamed: 13'], axis=1)

df.shape

# target -> purchased bike (yes/no)

df['Purchased Bike'].value_counts().plot.bar(figsize=(10, 4))
plt.xlabel('Purchased Bike')
plt.ylabel('Number of Customers')
plt.show()

print(df['Purchased Bike'].value_counts())


# variables

cat_vars = df.select_dtypes("object").columns

# Bivariate Analysis #########################################################

plt.style.use("ggplot")
for column in cat_vars:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.countplot(df[column], hue=df["Purchased Bike"])
    plt.title(column)    
    plt.xticks(rotation=90)
    
# check dupes
print(len(pd.unique(df['ID'])))

num_vars = df.drop('ID', axis=1)._get_numeric_data().columns
df = df.drop('ID', axis=1)

for var in num_vars:
    # make boxplot with Catplot
    sns.catplot(x= 'Purchased Bike', y= var, data=df, kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x='Purchased Bike', y=var, data=df, jitter=0.1, alpha=0.3, color='k')
    plt.show()
    
    
plt.style.use("ggplot")
for var in num_vars:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.distplot(df[var], kde=True)
    plt.title(var)

# ensure all num vars are positive
for var in num_vars:
    df[var] = np.where(df[var] < 0, 0 , df[var])
    

data_descrip = df.describe()

'''
# try making age into cats
df['Age'] = np.where(df['Age'] <= 35, 1 , np.where((df['Age'] > 35) & (df['Age'] <= 45), 2, np.where((df['Age'] > 45) & (df['Age'] <= 55), 3, 4)))

plt.figure(figsize=(20,4))
plt.subplot(121)
sns.countplot(df['Age'])#, hue=df["Purchased Bike"])
plt.title(column)    
plt.xticks(rotation=90)

# try making income into cats
df['Income'] = np.where(df['Income'] <= 25000, 1 , np.where((df['Income'] > 25000) & (df['Income'] <= 50000), 2, np.where((df['Income'] > 50000) & (df['Income'] <= 75000), 3, 4)))

plt.figure(figsize=(20,4))
plt.subplot(121)
sns.countplot(df['Income'])#, hue=df["Purchased Bike"])
plt.title(column)    
plt.xticks(rotation=90)
'''

# missing data

vars_with_na = [var for var in df.columns if df[var].isnull().sum() > 0]

for var in vars_with_na:
    tmp = df.copy()

    tmp = tmp[tmp['Education'].isnull()]
    
    print('Number of missing values for %s is %d' % (var, len(tmp)))
    tmp['Purchased Bike'].value_counts().plot.bar(figsize=(10, 4))
    plt.xlabel('Purchased Bike')
    plt.ylabel('Number of Customers')
    plt.show()

    plt.show()
    
    

for var in ['Gender', 'Education', 'Region']:
    mode = df[var].mode()[0]
    
    df[var].fillna(mode, inplace=True)
    df[var].fillna(mode, inplace=True)


# encode variables

# Encoding target
df['Purchased Bike'] =df['Purchased Bike'].apply(lambda x: 0 if x == 'No' else 1)

# Ordinal Vars
# Commute Distance
cd_map = {'0-1 Miles': 0, '1-2 Miles': 1, '2-5 Miles': 2, '5-10 Miles': 3, '10+ Miles': 4}

df['Commute Distance'] = df['Commute Distance'].map(cd_map)


encoder = OneHotEncoder(handle_unknown='ignore')
df_enc = pd.DataFrame(encoder.fit_transform(df[['Education', 'Occupation','Region']]).toarray(), columns=encoder.get_feature_names_out(), index=df.index)
df = df.drop(['Education', 'Occupation','Region'], axis=1)

df =pd.concat([df, df_enc], axis=1)

labelencoder = LabelEncoder()

df['Marital Status'] = labelencoder.fit_transform(df['Marital Status'])
df['Gender'] = labelencoder.fit_transform(df['Gender'])
df['Home Owner'] = labelencoder.fit_transform(df['Home Owner'])

# log transform skewed vars (Age, Income)
df['Income'] = np.log(df['Income'])
df['Age'] = np.log(df['Age'])


# look at correlations

plt.figure(figsize=(20, 20))
s = sns.heatmap(df.corr(),
                annot=True,
                cmap='RdBu',
                vmin=-1,
                vmax=1)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.title('Correlation Matrix')
plt.show()

# look at correlations

plt.figure(figsize=(4, 4))
s = sns.heatmap(df[['Income', 'Home Owner', 'Children']].corr(),
                annot=True,
                cmap='RdBu',
                vmin=-1,
                vmax=1)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.title('Correlation Matrix')
plt.show()

# correlation for target
df[df.columns[1:]].corr()['Purchased Bike'][:].sort_values(ascending=False)

df.to_csv('df_data_cleaned.csv')

































    