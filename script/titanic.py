
# coding: utf-8

# importは1つ目のセルに入れる。
# 依存ライブラリを明白にするため。

# In[75]:


import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # データimport

# In[76]:


for dirname, _, filenames in os.walk('../input/titanic'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[77]:


df_train = pd.read_csv('../input/titanic/train.csv')
df_train.head()


# In[78]:


df_test = pd.read_csv('../input/titanic/train.csv')
df_test.head()


# In[79]:


print(df_train.shape)


# In[80]:


df_train.describe()


# In[81]:


null_val=df_train.isnull().sum()
null_val


# # データ整形

# 文字列データを数値に変換する。
# 
# ```
# df.loc[条件, 列] = 列への代入値
# ```
# を用いる。

# In[82]:


def convert_string_to_number(df):
    df.loc[df['Sex'] == 'male', 'n_Sex'] = 0
    df.loc[df['Sex'] == 'female', 'n_Sex'] = 1
    df.loc[df['Embarked'] == 'S', 'n_Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'n_Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'n_Embarked'] = 2
    return df

df_train = convert_string_to_number(df_train)
df_train.head()


# 欠損データを保管する。
# 
# ```
# df['列'].fillna(保管データ)
# ```
# 
# を用いる。

# In[83]:


def store_null_to_median(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['n_Embarked'] = df['n_Embarked'].fillna(df['n_Embarked'].median())
    return df
    
df_train = store_null_to_median(df_train)
df_train.head()


# In[84]:


null_val=df_train.isnull().sum()
null_val


# # データ可視化
# 
# 参考： [【Kaggle】Titanicを可視化してみる【Seaborn】 | 趣味で始める機械学習](https://ct-innovation01.xyz/DL-Freetime/kaggle-003/)

# In[85]:


sns.pairplot(df_train[['Survived', 'Pclass', 'SibSp', 'Parch', 'Age', 'Fare', 'n_Sex', 'n_Embarked']], hue='Survived')
plt.show()

