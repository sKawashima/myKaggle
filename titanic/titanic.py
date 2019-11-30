
# coding: utf-8

# importは1つ目のセルに入れる。
# 依存ライブラリを明白にするため。

# In[ ]:


import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import tree


# # データimport

# In[ ]:


for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv('input/train.csv')
df_train.head()


# In[ ]:


df_test = pd.read_csv('input/test.csv')
df_test.head()


# In[ ]:


print(df_train.shape)


# In[ ]:


df_train.describe()


# In[ ]:


df_train.isnull().sum()


# # データ整形

# 欠損データを保管する。
# 
# ```
# df['列'].fillna(保管データ)
# ```
# 
# を用いる。

# In[ ]:


def store_null_to_median(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Cabin'].fillna('0', inplace=True)
    df['Embarked'].fillna('0', inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    return df

df_train['Embarked'].dropna()

df_train = store_null_to_median(df_train)
df_train.head()


# In[ ]:


null_val=df_train.isnull().sum()
null_val


# 文字列データを数値に変換する。
# 
# ```
# df.loc[条件, 列] = 列への代入値
# ```
# を用いる。

# In[ ]:


def convert_string_to_number(df):
    df.loc[df['Sex'] == 'male', 'n_Sex'] = 0
    df.loc[df['Sex'] == 'female', 'n_Sex'] = 1
    df.loc[df['Embarked'] == 'S', 'n_Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'n_Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'n_Embarked'] = 2
    df['n_Embarked'] = df['n_Embarked'].fillna(df['n_Embarked'].median())
    df.loc[df['Cabin'].str.startswith('A'), 'n_Cabin'] = 1
    df.loc[df['Cabin'].str.startswith('B'), 'n_Cabin'] = 2
    df.loc[df['Cabin'].str.startswith('C'), 'n_Cabin'] = 3
    df.loc[df['Cabin'].str.startswith('D'), 'n_Cabin'] = 4
    df.loc[df['Cabin'].str.startswith('E'), 'n_Cabin'] = 5
    df.loc[df['Cabin'].str.startswith('F'), 'n_Cabin'] = 6
    df.loc[df['Cabin'].str.startswith('G'), 'n_Cabin'] = 7
    df.loc[df['Cabin'].str.startswith('T'), 'n_Cabin'] = 8
    df.loc[df['Cabin']==('0'), 'n_Cabin'] = df['n_Cabin'].median()
    return df

df_train = convert_string_to_number(df_train)
df_train.head()


# # データ可視化
# 
# 参考： [【Kaggle】Titanicを可視化してみる【Seaborn】 | 趣味で始める機械学習](https://ct-innovation01.xyz/DL-Freetime/kaggle-003/)

# In[ ]:


sns.pairplot(df_train[['Survived', 'Pclass', 'SibSp', 'Parch', 'Age', 'Fare', 'n_Sex', 'n_Embarked', 'n_Cabin']], hue='Survived')
plt.show()


# In[ ]:


sns.factorplot(
    x = 'Sex',
    y = 'Survived',
    data = df_train,
    kind='bar'
)

plt.show()


# In[ ]:


sns.countplot(
    x = 'Sex',
    hue = 'Survived',
    data = df_train
)

plt.show()


# In[ ]:


sns.factorplot(
    x='Pclass',
    y='Survived',
    hue='Sex',
    data=df_train,
    kind='bar'
)

plt.show()


# # 予測モデル作成

# In[ ]:


target = df_train['Survived'].values

def setData(df):
    return df[[
        'Pclass',
        'n_Sex',
        'Age',
        'Fare',
        'n_Embarked',
        'n_Cabin'
    ]].values

features = setData(df_train)

titanic_tree = tree.DecisionTreeClassifier()
titanic_tree = titanic_tree.fit(features, target)


# # 対象データ整形

# In[ ]:


df_test = store_null_to_median(df_test)
df_test = convert_string_to_number(df_test)
df_test.isnull().sum()


# # データ推測

# In[ ]:


test_features = setData(df_test)

my_prediction = titanic_tree.predict(test_features)


# In[ ]:


PassengerId = np.array(df_test['PassengerId']).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

my_solution.to_csv('gender_submission.csv', index_label = ["PassengerId"])

