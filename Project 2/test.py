import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

batData = ['a','b','c','a','c']
bowlData = ['b','a','d','d','a']

df=pd.DataFrame()
df['batting']=batData
df['bowling']=bowlData


fig, ax =plt.subplots(1,2)
sns.countplot(df['batting'], ax=ax[0])
sns.countplot(df['bowling'], ax=ax[1])

plt.show()