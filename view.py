import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set(style='darkgrid', font_scale=1.5)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('./data/train.csv')

corr = train.corr()
plt.subplots(figsize=(10, 4))
sns.heatmap(corr, annot=True)

sns.countplot(x='Pclass', hue='Survived', data=train)

sns.countplot(x='Sex', hue='Survived', data=train)

train_age = sns.FacetGrid(train, col='Survived', height=5)
train_age.map(plt.hist, 'Age', bins=40)

fig = plt.figure(figsize=(10, 6))
sns.violinplot(x='Survived', y='Age', data=train
               , split=True, palette={0: 'r', 1: 'g'})
plt.xlabel('是否存活')
plt.ylabel('年龄')
plt.title('年龄和存活')
plt.show()

sns.countplot(x='SibSp', hue='Survived', data=train)

sns.countplot(x='Parch', hue='Survived', data=train)

train['members'] = train['Parch'] + train['SibSp']
sns.countplot(x='members', hue='Survived', data=train)

train_fare = sns.FacetGrid(train, col='Survived', height=5)
train_fare.map(plt.hist, 'Fare', bins=5)

train.groupby('Pclass').agg('mean')['Fare'].plot(kind='bar', figsize=(10, 6))
plt.title('等级和票价')
plt.xlabel('等级')
plt.ylabel('票价')
plt.show()

train['Cabin'] = train['Cabin'].map(lambda x: 'y' if type(x) == str else 'n')
sns.countplot(x='Cabin', hue='Survived', data=train)

sns.countplot(x='Embarked', hue='Survived', data=train)

plt.pie(train['Survived'].value_counts(), labels=['死亡', '获救'], autopct='%1.2f%%')

ax5 = fig.add_subplot(2, 3, 5)
train['Age'][train['Pclass'] == 1].plot(kind='kde', label='头等舱')
train['Age'][train['Pclass'] == 2].plot(kind='kde', label='二等舱')
train['Age'][train['Pclass'] == 3].plot(kind='kde', label='三等舱')
plt.ylabel('密度')
ax5.legend(loc='best')
plt.show()

survived_0 = train['Pclass'][train['Survived'] == 0].value_counts()
survived_1 = train['Pclass'][train['Survived'] == 1].value_counts()
p_survived = pd.DataFrame({'获救': survived_1, '死亡': survived_0})
p_survived.plot(kind='bar', stacked=True)
plt.title('各舱位获救情况')
plt.ylabel('人数')
plt.show()

fig = plt.figure()
plt.subplot(131)
plt.pie([p_survived.iloc[0, 0], p_survived.iloc[0, 1]], labels=['获救', '死亡'], autopct='%1.0f%%')
plt.title('一等舱获救概率')
fig = plt.figure()
plt.subplot(132)
plt.pie([p_survived.iloc[1, 0], p_survived.iloc[1, 1]], labels=['获救', '死亡'], autopct='%1.0f%%')
plt.title('二等舱获救概率')
fig = plt.figure()
plt.subplot(133)
plt.pie([p_survived.iloc[2, 0], p_survived.iloc[2, 1]], labels=['获救', '死亡'], autopct='%1.0f%%')
plt.title('三等舱获救概率')
plt.show()

sns.set_style('dark')
plt.figure(figsize=(10, 8))
colnm = train.columns.tolist()
mcorr = train[colnm].corr()
mask = np.zeros_like(mcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
plt.show()

age = train['Age'].dropna(axis=0)
plt.subplot(211)
sns.distplot(age)

plt.subplot(212)
sns.distplot(train[train.Survived == 1].Age.dropna(axis=0), hist=False, color='r', label='live')
sns.distplot(train[train.Survived == 0].Age.dropna(axis=0), hist=False, color='g', label='dead')
plt.show()

sns.distplot(train[train.Sex == 'male'].Age.dropna(axis=0), hist=False, color='r', label='male')
sns.distplot(train[train.Sex == 'female'].Age.dropna(axis=0), hist=False, color='g', label='female')
plt.show()

data = train[['Sex', 'Survived']]
y_male = [0, 0]
y_female = [0, 0]
for i in range(len(data.Sex)):
    if data.Sex[i] == 'male':
        if data.Survived[i] == 0:
            y_male[0] += 1
        else:
            y_male[1] += 1
    else:
        if data.Survived[i] == 0:
            y_female[0] += 1
        else:
            y_female[1] += 1
x_number = range(1, 3)
x_word = ['dead', 'live']
plt.bar(x_number, y_male, label='male')
plt.bar(x_number, y_female, bottom=y_male, label='female')
plt.legend()
plt.xticks(x_number, x_word)
plt.show()

data = train[['Fare', 'Survived']]
sns.distplot(data.Fare, hist=False)
plt.title('Fare')
plt.show()
sns.distplot(data[data.Survived == 1].Fare, hist=False, label='Survived')
sns.distplot(data[data.Survived == 0].Fare, hist=False, label='Dead')
plt.title('Fare and Survived')
plt.show()

data = train[['SibSp', 'Parch', 'Survived']]
plt.subplot(211)
sns.countplot(train.SibSp)
plt.subplot(212)
sns.countplot(train.Parch)
plt.suptitle('SibSp and Parch')
plt.show()

plt.subplot(211)
sns.countplot(train[train.Survived == 1].SibSp)
plt.xlabel('live')
plt.subplot(212)
sns.countplot(train[train.Survived == 0].SibSp)
plt.xlabel('dead')
plt.suptitle('SibSp live and dead')
plt.show()

plt.subplot(211)
sns.countplot(train[train.Survived == 1].Parch)
plt.xlabel('live')
plt.subplot(212)
sns.countplot(train[train.Survived == 0].Parch)
plt.xlabel('dead')
plt.suptitle('Parch live and dead')
plt.show()

data = train[['Embarked', 'Survived']]
y_dead = [0, 0, 0]
y_live = [0, 0, 0]
pos = [1, 2, 3]
for i in range(len(data)):
    if data.Survived[i] == 1:
        if data.Embarked[i] == 'C':
            y_live[0] += 1
        elif data.Embarked[i] == 'Q':
            y_live[1] += 1
        else:
            y_live[2] += 1
    else:
        if data.Embarked[i] == 'C':
            y_dead[0] += 1
        elif data.Embarked[i] == 'Q':
            y_dead[1] += 1
        else:
            y_dead[2] += 1
plt.bar(pos, y_live, label='live')
plt.bar(pos, y_dead, bottom=y_live, label='dead')
defination = ['C', 'Q', 'S']
plt.xticks(pos, defination)
plt.legend()
plt.title('Embarked and Survived')
plt.show()

plt.rcParams['font.sans-serif'] = ['SimHei']
n = train['Survived'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(n, autopct='%.2f%%', labels=['死亡', '存活']
        , pctdistance=0.4, labeldistance=0.6, shadow=True
        , explode=[0, 0.1], textprops=dict(size=15))
plt.title('总体生还率')
plt.show()

sex_count = train.groupby(by='Sex')['Survived'].value_counts()
plt.figure(figsize=(2 * 5, 5))
axs1 = plt.subplot(1, 2, 1)
axs1.pie(sex_count.loc['female'][::-1], autopct='%.2f%%', labels=['死亡', '存活']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['b', 'y'], startangle=90)
axs1.set_title('女性生还率')
plt.show()

axs2 = plt.subplot(1, 2, 2)
axs2.pie(sex_count.loc['male'], autopct='%.2f%%', labels=['死亡', '生存']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['#FFB6C1', '#AFEEEE'])
axs2.set_title('男性生还率')
plt.show()

age_range = train['Age']
age_num, _ = np.histogram(age_range, range=[0, 80], bins=16)
age_survived = []
for age in range(5, 81, 5):
    survived_num = train.loc[(age_range >= age - 5) & (age_range <= age)]['Survived'].sum()
    age_survived.append(survived_num)
plt.figure(figsize=(12, 6))
plt.bar(np.arange(2, 78, 5) + 0.5, age_num, width=5, label='总人数', alpha=0.8)
plt.bar(np.arange(2, 78, 5) + 0.5, age_survived, width=5, label='生还人数')
plt.xticks(range(0, 81, 5))
plt.yticks(range(0, 121, 10))
plt.xlabel('年龄', position=(0.95, 0), fontsize=15)
plt.ylabel('人数', position=(0, 0.95), fontsize=15)
plt.title('各年龄生还人数')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

embarked_count = train.groupby(by='Embarked')['Survived'].value_counts()
plt.figure(figsize=(3 * 5, 5))
axs1 = plt.subplot(1, 3, 1)
axs1.pie(embarked_count.loc['C'][::-1], autopct='%.2f%%', labels=['死亡', '存活']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['r', 'g'])
axs1.set_title('法国生还率')
axs2 = plt.subplot(1, 3, 2)
axs2.pie(embarked_count.loc['Q'], autopct='%.2f%%', labels=['死亡', '存活']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['y', 'b'])
axs2.set_title('爱尔兰生还率')
axs3 = plt.subplot(1, 3, 3)
axs3.pie(embarked_count.loc['S'], autopct='%.2f%%', labels=['死亡', '存活']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['#698B69', '#76EE00'])
axs3.set_title('英国生还率')
plt.show()

pclass_count = train.groupby(by='Pclass')['Survived'].value_counts()
plt.figure(figsize=(3 * 5, 5))
axs1 = plt.subplot(1, 3, 1)
axs1.pie(pclass_count.loc[1][::-1], autopct='%.2f%%', labels=['死亡', '存活']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['r', 'g'])
axs1.set_title('一等舱乘客生还率')
axs2 = plt.subplot(1, 3, 2)
axs2.pie(pclass_count.loc[2], autopct='%.2f%%', labels=['死亡', '存活']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['b', 'y'])
axs2.set_title('二等舱乘客生还率')
axs3 = plt.subplot(1, 3, 3)
axs3.pie(pclass_count.loc[3], autopct='%.2f%%', labels=['死亡', '存活']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1]
         , textprops=dict(size=15), colors=['c', 'm'])
axs3.set_title('三等舱乘客生还率')
plt.show()

embarked_pclass = train.groupby(by='Embarked')['Pclass'].value_counts()
plt.figure(figsize=(3 * 5, 5))
axs1 = plt.subplot(131)
axs1.pie(embarked_pclass.loc['C'], autopct='%.2f%%', labels=['一等舱', '三等舱', '二等舱']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1, 0.1]
         , textprops=dict(size=15), colors=['c', 'm', 'y'])
axs1.set_title('法国乘客各舱位占比')
axs2 = plt.subplot(132)
axs2.pie(embarked_pclass.loc['Q'], autopct='%.2f%%', labels=['三等舱', '二等舱', '一等舱']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1, 0.1]
         , textprops=dict(size=10), colors=['r', 'g', 'b'])
axs2.set_title('爱尔兰乘客各舱位占比')
axs3 = plt.subplot(133)
axs3.pie(embarked_pclass.loc['S'], autopct='%.2f%%', labels=['三等舱', '二等舱', '一等舱']
         , pctdistance=0.4, labeldistance=0.6, shadow=True, explode=[0, 0.1, 0.1]
         , textprops=dict(size=15), colors=['#698B69', '#76EE00', '#76EEC6'], startangle=180)
axs3.set_title('英国乘客各舱位占比')
plt.show()
