import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv('./data/bike_train.csv')
test = pd.read_csv('./data/bike_test.csv')

# print(train.isnull().sum().sort_values(ascending=False))
# print(test.isnull().sum().sort_values(ascending=False))
# print(train.info())
# print(test.info())

train.datetime = pd.to_datetime(train.datetime)
test.datetime = pd.to_datetime(test.datetime)

# print(train.info())
# print(test.info())

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['week'] = train['datetime'].dt.week

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['week'] = test['datetime'].dt.week

# print(train.tail(3))
# print(test.tail(3))

plt.figure(figsize=(16, 8))
sns.heatmap(train.corr(), annot=True)
# plt.show()

plt.figure(figsize=(16, 8))
sns.distplot(train['count'])
# plt.show()

plt.figure(figsize=(16, 8))
plt.plot(train['datetime'][0:500], train['count'][0:500])
# plt.show()

plt.hist(x='workingday', data=train)

plt.figure(figsize=(16, 8))
sns.boxplot(x='season', y='count', data=train)

plt.figure(figsize=(16, 8))
sns.boxplot(x='week', y='count', data=train)

plt.figure(figsize=(16, 8))
sns.boxplot(x='hour', y='count', data=train)

plt.figure(figsize=(16, 8))
sns.boxplot(x='year', y='count', data=train)

plt.figure(figsize=(16, 8))
plt.hist(train['count'][train['year'] == 2011], alpha=0.5, label='2011')
plt.hist(train['count'][train['year'] == 2012], alpha=0.5, label='2012', color='red')
plt.scatter(train['hour'], train['count'])
# print(train.head(3))

del train['datetime']

Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
# print(IQR)

train_wind = train[~((train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))).any(axis=1)]
train_wind.dropna(inplace=True)

# print(train.info())
# print(train_wind.info())
# print(train_wind.head(3))

plt.figure(figsize=(12, 7))
sns.boxplot(x='season', y='windspeed', data=train_wind, palette='winter')


def wind(cols):
    windspeed = cols[0]
    season = cols[1]
    if windspeed == 0:
        if season == 1:
            return 14
        elif season == 2:
            return 14
        else:
            return 13
    else:
        return windspeed


train_wind['wind'] = train_wind[['windspeed', 'season']].apply(wind, axis=1)
test['wind'] = test[['windspeed', 'season']].apply(wind, axis=1)

# print(test.head(3))
# print(train_wind.head(3))

train_wind[['season', 'holiday', 'workingday', 'weather', 'year', 'month'
    , 'day', 'hour', 'week']] = train_wind[['season', 'holiday', 'workingday'
    , 'weather', 'year', 'month', 'day', 'hour', 'week']].astype('category')
test[['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'day'
    , 'hour', 'week']] = test[['season', 'holiday', 'workingday', 'weather'
    , 'year', 'month', 'day', 'hour', 'week']].astype('category')
# print(train_wind.info())

X = train_wind[['season', 'holiday', 'workingday', 'weather', 'temp'
    , 'atemp', 'humidity', 'year', 'month', 'day', 'hour', 'week', 'wind']]
y = train_wind['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# sc_X = MinMaxScaler()
# sc_y = MinMaxScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)
# y_train = sc_X.fit_transform(y_train)
# y_test = sc_y.fit_transform(y_test)

sc_X = MinMaxScaler()
sc_X.fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = MinMaxScaler()
sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
print('MSE:', metrics.mean_squared_error(y_test, rf_prediction))

plt.scatter(y_test, rf_prediction)

plt.figure(figsize=(16, 8))
plt.plot(rf_prediction[0:200], 'r')
plt.plot(y_test[0:200])

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
dt_prediction = dt_reg.predict(X_test)
print('MSE:', metrics.mean_squared_error(y_test, dt_prediction))

plt.scatter(y_test, dt_prediction)
# print(test.head(3))

test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity'
    , 'year', 'month', 'day', 'hour', 'week', 'wind']] = sc_X.fit_transform(test[['season'
    , 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity'
    , 'year', 'month', 'day', 'hour', 'week', 'wind']])
test_pred = rf.predict(test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp'
    , 'humidity', 'year', 'month', 'day', 'hour', 'week', 'wind']])
# print(test_pred)

test_pred = test_pred.reshape(-1, 1)
test_pred = sc_y.inverse_transform(test_pred)
test_pred = pd.DataFrame(test_pred, columns=['count'])
df = pd.concat([test['datetime'], test_pred], axis=1)
# print(df.head(3))

df['count'] = df['count'].astype('int')
df.to_csv('submission.csv', index=False)
