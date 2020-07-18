from time import time
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier

warnings.filterwarnings('ignore')

'''获取头衔'''


def get_title(name):
    s1 = name.split(',')[1]
    s2 = s1.split('.')[0]
    s = s2.strip()
    return s


'''处理数据'''


def deal_with():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    full = train.append(test, ignore_index=True)
    # print(full.isnull().sum())

    full['Age'] = full['Age'].fillna(full['Age'].mean())
    full['Fare'] = full['Fare'].fillna(full['Fare'].mean())

    # print(full['Embarked'].value_counts())
    full['Embarked'] = full['Embarked'].fillna('S')

    # print(full['Cabin'].value_counts())
    full['Cabin'] = full['Cabin'].fillna('U')

    # corr = train.corr()   #相关系数
    # print(corr)

    sex_map = {'male': 1, 'female': 0}
    full['Sex'] = full['Sex'].map(sex_map)

    # embarked = pd.DataFrame()
    embarked = pd.get_dummies(full['Embarked'], prefix='Embarked')
    full = pd.concat([full, embarked], axis=1)
    full.drop('Embarked', axis=1, inplace=True)

    # pclass = pd.DataFrame()
    pclass = pd.get_dummies(full['Pclass'], prefix='Pclass')
    full = pd.concat([full, pclass], axis=1)
    full.drop('Pclass', axis=1, inplace=True)

    title = pd.DataFrame()
    title['Title'] = full['Name'].map(get_title)
    # title.groupby('Title').size()

    title_map = {
        'Capt': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Jonkheer': 'Royalty',
        'Don': 'Royalty',
        'Sir': 'Royalty',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'the Countess': 'Royalty',
        'Dona': 'Royalty',
        'Mme': 'Mrs',
        'Mlle': 'Miss',
        'Ms': 'Mrs',
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Lady': 'Royalty'
    }
    title['Title'] = title['Title'].map(title_map)
    title = pd.get_dummies(title['Title'])
    full = pd.concat([full, title], axis=1)
    full.drop('Name', axis=1, inplace=True)

    cabin = pd.DataFrame()
    cabin['Cabin'] = full['Cabin'].map(lambda c: c[0])
    cabin = pd.get_dummies(cabin['Cabin'], prefix='Cabin')
    full = pd.concat([full, cabin], axis=1)
    full.drop('Cabin', axis=1, inplace=True)

    family = pd.DataFrame()
    family['FamilySize'] = full['Parch'] + full['SibSp'] + 1
    family['Family_Single'] = family['FamilySize'].map(lambda x: 1 if x == 1 else 0)
    family['Family_Small'] = family['FamilySize'].map(lambda x: 1 if 2 <= x <= 4 else 0)
    family['Family_Large'] = family['FamilySize'].map(lambda x: 1 if x >= 5 else 0)
    full = pd.concat([full, family], axis=1)
    # print(full.columns)

    # corr = full.corr()
    # print(corr['Survived'].sort_values(ascending=False))

    full_X = pd.concat([title
                           , pclass
                           , family
                           , full['Fare']
                           , cabin
                           , embarked
                           , full['Sex']
                        ], axis=1)

    global row
    row = 891
    data_X = full_X.loc[0:row - 1, :]
    data_y = full.loc[0:row - 1, 'Survived']
    pred_X = full_X.iloc[row:, :]
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25)
    return full, pred_X, train_X, test_X, train_y, test_y, data_X, data_y


'''逻辑回归'''


def Logistic(train_X, train_y, test_X, test_y, pred_X, full):
    model = LogisticRegression()
    model.fit(train_X, train_y)
    print('Logistic: ', model.score(test_X, test_y))
    pred_Y = model.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row, 'PassengerId']
    pred = pd.DataFrame({'PassengerId': passenger_id
                            , 'Survived': pred_Y})
    pred.to_csv('Logistic.csv', index=False)


'''朴素贝叶斯'''


def GNB(train_X, train_y, test_X, test_y, pred_X, full):
    model = GaussianNB()
    model.fit(train_X, train_y)
    print('GaussianNB: ', model.score(test_X, test_y))
    pred_Y = model.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({'PassengerId': passenger_id
                            , 'Survived': pred_Y})
    pred.to_csv('GNB.csv', index=False)


'''训练模型'''


def fit_model(model, parameters, data_X, data_y):
    scorer = make_scorer(roc_auc_score)
    grid = GridSearchCV(model, parameters, scoring=scorer, cv=4)
    start = time()
    grid = grid.fit(data_X, data_y)
    end = time()
    t = round(end - start, 3)
    # print(grid.best_params_)
    print(model.__class__.__name__ + ':', grid.best_score_, 'time:', t)
    return grid


'''决策树'''


def DecisionTree(pred_X, full, data_X, data_y):
    dict1 = {
        'max_depth': range(5, 20)
        , 'min_samples_split': range(5, 10)
        , 'min_samples_leaf': range(1, 10)
    }
    model = DecisionTreeClassifier(random_state=0)
    tree = fit_model(model, dict1, data_X, data_y)
    pred_Y = tree.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({'PassengerId': passenger_id
                            , 'Survived': pred_Y})
    pred.to_csv('DecisionTree.csv', index=False)


'''支持向量机'''


def SVM(pred_X, full, data_X, data_y):
    dict2 = {'C': range(1, 20)
        , 'gamma': np.arange(0.01, 0.3)}
    model = SVC(probability=True
                , random_state=0)
    svm = fit_model(model, dict2, data_X, data_y)
    pred_Y = svm.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({'PassengerId': passenger_id
                            , 'Survived': pred_Y})
    pred.to_csv('SVM.csv', index=False)


'''随机森林'''


def RandomForest(data_X, data_y, pred_X, full):
    dict3 = {'n_estimators': range(30, 200, 10)}
    model = RandomForestClassifier(random_state=0)
    forest = fit_model(model, dict3, data_X, data_y)
    pred_Y = forest.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({'PassengerId': passenger_id
                            , 'Survived': pred_Y})
    pred.to_csv('RandomForest.csv', index=False)


'''AdaBoost'''


def AdaBoost(data_X, data_y, pred_X, full):
    dict4 = {'n_estimators': range(10, 200, 10)
        , 'learning_rate': np.arange(0.5, 2, 0.1)}
    model = AdaBoostClassifier(random_state=0)
    adaboost = fit_model(model, dict4, data_X, data_y)
    pred_Y = adaboost.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': pred_Y
    })
    pred.to_csv('AdaBoost.csv', index=False)


'''K近邻'''


def KNN(data_X, data_y, pred_X, full):
    dict5 = {
        'n_neighbors': range(2, 10),
        'leaf_size': range(10, 100, 10)
    }
    model = KNeighborsClassifier(n_jobs=1)
    knn = fit_model(model, dict5, data_X, data_y)

    knn1 = KNeighborsClassifier(weights='distance'
                                , n_neighbors=7
                                , leaf_size=10)
    knn1.fit(data_X, data_y)
    print('knn1:', knn1.score(data_X, data_y))
    pred_Y = knn1.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': pred_Y
    })
    pred.to_csv('KNN.csv', index=False)


'''XGBoost'''


def XGB(data_X, data_y, full, pred_X):
    dict6 = {'n_estimators': range(10, 200, 10)}
    model = XGBClassifier(random_state=0
                          , n_jobs=1)
    xgb1 = fit_model(model, dict6, data_X, data_y)

    dict7 = {
        'max_depth': range(2, 20),
        'min_child_weight': range(2, 10)
    }
    model = XGBClassifier(n_estimators=15
                          , random_state=0
                          , n_jobs=1)
    xgb2 = fit_model(model, dict7, data_X, data_y)

    dict8 = {
        'reg_lambda': np.arange(0.2, 2, 0.1)
        , 'reg_alpha': np.arange(0.2, 2, 0.1)
    }
    model = XGBClassifier(n_estimators=15
                          , max_depth=5
                          , min_child_weight=7
                          , random_state=0
                          , n_jobs=1)
    xgb4 = fit_model(model, dict8, data_X, data_y)

    dict9 = {
        'max_depth': range(2, 20)
    }
    model = XGBClassifier(n_estimators=20
                          , random_state=0
                          , n_jobs=1)
    xgb5 = fit_model(model, dict9, data_X, data_y)

    pred_Y = xgb4.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': pred_Y
    })
    pred.to_csv('XGB.csv', index=False)


'''梯度提升'''


def GBDT(data_X, data_y, pred_X, full):
    dict10 = {
        'n_estimators': range(10, 200, 10)
    }
    model = GradientBoostingClassifier(random_state=0)
    gbc1 = fit_model(model, dict10, data_X, data_y)
    pred_Y = gbc1.predict(pred_X)
    pred_Y = pred_Y.astype(int)
    passenger_id = full.loc[row:, 'PassengerId']
    pred = pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': pred_Y
    })
    pred.to_csv('GBDT.csv', index=False)


def main():
    full, pred_X, train_X, test_X, train_y, test_y, data_X, data_y = deal_with()
    Logistic(train_X, train_y, test_X, test_y, pred_X, full)
    GNB(train_X, train_y, test_X, test_y, pred_X, full)
    DecisionTree(pred_X, full, data_X, data_y)
    SVM(pred_X, full, data_X, data_y)
    RandomForest(data_X, data_y, pred_X, full)
    AdaBoost(data_X, data_y, pred_X, full)
    KNN(data_X, data_y, pred_X, full)
    XGB(data_X, data_y, full, pred_X)
    GBDT(data_X, data_y, pred_X, full)


if __name__ == '__main__':
    main()
