'''导入需要的包'''
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGBC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
plt.rcParams['font.sans-serif'] = ['simHei']
plt.rcParams['axes.unicode_minus'] = False


class titanic():

    def __init__(self):
        '''导入数据'''
        self.train = pd.read_csv('./data/titanic_train.csv')
        self.test = pd.read_csv('./data/titanic_test.csv')
        self.train.drop('PassengerId', axis=1, inplace=True)
        self.sex_map = {'male': 1, 'female': 0}
        self.title_map = {
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Sir": "Royalty",
            "Dr": "Officer",
            "Rev": "Officer",
            "the Countess": "Royalty",
            "Dona": "Royalty",
            "Mme": "Mrs",
            "Mlle": "Miss",
            "Ms": "Mrs",
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Master": "Master",
            "Lady": "Royalty"
        }

    '''数据预处理'''

    def process(self, data):
        data['Age'].fillna(data['Age'].median(), inplace=True)  # 年龄缺失用中位数填补
        data['Fare'].fillna(data['Fare'].mean(), inplace=True)  # 票价缺失用均值填补
        data['Embarked'].fillna(data['Embarked'].mode(), inplace=True)  # 码头缺失用众数填补
        data['Cabin'].fillna('U', inplace=True)  # 座位号用U(未知)填补
        data['Sex'] = data['Sex'].map(self.sex_map)  # 性别映射成0, 1
        embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')  # one_hot编码
        data = pd.concat([data, embarked], axis=1)  # 合并数据
        data.drop('Embarked', axis=1, inplace=True)  # 删除列
        pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
        data = pd.concat([data, pclass], axis=1)
        data.drop('Pclass', axis=1, inplace=True)
        title = pd.DataFrame()
        title['Title'] = data['Name'].map(self.get_title)
        title['Title'] = title['Title'].map(self.title_map)
        title = pd.get_dummies(title['Title'], prefix='Title')
        data = pd.concat([data, title], axis=1)
        data.drop('Name', axis=1, inplace=True)
        data['Cabin'] = data['Cabin'].map(lambda c: c[0])
        cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
        data = pd.concat([data, cabin], axis=1)
        data.drop('Cabin', axis=1, inplace=True)
        family = pd.DataFrame()
        family['FamilySize'] = data['Parch'] + data['SibSp'] + 1  # 对家庭成员数量进行分箱
        family['Family_Single'] = family['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        family['Family_Single'] = family['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        family['Family_Single'] = family['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
        data = pd.concat([data, family], axis=1)
        data['Ticket'] = data['Ticket'].map(self.clean_ticket)
        ticket = pd.get_dummies(data['Ticket'], prefix='Ticket')
        data = pd.concat([data, ticket], axis=1)
        data.drop('Ticket', axis=1, inplace=True)
        return data

    '''提取头衔'''

    def get_title(self, name):
        str1 = name.split(',')[1]
        title = str1.split('.')[0]
        return title.strip()

    '''提取票号'''

    def clean_ticket(self, ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))  # 取出票号中的字母
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'N'

    '''逻辑回归'''

    def Logistic(self, x, y):
        lr = LogisticRegression()
        score = cross_val_score(lr, x, y, cv=10).mean()
        print('Logistic:', score)

    '''决策树'''

    def DecisionTree(self, x, y):
        dt = DecisionTreeClassifier(random_state=0)
        score = cross_val_score(dt, x, y, cv=10).mean()
        print('DecisionTree:', score)

    '''随机森林'''

    def RandomForest(self, x, y):
        rfc = RandomForestClassifier(n_estimators=25)
        score = cross_val_score(rfc, x, y, cv=10).mean()
        print('FandomForest:', score)

    '''支持向量机'''

    def SVC(self, x, y):
        Kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in Kernel:
            clf = SVC(kernel=kernel
                      , gamma='auto'
                      , degree=1
                      , cache_size=5000
                      )
            score = cross_val_score(clf, x, y, cv=5).mean()
            print('SVC {}:'.format(kernel), score)

    '''XGBoost'''

    def XGB(self, x, y):
        xgb = XGBC(n_estimators=100)
        score = cross_val_score(xgb, x, y, cv=10).mean()
        print('XGB:', score)

    '''朴素贝叶斯'''

    def Naive_bayes(self, x, y):
        bnb = BernoulliNB()
        score = cross_val_score(bnb, x, y).mean()
        print('Naive_bayes:', score)

    '''梯度提升决策树'''

    def GBDT(self, x, y):
        gbdt = GradientBoostingClassifier()
        score = cross_val_score(gbdt, x, y).mean()
        print('GBDT:', score)

    '''K近邻'''

    def KNN(self, x, y):
        knn = KNeighborsClassifier()
        score = cross_val_score(knn, x, y).mean()
        print('KNN:', score)

    def main(self):
        data = self.process(self.train)
        # targets = pd.read_csv('./data/train.csv', usecols=['Survived'])['Survived'].values
        # print(targets)
        # corr = data.corr()
        # print(corr['Survived'].sort_values(ascending=False))

        x = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        x = StandardScaler().fit_transform(x)
        self.Logistic(x, y)
        self.DecisionTree(x, y)
        self.RandomForest(x, y)
        self.SVC(x, y)
        self.XGB(x, y)
        self.Naive_bayes(x, y)
        self.GBDT(x, y)
        self.KNN(x, y)


if __name__ == '__main__':
    t = titanic()
    t.main()
