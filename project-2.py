#!/usr/bin/env python
# coding: utf-8

# ## Шаг 1. Загрузка данных

# In[1]:


get_ipython().system('pip install shap ')


# In[2]:


get_ipython().system(' pip install phik==0.10.0')


# In[3]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5,5)
from pylab import rcParams
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import recall_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder, 
    StandardScaler, 
    MinMaxScaler,
    RobustScaler)


# In[4]:


import phik
from phik.report import plot_correlation_matrix
from phik import report
import shap


# Загружаем датасеты и проверяем форматирование, классы данных, пропуски

# In[5]:


market = pd.read_csv("/datasets/market_file.csv")
money = pd.read_csv("/datasets/market_money.csv")
time = pd.read_csv("/datasets/market_time.csv")
profit = pd.read_csv("/datasets/money.csv",delimiter=";", decimal=",")


# In[6]:


market.head()


# In[7]:


profit.head()


# Замечаем аномально низкие показатели в Прибыли в profit. Cкорре всего в значениях не хватет 2 нулей. Исправим ошибку.

# In[8]:


profit["Прибыль"] = profit["Прибыль"] * 1000
len(profit)


# In[9]:


profit


# In[10]:


market.info()


# In[11]:


for a in [market,money,time,profit]:
    a.info()


# In[12]:


for a in [market,money,time,profit]:
    print(a.isna().sum())


# Фиксируем отсутствие пропусков, классы данных тоже в порядке, форматирование корректное.

# ## Шаг 2. Предобработка данных

# In[13]:


non_str = market.select_dtypes(include=['int64','float64'])
non_str.info()


# In[14]:


str_v = market.select_dtypes(exclude=['int64','float64'])


# In[15]:


cols_list = non_str.columns.tolist()


# In[16]:


for a in str_v:
            b = market.pivot_table(index=[a],values="id",aggfunc="count")
            plt.figure()
            plt.title(a,loc='right')
            plt.pie(b["id"], labels=market[a].unique(), autopct='%.2f')
            plt.show




# In[17]:


a = market.drop("id",axis=1).hist(figsize=(15,7));


# In[18]:


for b in non_str.drop("id",axis=1):
    plt.figure();
    plt.title(b)
    market[b].hist(figsize=(7,3));
    plt.xlabel("значения");
    plt.ylabel("клиенты");

    


# In[19]:


for a in [money,time,profit]:
    a.drop("id",axis=1).hist()
    plt.xlabel("значение")
    plt.ylabel("клиенты")


# Строим гистораммы, замечаем аномалии в показатели выручка в датасете money: нулевые значения (которые по тз необходимо удалить и вброс в 100.000). Удалим их позже после объединения датасетов. Также замечаем ошибку в названии категорий в признаке тип сервиса, исправляем ее. Проверям категорияльные признаки в остальных датасетах.

# In[20]:


for a in [market,money,time]:
    b = a.select_dtypes(exclude=['int64','float64'])
    for c in b.columns:
        print(a[c].unique())


# In[21]:


market['Тип сервиса'] = market['Тип сервиса'].replace('стандартт', 'стандарт') 


# In[22]:


market['Тип сервиса'].unique()


# In[23]:


time["Период"] = time["Период"].replace('предыдцщий_месяц', 'предыдущий_месяц') 


# In[24]:


time["Период"].unique()


# Объединяем данные таким образом чтобы в итоговом датасете были столбцы по показателям выручки/минут по месяцам

# In[25]:


money1 = money.query("Период == 'текущий_месяц'")
money2 = money.query("Период == 'предыдущий_месяц'")
money3 = money.query("Период == 'препредыдущий_месяц'")


# In[26]:


market = market.merge(money1,on="id")


# In[27]:


market.rename(columns={"Выручка": "Выручка_текущий_месяц"},inplace=True)
market = market.drop("Период",axis=1)


# In[28]:


market = market.merge(money2,on="id")
market.rename(columns={"Выручка": "Выручка_предыдущий_месяц"},inplace=True)
market = market.drop("Период",axis=1)


# In[29]:


market = market.merge(money3,on="id")
market.rename(columns={"Выручка": "Выручка_препредыдущий_месяц"},inplace=True)
market = market.drop("Период",axis=1)


# In[30]:


market.info()


# In[31]:


time1 = time.query("Период == 'текущий_месяц'")
time2 = time.query("Период == 'предыдущий_месяц'")


# In[32]:


market = market.merge(time1,on="id")
market.rename(columns={"минут": "Минуты_текущий_месяц"},inplace=True)


# In[33]:


market = market.merge(time2,on="id")
market.rename(columns={"минут": "Минуты_предыдущий_месяц"},inplace=True)


# In[34]:


market.head(20)
market = market.drop(["Период_y","Период_x"],axis=1)


# In[35]:


market.head()


# Проверяем датасет после объединения на тип данных и пропуски.

# In[36]:


market.info()


# In[37]:


market.isna().sum()


# Теперь избавляемся от id, с нулевыми показателями выручки и вброс в 100.000

# In[38]:


u = market.query("Выручка_препредыдущий_месяц == 0")


# In[39]:


for a in u["id"].unique():
    market = market.drop(market[market["id"]==a].index)


# In[40]:


u1 = market.query("Выручка_текущий_месяц > 20000")


# In[41]:


for a in u1["id"].unique():
    market = market.drop(market[market["id"]==a].index)


# In[42]:


market.drop("id",axis=1).hist(figsize=(20,10));


# ## Шаг 3. Анализ корреляций

# Кодируем целевой признак целочисленным значением, чтобы посмотреть корреляцию, избавляемся от неинформативного id

# In[43]:


market.info()


# In[44]:


market.set_index("id",inplace=True)


# In[45]:


market


# In[46]:


phik_overview = market.phik_matrix()

plot_correlation_matrix(
    phik_overview.values,
    x_labels=phik_overview.columns,
    y_labels=phik_overview.index,
    title=r"correlation $\phi_K$",
    fontsize_factor=1.5,
    figsize=(15, 12)
)


# У таргета наибольшая корреляция с показателями Cтраниц_за_визит (0.75), Акционные_покупки(0.51) Маркет_актив_6_мес (0.54), Минут_предыдущий_месяц (0.52), Неоплаченные_продукты (0.51),Минут_текущий_месяц (0.58), Средний_просмотр_категорий_за_визит (0.54), Выручка_препредыдущий_месяц(0.69). Нулевая корреляция с таргетом у признаков разрешить сообщать и маркет активность текущий месяц.
# 
# Также стоит отметить сильную корреляцию между показателями Выручка_текущий_месяц и Выручка_предыдущий_месяц.

# Группируем признаки по группам (коммуникация, продукт, поведение на сайте, выручка (сom,product,site, revenue) и проверям, какая сильнее всего коррелирует с таргетом:

# In[47]:


com = market[["Покупательская активность","Тип сервиса","Разрешить сообщать","Маркет_актив_6_мес","Маркет_актив_тек_мес","Длительность"]]
product=market[["Покупательская активность","Акционные_покупки","Популярная_категория","Средний_просмотр_категорий_за_визит","Неоплаченные_продукты_штук_квартал"]]
site = market[["Покупательская активность","Минуты_предыдущий_месяц","Минуты_текущий_месяц","Страниц_за_визит","Ошибка_сервиса"]]
revenue = market[["Покупательская активность","Выручка_предыдущий_месяц","Выручка_препредыдущий_месяц","Выручка_текущий_месяц"]]


# In[48]:


for a in [com,product,site,revenue]:
    for b in ["Коммуникация с клиентом","Продуктовое поведение","Поведение на сайте","Вырчка"]:
        pp = sns.pairplot(a,hue="Покупательская активность");
        pp.fig.suptitle(b, y=1.08)
        plt.show()


# Замечаем наибольшую корреляцию у продуктовых признаков и признаков, cвязанных с поведением на сайте

# In[49]:


market.head()


# ## Шаг 4. Моделирование

# В рамках анализа корреляций мы выявили два признака с нулевой корреляцией с таргетом, удаляем их перед подготовкой данных к модели во избежаний шума.

# In[50]:


RANDOM_STATE = 42
TEST_SIZE = 0.25
y = market[['Покупательская активность']]
X = market.drop(['Покупательская активность','Разрешить сообщать','Маркет_актив_тек_мес'], axis=1)
y['Покупательская активность'] = y['Покупательская активность'].apply(lambda x: 1 if x == 'Снизилась' else 0)
scoring = "recall"


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify = market['Покупательская активность']
)


# In[52]:


ord_columns = ['Тип сервиса']
ohe_columns = ['Популярная_категория']

num_columns = ['Маркет_актив_6_мес','Длительность','Акционные_покупки','Средний_просмотр_категорий_за_визит','Неоплаченные_продукты_штук_квартал','Ошибка_сервиса','Страниц_за_визит','Выручка_текущий_месяц','Выручка_предыдущий_месяц','Выручка_препредыдущий_месяц','Минуты_текущий_месяц','Минуты_предыдущий_месяц'] 
ohe_pipe = Pipeline(
    [
        (
            'simpleImputer_ohe', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        (
            'ohe', 
            OneHotEncoder(drop='first', handle_unknown='error', sparse=False)
        )
    ]
)

ord_pipe = Pipeline(
    [
        (
            'simpleImputer_before_ord', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        (
            'ord',
            OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
        ),
        (
            'simpleImputer_after_ord', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        )
    ])


data_preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_columns),
        ('ord', ord_pipe, ord_columns),
        ('num', MinMaxScaler(), num_columns)
    ], 
    remainder='passthrough'
)


pipe_final= Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]
)


param_grid = [
       {
        'models': [DecisionTreeClassifier(random_state=RANDOM_STATE)],
        'models__max_depth': range(2, 5),
        'models__max_features': range(2,5),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    
    {
        'models': [KNeighborsClassifier()],
        'models__n_neighbors': range(2,5),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']   
    },

    {
        'models': [LogisticRegression(
            random_state=RANDOM_STATE, 
            solver='liblinear', 
            penalty='l1'
        )],
        'models__C': range(1,5),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },

 
     {
        'models': [SVC(probability=True,random_state=RANDOM_STATE)],
            'C':range(0,1), 
            'degree':range(3,4),
            'kernel': ['rbf'],

        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    }
]

randomized_search = RandomizedSearchCV(
    pipe_final, 
    param_grid, 
    cv=5,
    scoring="recall",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
randomized_search.fit(X_train, y_train)

print('Лучшая модель и её параметры:\n\n', randomized_search.best_estimator_)
print ('Метрика лучшей модели на тренировочной выборке:', randomized_search.best_score_)

y_test_pred = randomized_search.predict(X_test)
print(f'Метрика recall на тестовой выборке: {recall_score(y_test, y_test_pred)}')


# In[53]:


prepros= Pipeline(
    [('preprocessor', data_preprocessor)])


# In[54]:


X_train_trans = prepros.fit_transform(X_train)


# In[55]:


X_test_trans = prepros.transform(X_test)


# In[56]:


model = LogisticRegression(C=3, penalty='l1', random_state=42, solver='liblinear')


# In[57]:


model = model.fit(X_train_trans,y_train)


# In[58]:


pred = model.predict(X_test_trans)


# In[59]:


recall_score(y_test,pred)


# Получаем названия столбцов после обработки для shap.explainer

# In[60]:


new_col = prepros.named_steps['preprocessor'].named_transformers_['ohe'].named_steps['ohe'].get_feature_names()
new_cols = np.concatenate([new_col,ord_columns,num_columns])


# In[61]:


new_cols


# In[62]:


pd.set_option('display.max_colwidth', None)


# In[63]:


models_result = pd.concat([pd.DataFrame(randomized_search.cv_results_["params"]),pd.DataFrame(randomized_search.cv_results_["mean_test_score"], columns=["recall"])],axis=1)


# In[64]:


models_result


# In[65]:


proba = model.predict_proba(X_test_trans)
probabilities_one = proba[:, 1]
print('Площадь ROC-кривой:', roc_auc_score(y_test, probabilities_one))


# In[66]:


market.head()


# In[67]:


explainer = shap.LinearExplainer(model,X_train_trans,feature_names=new_cols)
shap_values = explainer(X_train_trans)

shap.plots.beeswarm(shap_values) 


# Страниц_за_визит, Средний_просмотр_категорий_за_визит, Минуты_предыдущий_месяц, Минуты_текущий_месяц, Маркет_актив_6_мес, Выручка_предыдущий_месяц - чем выше эти признаки, тем больше вероятность присвоения класса 1. Обратная корреляция с у признаков Неоплаченные_продукты_штук_квартал,Акционные_покупки - высокое значение этих признаков означает высокую вероятность принадлежности к классу 0. Категория Мелкая бытовая техника и электроника повышает вероятность присвоения класса 1. Уверенне всего модель присваивает класс на основе признаков Акционные_покупки, Маркет_актив_6_мес, Выручка_предыдущий_месяц	

