import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from prettytable import PrettyTable

from sklearn import preprocessing
import seaborn as sns

#remove os future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#carrega os dados do arquivo abalone.data
column_names = ['sex','length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','rings']
data = pd.read_csv('abalone.data',names = column_names)

# print("Formato da base de dados")
# print(data.head(5))
# print(data.describe())

#pré-processamento dos dados, criando recursos binários para as 3 categorias de gênero
#e convertendo a idade em anéis em grupos de idade (jovem, médio e velho).
for value in "MFI":
    data[value] = data["sex"] == value
del data["sex"]

for i in data.index:
    row = data.loc[i]
    if row.rings <= 8:
        data.loc[i, 'rings'] = 'young'
    elif row.rings >= 11:
        data.loc[i, 'rings'] = 'old'
    elif row.rings >=9 & row.rings <= 10:
        data.loc[i, 'rings'] = 'medium'
        
data = data[['length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','M','F','I','rings']]
# print(data.head(5))

#verifica se o dataset tem problemas com dados não balanceados
age_group = data.groupby('rings').rings.count()
ax = age_group.plot(kind='bar')
plt.ylabel('Numbers')
plt.xlabel('Age group')
plt.title('Age Distribution Of Abalones')
# plt.show()

#divide os dados em dados de treinamento e teste
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_available,x_inbox,y_available,y_inbox = train_test_split(x,y,test_size=0.2,random_state=1)
x_train, x_test, y_train, y_test = train_test_split( x_available, y_available, test_size=0.2, random_state=1)


#cross validation para encontrar os melhores parâmetros para cada classificador
#na execução da GridSearchCV, não foi especificado o número de folds
#sendo assim, foi utilizado o valor padrão (3-fold)

############ KNN ############
parameters_knn = {'n_neighbors':range(1,50)}
gridsearch_knn = GridSearchCV(KNeighborsClassifier(),parameters_knn, cv=5)
gridsearch_knn.fit(x_train,y_train)

knn_model = gridsearch_knn.best_estimator_
print('KNN')
print(gridsearch_knn.best_score_,gridsearch_knn.best_params_)
print('\n')

############ Decision Tree ############
parameters_decision_tree = {'max_depth':range(3,20)} 
gridsearch_decision_tree = GridSearchCV(DecisionTreeClassifier(), parameters_decision_tree, cv=5)
gridsearch_decision_tree.fit(x_train, y_train)

decision_tree_model = gridsearch_decision_tree.best_estimator_
print('Decision Tree')
print (gridsearch_decision_tree.best_score_, gridsearch_decision_tree.best_params_)
print('\n')

############ Random Forest ############
parameters_random_forest = {'n_estimators': range(10,100,10),'max_features': ["sqrt"],'max_depth': range(2,30,2)} 
gridsearch_random_forest = GridSearchCV(RandomForestClassifier(),parameters_random_forest, cv=5)
gridsearch_random_forest.fit(x_train,y_train)

random_forest_model = gridsearch_random_forest.best_estimator_
print('Random Forest')
print(gridsearch_random_forest.best_score_,gridsearch_random_forest.best_params_)
print('\n')

############ SVM ############
parameters_svm = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'C':[0.1,1,10],'gamma':[0.01,0.1,0.5,1,2]}
gridsearch_svm = GridSearchCV(SVC(),parameters_svm, cv=5)
gridsearch_svm.fit(x_train,y_train)

smv_model = gridsearch_svm.best_estimator_
print('SVM')
print(gridsearch_svm.best_score_,gridsearch_svm.best_params_)
print('\n')

#acurácia de cada classificador
models_score = PrettyTable()
models_score.add_column("Classificador",["KNN","Decision Tree","Random Forest","SVM"])
models_score.add_column("Acurácia",[gridsearch_knn.best_score_,gridsearch_decision_tree.best_score_,gridsearch_random_forest.best_score_,gridsearch_svm.best_score_])
print(models_score)

#acurácia para dados de teste do classificador random forest (algoritmo com maior acurácia obtida após diversos testes)
prediction = gridsearch_random_forest.predict(x_inbox)
print("Acurácia Random Forest aos dados de teste:",metrics.accuracy_score(prediction, y_inbox))

x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
clf = RandomForestClassifier(max_depth=8, max_features='sqrt', n_estimators=60)
clf.fit(x_train, y_train)

imp = pd.DataFrame(clf.feature_importances_).reset_index()
imp['index'] = data.columns.values[:-1]
imp.columns=['index','importance']
order = imp.sort_values('importance',ascending = False)['index'].values
fig = plt.subplots(figsize=(11.7, 8.27))# a4 size
sns.barplot(data=imp,x='index',y='importance',order = order,palette="Reds_d")
plt.show()