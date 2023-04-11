import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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

data = data[['length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','M','F','I','rings']]
# print(data.head(5))

#divide os dados em dados de treinamento e teste
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_available,x_inbox,y_available,y_inbox = train_test_split(x,y,test_size=0.2,random_state=1)
x_train, x_test, y_train, y_test = train_test_split( x_available, y_available, test_size=0.2, random_state=1)


parameters_knn = {'n_neighbors': [17]}
knn_model = GridSearchCV(KNeighborsRegressor(), parameters_knn, cv=3, scoring='neg_mean_absolute_error')
knn_model.fit(x_train, y_train)
knn_best_params = knn_model.best_params_

parameters_svm = {'kernel':['poly'],'C':[10],'gamma':[2]}
svm_model = GridSearchCV(SVR(), parameters_svm, cv=3, scoring='neg_mean_absolute_error')
svm_model.fit(x_train, y_train)
svm_best_params = svm_model.best_params_

parameters_decision_tree = {'max_depth': [5]} 
tree_model = GridSearchCV(DecisionTreeRegressor(), parameters_decision_tree, cv=3, scoring='neg_mean_absolute_error')
tree_model.fit(x_train, y_train)
tree_best_params = tree_model.best_params_

parameters_random_forest = {'n_estimators': [40],'max_features': ["sqrt"],'max_depth': [6]} 
rf_model = GridSearchCV(RandomForestRegressor(), parameters_random_forest, cv=3, scoring='neg_mean_absolute_error')
rf_model.fit(x_train, y_train)
rf_best_params = rf_model.best_params_

knn_pred = knn_model.predict(x_test)
svm_pred = svm_model.predict(x_test)
tree_pred = tree_model.predict(x_test)
rf_pred = rf_model.predict(x_test)

knn_mae = mean_absolute_error(y_test, knn_pred)
svm_mae = mean_absolute_error(y_test, svm_pred)
tree_mae = mean_absolute_error(y_test, tree_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print("KNN MAE: ", mean_absolute_error(y_test, knn_pred))
print("Decision Tree MAE: ", mean_absolute_error(y_test, tree_pred))
print("Random Forest MAE: ", mean_absolute_error(y_test, rf_pred))
print("SVM MAE: ", mean_absolute_error(y_test, svm_pred))

print("Melhores parâmetros KNN: ", knn_model.best_params_)
print("Melhores parâmetros SVM: ", svm_model.best_params_)
print("Melhores parâmetros Decision Tree: ", tree_model.best_params_)
print("Melhores parâmetros Random Forest: ", rf_model.best_params_)

models = {"KNN": knn_model, "SVM": svm_model, "Decision Tree": tree_model, "Random Forest": rf_model}
best_model_name = min(models, key=lambda x: mean_absolute_error(y_test, models[x].predict(x_test)))
print(f"O melhor algoritmo para prever a idade dos abalones foi o {best_model_name}.")

fig, ax = plt.subplots()

ax.scatter(y_test, knn_pred, color='blue', label='KNN')
ax.scatter(y_test, svm_pred, color='red', label='SVM')
ax.scatter(y_test, tree_pred, color='green', label='Tree')
ax.scatter(y_test, rf_pred, color='orange', label='RF')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Idade Real')
ax.set_ylabel('Idade Prevista')
ax.legend(loc='best')
plt.show()