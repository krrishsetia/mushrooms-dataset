import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing

pd.options.display.max_columns = 5
pd.options.display.max_rows = 10000000

data = pd.read_csv('mushrooms.csv')

def classify(var):
    if var == 'a':
        return 1
    elif var == 'b':
        return 2
    elif var == 'c':
        return 3
    elif var == 'd':
        return 4
    elif var == 'e':
        return 5
    elif var == 'f':
        return 6
    elif var == 'g':
        return 7
    elif var == 'h':
        return 8
    elif var == 'i':
        return 9
    elif var == 'j':
        return 10
    elif var == 'k':
        return 11
    elif var == 'l':
        return 12
    elif var == 'o':
        return 13
    elif var == 'm':
        return 14
    elif var == 'n':
        return 15
    elif var == 'p':
        return 16
    elif var == 'q':
        return 17
    elif var == 'r':
        return 18
    elif var == 's':
        return 19
    elif var == 't':
        return 20
    elif var == 'u':
        return 21
    elif var == 'v':
        return 22
    elif var == 'w':
        return 23
    elif var == 'x':
        return 24
    elif var == 'y':
        return 25
    elif var == 'z':
        return 26

data['class'] = data['class'].apply(classify)
data['cap-shape'] = data['cap-shape'].apply(classify)
data['cap-surface'] = data['cap-surface'].apply(classify)
data['cap-color'] = data['cap-color'].apply(classify)
data['bruises'] = data['bruises'].apply(classify)
data['odor'] = data['odor'].apply(classify)
data['gill-attachment'] = data['gill-attachment'].apply(classify)
data['gill-spacing'] = data['gill-spacing'].apply(classify)
data['gill-size'] = data['gill-size'].apply(classify)
data['gill-color'] = data['gill-color'].apply(classify)
data['stalk-shape'] = data['stalk-shape'].apply(classify)
data['stalk-root'] = data['stalk-root'].apply(classify)
data['stalk-surface-above-ring'] = data['stalk-surface-above-ring'].apply(classify)
data['stalk-surface-below-ring'] = data['stalk-surface-below-ring'].apply(classify)
data['stalk-color-above-ring'] = data['stalk-color-above-ring'].apply(classify)
data['stalk-color-below-ring'] = data['stalk-color-below-ring'].apply(classify)
data['veil-type'] = data['veil-type'].apply(classify)
data['veil-color'] = data['veil-color'].apply(classify)
data['ring-number'] = data['ring-number'].apply(classify)
data['ring-type'] = data['ring-type'].apply(classify)
data['spore-print-color'] = data['spore-print-color'].apply(classify)
data['population'] = data['population'].apply(classify)
data['habitat'] = data['habitat'].apply(classify)

x = data['ring-number'].values.reshape(-1,1)
y = data['habitat'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train,y_train)
x_test_scaled = scaler.transform(x_test)

regression = Ridge()

regression.fit(x_train_scaled,y_train)
y_pred = regression.predict(x_test_scaled)

plt.scatter(x_test_scaled,y_test)
plt.plot(x_test_scaled,y_pred)
plt.show()
y_pred_round = np.round(y_pred)
print('mean squared error:',metrics.mean_squared_error(y_true=y_test,y_pred=y_pred))
print('regular accuracy:',metrics.accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('balanced accuracy:',metrics.balanced_accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('f1:',metrics.f1_score(y_true=y_test,y_pred=y_pred_round,average='weighted')*100,'%')
print('precision:',metrics.precision_score(y_true=y_test,y_pred=y_pred_round,average='weighted',zero_division=0)*100,'%')
print('kappa:',metrics.cohen_kappa_score(y1=y_test,y2=y_pred_round)*100,'%')

matrix = metrics.confusion_matrix(y_true=y_test,y_pred=y_pred_round)
display = metrics.ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()
