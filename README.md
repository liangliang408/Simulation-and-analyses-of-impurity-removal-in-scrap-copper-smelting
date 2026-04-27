# Simulation-and-analyses-of-impurity-removal-in-scrap-copper-smelting
# machine_learning
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import AdaBoostRegressor  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import mean_squared_error, r2_score 
from hyperopt import fmin, tpe, hp, Trials 
import matplotlib.pyplot as plt 


plt.rcParams['figure.dpi'] = 300 


plt.rcParams['font.sans-serif'] = ['SimHei']  

plt.rcParams['axes.unicode_minus'] = False 


parameter_space_adaboost = {
    'n_estimators': hp.choice('n_estimators', range(min, max)),  #min to max
    'learning_rate': hp.uniform('learning_rate', min, max),  #min to max
    'loss': hp.choice('loss', ['linear', 'square', 'exponential']),  
    'base_estimator': hp.choice('base_estimator', [DecisionTreeRegressor(max_depth=depth) for depth in range(1, 11)]),  
}

# obj 
def objective(params):
    model = AdaBoostRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        loss=params['loss'],
        base_estimator=params['base_estimator'],
        random_state=42,
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -np.mean(scores)  # 目标是最小化均方误差 


data = pd.read_excel('数据.xlsx')

data['X10'] = data['X10'].replace('                            ', np.nan)
data = data.dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
columns=X.columns.values

X = X.astype(np.float32).values 
y = y.astype(np.float32).values 

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
trials = Trials()
best_params = fmin(
    fn=objective,
    space=parameter_space_adaboost,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials 
)


n_estimators_best = best_params['n_estimators']
learning_rate_best = best_params['learning_rate']
loss_best = ['linear', 'square', 'exponential'][best_params['loss']]  
base_estimator_best = [DecisionTreeRegressor(max_depth=depth) for depth in range(1, 11)][best_params['base_estimator']]  

print('最优的 n_estimators:', n_estimators_best)
print('最优的 learning_rate:', learning_rate_best)
print('最优的 loss:', loss_best)
print('最优的 base_estimator:', base_estimator_best)


best_model = AdaBoostRegressor(
    n_estimators=n_estimators_best,
    learning_rate=learning_rate_best,
    loss=loss_best,
    base_estimator=base_estimator_best,
    random_state=42,
)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'优化后的 AdaBoost 回归均方误差 (MSE): {mse:.4f}')
print(f'优化后的 AdaBoost 回归 R²: {r2:.4f}')

import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  


losses = [t['result']['loss'] for t in trials.trials]
min_loss = min(losses)
min_loss_index = losses.index(min_loss)

plt.plot(range(len(losses)), losses, marker='o', linestyle='-', color='b', label='Loss (MSE)')
plt.scatter(min_loss_index, min_loss, marker='*', color='r', s=200, label='Optimal Loss',zorder=2)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Bayesian Optimization of AdaBoost Regression')
plt.legend()
plt.show()
