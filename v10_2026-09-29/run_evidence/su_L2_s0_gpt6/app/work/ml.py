import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
A=np.load('/app/data/sample.npz');D=np.load('/app/data/dev.npz');X,y=A['X'],A['y'];V,z=D['X'],D['y'];
for model in [LogisticRegression(C=.01,max_iter=1000),LogisticRegression(C=.1,max_iter=1000),LogisticRegression(C=1,max_iter=1000),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=1.0,n_jobs=1),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=5,max_features=1.0,n_jobs=1),RandomForestClassifier(n_estimators=300,min_samples_leaf=3,max_features=1.0,n_jobs=1),HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=3)]:
 model.fit(X,y); print(model,np.mean(model.predict(V)==z),np.mean(model.predict(X)==y),flush=True)
