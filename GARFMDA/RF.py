import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def train11(x_train,y_train,x_test,y_test):
  x_train_2D = np.array(x_train).reshape(np.array(x_train).shape[0], np.array(x_train).shape[1] * np.array(x_train).shape[2])
  x_test_2D =np.array(x_test).reshape(np.array(x_test).shape[0], np.array(x_test).shape[1] * np.array(x_test).shape[2])
  rf = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=0)
  rf.fit(x_train_2D, y_train)
  importances = rf.feature_importances_
  inl = 0.00104
  x_train = x_train_2D[:,importances > inl]
  x_test = x_test_2D[:,importances > inl]
  rf2 = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=0)
  rf2.fit(x_train,y_train)

  y_pred_proba = rf.predict_proba(x_test)
  y_pred_proba = y_pred_proba[:, 1]
  return  y_pred_proba,y_test
