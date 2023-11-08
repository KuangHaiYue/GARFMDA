import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

fpr=np.loadtxt("fpr5.txt")
tpr=np.loadtxt("tpr5.txt")
auc_val=auc(fpr,tpr)
print(auc_val)
plt.figure()
plt.plot(fpr,tpr)
plt.show()