import matplotlib.pyplot as plt
import numpy as np

# 加载保存的指标
with open('metrics.txt', 'r') as file:
    lines = file.readlines()

fpr = np.fromstring(lines[0].split(': ')[1], sep=' ')
tpr = np.fromstring(lines[1].split(': ')[1], sep=' ')
roc_auc = float(lines[2].split(': ')[1])
precision = np.fromstring(lines[3].split(': ')[1], sep=' ')
recall = np.fromstring(lines[4].split(': ')[1], sep=' ')
pr_auc = float(lines[5].split(': ')[1])

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-1.0, 1.5])
plt.ylim([-1.0, 1.5])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
