import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
fig, ax = plt.subplots()
GARFMDAfpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/GARFMDAfpr.txt")
GARFMDAtpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/GARFMDAtpr.txt")
GSAMDAfpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/GSAMDAfpr.txt")
GSAMDAtpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/GSAMDAtpr.txt")

LAGCNfpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/LAGCNtpr.txt")
LAGCNtpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/LAGCNfpr.txt")
LRLSHMDAfpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/LRLSHMDAfpr.txt")
LRLSHMDAtpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/LRLSHMDAtpr.txt")
MDASAEfpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/MDASAEfpr.txt")
MDASAEtpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/MDASAEtpr.txt")
SCSMDAfpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/SCSMDAfpr.txt")
SCSMDAtpr = np.loadtxt("E:/代码/GARFMDA/data/新建文件夹/SCSMDAtpr.txt")
GARFMDA_auc = auc(GARFMDAfpr, GARFMDAtpr)
ax.plot(GARFMDAfpr, GARFMDAtpr, label=f'GARFMDA (AUC={GARFMDA_auc:.4f})')

GSAMDA_auc = auc(GSAMDAfpr, GSAMDAtpr)
ax.plot(GSAMDAfpr, GSAMDAtpr, label=f'GSAMDA (AUC={GSAMDA_auc:.4f})')

LAGCN_auc = auc(LAGCNfpr, LAGCNtpr)
ax.plot(LAGCNfpr, LAGCNtpr, label=f'LAGCN (AUC={LAGCN_auc:.4f})')

LRLSHMDA_auc = auc(LRLSHMDAfpr, LRLSHMDAtpr)
ax.plot(LRLSHMDAfpr, LRLSHMDAtpr, label=f'LRLSHMDA (AUC={LRLSHMDA_auc:.4f})')

MDASAE_auc = auc(MDASAEfpr, MDASAEtpr)
ax.plot(MDASAEfpr, MDASAEtpr, label=f'MDASAE (AUC={MDASAE_auc:.4f})')

SCSMDA_auc = auc(SCSMDAfpr, SCSMDAtpr)
ax.plot(SCSMDAfpr, SCSMDAtpr, label=f'SCSMDA (AUC={SCSMDA_auc:.4f})')
ax.legend(loc='lower right')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.savefig('roc_curves.png')
plt.show()