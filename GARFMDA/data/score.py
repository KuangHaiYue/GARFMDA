import numpy as np

score=np.loadtxt("E:/代码/GARFMDA/data/fpr_tpr/scores0.txt")
row = score[:,108]
sorted_row = sorted(row, reverse=True)
output_file = "./fpr_tpr/s6.txt"
with open(output_file, 'w') as file:
    for num in sorted_row:
        file.write(str(num) + '\n')