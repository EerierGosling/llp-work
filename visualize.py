import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results.csv')

for i, row in df.iterrows():
    plt.plot(df.columns[1:], row[1:].values, label=f'LR = {row["epoch"]}', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Epoch for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()