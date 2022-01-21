#%%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

operator_size = 3
alpha = 0.5
gama = 0.3
run = 1
w = 25
pmin=0.5
lmode = 0
reward = 'extreme'
pName='sukp 100_100_0.10_0.75.txt'
# pName='1000'
o="CLRL"
learning = 0

sns.set_theme(palette="tab10")
# # Visualize Cluster Centers
# file_path = f"results/cluster_history-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-{learning}-{pName}.csv"
# columns = ["op_no","iteration","run","f0","f1","f2","f3","f4","f5"]  
# df = pd.read_csv(file_path, header=None,names=columns)
# df = df[df["run"]==run]
# fig, axes = plt.subplots(3,2)
# sns.scatterplot(data=df, x="iteration", y="f0", hue="op_no", ax=axes[0][0], legend = True, palette="tab10")
# sns.scatterplot(data=df, x="iteration", y="f1", hue="op_no", ax=axes[0][1] ,legend = False, palette="tab10")
# sns.scatterplot(data=df, x="iteration", y="f2", hue="op_no", ax=axes[1][0], legend = False, palette="tab10")
# sns.scatterplot(data=df, x="iteration", y="f3", hue="op_no", ax=axes[1][1] ,legend = False, palette="tab10")
# sns.scatterplot(data=df, x="iteration", y="f4", hue="op_no", ax=axes[2][0], legend = False, palette="tab10")
# sns.scatterplot(data=df, x="iteration", y="f5", hue="op_no", ax=axes[2][1], legend = False, palette="tab10")
# plt.show()

#Visualize Operator Informations
file_path = f"results/operator_informations-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-{learning}-{pName}.csv"
columns = ["op_no","run","iteration","credits","rewards","usages","success"]  
df = pd.read_csv(file_path, header=None,names=columns)
df = df[df["run"]==run]
fig, axes = plt.subplots(2,2)
sns.lineplot(data=df, x="iteration", y="credits", hue="op_no", ax=axes[0][0], legend = True, palette="tab10")
sns.lineplot(data=df, x="iteration", y="rewards", hue="op_no", ax=axes[0][1] ,legend = False, palette="tab10")
sns.lineplot(data=df, x="iteration", y="usages", hue="op_no", ax=axes[1][0], legend = False, palette="tab10")
sns.lineplot(data=df, x="iteration", y="success", hue="op_no", ax=axes[1][1] ,legend = False, palette="tab10")
plt.show()