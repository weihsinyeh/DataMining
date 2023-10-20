import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# High support, high confidence 0.2 0.2   'viridis'
# High support, low confidence 0.2 0.05   'plasma'
# Low support, high confidence 0.05 0.2   'inferno'
# Low support, low confidence 0.05 0.05   'cividis'
path = "D:\GitHub\HW1DM\hw1_example_2023\hw1_example_2023\outputs\ibm-2023-released-apriori-0.05-0.05.csv"
colormap = 'cividis'
rules = pd.read_csv(path)
file_name = path.split("\\")[-1]
file_name = file_name.split(".c")[0]
print(file_name)
# extract support and confidence from file name
support = file_name.split("-")[4]
confidence = file_name.split("-")[5]

# rules = read.csv("D:\GitHub\DataMining\hw1_example_2023\hw1_example_2023\outputs\ibm-2023-released-apriori-0.1-0.3.csv")

# Generate scatterplot using support and confidence
#plt.figure(figsize=(10,6))
#sns.scatterplot(x = "support", y = "confidence", data = rules)
#plt.margins(0.01,0.01)
# plt.show()

# df = pd.read_csv('your_association_rules.csv')

# 將antecedent, consequent, support, confidence, lift列設置為索引

rules.set_index(['antecedent', 'consequent', 'support', 'confidence', 'lift'])
df_cleaned = rules[::2].dropna()
print(df_cleaned)
df_cleaned = df_cleaned.drop(columns=['antecedent', 'consequent'])



# 绘制散点图，根据lift调整点的大小
size_multiplier = df_cleaned['lift'] ** 10
plt.scatter(df_cleaned['support'], df_cleaned['confidence'], c=df_cleaned['lift'], cmap=colormap, s=size_multiplier*5,alpha=0.8)
plt.xlabel('Support')
plt.ylabel('Confidence')
# file name add min_sup and min_conf
figure_name = file_name.split(".")[0] + "-" + support + "-" + confidence + ".png"
# add min_sup and min_conf on title for the figure
# smaller size of the title
plt.rcParams.update({'font.size': 8})
plt.title('Association Rules Scatterplot (min_support : ' + support + ',min_confidence : ' + confidence + ')')
plt.colorbar(label='Lift')  # 添加颜色条
plt.savefig(figure_name)
plt.show()
#在这个示例中，df.drop(columns=['antecedent', 'consequent']) 用于删除'antecedent'和'consequent'列。然后，你可以使用Seaborn的热图函数（heatmap）绘制热图，而不包括这两列的数据。annot=True参数用于


'''
# 使用熱圖呈現數據
plt.figure(figsize=(12, 8))
sns.heatmap(df_cleaned, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Association Rules Heatmap')
plt.show()'''