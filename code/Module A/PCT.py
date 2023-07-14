import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import  pandas as pd
#Setting the font to Times New Roman
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

wind_data=pd.read_excel(r"...\Mali.xlsx")
wind_data=wind_data.iloc[:,3:]
wind_data_values=wind_data.values
cc_zscore=wind_data_values
data = pd.DataFrame(cc_zscore)
corr = data.corr(method='pearson')
class_label=["Mali Power","Humidity at 10m","Temperature at 10m","Air pressure at 10m","Wind speed at 10m","Wind speed at 50m","Wind speed at 70m","Taonan Power","Taobei Power","Chaganhaote Power","Xiangyang Power"]
corr.index=class_label
corr.columns=class_label
f=plt.figure(figsize=(5, 4))
heatmap=sns.heatmap(corr, annot=True,cmap="YlGnBu",annot_kws={"fontsize":6},cbar=False,linewidths=0.1,fmt='.2f')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=30, ha='right',size=6)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=20, ha='right',size=6)

# Setting title
ax.set_title('PCT')
plt.ylabel('True label')
plt.xlabel('Correlation between features')
plt.show()
plt.close()
f.savefig('mali.jpg', dpi=100, bbox_inches='tight')
