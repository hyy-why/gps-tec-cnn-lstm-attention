import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('default')
import matplotlib.colorbar as mpl_colorbar
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

import shap
shap.initjs()

# 导入数据集
# df = pd.read_csv(r"E:\code\workspace\demo1\pythonProject1\cnn-lstm-attention\dataset\data_TEC - 全部特征_数据未处理.csv")
df = pd.read_csv(r"C:\Users\HP\Desktop\测试数据.csv")

# 提取特征和目标变量
y = df["9"]
# X = df[["f10.7", "Scalar B, nT", "Lat. Angle of B (GSE)", " Long. Angle of B (GSE)", "BX(GSE, GSM)", "BY(GSE)", "BZ(GSE)", "BY(GSM)", "BZ", "Kp", "Dst", "ap"]]
# X = df.drop(columns=['9', df.columns[5]]).values
X = df[["7", "10", "11", "12", "13", "14", "15", "17", "18", "19", "20", "22", "23", "24", "25"]]
print(X)

# 训练XGBoost模型
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X, y)

# 计算SHAP值
explainer = shap.Explainer(model)
shap_values = explainer(X)

# ===================== 全局字体设置 =====================
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams["font.size"] = 16
matplotlib.rcParams["axes.labelsize"] = 18
matplotlib.rcParams["xtick.labelsize"] = 18  # X轴刻度字体放大到24
matplotlib.rcParams["ytick.labelsize"] = 18

# 创建更大的画布，给刻度留出足够空间
plt.figure(figsize=(22, 18))

# 绘制SHAP蜂窝图
ax = shap.plots.beeswarm(shap_values, show=False, color_bar=True)

# ===================== 核心：强制显示完整刻度（-10、0、10、20） =====================
# 1. 获取当前X轴的数值范围，手动设置刻度位置
x_min, x_max = ax.get_xlim()
# 强制指定要显示的刻度：-10、0、10、20（可根据你的实际数据调整）
custom_xticks = [-10, 0, 10, 20]
# 只保留在X轴范围内的刻度（避免超出范围报错）
custom_xticks = [tick for tick in custom_xticks if x_min <= tick <= x_max]
# 强制设置X轴刻度
ax.set_xticks(custom_xticks)
# 强制显示刻度标签（避免Matplotlib自动隐藏）
ax.set_xticklabels(custom_xticks, fontsize=24, fontfamily='Times New Roman')

# 2. 强制放大X轴刻度数字（双重保障）
ax.tick_params(axis='x', labelsize=18, pad=10)  # pad增加间距，避免重叠

# 3. 放大Y轴特征名称字体
ax.tick_params(axis='y', labelsize=18, pad=8)
for label in ax.get_yticklabels():
    label.set_fontsize(24)
    label.set_fontfamily('Times New Roman')

# 4. 放大X轴标题字体
ax.set_xlabel(ax.get_xlabel(), fontsize=18, fontfamily='Times New Roman')

# 5. 放大右侧Feature Value色条刻度字体
cbar = None
for obj in plt.gcf().get_children():
    if isinstance(obj, mpl_colorbar.Colorbar):
        cbar = obj
        break
if cbar is not None:
    cbar.ax.tick_params(labelsize=45, pad=8)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(45)
        label.set_fontfamily('Times New Roman')

# ===================== 布局调整（避免刻度被截断） =====================
# 手动调整边距，重点增大底部边距（给X轴刻度留空间）
plt.subplots_adjust(left=0.4, right=0.95, bottom=0.25, top=0.95)
# 关闭自动tight_layout，防止重置刻度/字体
# plt.tight_layout()

# 保存图片
plt.savefig(
    r'E:\code\workspace\demo1\pythonProject1\cnn-lstm-attention\result_picture\蜂窝.png',
    bbox_inches='tight',
    dpi=600,
    facecolor='white'
)
# plt.close()
plt.show()