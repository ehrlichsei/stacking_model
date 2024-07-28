from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def print_dataset_info(data):
    # 查看数据集的键
    print(data.keys())

    # 特征数据的大小和形式
    print("Feature data shape:", data.data.shape)
    print("Feature names:", data.feature_names)

    # 目标数据的大小和形式
    print("Target data shape:", data.target.shape)
    print("Target names:", data.target_names)

    # 查看前几个数据点
    print("First 5 feature data:\n", data.data[:5])
    print("First 5 target data:", data.target[:5])

# 加载数据
data = load_iris()

print_dataset_info(data)      



X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 基础模型
model1 = make_pipeline(StandardScaler(), LogisticRegression())
model2 = make_pipeline(StandardScaler(), RandomForestClassifier())
model3 = make_pipeline(StandardScaler(), SVC(probability=True))

# 训练基础模型
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# 获取基础模型预测
pred1 = model1.predict_proba(X_test)
pred2 = model2.predict_proba(X_test)
pred3 = model3.predict_proba(X_test)

# 组合预测
stacked_predictions = np.column_stack((pred1, pred2, pred3))

# 元模型
meta_model = LogisticRegression()
meta_model.fit(stacked_predictions, y_test)

# 评估
final_predictions = meta_model.predict(stacked_predictions)
print(f"Stacking Model Accuracy: {accuracy_score(y_test, final_predictions)}")