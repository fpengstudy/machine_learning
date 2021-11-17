import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

##将数据进行数字划分
data = pd.read_csv('C:/Users/616-9/Downloads/breast-cancer.data',header=None)
##将数据规整确立
data.columns=["class","age","menopause","tumor-szie","invnodes","nodecaps","degmalig","breastleft","breastquad","irradiat"]
##将所有的字符串数据进行数字化处理，否则sklearn包无法识别
for i in data.columns:
    data[i] = pd.factorize(data[i])[0].astype(np.uint16)
##将dataframe数据矩阵化处理
npdata = data.values
print("数字化后数据","\n",npdata)

##分割样本和标签，第一列为标签，后续为数据
x,y = np.split(npdata,(1,),axis=1) ##x为标签，y为数据
##分别划分样本特征集，样本类别，随机数编号，样本占比（可改），训练占比（可改）
train_data,test_data,train_label,test_label = train_test_split(y,x, random_state=1, train_size=0.6,test_size=0.4)

##训练SVM分类器，kernel选择是线性核或是高斯核
classifier = svm.SVC(C=2,kernel='rbf',gamma=5,decision_function_shape='ovr')
#classifier = svm.SVC(C=1,kernel='linear',decision_function_shape='ovr')
classifier.fit(train_data,train_label.ravel())
#print(classifier.fit(train_data,train_label.ravel()))
print("SVM训练集",classifier.score(train_data,train_label))
print("SVM测试集",classifier.score(test_data,test_label))


test_predict_label = classifier.decision_function(test_data)
fpr, tpr, threshold = roc_curve(test_label, test_predict_label)  ###计算真正率和假正率
#print(fpr)#print(tpr)#print(threshold)
roc_auc = auc(fpr, tpr)  ###计算auc的值，auc就是曲线包围的面积，越大越好
##开始绘制ROC图片
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
plt.savefig('C:/Users/616-9/Desktop/svm_roc.png')



####决策树模型构建和预测
hpv_feature_E= "age","menopause","tumor-szie","invnodes","nodecaps","degmalig","breastleft","breastquad","irradiat"
hpv_class = "yes","no"
##通过对x和y的训练和样本进行再次比例调整
x_train, x_test, y_train, y_test = train_test_split(y, x, test_size=0.2)
##使用信息熵作为划分标准，调用tree模板
classifier1 = tree.DecisionTreeClassifier(criterion='entropy')
clafit = classifier1.fit(x_train, y_train)
print(clafit)
# 把决策树结构写入文件
dot_data = tree.export_graphviz(classifier1, out_file=None, feature_names=hpv_feature_E, class_names=hpv_class,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_png('C:/Users/616-9/Desktop/tree.png')
# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
print(classifier1.feature_importances_)
# 使用训练数据预测，预测结果完全正确
answer1 = classifier1.predict(x_train)
y_train = y_train.reshape(-1)
#print(answer)#print(y_train)
print(np.mean(answer1 == y_train))
# 对测试数据进行预测
answer2 = classifier1.predict(x_test)
y_test = y_test.reshape(-1)
#print(answer)#print(y_test)
print(np.mean(answer2 == y_test))

