import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy import interp
import time

from Model import preprocessing
from Model import metrics

data = pd.read_csv('../IGdata/IG_50.csv', delimiter=',', header=None)
cls = data.shape[1]-1
# df = df.drop(74, axis=1)
# data = df.set_index(0).transpose()

test_acc = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
time_av = []
precision_av = []
f1_av = []
recall_av = []
actual_classes = np.empty([0], dtype=int)
predicted_classes = np.empty([0], dtype=int)
start_time_all = time.perf_counter()
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
df = data.transpose()

y = df.loc[50]
x = df.iloc[:-1]
X = x.transpose()
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    # %%time
    print("{}st fold".format(i))
    start_time = time.perf_counter()

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train = X_train.apply(pd.to_numeric)
    y_train = y_train.apply(pd.to_numeric)
    X_test = X_test.apply(pd.to_numeric)
    y_test = y_test.apply(pd.to_numeric)

    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=i)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    accuracy, precision, recall, f1 = metrics.metrics(y_test, xgb_pred)

    test_acc.append(accuracy)
    precision_av.append(precision)
    f1_av.append(f1)
    recall_av.append(recall)

    predicted_classes = np.append(predicted_classes, y_test)
    actual_classes = np.append(actual_classes, xgb_pred)

    fpr, tpr, t = roc_curve(y_test, xgb_pred)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i = i + 1

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("CPU Time = ", elapsed_time)
    time_av.append(elapsed_time)

end_time_all = time.perf_counter()
elapsed_time_all = end_time_all - start_time_all
print("all CPU Time = ", elapsed_time_all)

print("cv")
print("Test acc = %.2f (+/- %.2f%%)" % (np.mean(test_acc) * 100, np.std(test_acc) * 100))
print("Test precision = %.2f (+/- %.2f%%)" % (np.mean(precision_av) * 100, np.std(precision_av) * 100))
print("Test recall_av = %.2f (+/- %.2f%%)" % (np.mean(recall_av) * 100, np.std(recall_av) * 100))
print("Test f1 = %.2f (+/- %.2f%%)" % (np.mean(f1_av) * 100, np.std(f1_av) * 100))
print("all time = ", np.sum(time_av))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

print("all acc")
print(test_acc)

print("all precision")
print(precision_av)

print("all f1")
print(f1_av)

print("all recall")
print(recall_av)

def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=[0, 1])
    print(matrix)
    plt.figure(figsize=(12, 6))
    sn.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted');
    plt.ylabel('Actual');
    plt.title('Confusion Matrix')

    plt.show()
print(actual_classes)
print(predicted_classes)
plot_confusion_matrix(actual_classes, predicted_classes, ["Control", "PD"])