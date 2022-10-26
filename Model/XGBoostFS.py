import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy import interp
import time
import preprocessing
import metrics

df = pd.read_csv('../data/newGA_parkinson_100_100_100.csv', delimiter='\t', header=None)
df = df.drop(74, axis=1)
data = df.set_index(0).transpose()

test_acc = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
time_av = []
precision_av = []
f1_av = []
recall_av = []
start_time_all = time.perf_counter()
for i in range(5):
    # %%time
    print("{}st fold".format(i))
    start_time = time.perf_counter()

    X_train, X_test, y_train, y_test = preprocessing.preprocess_inputscv_FS(data, i)

    X_train = X_train.apply(pd.to_numeric)
    y_train = y_train.apply(pd.to_numeric)
    X_test = X_test.apply(pd.to_numeric)
    y_test = y_test.apply(pd.to_numeric)

    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    accuracy, precision, recall, f1 = metrics.metrics(y_test, xgb_pred)

    test_acc.append(accuracy)
    precision_av.append(precision)
    f1_av.append(f1)
    recall_av.append(recall)

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
