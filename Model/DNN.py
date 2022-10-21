import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

from Model import preprocessing
from Model.metrics import metrics


#WEKA로 SVM, Decision Tree


use_mps = torch.backends.mps.is_available()
DEVICE = torch.device('mps' if use_mps else 'cpu')
print(DEVICE)

df = pd.read_csv('../data/GSE68719_mlpd_PCG_DESeq2_norm_counts.csv')
data = df.drop(['EnsemblID', 'symbol'], axis=1)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(17580, 128)
        self.fc12 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc11 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = x.float()
        x = self.dropout(F.relu(self.fc1(x.view(-1, 17580))))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc6(x))
        x = F.sigmoid(self.fc7(x))

        return x


def train_model(X_train, y_train, model):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    epoch = 250

    for step in range(epoch):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred.squeeze().to(torch.float64), y_train)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(step, loss.item())



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

    X_train, X_test, y_train, y_test = preprocessing.preprocess_inputscv(data, 15)

    X_train = torch.tensor(X_train.values)
    X_test = torch.tensor(X_test.values)
    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)

    model = Net()
    train_model(X_train, y_train, model)
    with torch.no_grad():
        hypothesis = model(X_test)
        predicted = (hypothesis > 0.5).float()
        accuracy, precision, recall, f1 = metrics(y_test, predicted)
    test_acc.append(accuracy)
    precision_av.append(precision)
    f1_av.append(f1)
    recall_av.append(recall)
    fpr, tpr, t = roc_curve(y_test, hypothesis)
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
