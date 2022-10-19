from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def metrics(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    f1 = f1_score(y_test, pred, average='macro')
    print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy * 100, precision * 100, recall * 100))
    return accuracy, precision, recall, f1
