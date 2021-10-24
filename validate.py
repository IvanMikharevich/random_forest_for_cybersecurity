from pathlib import Path
import os
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from train import pipline


workdir = Path(os.getcwd())
data_dir = workdir / 'data'

X_val, Y_val = pipline(data_dir / 'val.tsv')

clf = joblib.load("my_random_forest.joblib")
predicted = clf.predict(X_val)


# The x function saves the model metrics to validation.txt

def save_metrics(predicted, Y_val, data_dir):
    ac = accuracy_score(predicted, Y_val)

    TN, FP, FN, TP = confusion_matrix(Y_val, predicted).ravel()
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    metrics = ['True positive: ' + str(TP),
               'False positive: ' + str(FP),
               'False negative: ' + str(FN),
               'True negative: ' + str(TN),
               'Accuracy: ' + str(round(ac, 4)),
               'Precision: ' + str(round(precision, 4)),
               'Recall: ' + str(round(recall, 4)),
               'F1: ' + str(round(F1, 4))]

    with open(data_dir / 'validation.txt', 'w') as filehandle:
        for listitem in metrics:
            filehandle.write('%s\n' % listitem)


save_metrics(predicted, Y_val, data_dir)





