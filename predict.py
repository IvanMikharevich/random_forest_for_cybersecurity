from pathlib import Path
import pandas as pd
import os
import joblib
from train import pipline

workdir = Path(os.getcwd())
data_dir = workdir / 'data'
clf = joblib.load("my_random_forest.joblib")

X_test = pipline(data_dir / 'test.tsv', y=False)
predicted = clf.predict(X_test)

df = pd.DataFrame({'predicted': predicted})
df.to_csv(data_dir / 'prediction.txt', index=False)

# In a message with an explanation, the program will display
# the probability that the file is viral in a user-friendly form.

proba = clf.predict_proba(X_test)
explain = []
counter = 0
for prediction in predicted:
    if prediction == 1:
        val = round(proba[counter][1] * 100, 0)
        message = 'System is ' + str(val) + '% sure that exe file is virus'
    else:
        message = ' '
    explain.append(message)
    counter += 1

df = pd.DataFrame({'explain': explain})
df.to_csv(data_dir / 'explain.txt', index=False)
