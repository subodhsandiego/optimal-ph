import argparse
import pandas as pd
from model import BaselineModel
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', default='input.csv')
args = parser.parse_args()

# Config
output_file_path = 'predictions.csv'

# Load input.csv
with open(args.input_csv) as input_csv:
    df = pd.read_csv(input_csv)

label_mapping = {}
ph = 0.0
ctr = 0
while ph <= 14.0:
    label_mapping[str(round(ph,1))] = ctr
    ctr += 1
    ph += 0.1
    
label_mapping_inv = {}
for key in label_mapping:
    label_mapping_inv[label_mapping[key]] = key

tfidfvectorizer = pickle.load(open('src/tfidf.pickle','rb'))
tfidf = tfidfvectorizer.transform(df.sequence)

# Run predictions
y_predictions = BaselineModel(model_file_path='src/model.pickle').predict(tfidf)
y_pred_classifier_float_2 = []

for i in range(len(df)):
    y_pred_classifier_float_2.append(float(label_mapping_inv[y_predictions[i]]))

# Save predictions to file
df_predictions = pd.DataFrame({'prediction': y_pred_classifier_float_2})
df_predictions.to_csv(output_file_path, index=False)

print(f'{len(y_predictions)} predictions saved to a csv file')
