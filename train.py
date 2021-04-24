import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from model import BaselineModel

# Load data set
with open('data/train_set.csv', 'rb') as train_data:
    df = pd.read_csv(train_data, nrows=1000)

new_df_2 = []
for idx,row in df.iterrows():
    new_df_2.append({
        'mean_growth_PH':row.mean_growth_PH,
        'sequence':row.sequence,
        'PH_class':label_mapping[str(round(row.mean_growth_PH,1))]
    })
new_df_2 =pd.DataFrame(new_df_2)

X_train, X_test,y_train, y_test = train_test_split(new_df_2.sequence, new_df_2.PH_class, test_size=0.33, random_state=42)

tfidfvectorizer = TfidfVectorizer(analyzer='char',ngram_range=(3,3))
tfidf = tfidfvectorizer.fit_transform(X_train)

BaselineModel(model_file_path='src/model.pickle').fit(tfidf,y_train)
