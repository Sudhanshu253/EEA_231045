import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open('/Users/sudhanshusaroj/myvenv/data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

print("Number of samples:", len(data_dict['data']))
print("Sample lengths:", [len(x) if hasattr(x, '__len__') else None for x in data_dict['data']])

try:
    max_len = max(len(x) for x in data_dict['data'])
    data = np.array([np.pad(x, (0, max_len - len(x)), mode='constant') for x in data_dict['data']])
except Exception as e:
    print("Error in standardizing data:", e)
    raise

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% of samples were classified correctly!")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
