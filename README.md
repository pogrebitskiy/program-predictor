# Deployment:
## https://program-predictor.herokuapp.com


## Data File:
### https://drive.google.com/file/d/1V6YqkS_II9xgfEOgkaPXnFrL9LPil9Nx/view?usp=sharing



```python
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
warnings.filterwarnings("ignore")

DATA_PATH = 'data.csv'
```


```python
# Read in data files and do some minor cleaning
df = pd.read_csv(DATA_PATH, index_col=0)
df = df.dropna()
df = df.drop_duplicates()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>language</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/*\n * Copyright (c) 1995-2001 Silicon Graphic...</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/* *******************************************...</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/* Interprocedural constant propagation\n   Co...</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/*\n * Copyright (c) 2004 Topspin Corporation....</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;?php\n\nnamespace Ojs\JournalBundle\Listeners...</td>
      <td>PHP</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Regular Expression that will tokenize our code snippets
regex = r"[A-Za-z_]\w*|[ \t\(\),;\{\}\[\]`\"']|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+"
```


```python
# Separate features from labels
X = df.content
y = df.language

# Assign numerical values to categorical labels
le = LabelEncoder()
y = le.fit_transform(y)

# vectorize contents of each code snippet for ML use
vectorizer = TfidfVectorizer(token_pattern=regex, max_features=3000)
X = vectorizer.fit_transform(X).toarray()

# Split features and labels into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
```


```python
# fit multinomial naive bayes' model using train sets
model = MultinomialNB()
model.fit(X_train, y_train)
```




    MultinomialNB()




```python
# run predict on test set to analyze predictive performance
y_pred = model.predict(X_test)
```


```python
print(f'Test Accuracy: {model.score(X_test, y_test):.3f}')
print(f'Training Accuracy: {model.score(X_train, y_train):.3f}')
```

    Test Accuracy: 0.816
    Training Accuracy: 0.823



```python
# use gridseach to final optimal alpha value in multinomialNB to increase predictive performance
params = {'alpha': [0.00000001,0.0000001,0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, ],}

multinomial_nb_grid = GridSearchCV(MultinomialNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5);
multinomial_nb_grid.fit(X,y);

print(f'Best Accuracy Through Grid Search : {multinomial_nb_grid.best_score_:.3f}')
print(f'Best Parameters : {multinomial_nb_grid.best_params_}')
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    Best Accuracy Through Grid Search : 0.867
    Best Parameters : {'alpha': 1e-06}



```python
best_alpha = multinomial_nb_grid.best_params_['alpha']
```


```python
# re-train model using optimal parameters
model_v2 = MultinomialNB(alpha = best_alpha);
model_v2.fit(X_train, y_train);
```


```python
print(f'Test Accuracy: {model_v2.score(X_test, y_test):.3f}')
print(f'Training Accuracy: {model_v2.score(X_train, y_train):.3f}')
```

    Test Accuracy: 0.865
    Training Accuracy: 0.885



```python
y_pred = model_v2.predict(X_test)
```


```python
print(f'Precision: {precision_score(y_test, y_pred, average = "macro"):.3f}')
print(f'Recall: {recall_score(y_test, y_pred, average = "macro"):.3f}')
```

    Precision: 0.900
    Recall: 0.818



```python
def predict(text):
    '''
    Function: predict
    Parameters: text, a string
    Returns: a string representing the predicted language
    
    This function vectorizes the input string, runs it through the classification model,
    then return the language that it predicted
    
    '''
    x = vectorizer.transform([text]).toarray()
    language = model.predict(x)
    language = le.inverse_transform(language)
    return language[0]
```


```python
# check to see it's working
code = r"const { Client } = require('discord.js'); client.on(msg => {})"
predict(code)
```




    'JavaScript'




```python
# builds dictionary that keeps track of how many times a prediction was made wrong as the value and 
# the correct language as the key

wrong = {}
for test, answer in zip(X_test, y_test):
    guess = le.inverse_transform(model_v2.predict(test.reshape(1,-1)))[0]
    correct_answer = le.inverse_transform([answer])[0]
    if guess != correct_answer:
        if correct_answer not in wrong.keys():
            wrong[correct_answer] = 1
        else:
            wrong[correct_answer] += 1
    
```


```python
# Plot to see which languages are getting predicted incorrectly 

wrong_full = {k: v for k, v in (sorted(wrong.items(), key = lambda x: x[1], reverse = True))}
names = list(wrong_full.keys())
values = list(wrong_full.values())
total_vals = sum(values)
rel_values = [i/total_vals for i in values]
plt.figure(figsize = (12,8));
sns.barplot(rel_values[:10], names[:10], orient = 'h', palette = 'icefire')
plt.ylabel('Programming Language')
plt.xlabel('Relative Frequency')
plt.title('Model Errors Across Programming Languages') 
plt.savefig('error.png', bbox_inches='tight')
```


    
![png](README_files/README_19_0.png)
    

