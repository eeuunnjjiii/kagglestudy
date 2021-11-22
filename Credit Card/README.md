# 신용카드 예측

# 전처리
## 특정 컬럼 데이터만 스케일링
```python
from sklearn.preprocessing import StandardScaler

data['new'] = StandardScaler().fit_transform(data['old'].values.reshape(-1, 1))
```

## 언더샘플링
```python
# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select 'x' number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_noraml_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices, :]

X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

# Showing ratio
print('Percentage of normal transactions: ', len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print('Percentage of fraud transactions: ', len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print('Total number of transacttions in resampled data: ', len(under_sample_data))
```

## Threshold 바꿔가며 데이터 확인

```python
lr = LogisticRegression(C = 1.0)
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10, 10))

j = 1
for i in thresholds :
  y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

  plt.subplot(3, 3, j)
  j += 1

  cnf_matrix = confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
  np.set_printoptions(precision=2)

  print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

  # Plot non-normalized confusion matrix
  class_names = [0,1]
  plot_confusion_matrix(cnf_matrix
                        , classes=class_names
                        , title='Threshold >= %s'%i) lr = LogisticRegression(C = 1.0)
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10, 10))

j = 1
for i in thresholds :
  y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

  plt.subplot(3, 3, j)
  j += 1

  cnf_matrix = confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
  np.set_printoptions(precision=2)

  print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

  # Plot non-normalized confusion matrix
  class_names = [0,1]
  plot_confusion_matrix(cnf_matrix
                        , classes=class_names
                        , title='Threshold >= %s'%i) 
```
![image](https://user-images.githubusercontent.com/75970111/142867818-da268d88-a5c0-4d14-a824-061921aec09c.png)

```python
from itertools import cycle

lr = LogisticRegression(C=1.0)
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black'])

plt.figure(figsize=(5, 5))

j = 1
for i, color in zip(thresholds, colors):
  y_test_predictions_prob = y_pred_undersample_proba[:, 1] > i

  precision, recall, thresholds = precision_recall_curve(y_test_undersample, y_test_predictions_prob)

  plt.plot(recall, precision, color=color, label='Threshold: %s'%i)

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall example')
  plt.legend(loc="lower left")
```
![image](https://user-images.githubusercontent.com/75970111/142867869-0c1bb961-3826-4675-991b-0a5114277a12.png)
