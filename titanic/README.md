# 타이타닉 생존률 예측

## 1. 시각화
### 1) 컬럼별 결측치 확인
```
for col in df_train.columns: 
  msg = 'column: {:>10}\t Percent of NaN value; {:.2}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
  ## {:>10}\t : 오른쪽 정렬, {:.2} : 소수점 둘째자리까지 출력
  print(msg)
```
참고 : Number Formatting(https://mkaz.blog/code/python-string-format-cookbook/)
<img width="930" alt="스크린샷 2021-10-12 오전 10 06 47" src="https://user-images.githubusercontent.com/75970111/136873801-a29b2ad1-9b11-48f4-b482-c2441a5f2dbc.png">

### 2) 결측치 - matrix
```
import missingno as msno
msno.matrix(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8, 0.5, 0.2) #color로 색상 지정
```
![image](https://user-images.githubusercontent.com/75970111/136874236-a038cbf1-db27-4e8c-987b-e9e24ea69e1a.png)

### 3) 결측치 - bar
```
import missingno as msno
msno.bar(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8, 0.5, 0.2))
```
![image](https://user-images.githubusercontent.com/75970111/136874283-bca59e61-2310-4946-9e90-979696af1f3e.png)

### 4) 파이차트
```
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True) #explode : 파이차트 조각 돌출 정도, autopct : 파이조각 전체 대비 백분율
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('') #y축 레이블 제거
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')
plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/136874926-26fafdd0-3572-4d93-8cba-8be9678ac94a.png)

### 5) 평균 꺾은선 그래프
```
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5) ## size:시각화 크기, aspect:시각화 비율
```
![image](https://user-images.githubusercontent.com/75970111/136875971-c47f0c6e-ad15-4b19-ba90-efbcf0c6bb75.png)

```
sns.factorplot('Sex', 'Survived', col='Pclass', data=df_train, satureation=.5, size=9, aspect=1) #satureation:채도의 비율
```
![image](https://user-images.githubusercontent.com/75970111/136876052-04135a5b-e3cf-490c-9451-ab38f735cfcd.png)

### 6) 분포 겹쳐 그리기
```
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_train[df_train['Survived']==1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived']==0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/136876434-ef900ddd-6043-4b0e-85d5-6f275ce7ced3.png)

### 7) violinplot
```
f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train, sclae='count', split=True, ax=ax[0]) 
# split : 하나로 합칠 것인지, 분리할 것인지
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot('Sex', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()
```
`split=True`
![image](https://user-images.githubusercontent.com/75970111/136898375-b4159d35-d735-42a3-bcfa-061a2e69dd90.png)
`split=False`
![image](https://user-images.githubusercontent.com/75970111/136898402-4f45e0d9-dc99-4a15-85e0-cff9d39f8502.png)

## 2. 결측치 대체
### 1) 정규표현식
```
df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.') #Mr, Ms 등 꺼내기
```

## 3. ML
### 1) SVM
- 참고 : https://hleecaster.com/ml-svm-concept/
- **결정경계(Decision Boundary)**, 즉 분류를 위한 기준 선을 정의하는 모델
- 결정경계는 데이터 군으로부터 최대한 멀리 떨어지는 것이 좋음
- 서포트 벡터(support vectors) : 결정경계와 가까이 있는 데이터 포인트
- 마진(Margin) : 결정경계와 서포트 벡터 사이의 거리 > Hard Margin 아웃라이어 허용X, 서포트벡터와 결정경계 사이 거리가 매우 좁음, 오버피팅 문제 발생 / Soft margin 서포트벡터와 결정경계 사이 거리가 멀어짐, 언더피팅 문제 발생 
- **최적의 결정경계는 마진을 최대화**
- 장점 : 서포트 벡터만 잘 고르면 되기 때문에 매우 빠름!
```
from sklearn import svm
model = svm.SVC()
model.fit(train_X, train_Y)
prediction1 = model.predict(test_X)
print('Accuracy of rbf SVM is ', metrics.accuracy_score(prediction1, test_Y))
```
하이퍼파라미터 조절
- `kernel='rbf'`(기본값), `kernel='linear'` 등 데이터 변환
- `C=1` 마진조절 (작을수록 소프트마진)
- `gamma=0.1` 결정경계를 얼마나 유연하게 그을 것인지. 낮추면 직선에 가깝고 높이면 구불구불한 결졍경계

### 2) K-Nearest Neighbours(KNN)
- 참고 : https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-6-K-%EC%B5%9C%EA%B7%BC%EC%A0%91%EC%9D%B4%EC%9B%83KNN
- k개의 데이터를 살펴본 뒤, 주변 데이터가 더 많이 포함된 범주로 분류
- 일반적으로 k는 홀수를 사용, 짝수일 경우 동점 발생 가능
- KNN은 Lazy Model이라 훈련이 필요없고, 바로 분류 > 빠른 장점
```
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_X, train_Y)
prediction5 = model.predict(test_X)
print('Accuracy of the KNN is ', metrics.accuracy_score(prediction5, test_Y))
```
하이퍼파라미터 조절
- `n_neighbors=6` 기본값 = 5

### 3) Gaussian Naive Bayes
- 참고 : https://jhryu1208.github.io/data/2020/11/14/naive_bayes/
- 선형 모델과 매우 유사하고 훈련속도가 더 빠르지만, 일반화 성능이 조금 뒤쳐짐
- scikit-learn naive bayes :`GaussianNB`(연속 데이터), `BernoulliNB`(이진 데이터), `MultinomialNB`(카운트데이터)
- Gaussian Naive Bayes는 클래스별로 각 특성의 표준편차와 평균 저장, 고차원 데이터에 사용
```
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_X, train_Y)
prediction6 = model.predict(test_X)
print('Accuracy of the NaiveBayes is ', metrics.accuracy_score(prediction6, test_Y))
```

## 4. Ensembling
### 1) Voting Classifier
- 가장 간단한 방법
- 모든 하위모델의 평균 예측 결과 기반
- 모든 하위모델이나 베이스모델은 모두 다른 타입
```
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)), 
                                                ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                            ], voting='soft').fit(train_X, train_Y)
print('The accuracy for ensembled model is:', ensemble_lin_rbf.score(test_X, test_Y))
cross = cross_val_score(ensemble_lin_rbf, X, Y, cv=10, scoring='accuracy')
print('The cross validated score is', cross.mean())
```
### 2) Bagging
- 작게 분할한 데이터로 비슷한 분류기를 만들어 모든 예측의 평균으로 결정
- voting과 달리 유사한 분류기를 사용
- 분산이 큰 모델에 잘 적용됨 > decision tree, Random Forests
```
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=0, n_estimators=100)
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print('The accuracy for bagged Decision Tree is', metrics.accuracy_score(prediction, test_Y))
result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
print('The cross validated score for bagged Decision Tree is', result.mean())
```

### 3) Boosting
- 약한 모델에 가중치를 적용해 강화하는 방식
```
# AdaBoost(Adaptive Boosting)
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.1)
result = cross_val_score(ada, X, Y, cv=10, scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())

# Stochastic Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier(n_estimators=500, random_state=0, learning_rate=0.1)
result = cross_val_score(grad, X, Y, cv=10, scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())

# XGBoost
import xgboost as xg
xgboost = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
result = cross_val_score(xgboost, X, Y, cv=10, scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())
```
