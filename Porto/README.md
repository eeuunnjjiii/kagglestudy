# 안전운전자 예측

## 1. 시각화
1) lmplot (회귀선 + 플롯)
```python
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show();
```
![image](https://user-images.githubusercontent.com/75970111/138887864-78cc665f-aa0d-4b0b-b29c-948bf4b4c49d.png)

## 2. 데이터 
### 1) 언더 샘플링과 오버 샘플링
- 분류문제에서 타겟 데이터가 불균형할 경우, 언더샘플링이나 오버샘플링 진행
- 언더 샘플링 : 높은 비율을 차지하는 클래스의 데이터 수를 줄임 > 학습에 사용되는 전체 데이터 수를 급격하게 감소시켜 오히려 성능이 떨어질 수 있음

```python
desired_apriori = 0.10

# Get the indices per target value
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# 언더샘플링 비율과 타겟=0인 행의 수
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# target=0인 행 랜덤 추출
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# 남은 인덱스 리스트
idx_list = list(undersampled_idx) + list(idx_1)

# 언더샘플링 데이터 프레임 반환
train = train.loc[idx_list].reset_index(drop=True)
```

- 오버 샘플링 : 낮은 비율 클래스의 데이터 수를 늘림 > SMOTE(Synthetic Minority Over Sampling Technique), 낮은 비율 클래스 데이터들의 최근접 이웃 이용
> 재현율, 정밀도에 주의!! 양성 데이터가 낮은 비율 데이터라고 예를 들면, 오버 샘플링시 양성 예측 비율이 높아져 정밀도 감소 & 재현율 증가

> 꼭 훈련 데이터에만 적용할 것!
- 참고 : https://hwi-doc.tistory.com/entry/%EC%96%B8%EB%8D%94-%EC%83%98%ED%94%8C%EB%A7%81Undersampling%EA%B3%BC-%EC%98%A4%EB%B2%84-%EC%83%98%ED%94%8C%EB%A7%81Oversampling
```python
from imlearn.over_sampling import SMOTE
smote = SMOTE(random_state=11)
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
```

### 2) 다차원에서 1차원 array 변환
`np.ravel()` : 다차원 배열을 1차원 배열로 변환하는 함수 <-> `np.reshape()`

참고 : https://rfriend.tistory.com/349
- `np.ravel(x, order='C')` : 기본값, 행 > 열 순서로 인덱싱하여 평평하게 배열
- `np.ravel(x, order='F')` : 열 > 행 순서로 인덱싱하여 배열
- `np.ravel(x, order='K')` : 메모리에서 발생하는 순서대로 인덱싱하여 배열

### 3) 다항변환
- 비선형 데이터를 추가하는 방법
- 데이터 형태가 비선형일 때, 특성이 추가된 비선형 데이터로 모델에 복잡성을 주고 선형 회귀 모델로 훈련시키는 방법
```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_poly = poly_features.fit_transform(x)
```
- `interaction_only = False` : 기본값 (X1, X2) > (1, X1, X2, X1^2, X1X2, X2^2)
- `interaction_only = TRUE` : (X1, X2) > (1, X1, X2, X1X2)
- `include_bias` : 상수항 생성 여부
- 참고 : https://mambo-coding-note.tistory.com/388

## 3. Feature selection
### 1) 분산 기반 선택 방법
- 예측 모형에서 중요한 특징 데이터란, 종속데이터와 상관관계가 크고 예측에 도움되는 데이터
- 하지만 특징데이터 값 자체가 표본에 따라 그다지 변하지 않는다면 종속 데이터 예측에도 도움이 되지 않을 가능성이 높음
- 따라서, 표본변화에 따른 데이터 값의 변화 즉, 분산이 기준치보다 낮은 특징데이터는 사용하지 않는 방법
- 분산에 의한 선택이 반드시 상관관계와 일치한다는 보장이 없으므로 신중하게 사용할 것
- 참고 : https://datascienceschool.net/03%20machine%20learning/14.03%20%ED%8A%B9%EC%A7%95%20%EC%84%A0%ED%83%9D.html
- 공식문서 : https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=.01) # threshold는 float으로 입력, default=0
selector.fit_transform(train, axis=1)
selector.get_support() # 선택된 피쳐 확인
```

### 2) 모델 기반 선택 방법
- 지도학습 모델을 사용해 특성의 중요도로 선택
- 특성 선택에 사용하는 지도 학습 모델은 최종적으로 사용할 지도 학습 모델과 같을 필요는 없음
- 특성 선택을 위한 모델은 각 특성의 중요도를 측정하여 순서를 매길 수 있어야 함

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

sfm = SelectFromModel(rf, threshold='median', prefit=True)
sfm.transform(X_train)
```
