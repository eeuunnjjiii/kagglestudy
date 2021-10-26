# 안전운전자 예측

## 1. 시각화
1) lmplot (회귀선 + 플롯)
```python
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show();
```
![image](https://user-images.githubusercontent.com/75970111/138887864-78cc665f-aa0d-4b0b-b29c-948bf4b4c49d.png)

## 2. Feature selection
1) 분산 기반 선택 방법
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

2) 모델 기반 선택 방법
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
