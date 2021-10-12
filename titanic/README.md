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
(1) 정규표현식
```
df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.') #Mr, Ms 등 꺼내기
```
