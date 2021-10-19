# 설문조사

## 1. 시각화
### 1) 트리맵 
```python
!pip install squarify
import squarify

tree = response['Country'].value_counts().to_frame()
squarify.plot(sizes=tree['Country'].values, label=tree.index, color=sns.color_palette('RdYlGn_r', 52))
plt.rcParams.update({'font.size':20})
fig = plt.gcf()
fig.set_size_inches(40, 15)
plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/137679149-3acef1b8-8029-4d2e-97af-aee9fcd5dba6.png)

### 2) 그래프 속 수치 표시
```python
sal_job = salary.groupby('CurrentJobTitleSelect')['Salary'].median().to_frame().sort_values(by='Salary', ascending=False)
ax = sns.barplot(sal_job.Salary, sal_job.index, palette=sns.color_palette('inferno', 20))
plt.title('Compensation By Job Title', size=15)
for i, v in enumerate(sal_job.Salary):
  ax.text(.5, i, v, fontsize=10, color='white', weight='bold')
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/137679243-98a06ff0-e1ce-4999-9b16-916d26cba8ba.png)

### 3) Matploblib 파이차트
```python
my_circle = plt.Circle((x, y), r, color='white') ##x,y 원의 중심, r 반지름 (default=5)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()
```

### 4) 지도 차트
```python
import plotly.offline as py
py.init_notebook_mode(connected=True)
```

### 5) 벤다이어그램
```python
from matplotlib_venn import venn2
venn2(subsets = (python.shape[0], R.shape[0], both.shape[0]), set_labels=('Python Users', 'R Users'))
plt.title('Venn Diagram for Users')
plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/137815523-1c29a759-f82b-49ad-9c09-8d7836e66889.png)

### 6) 컬럼 별 반복 시각화
```python
import itertools
plt.subplots(figsize=(22, 10))
time_spent = ['TimeFindingInsights','TimeVisualizing','TimeGatheringData','TimeModelBuilding']
length = len(time_spent)
for i, j in itertools.zip_longest(time_spent, range(length)): ## 루프돌면서 자료 묶음
  plt.subplot((length/2), 2, j+1) ## nrows, ncols, index
  plt.subplots_adjust(wspace=0.2, hspace=0.5)
  scientist[i].hist(bins=10, edgecolor='black')
  plt.axvline(scientist[i].mean(), linestyle='dashed', color='r')
  plt.title(i, size=20)
  plt.xlabel('% TIme')

plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/137828954-f08d1de4-913b-47f1-a7af-5e1605739d3e.png)

### Seaborn Palette
https://hleecaster.com/python-seaborn-color/

## 2. 기타
### 1) 특성 하나의 데이터 확인 시 리스트 > 시리즈 변경
```python
plt.subplots(figsize=(10, 10))
hard = response['HardwarePersonalProjectsSelect'].str.split(',')
hardware = []
for i in hard.dropna():
  hardware.extend(i)
pd.Series(hardware).value_counts().sort_values(ascending=True).plot.barh(width=0.9, color=sns.color_palette('inferno', 10))
plt.title('Machines Used')
plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/137733222-a8e09917-cc6d-4b55-ab2c-1c162286fa78.png)
