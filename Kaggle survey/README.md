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
plt.figure(figsize=(10, 8))

temp = data['FirstTrainingSelect'].value_counts()
labels = temp.index
sizes = temp.values

plt.pie(sizes, labels=labels, autopct='%1.1f%%')

centre_circle = plt.Circle((0, 0), 0.75, color='black', fc='white', linewidth=1.25) ## (0,0):원의 중심, 0.75:반지름
fig = plt.gcf() 
fig.gca().add_artist(centre_circle)

# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show();
```

### 4) 지도 차트
```python
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff

df = pd.DataFrame(mcr['Country'].value_counts())
df['country'] = df.index
df.columns = ['number', 'country']
df = df.reset_index().drop('index', axis=1)

data = [dict(
        type = 'choropleth',
        locations = df['country'],
        locationmode = 'country names',
        z = df['number'],
        text = df['country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],
                      [0.5,"rgb(70, 100, 245)"],
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],
                      [1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) 
        ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Survey Respondents')
)
       ]
layout = dict(
    title = 'The Nationality of Respondents',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator')
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
```
![newplot](https://user-images.githubusercontent.com/75970111/138223550-2afd03d4-fd43-4946-8da8-604836297aed.png)

### 5) 벤다이어그램
```python
from matplotlib_venn import venn2
venn2(subsets = (python.shape[0], R.shape[0], both.shape[0]), set_labels=('Python Users', 'R Users'))
plt.title('Venn Diagram for Users')
plt.show()
```
![image](https://user-images.githubusercontent.com/75970111/137815523-1c29a759-f82b-49ad-9c09-8d7836e66889.png)

### 5-1) 벤다이어그램(pyplot)
```python
import plotly.offline as py
py.init_notebook_mode(connected=True)

car = data['CareerSwitcher'].value_counts()
labels = (np.array(car.index))
proportions = (np.array((car/car.sum())*100))
colors = ['#FEBFB3', '#E1396C']

trace = go.Pie(labels=labels, values=proportions, hoverinfo='label+percent', marker=dict(colors=colors, line=dict(color='#000000', width=2)))

layout = go.Layout(title='Working people looking to switch careers to data science')
data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename='Career_Switcher')
fig.show(renderer="colab") # 코랩에서 실행시 입력해줘야함
```
![newplot](https://user-images.githubusercontent.com/75970111/138034847-e455ba8f-74b9-47f2-a7c5-1f6666be98ff.png)


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

### colab에서 plotly 실행
```python
import plotly.io as pio
pio.renderers.default = 'colab'
```
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
