# 노숙자 

# 시각화
## 1) kdeplot 색상별로 반복해서 출력
```python
from collections import OrderedDict

plt.figure(figsize=(20, 16))
plt.style.use('fivethirtyeight')

# Color mapping
colors = OrderedDict({1:'red', 2:'orange', 3:'blue', 4:'green'})
poverty_mapping = OrderedDict({1:'extreme', 2:'moderate', 3:'vulnerable', 4:'non vulnerable'})

# Iterate through the float columns
for i, col in enumerate(train.select_dtypes('float')):
  ax = plt.subplot(4, 2, i+1)
  # Iterate through the poverty levels
  for poverty_level, color in colors.items():
    # Plot each poverty level as a separate line
    sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), ax=ax, color=color, label=poverty_mapping[poverty_level])
    plt.title(f'{col.capitalize()} Distribution')
    plt.xlabel(f'{col}')
    plt.ylabel('Density')

plt.subplots_adjust(top=2)
```
![image](https://user-images.githubusercontent.com/75970111/140924886-3cedf44b-b66e-401c-b771-9e9a33755b5c.png)

# 기타
## 1) 데이터 형별 컬럼 출력
```python
train.select_dtypes('object').head() #float, np.int64 ...
```
