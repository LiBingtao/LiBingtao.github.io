---

layout: blog
Code: true
title:  "高维数据可视化"
background: green
background-image: https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_40_2.png
date:   2018-03-26 22:07:03
category: Code
tags:

- 数据可视化

---

# 高维测井数据可视化

## 介绍 
>描述性分析（descriptive analytics）是分析任何数据科学项目或特定研究的核心组成部分。数据聚合（aggregation）、汇总（summarization）和可视化（visualization）是支撑数据分析领域的主要支柱。从传统商业智能（Business Intelligence）开始，甚至到如今人工智能时代，数据可视化都是一个强有力的工具；由于其能有效抽取正确的信息，同时清楚容易地理解和解释结果，可视化被业界组织广泛使用。  

对于测井数据的分析来说也是如此。尤其是目前处理多系列的测井数据也即处理多维数据集开始出现问题，因为我们的数据分析通常限于 2 个维度。在本文中，我们将探索一些有效的多维数据可视化策略。
    
>一图胜千言

无论是作为分析数据特征的工具，还是作为展示项目成果的利器，这都是亘古不变的真理。
    
>「一张图片的最大价值在于，它迫使我们注意到我们从未期望看到的东西。」  
——John Tukey

首先加载以下必要的依赖包进行分析


```python
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
%matplotlib inline
```

我们将主要使用 matplotlib 和 seaborn 作为我们的可视化框架。首先进行基本的数据预处理步骤


```python
sand = pd.read_csv('sandstone.csv', sep=',')
mud = pd.read_csv('mudstone.csv', sep=',')
sand['rock_type'] = 'sandstone'
mud['rock_type'] = 'mudstone'
rock = pd.concat([sand,mud])
rock = rock.sample(frac=1, random_state=42).reset_index(drop=True)
```

看一下数据的前几行：


```python
rock.head()
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
      <th>SP</th>
      <th>GR</th>
      <th>CAL</th>
      <th>RC</th>
      <th>CNL</th>
      <th>DEN</th>
      <th>rock_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.687999</td>
      <td>0.781</td>
      <td>0.330702</td>
      <td>0.324</td>
      <td>36.559</td>
      <td>2.256</td>
      <td>mudstone</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.661169</td>
      <td>0.528</td>
      <td>0.577179</td>
      <td>0.126</td>
      <td>34.610</td>
      <td>2.127</td>
      <td>mudstone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.313065</td>
      <td>0.512</td>
      <td>0.058519</td>
      <td>2.186</td>
      <td>18.752</td>
      <td>2.451</td>
      <td>sandstone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.635486</td>
      <td>0.685</td>
      <td>0.312557</td>
      <td>0.392</td>
      <td>34.511</td>
      <td>2.358</td>
      <td>mudstone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.439296</td>
      <td>0.486</td>
      <td>0.013156</td>
      <td>1.964</td>
      <td>17.105</td>
      <td>2.415</td>
      <td>sandstone</td>
    </tr>
  </tbody>
</table>
</div>



现在，我们用几个测井深度点的数值和分类属性。每个观测样本属于砂岩或泥岩，属性是从井眼中测量和获得的特定测井数据。

## 单变量分析 

单变量分析基本上是数据分析或可视化的最简单形式，因为只关心分析一个数据属性或变量并将其可视化（1 维）。

### 可视化 1 维数据（1-D）  


使所有数值数据及其分布可视化的最快、最有效的方法之一是利用 pandas 画直方图（histogram）。


```python
rock.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.5, 1.5)) 
```


![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_9_0.png)


上图方便的可视化任何属性的基本数据分布。

而直方图或核密度图能够很好地帮助理解该属性数据的连续性分布。


```python
# Histogram
fig = plt.figure(figsize = (12,4))
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,2,1)
ax.set_xlabel("SP")
ax.set_ylabel("Frequency") 
ax1.set_title("SP Distribution in rock")
ax.text(0.8, 40, r'$\mu$='+str(round(rock['SP'].mean(),2)), fontsize=12)
freq, bins, patches = ax.hist(rock['SP'], color='steelblue', bins=15, edgecolor='black', linewidth=1)


# Density Plot
ax1 = fig.add_subplot(1,2,2)
ax1.set_xlabel("SP")
ax1.set_ylabel("Frequency") 
ax1.set_title("SP Distribution in rock")
sns.kdeplot(rock['SP'], ax=ax1, shade=True, color='steelblue')
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_11_1.png)
![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_11_1.png)


从上面的图表中可以看出，数据中的SP属性分布是明显的双峰，跟砂泥岩相对应。

## 多测井属性分析 
多元分析才是真正有意思且复杂领域。这里我们分析多个数据维度或属性（2 个或更多）。多变量分析不仅包括检查分布，还包括这些属性之间的潜在关系、模式和相关性。你也可以根据需要解决的问题，利用推断统计（inferential statistics）和假设检验，检查不同属性、群体等的统计显著性（significance）  
### 可视化2维数据（2D）
检查不同数据属性之间的潜在关系或相关性的最佳方法之一是利用配对相关性矩阵（pair-wise correlation matrix）并将其可视化为热力图。这里需要用到seaborn库的heatmap方法。


```python
# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = rock.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Rock well log data Correlation Heatmap', fontsize=14)
```


![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_14_0.png)


热力图中的颜色根据相关性的强度而变化，可以很容易发现彼此之间具有强相关性的潜在属性。另一种可视化的方法是在感兴趣的属性之间使用pairplot


```python
pp = sns.pairplot(rock, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Rock well log data Pairwise Plots', fontsize=14)
```


![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_16_0.png)


根据上图，可以看到散点图也是观察数据属性的 2 维潜在关系或模式的有效方式。另一种将多元数据可视化为多个属性的方法是使用平行坐标图。 

首先对数据进行处理


```python
cols = ['SP', 'GR', 'CAL', 'RC', 'CNL', 'DEN']
subset_df = rock[cols]

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
final_df = pd.concat([scaled_df, rock['rock_type']], axis=1)
final_df.head()
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
      <th>SP</th>
      <th>GR</th>
      <th>CAL</th>
      <th>RC</th>
      <th>CNL</th>
      <th>DEN</th>
      <th>rock_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.694273</td>
      <td>2.078427</td>
      <td>0.085873</td>
      <td>-0.841257</td>
      <td>0.998660</td>
      <td>-0.585282</td>
      <td>mudstone</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.566916</td>
      <td>-0.117041</td>
      <td>1.001720</td>
      <td>-1.107031</td>
      <td>0.773486</td>
      <td>-1.634168</td>
      <td>mudstone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.085448</td>
      <td>-0.255885</td>
      <td>-0.925491</td>
      <td>1.658088</td>
      <td>-1.058642</td>
      <td>1.000243</td>
      <td>sandstone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.445004</td>
      <td>1.245364</td>
      <td>0.018449</td>
      <td>-0.749981</td>
      <td>0.762048</td>
      <td>0.244069</td>
      <td>mudstone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.486259</td>
      <td>-0.481506</td>
      <td>-1.094052</td>
      <td>1.360099</td>
      <td>-1.248925</td>
      <td>0.707530</td>
      <td>sandstone</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pandas.plotting import parallel_coordinates

pc = parallel_coordinates(final_df, 'rock_type', color=('#FFE888', '#FF9999'))
```


![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_19_0.png)


上图中，点被表征为连接的线段。每条垂直线代表一个数据属性。所有属性中的一组完整的连接线段表征一个数据点。因此，趋于同一类的点将会更加接近。通过观察就可以清楚看到，砂岩的自然电位（SP）,自然伽马（GR）,井径（CAL）,中子测井（CNL）明显偏低，而深浅电阻率差（RC）和密度测井（DEN）明显偏高。  

接下来我们看看可视化两个连续型数值属性的方法。散点图和联合分布图（joint plot）是检查模式、关系以及属性分布的好方法。


```python
# Scatter Plot
plt.scatter(rock['SP'], rock['GR'],
            alpha=0.4, edgecolors='w')

plt.xlabel('SP')
plt.ylabel('GR')
plt.title('Rock SP - GR Content',y=1.05)
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_21_1.png)

```python
# Joint Plot
sns.jointplot(x='SP', y='GR', data=rock,
                   kind='kde', space=0, size=5, ratio=4)
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_22_1.png)


接下来，我们看一下如何进行二维混合数据的可视化（连续的GR和离散的rock type）


```python
# Using multiple Histograms 
fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("GR", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("GR")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(rock, hue='rock_type', palette={"sandstone": "y", "mudstone": "b"})
g.map(sns.distplot, 'GR', kde=False, bins=15, ax=ax)
ax.legend(title='Rock Type')
plt.close(2)
```


![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_24_0.png)


可以看到上面生成的图形清晰简洁，我们可以轻松地比较各种分布。  
除此之外，箱线图（box plot）是根据分类属性中的不同数值有效描述数值数据组的另一种方法。箱线图是了解数据中四分位数值以及潜在异常值的好方法。  
另一个类似的可视化是小提琴图，这是使用核密度图显示分组数值数据的另一种有效方法（描绘了数据在不同值下的概率密度）。


```python
f = plt.figure(figsize=(16, 6))
# Box Plots
ax = f.add_subplot(121)
sns.boxplot(data=rock.iloc[:,[0,1,2,3,5]],  ax=ax)
# Violin Plots
ax1 = f.add_subplot(122)
sns.violinplot(data=rock.iloc[:,[0,1,2,3,5]],  ax=ax1, width=1.1)
f.suptitle('Well log data distribution in box and violin plot', fontsize=14)
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_26_1.png)


### 可视化3维数据（3D）
这里研究有 3 个属性或维度的数据，我们可以通过考虑配对散点图并引入颜色或色调将分类维度中的值分离出来。


```python
pp = sns.pairplot(rock, hue='rock_type', size=1.8, aspect=1.8, 
                  palette={"mudstone": "#FF9999", "sandstone": "#FFE888"},
                  plot_kws=dict(edgecolor="black", linewidth=0.5))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Well log data Pairwise Plots', fontsize=14)
```


![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_28_0.png)


上图可以查看相关性和模式，也可以比较岩性。我们可以清楚地看到砂岩的SP和GR比泥岩低。

让我们来看看可视化 3 个连续型数值属性的策略。一种方法是将 2 个维度表征为常规长度（x 轴）和宽度（y 轴）并且将第 3 维表征为深度（z 轴）的概念。


```python
# Visualizing 3-D numeric data with Scatter Plots
# length, breadth and depth
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = rock['SP']
ys = rock['GR']
zs = rock['CAL']
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('SP')
ax.set_ylabel('GR')
ax.set_zlabel('CAL')
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_30_1.png)


我们还可以利用常规的 2 维坐标轴，并将尺寸大小的概念作为第 3 维（本质上是气泡图），其中点的尺寸大小表征第 3 维的数量。


```python
plt.scatter(rock['SP'], rock['GR'], s=rock['CAL']*300, 
            alpha=0.4, edgecolors='w')

plt.xlabel('SP')
plt.ylabel('GR')
plt.title('Rock SP - GR - CAL',y=1.05)
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_32_1.png)


### 可视化4维数据（4D）

基于上述讨论，我们利用图表的各个组件可视化多个维度。一种可视化 4 维数据的方法是在传统图如散点图中利用深度和色调表征特定的数据维度。


```python
# Visualizing 4-D mix data using scatter plots
# leveraging the concepts of hue and depth
fig = plt.figure(figsize=(8, 6))
t = fig.suptitle('Rock SP - GR - CAL - Type', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

xs = list(rock['SP'])
ys = list(rock['GR'])
zs = list(rock['CAL'])
data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]
colors = ['red' if wt == 'sandstone' else 'yellow' for wt in list(rock['rock_type'])]

for data, color in zip(data_points, colors):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.4, c=color, edgecolors='none', s=30)

ax.set_xlabel('SP')
ax.set_ylabel('GR')
ax.set_zlabel('CAL') 
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_34_1.png)


rock_type 属性由上图中的色调表征得相当明显。此外，由于图的复杂性，解释这些可视化开始变得困难，但我们仍然可以看出，例如泥岩的SP，GR，CAL均较泥岩更高。当然，如果SP和GR之间有某种联系，我们可能会看到一个逐渐增加或减少的数据点趋势。

另一个策略是使用二维图，但利用色调和数据点大小作为数据维度。通常情况下，这将类似于气泡图等我们先前可视化的图表。


```python
# Visualizing 4-D mix data using bubble plots
# leveraging the concepts of hue and size
size = rock['CAL']*300
fill_colors = ['#FF9999' if wt=='sandstone' else '#FFE888' for wt in list(rock['rock_type'])]
edge_colors = ['red' if wt=='sandstone' else 'orange' for wt in list(rock['rock_type'])]

plt.scatter(rock['SP'], rock['GR'], s=size, 
            alpha=0.4, color=fill_colors, edgecolors=edge_colors)

plt.xlabel('SP')
plt.ylabel('GR')
plt.title('Rock SP - GR - CAL - Type',y=1.05) 
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_36_1.png)


### 可视化 5 维数据（5D）

我们照旧遵从上文提出的策略，要想可视化 5 维数据，我们要利用各种绘图组件。我们使用深度、色调、大小来表征其中的三个维度。其它两维仍为常规轴。因为我们还会用到大小这个概念，并借此画出一个三维气泡图。


```python
# Visualizing 5-D mix data using bubble charts
# leveraging the concepts of hue, size and depth
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
t = fig.suptitle('Rock SP - GR - CAL - RC - Type', fontsize=14)

xs = list(rock['SP'])
ys = list(rock['GR'])
zs = list(rock['CAL'])
data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]

ss = list(rock['RC']*300)
colors = ['red' if wt == 'sandstone' else 'yellow' for wt in list(rock['rock_type'])]

for data, color, size in zip(data_points, colors, ss):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.4, c=color, edgecolors='none', s=size)

ax.set_xlabel('SP')
ax.set_ylabel('GR')
ax.set_zlabel('CAL')
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_38_2.png)


可视化 6 维数据（6-D）

我们继续在可视化中添加一个数据维度。我们将利用深度、色调、大小和形状及两个常规轴来描述所有 6 个数据维度。


```python
# Visualizing 6-D mix data using scatter charts
# leveraging the concepts of hue, size, depth and shape
fig = plt.figure(figsize=(8, 6))
t = fig.suptitle('Rock SP - GR - CAL - RC - DEN - Type', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

xs = list(rock['SP'])
ys = list(rock['GR'])
zs = list(rock['CAL'])
data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]

ss = list(rock['RC']*200)
co = np.array(['black', 'midnightblue', 'blue', 'darkgreen', 'green', 'greenyellow', 
            'yellow', 'gold', 'orange', 'darkorange', 'red', 'darkred', 'purple'])
interval = (rock['DEN'].max()-rock['DEN'].min())/13
colors = list(co[((rock['DEN']-rock['DEN'].min())/interval).astype(int)-1])
markers = ['o' if q == 'sandstone' else 'x' for q in list(rock['rock_type'])]

for data, color, size, mark in zip(data_points, colors, ss, markers):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.4, c=color, edgecolors='none', s=size, marker=mark)

ax.set_xlabel('SP')
ax.set_ylabel('GR')
ax.set_zlabel('CAL')
```

![png](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/output_40_2.png)


### 更高维度……
暂时没有想出更高维度的数据直接可视化的方法，折中的办法是利用PCA和TSNE等方法进行降维之后再进行可视化，但可解释性降低

## Reference

*https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57*

