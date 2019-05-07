# Predicting Home Values with Linear Regression

In this project, I used Python to build a linear regression model to try to predict the price of houses in Kings County.

#### Contents of the Jupyter Notebook:

0. Methodology
1. Exploring the Data
2. Splitting, Selecting, and Scaling Data
3. Linear Regression
4. Takeaways

## Exploratory Data Analysis

Excerpts:

```python
df = pd.read_csv('kc_house_data.csv')
df.head()
```
![excerpt](https://github.com/lorarjohns/dsc-v2-mod1-final-project/blob/master/excerpt.png)

Checking for collinearity:

![heatmap](https://github.com/lorarjohns/dsc-v2-mod1-final-project/blob/master/heatmap.png)

Creating dummy variables:

```python
# Create dummy variables
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer

waterfront_dummies = pd.get_dummies(df['waterfront'], prefix='wf', drop_first=True)
condition_dummies = pd.get_dummies(df['condition'], prefix='cond', drop_first=True)
```

Binning continuous values:

```python
disc_5k = KBinsDiscretizer(encode='onehot-dense', strategy='kmeans')
disc_5 = KBinsDiscretizer(encode='onehot-dense')
disc_3 = KBinsDiscretizer(encode='onehot-dense')

bedrooms_bin = disc_5k.fit_transform(df[['bedrooms']])
bathrooms_bin = disc_3.fit_transform(df[['bathrooms']])
```
Scaling factors:

```python
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, StandardScaler

scale_robust = RobustScaler(copy=False, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)

scale_std = StandardScaler(copy=False)
```

## Results

+ Linear regression with two variables: R-squared ~ 0.5
+ Linear regression with additional factors and scaling: R-squared ~ 0.7
+ RMSE still very high

![regression](https://github.com/lorarjohns/dsc-v2-mod1-final-project/blob/master/regression.png)

## Takeaways

+ In this dataset, sqft_living and grade are the variables most highly correlated with price.
+ R-squared and MSE/RMSE can diverge sharply, and multiple metrics of accuracy must be taken into account.
+ The choice of which variables to keep and which to eliminate for multicollinearity has significant impacts on the accuracy of the final model.
+ Data exploration and cleaning take up more time than running the actual regressions, but are crucial precursors to manipulating and processing the information.
+ Pandas and scikitlearn contain many powerful tools with extensive documentation to aid in statistical analysis of data, but these libraries present a surprising number of incompatibilities that must be resolved via ad-hoc methods (or discovery of additional libraries like sklearn-pandas).
