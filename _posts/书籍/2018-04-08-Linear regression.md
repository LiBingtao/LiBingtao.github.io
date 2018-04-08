---
layout: blog
book: true
background-image: https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera.s3.amazonaws.com/topics/ml/large-icon.png?auto=format%2Ccompress&dpr=1&w=320&h=320&fit=fill&bg=FFF
category: 书籍
date:   2018-04-08 14:09:03
title: Linear regression
tags:

- Linear regression

---



#Linear regression:

Hypothesis:           

$$ h_\theta(x) = \theta^Tx = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n $$

Parameters:          

$$\theta = [\theta_0,\theta_1,\ldots,\theta_n]^T $$

Cost function:        

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 = X\theta - y $$

Gradient descent: 

$$\theta_j = \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta) = \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$$

​			   or:

$$\theta = \theta - \frac \alpha mX^T(X\theta-y)$$









