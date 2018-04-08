---
layout: blog
book: true
background-image: https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera.s3.amazonaws.com/topics/ml/large-icon.png?auto=format%2Ccompress&dpr=1&w=320&h=320&fit=fill&bg=FFF
category: 书籍
date:   2018-04-08 22:46:03
title: Linear regression
tags:

- Linear regression
---

# Logistic regression

## Hypothesis

$$ h_\theta(x) = g(\theta^Tx) $$

$$ g(z) = \frac{1}{1+e^{-z}} $$

## Cost Function

$$ J(\theta) = \frac1m\sum_{i=1}^mCost(h_\theta(x^{i}),y^{(i)}) $$

$$Cost(h_\theta(x),y) = \begin{cases} -log(h_\theta(x)) & y=1 \\ -log(1-h_\theta(x)) & y=0\end{cases} = -ylog(h_\theta)-(1-y)log(1-h_\theta) $$

## Gradient Descent

$$J(\theta) = \frac 1m[\sum_{i=1}^m y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$$

Want:

$$\min_\theta J(\theta)$$

Repeat:

$$\theta_j = \theta_j - \alpha \frac{\partial}{\partial\theta}J(\theta) = \theta_j - \frac \alpha m \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$

Or:

$$\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$$

## Derivation

![image](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/LR_GD.png)