---
layout: blog
book: true
background-image: https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera.s3.amazonaws.com/topics/ml/large-icon.png?auto=format%2Ccompress&dpr=1&w=320&h=320&fit=fill&bg=FFF
category: 书籍
date:   2018-04-12 15:59:03
title: BP neurual network
tags:

- BP neurual network

---



# Neural Networks: Learning:

## Cost Function

$$\begin{gather*}\large J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}$$

## Backpropagation Algorithm

Our goal is to compute:

$$\min_\Theta J(\Theta)$$

That is, we want to minimize our cost function J using an optimal set of parameters in theta.

In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

$$\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$$

![image](https://raw.githubusercontent.com/LiBingtao/LiBingtao.github.io/master/image/BP.PNG)









