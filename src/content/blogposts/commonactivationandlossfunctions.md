---
slug: "/blog/commonactivationandlossfunctions"
date: "2021-04-07"
title: "Common Activation and Loss Functions"
category: "2 Deep Learning"
order: 2
---

### Introduction

In my previous post on deep learning, I discussed how to create feed-forward neural network layers, and how create models of arbitrary size with those layers.
I also mention that a variety of activation functions can be used in the hidden layers of a neural network, and various loss functions can be used depending on the desired goals for a model.
In this post, I will go over some common activation and loss functions and derive their local gradients.

### Common Activation Functions

#### Sigmoid Function

The sigmoid function takes in real-valued input $x$ and returns a real-valued output in the range $[0, 1]$.
This activation function is most often used in output layers for binary classification models, although it could technically be used in hidden layers as well.

$$
\begin{aligned}
    \sigma(x) &= \frac{1}{1 + e^{-x}} \\
    \frac{\partial \sigma(x)}{\partial x} &= \frac{\partial}{\partial x} (1+e^{-x})^{-1}\\
    &= -(1+e^{-x})^{-2} \frac{\partial}{\partial x} (1 + e^{-x})\\
    &= -(1+e^{-x})^{-2} \frac{\partial}{\partial x} e^{-x}\\
    &= -(1+e^{-x})^{-2} e^{-x} \frac{\partial}{\partial x} -x\\
    &= (1+e^{-x})^{-2} e^{-x}\\
    &= \frac{e^{-x}}{(1+e^{-x})^{2}}\\
    &= \frac{1}{1+e^{-x}} \frac{e^{-x}}{1+e^{-x}}\\
    &= \frac{1}{1+e^{-x}} \frac{1 + e^{-x} - 1}{1+e^{-x}}\\
    &= \frac{1}{1+e^{-x}} \left[\frac{1 + e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right]\\
    &= \frac{1}{1+e^{-x}} \left[1 - \frac{1}{1+e^{-x}}\right]\\
    &= \sigma(x) [1 - \sigma(x)]\\
\end{aligned}
$$

#### Tanh Function

The tanh function takes in real-valued input $x$ and returns a real-valued output in the range $[-1, 1]$.
This activation function is most often seen as an activation function for hidden layers.

$$
\begin{aligned}
    \tanh(x) &= \frac{\sinh(x)}{\cosh(x)}\\
    &= \frac{e^x - e^{-x}}{e^x + e^{-x}} \\
    \frac{\partial \tanh(x)}{\partial x} &= 
    \frac{
        \left[\frac{\partial}{\partial x} (e^x - e^{-x})\right]
        (e^x + e^{-x})
        - 
        (e^x - e^{-x})
        \left[\frac{\partial}{\partial x} (e^x + e^{-x})\right]
    }{
        (e^x + e^{-x})^2
    }\\
    &= 
    \frac{
        (e^x + e^{-x})^2 - (e^x - e^{-x})^2
    }{
        (e^x + e^{-x})^2
    }\\
    &= 1 - 
    \frac{
        (e^x - e^{-x})^2
    }{
        (e^x + e^{-x})^2
    }\\
    &= 1 - \tanh^2(x)\\
\end{aligned}
$$

#### Rectified Linear Activation Function (ReLU)

The ReLU function takes in real-valued input $x$ and returns real-valued output in the range $[0, \infty)$. 
The ReLU function is most often seen as an activation function for hidden layers.
It should be noted that formally, the derivative for the ReLU function at 0 is undefined. 
However, in practice, it is often set to 0 when $x$ is 0.

$$
\begin{aligned}
    r(x) &= 
    \begin{cases}
        x & x > 0\\
        0 & x < 0
    \end{cases}\\
    \frac{\partial r(x)}{\partial x} &= 
    \begin{cases}
        1 & x > 0\\
        0 & x < 0\\
    \end{cases}\\
\end{aligned}
$$

#### Softmax Function

The softmax function takes in a vector of real-values and returns a vector of real values, each in the range $[0, 1]$.
The vector $\vec{s(x)}$ obtained from running vector $x$ through the softmax function always sums to 1.
Due to this, the softmax function is most often used as the activation function for output layers in multinomial classification problems.

$$
\begin{aligned}
    s(x_i) &= \frac{e^{x_i}}{\sum^K_{k=1} e^{x_k}} \\
    \frac{\partial s(x_j)}{\partial x_i} &= \frac{\partial}{\partial x_i} \frac{e^{x_j}}{\sum^K_{k=1} e^{x_k}}\\
    &= \frac{
        \left(\frac{\partial}{\partial x_i} e^{x_j}\right)\sum^K_{k=1}e^{x_k} - 
        e^{x_j}\left(\frac{\partial}{\partial x_i}\sum^K_{k=1}e^{x_k}\right) 
    }{
        (\sum^K_{k=1}e^{x_k})^2
    }\\
    &= 
    \begin{cases}
        \frac{
            \left(\frac{\partial}{\partial x_i} e^{x_j}\right)\sum^K_{k=1}e^{x_k} - 
            e^{x_j}\left(\frac{\partial}{\partial x_i}\sum^K_{k=1}e^{x_k}\right) 
        }{
            (\sum^K_{k=1}e^{x_k})^2
        } & i=j\\
        \frac{
            \left(\frac{\partial}{\partial x_i} e^{x_j}\right)\sum^K_{k=1}e^{x_k} - 
            e^{x_j}\left(\frac{\partial}{\partial x_i}\sum^K_{k=1}e^{x_k}\right) 
        }{
            (\sum^K_{k=1}e^{x_k})^2
        } & i\neq j\\
    \end{cases}\\
    &= 
    \begin{cases}
        \frac{
            e^{x_i}\sum^K_{k=1}e^{x_k} - 
            e^{x_i}e^{x_i} 
        }{
            (\sum^K_{k=1}e^{x_k})^2
        } & i=j\\
        \frac{
            0\sum^K_{k=1}e^{x_k} - 
            e^{x_j}e^{x_i}
        }{
            (\sum^K_{k=1}e^{x_k})^2
        } & i\neq j\\
    \end{cases}\\
    &= 
    \begin{cases}
        \frac{
            e^{x_i}(\sum^K_{k=1}e^{x_k} -  e^{x_i}) 
        }{
            (\sum^K_{k=1}e^{x_k})^2
        } & i=j\\
        \frac{
            e^{x_i}(-e^{x_j})
        }{
            (\sum^K_{k=1}e^{x_k})^2
        } & i\neq j\\
    \end{cases}\\
    &= 
    \begin{cases}
        s(x_i)(1 - s(x_j))
        & i=j\\
        s(x_i)(0 - s(x_j))
        & i\neq j\\
    \end{cases}\\
    &= s(x_i)(\delta_{i=j} - s(x_j))\\
\end{aligned}
$$

### Common Loss Functions

#### Binary Cross-Entropy

The binary cross-entropy loss function is used for models where the desired output is a binary probability.
This is necessary in binary classification models.

$$
\begin{aligned}
    L_{\text{BCE}}(y, \hat{y}) &= -\left[y\log \hat{y} + (1-y)\log(1-\hat{y})\right]\\
    \frac{\partial L}{\partial \hat{y}} &= 
    \frac{\partial}{\partial \hat{y}} -\left[y\log \hat{y} + (1-y)\log(1-\hat{y})\right]\\
    &= -\left[
        \frac{\partial}{\partial \hat{y}} y\log \hat{y} + 
        \frac{\partial}{\partial \hat{y}}(1-y)\log(1-\hat{y}) 
    \right]\\
    &= -\left[  \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}  \right]\\
    &= \frac{1-y}{1-\hat{y}} - \frac{y}{\hat{y}}\\
\end{aligned}
$$

#### Cross-Entropy

The cross-entropy function is used for models where the desired output is a probability distribution over $K$ possible classes.
This is necessary in multinomial classification models.

$$
\begin{aligned}
    L_{CE}(y, \hat{y}) &= - \left[ \sum^{K}_{k=1} y_k \log(\hat{y}_k) \right] \\
    \frac{\partial L}{\partial \hat{y}_i} &= 
    \frac{\partial}{\partial \hat{y}_i} - \left[ \sum^{K}_{k=1} y_k \log(\hat{y}_k) \right]\\
    &= -\left[ \frac{\partial}{\partial \hat{y_i}} y_i \log(\hat{y}_i)  \right]\\
    &= -\frac{y_i}{\hat{y}_i}\\
\end{aligned}
$$

#### Mean Squared Error

The mean squared error loss function is used for models where the desired output is a real number.
This is necessary for regression models.

$$
\begin{aligned} 
    L_{\text{MSE}}(y,\hat{y}) &= (y - \hat{y})^2\\
    \frac{\partial L}{\partial \hat{y}} &= \frac{\partial}{\partial \hat{y}} (y - \hat{y})^2\\
    &= 2(y-\hat{y})\frac{\partial}{\partial \hat{y}} (y - \hat{y})\\
    &= 2(\hat{y}-y)\\
    &\propto \hat{y}-y\\
\end{aligned}
$$
