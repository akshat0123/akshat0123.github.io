---
slug: "/blog/softmaxregression"
date: "2021-03-04"
title: "Softmax Regression"
category: "Machine Learning"
order: 3
---

$$
\begin{aligned}
    S(z_i) &= \frac{e^{z_i}}{\sum^{K}_{j=1} e^{z_j}} & \text{[Softmax Function]}\\
    L_{\text{CE}}(y, S(z)) &= - \sum^{K}_{j=1} y_j \log S(z_j)& \text{[Cross Entropy Loss]}\\
    w_i &= w_i - \alpha \times \frac{\partial}{\partial w_i} L_{\text{CE}}(y_i, S(z_i))& \text{[Weight Update]}\\
    &= w_i - \alpha \times x_i(S(z_i) - y_i) &\\
    w &= w - \alpha \times \frac{1}{B} \sum^{B}_{i=1}x_i(S(z_i) - y_i) & \text{[Batch Weight Update]}\\
\end{aligned}
$$

### Derivations
Derivative of arbitrary sigmoid output $S(z_j)$ with respect to arbitrary linear combination output $z_i$:

$$
\begin{aligned}
    \frac{\partial S(z_j) }{\partial z_i} &= \frac{\partial}{\partial z_i} \frac{e^{z_j}}{\sum^{K}_{k=1}e^{z_k}}\\\\
    &= \frac{\frac{\partial}{\partial z_i} e^{z_j} (\sum^{K}_{k=1}e^{z_k}) - e^{z_j} (\frac{\partial}{\partial z_i} \sum^{K}_{k=1}e^{z_k})}{(\sum^{K}_{k=1}e^{z_k})^2}\\\\
    &= 
    \begin{cases}
        \frac{\frac{\partial}{\partial z_i} e^{z_j} (\sum^{K}_{k=1}e^{z_k}) - e^{z_j} (\frac{\partial}{\partial z_i} \sum^{K}_{k=1}e^{z_k})}{(\sum^{K}_{k=1}e^{z_k})^2} & i = j\\\\
        \frac{\frac{\partial}{\partial z_i} e^{z_j} (\sum^{K}_{k=1}e^{z_k}) - e^{z_j} (\frac{\partial}{\partial z_i} \sum^{K}_{k=1}e^{z_k})}{(\sum^{K}_{k=1}e^{z_k})^2} & i \neq j\\
    \end{cases}\\\\
    &=
    \begin{cases}
        \frac{e^{z_i} (\sum^{K}_{k=1}e^{z_k}) - (e^{z_i})^2 }{(\sum^{K}_{k=1}e^{z_k})^2} & i = j\\\\
        \frac{0 - e^{z_j}e^{z_i} }{(\sum^{K}_{k=1}e^{z_k})^2} & i \neq j\\
    \end{cases}\\\\
    &=
    \begin{cases}
        S(z_i)(1 - S(z_i)) & i = j\\
        -S(z_j)S(z_i) & i \neq j\\
    \end{cases}\\\\
    &= S(z_i)(\delta_{i,j} - S(z_j)) \\
\end{aligned}
$$

Derivative of loss with respect to arbitrary linear combination output $z_i$:

$$
\begin{aligned}
    \frac{\partial L}{\partial z_i} &= - \sum^{K}_{k=1} y_k \log S(z_k) \\
    &= - \left[ \frac{y_i}{S(z_i)}S(z_i)(1 - S(z_i)) - \sum^{K}_{k\neq i} \frac{y_k}{S(z_k)}(S(z_k)S(z_i))  \right]\\\\
    &= - \left[ y_i(1 - S(z_i)) - \sum^{K}_{k\neq i} y_k S(z_i) \right]\\\\
    &= - \left[ y_i - S(z_i) y_i  - \sum^{K}_{k\neq i} S(z_i) y_k  \right]\\\\
    &= - \left[ y_i - \sum^{K}_{k=1}  S(z_i) y_k \right]\\\\
    &= - \left[ y_i - S(z_i) \sum^{K}_{k=1}  y_k \right]\\\\
    &= S(z_i) - y_i\\\\
\end{aligned}
$$

Derivative of weight $w_i$ with respect to linear combination $z_i$:

$$
\begin{aligned}
    \frac{\partial z_i}{\partial w_i} &= \frac{\partial}{\partial w_i} w_i \times x_i\\
    &= x_i
\end{aligned}
$$

Derivative of weight $w_i$ with respect to loss:

$$
\begin{aligned}
    \frac{\partial L}{\partial w_i} &= \frac{\partial z_i}{\partial w_i}\frac{\partial L}{\partial z_i}\\
    &= x_i (S(z_i) - y_i)\\
\end{aligned}
$$
