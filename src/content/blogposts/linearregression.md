---
slug: "/blog/linearregression"
date: "2021-03-04"
title: "Linear Regression"
category: "Machine Learning"
order: 1
---

$$
\begin{aligned}
    h(x) &= w^T \cdot x &\\ 
    L_{\text{MSE}}(y, h(x)) &= (y-h(x))^2 & [\text{Mean Squared Error Loss}]\\
    w_i &= w_i - \alpha \times \frac{\partial}{\partial w_i} L_{\text{MSE}}(y_i, h(x_i)) & [\text{Weight Update}]\\
    &= w_i - \alpha \times x_i(h(x_i) - y_i)\\
    w &= w -\alpha \times \sum^{B}_{i=0} x_i(h(x_i) - y_i)\\
\end{aligned}
$$

### Derivations
Derivative of Mean Squared Error with respect to weight $w_i$

$$
\begin{aligned}
    \frac{\partial}{\partial w_i} L_{\text{MSE}}(y_i, h(x_i)) &= \frac{\partial}{\partial w_i} (y_i - h(x_i))^2\\
    &= 2(y_i - h(x_i)) \times \frac{\partial}{\partial w_i} (y_i - h(x_i))\\
    &= 2(y_i - h(x_i)) \times \left(\frac{\partial}{\partial w_i} y_i - \frac{\partial}{\partial w_i}h(x_i)\right)\\
    &= 2(y_i - h(x_i)) \times \left(-\frac{\partial}{\partial w_i}w_i \times x_i\right)\\
    &= -2x_i(y_i - h(x_i))\\
    &= 2x_i(h(x_i)-y_i)\\
    &\propto x_i(h(x_i)-y_i)\\
\end{aligned}
$$
