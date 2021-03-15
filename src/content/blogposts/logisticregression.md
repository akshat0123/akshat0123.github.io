---
slug: "/blog/logisticregression"
date: "2021-03-04"
title: "Logistic Regression"
category: "Machine Learning"
order: 2
---

$$
\begin{aligned}
    h(z) &= \frac{1}{1 + e^{-z}} & [\text{Logistic Function}] \\
    L_{\text{CE}}(y, h(z)) &= -\left[y\log h(z) + (1-y)\log(1-h(z))\right] & [\text{Cross Entropy Loss}] \\
    w_i &= w_i - \alpha \times \frac{\partial}{\partial w_i} L_{\text{CE}}(y_i, h(z_i)) & [\text{Weight Update}]\\
    &= w_i - \alpha \times (x_i (h(z_i) - y_i)) \\
    w &= w - \alpha \times \frac{1}{B}\sum^{B}_{i=1} x_i (h(z_i) - y_i) & [\text{Batch Weight Update}]\\
\end{aligned}
$$

Derivative of Logistic Function
$$
\begin{aligned}
    \frac{\partial}{\partial z} h(z) &= \frac{\partial}{\partial z} \frac{1}{1+e^{-z}} \\
    &= \frac{\partial}{\partial z} (1 + e^{-z})^{-1} \\
    &= -(1 + e^{-z})^{-2} \left(\frac{\partial}{\partial z} 1 + e^{-z}\right) \\
    &= -(1 + e^{-z})^{-2} \left(\frac{\partial}{\partial z} e^{-z}\right) \\
    &= -(1 + e^{-z})^{-2} \left(e^{-z} \frac{\partial}{\partial z} -z\right) \\
    &= -(1 + e^{-z})^{-2} (-e^{-z})\\
    &= \frac{e^{-z}}{(1+e^{-z})^2} \\
    &= \frac{e^{-z}}{(1+e^{-z})}\frac{1}{(1 + e^{-z})}\\
    &= \left(\frac{1+e^{-z}}{(1+e^{-z})} - \frac{1}{(1+e^{-z})}\right)\frac{1}{(1 + e^{-z})}\\
    &= \left(1 - \frac{1}{(1+e^{-z})}\right)\frac{1}{(1 + e^{-z})}\\
    &= (1-h(z))h(z)\\
\end{aligned}
$$

Derivative of Logistic Function with respect to weight $w_i$
$$
\begin{aligned}
    z_i &= w_i \times x_i \\
    \frac{\partial}{\partial w_i} z_i &= \frac{\partial}{\partial w_i} w_i \times x_i\\
    &= x_i\\
    \frac{\partial}{\partial w_i} h(z_i) &= \frac{\partial z_i}{\partial w_i}\frac{\partial h(z_i)}{\partial z_i} \\
    &= x_i (1 - h(z_i)) h(z_i)\\
\end{aligned}
$$

Derivative of Cross Entropy Loss with respect to weight $w_i$
$$
\begin{aligned}
    \frac{\partial}{\partial w_i} L_{\text{CE}}(y_i, h(z_i)) &=
    \frac{\partial}{\partial w_i} -\left[y_i \log h(z_i) + (1-y_i) \log (1-h(z_i))\right]\\
    &= -\left[ y_i \frac{\partial}{\partial w_i} \log h(z_i) + (1-y_i) \frac{\partial}{\partial w_i} \log (1 - h(z_i)) \right]\\
    &= -\left[ \frac{y_i}{h(z_i)} \frac{\partial}{\partial w_i} h(z_i) + \frac{(1-y_i)}{(1-h(z_i))} \frac{\partial}{\partial w_i} (1 - h(z_i)) \right]\\
    &= -\left[ \frac{y_i}{h(z_i)} \frac{\partial}{\partial w_i} h(z_i) - \frac{(1-y_i)}{(1-h(z_i))} \frac{\partial}{\partial w_i} h(z_i)\right]\\
    &= -\left[ \frac{\partial}{\partial w_i} h(z_i) \left(\frac{y_i}{h(z_i)} - \frac{(1-y_i)}{(1-h(z_i))}\right)\right]\\
    &= -\left[ x_i(1-h(z_i))h(z_i) \left(\frac{y_i}{h(z_i)} - \frac{(1-y_i)}{(1-h(z_i))}\right)\right]\\
    &= -\left[ x_i \left(y_i(1-h(z_i)) - (1-y_i)h(z_i)\right)\right]\\
    &= -\left[ x_i (y_i - h(z_i))\right]\\
    &= x_i (h(z_i) - y_i) \\
\end{aligned}
$$
