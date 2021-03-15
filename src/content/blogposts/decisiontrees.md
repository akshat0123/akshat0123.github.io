---
slug: "/blog/decisiontrees"
date: "2021-03-04"
title: "Decision Trees"
category: "Machine Learning"
order: 4
---

$$
\begin{aligned}
   N_m &= \#\{x_i \in R_m\} & [\text{Number of points in node}]\\
   j &= \text{attribute} &\\
   s &= \text{value} &\\
\end{aligned}
$$

### Regression
$$
\begin{aligned}
    \hat{c}_m &= \frac{1}{N_m} \sum_{x_i \in R_m} y_i & [\text{Output}]\\
    j, s &= \min_{j, s} \left[\min_{c_1} \sum_{x_i \in R_1 (j, s)} (y_i - c_1)^2 + \min_{c_2} \sum_{x_i \in R_2 (j, s)} (y_i - c_2)^2\right] & [\text{Splitting Condition}]\\
\end{aligned}
$$

### Classification
$$
\begin{aligned}
    \hat{p}_{mk} &= \frac{1}{N_m} \sum_{x_i \in R_m} I(y_i = k) & [\text{Probability of class $k$ in node $m$}]\\
    \hat{c}_m &=  \max_{k} \hat{p}_{mk} & [\text{Output}]\\
    j, s &= \min_{j, s} \left[ R_1 (j, s)\sum^{K}_{k=1} \hat{p}_{1k} (1 - \hat{p}_{1k}) + R_2 (j, s)\sum^{K}_{k=1} \hat{p}_{2k} (1 - \hat{p}_{2k}) \right] & [\text{Gini Index Splitting condition}]\\
    j, s &= \min_{j, s} \left[ -R_1 (j, s)\sum^{K}_{k=1} \hat{p}_{1k} \log \hat{p}_{1k} - R_2 (j, s)\sum^{K}_{k=1} \hat{p}_{2k} \log \hat{p}_{2k} \right] & [\text{Cross Entropy Splitting condition}]\\
\end{aligned}
$$
