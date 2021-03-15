---
slug: "/blog/ensemblelearning"
date: "2021-03-04"
title: "Ensemble Learning"
category: "Machine Learning"
order: 5
---

### Random Forests

```
for b = 1 to B:
    Retrieve a bootstrap sample
    Grow a full decision tree on the sample with random features
return ensemble 
```

$$
\begin{aligned}
    f(x) &= \frac{1}{B}\sum^{B}_{b=1} T_b(x) & \text{[Regression]}\\
    f(x) &= \text{majority vote} \{T_b(x)\}^{B}_{1} & \text{[Classification]}\\
\end{aligned}
$$

### Gradient Boosted Trees
```
fit tree to data
calculate gradient

while loss not acceptable:
    fit new tree to negative gradient
```
