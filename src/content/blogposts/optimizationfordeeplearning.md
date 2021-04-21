---
slug: "/blog/optimizationfordeeplearning"
date: "2021-04-12"
title: "Optimization for Deep Learning"
category: "2 Deep Learning"
order: 4
---

### Optimization

Optimization is the process used to minimize the loss function during the training process of a neural network.
There are a variety of different approaches to optimization.
This post will discuss some of those approaches, including gradient descent, stochastic gradient descent, RMSProp, and Adam.
This post will also cover generalized optimization practices, such as momentum as well as adaptive learning.

### Basic Optimization Methods

#### Gradient Descent

Vanilla gradient descent is one of the simplest approaches to optimization.
The general process is to reduce the loss function by moving the weights in the opposite direction of the gradient.
The weight update performed in gradient descent is shown below:

$$
\begin{aligned}
    w &= w - \alpha \frac{\partial L}{\partial w}&
    \text{Weight update}\\
    \alpha &\rightarrow \text{model hyperparameter} &\\
\end{aligned}
$$

#### Stochastic Gradient Descent

Stochastic gradient descent is a variant of gradient descent and is one of the most popular optimization techniques in machine learning.
The basic difference between stochastic gradient descent and plain gradient descent is that in stochastic gradient descent, weights are updated after a single randomly-drawn point of data is seen (or a randomly-drawn batch of data points), rather than only when the entire dataset has been seen, as in plain gradient descent.
Minibatch stochastic gradient descent is a variant of stochastic gradient descent where the weights of a network are updated using a batch of randomly-drawn input data rather than a single data point.
The weight updates for stochastic gradient descent and minibatch stochastic gradient descent are shown below:

$$
\begin{aligned}
    w &= w - \alpha \frac{\partial L(y_i, f(x_i; w))}{\partial w}
    &\text{Weight update}\\
    w &= w - \alpha \left[\frac{1}{B}\sum^B_{i=1} \frac{\partial L(y_i, f(x_i;w))}{\partial w}\right]
    &\text{Minibatch weight update}\\
    \alpha &\rightarrow \text{model hyperparameter} &\\
\end{aligned}
$$

#### Momentum

Momentum is a genralizable method to accelerate optimization that uses a moving average of past gradients to update weights rather than simply the last calculated gradient.
The exponentially decaying moving average of gradients, $v$, is the velocity at which the weights of the model move.
Nesterov momentum is a momentum method variant that evaluates the gradient after applying the current velocity, rather than before, as in standard momentum.

$$
\begin{aligned}
    v &= \epsilon v - \alpha \left[\frac{1}{B}\sum^B_{i=1}\frac{\partial L(y_i, f(x_i; w))}{\partial w}\right]
    & \text{SGD with Momentum}\\
    v &= \epsilon v - \alpha \left[\frac{1}{B}\sum^B_{i=1}\frac{\partial L(y_i, f(x_i; w + \epsilon v))}{\partial w}\right]
    & \text{SGD with Nesterov Momentum}\\
    w &= w + v 
    &\text{Weight update}\\
    \alpha &\rightarrow \text{model hyperparameter}&\\
    \epsilon &\rightarrow \text{model hyperparameter}&\\
    v & \rightarrow \text{velocity}&\\
\end{aligned}
$$

### Code

The code below displays an `Optimizer` class its subclasses for stochastic gradient descent and stochastic gradient descent with momentum.
These classes are meant to be used in conjunction with the `Network` and `Layer` classes I describe in my earlier post, titled *Classification and Regression with Neural Networks*.
The code for `Network` and `Layer` to be used with the `Optimizer` class can be found in the following package from my github page [here](https://github.com/akshat0123/MLReview/tree/main/Packages/mlr/NN).


```python
from abc import ABC, abstractmethod

import torch


class Optimizer(ABC):
    """ Abstract base class for optimizers
    """

    @abstractmethod
    def __init__(self):
        """ Initialize optimizer
        """
        pass            


    @abstractmethod
    def __copy__(self):
        """ Copy class instance
        """
        pass


    @abstractmethod
    def update(self):
        """ Update weights
        """
        pass


def SGDOptimizer(momentum: bool=False, epsilon: float=1e-4) -> Optimizer:
    """ Return stochastic gradient descent optimizer

    Args:
        momentum: whether to include momentum or not
        epsilon: epsilon parameter for momentum

    Returns:
        stochastic gradient descent optimizer
    """

    optimizer = SGDMomentumOptimizer(epsilon=epsilon) if momentum else DefaultSGDOptimizer()
    return optimizer


class DefaultSGDOptimizer(Optimizer):
    """ Stochastic Gradient Descent optimizer (without momentum)
    """
    

    def __init__(self) -> None:
        """ Initialize default SGD optimizer
        """
        pass


    def __copy__(self):
        """ Return copy of default SGD optimizer

        Returns: 
            copy of optimizer
        """

        return DefaultSGDOptimizer()


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        return w - (alpha * (dw + (lambdaa * dr)))


class SGDMomentumOptimizer(Optimizer):
    """ Stochastic Gradient Descent optimizer (with momentum)
    """
        

    def __init__(self, epsilon: float=1e-4) -> None:
        """ Initialize default SGD optimizer
        """

        self.epsilon = epsilon
        self.v = None


    def __copy__(self):
        """ Return copy of default SGD optimizer

        Returns: 
            copy of optimizer
        """

        return SGDMomentumOptimizer(epsilon=self.epsilon)            


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        if self.v is None: 
            self.v = torch.zeros(w.shape)

        self.v = (self.epsilon * self.v) - (alpha * (dw + (lambdaa * dr)))
        return w + self.v
```

### Adaptive Learning Optimization Methods

#### AdaGrad

AdaGrad is an optimization approach that has a learning rate for all parameters ($r$ in the equation below), and updates the learning rates continuously.
Learning rates are decayed proportionally with regard to how often updates are made.

$$
\begin{aligned}
    r &= r + \left[\frac{1}{B}\sum^B_{i=1}\frac{\partial L(y_i, f(x_i; w))}{\partial w}\right]^2\\
    w &= w - \frac{\alpha}{\delta + \sqrt{r}} \left[
         \frac{1}{B} \sum^B_{i=1} \frac{\partial L(y_i, f(x_i; w))}{\partial w}
     \right]\\
     \alpha &\rightarrow \text{model hyperparameter}\\
     \delta &\rightarrow \text{small constant, usually }10^{-6}\\
\end{aligned}
$$

#### RMSProp

RMSProp is an optimization approah that also has a learning rate for all parameters ($r$ in the equation below), and updates the learning rates continuously.
When compared to AdaGrad, learning rates in RMSProp do not diminish nearly as fast.

$$
\begin{aligned}
    r &= \rho r + (1-\rho) \left[
         \frac{1}{B} \sum^B_{i=1} \frac{\partial L(y_i, f(x_i; w))}{\partial w}
     \right]^2\\
     w &= w - \frac{\alpha}{\sqrt{\delta + r}} \left[
         \frac{1}{B} \sum^B_{i=1} \frac{\partial L(y_i, f(x_i; w))}{\partial w}
     \right]\\
     \alpha &\rightarrow \text{model hyperparameter}\\
     \rho &\rightarrow \text{model hyperparameter}\\
     \delta &\rightarrow \text{small constant, usually }10^{-6}\\
\end{aligned}
$$

#### RMSProp with Momentum

Momentum can also be added to the RMSProp optimization method as shown below:

$$
\begin{aligned}
    r &= \rho r + (1-\rho) \left[
         \frac{1}{B} \sum^B_{i=1} \frac{\partial L(y_i, f(x_i; w))}{\partial w}
     \right]^2\\
    v &= \epsilon v - \frac{\alpha}{\sqrt{\delta + r}} \left[
         \frac{1}{B} \sum^B_{i=1} \frac{\partial L(y_i, f(x_i; w))}{\partial w}
     \right]\\
    w &= w + v\\
    \alpha &\rightarrow \text{model hyperparameter}\\
    \rho &\rightarrow \text{model hyperparameter}\\
    \delta &\rightarrow \text{small constant, usually }10^{-6}\\
\end{aligned}
$$

#### Adam

Adam is an adaptive learning optimization method that limits the quick diminishing of learning rates, similar to RMSProp.
In addition, unlike RMSProp or AdaGrad, Adam also keeps a decaying average of nonsquared gradients ($s$ in the equation below), which can be seen to serve a purpose similar to momentum.

$$
\begin{aligned}
    s &= \rho_1 s + (1 - \rho_1) \left[ 
        \frac{1}{B} \sum^B_{i=1} 
        \frac{\partial L(y_i, f(x_i; w))}{\partial w} 
    \right]\\
    r &= \rho_2 r + (1 - \rho_2) \left[ 
        \frac{1}{B} \sum^B_{i=1} 
        \frac{\partial L(y_i, f(x_i; w))}{\partial w} 
    \right]^2\\
    \hat{s} &= \frac{s}{1-\rho_1}\\
    \hat{r} &= \frac{r}{1-\rho_2}\\
    w &= w - \alpha \left[ 
        \frac{\hat{s}}{\sqrt{\hat{r}} + \delta}
    \right]\\
    \alpha &\rightarrow \text{model hyperparameter}\\
    \rho_1 &\rightarrow \text{model hyperparameter}\\
    \rho_2 &\rightarrow \text{model hyperparameter}\\
    \delta &\rightarrow \text{small constant, usually }10^{-6}\\
\end{aligned}
$$

### Code

The code for AdaGrad, RMSProp, and Adam optimizers is displayed in the block below.
These classes are meant to be used in conjunction with the `Network` and `Layer` classes I describe in my earlier post, titled *Classification and Regression with Neural Networks*.
The code for `Network` and `Layer` to be used with the `Optimizer` class can be found in the following package from my github page [here](https://github.com/akshat0123/MLReview/tree/main/Packages/mlr/NN).


```python
class AdaGradOptimizer(Optimizer):
    """ AdaGrad optimizer
    """


    def __init__(self) -> None:
        """ Initialize AdaGrad optimizer
        """

        self.delta = 1e-5 
        self.r = None


    def __copy__(self):
        """ Return copy of default SGD optimizer

        Returns: 
            copy of optimizer
        """

        return AdaGradOptimizer()


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        if self.r is None: 
            self.r = torch.zeros(w.shape)

        self.r = self.r + (dw)**2
        return w - (alpha * ((dw + (lambdaa * dr)) / (self.delta + torch.sqrt(self.r))))


def RMSPropOptimizer(momentum: bool=False, rho: float=0.9, epsilon: float=1e-4) -> Optimizer:
    """ Return RMSProp optimizer

    Args:
        momentum: whether to include momentum or not
        epsilon: epsilon parameter for momentum
        rho: rho parameter for RMSProp

    Returns:
        RMSProp optimizer
    """

    optimizer = RMSPropMomentumOptimizer(rho=rho, epsilon=epsilon) if momentum else DefaultRMSPropOptimizer(rho=rho)
    return optimizer


class DefaultRMSPropOptimizer:
    """ RMSProp optimizer (without momentum)
    """


    def __init__(self, rho: float=0.9) -> None:
        """ Initialize optimizer
        """

        self.delta = 1e-5
        self.rho = rho
        self.r = None


    def __copy__(self):
        """ Return copy of DefaultRMSPropOptimizer
        """

        return DefaultRMSPropOptimizer(self.rho)


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        if self.r is None:
            self.r = torch.zeros(w.shape)

        self.r = (self.rho * self.r) + ((1 - self.rho) * (dw**2))
        return w - (alpha * ((dw + (lambdaa * dr)) / (torch.sqrt(self.delta + self.r))))


class RMSPropMomentumOptimizer:
    """ RMSProp optimizer (without momentum)
    """


    def __init__(self, rho: float=0.9, epsilon: float=1e-4):
        """ Initialize optimizer
        """

        self.epsilon = epsilon
        self.delta = 1e-05
        self.rho = rho
        self.r = None
        self.v = None


    def __copy__(self):
        """ Return copy of RMSPropMomentumOptimizer
        """

        return RMSPropMomentumOptimizer(self.rho, self.epsilon)


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        if self.r is None:
            self.r = torch.zeros(w.shape)                
            self.v = torch.zeros(w.shape)

        r = (self.rho * self.r) + ((1 - self.rho) * (dw**2))
        v = (self.epsilon * self.v) - (alpha * ((dw + (lambdaa * dr)) / (torch.sqrt(self.delta + self.r))))

        return w + v


class AdamOptimizer(ABC):
    """ Adam optimizer
    """

    def __init__(self, rho1: float=0.9, rho2: float=0.999):
        """ Initialize optimizer
        """

        self.delta = 1e-5
        self.rho1 = rho1
        self.rho2 = rho2
        self.s = None
        self.r = None


    def __copy__(self):
        """ Return copy of AdamOptimizer
        """

        return AdamOptimizer(self.rho1, self.rho2)


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        if self.s is None:
            self.s = torch.zeros(w.shape)                
            self.r = torch.zeros(w.shape)                

        self.s = (self.rho1 * self.s) + ((1 - self.rho1) * (dw + (lambdaa * dr)))
        self.r = (self.rho2 * self.r) + ((1 - self.rho2) * (dw + (lambdaa * dr))**2)
        shat = self.s / (1 - self.rho1)
        rhat = self.r / (1 - self.rho2)

        return w - (alpha * (shat / (torch.sqrt(rhat) + self.delta)))
```

### Resources

- Goodfellow, Ian, et al. *Deep Learning*. MIT Press, 2017.
