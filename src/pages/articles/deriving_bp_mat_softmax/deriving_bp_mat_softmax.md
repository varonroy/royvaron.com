---
layout: ../../../layouts/ArticleLayout.astro
title: Deriving the Back Propagation of the Softmax Function in Matrix Form
author: Roy Varon Weinryb
email: varonroy@gmail.com
date: 16/12/2021
category: ai
---

## Introduction

Softmax is a very common function in machine learning. It takes a set of real numbers, and converts them
to a probability distribution.
$$ \sigma(v)\_i = \frac{e^{v_i}}{\sum e^{v_j}} $$
This article shows how the chain rule can be applied to propagate back derivatives though this function.

## Forward Propagation

Consider the following vectors:
$$ v^1 \in \mathbb{R}^n $$
$$ v^2 \in \mathbb{R}^n $$
Which are related using the softmax function:
$$ v^2*i = \sigma(v^1)\_i = \frac{ \exp {v^1_i}}{\sum*{k=0}^{n} \exp{v^1_k} } $$
An important observation is that unlike other common functions in machine learning like $tanh$, every
element in $v^1$ is dependent on every element in $v^2$.

## Back Propagation

Let the derivative of every element of $v^2$ with respect to some function $L$ be given.
$$\frac{\partial L}{\partial v^2}$$
$$\frac{\partial L}{\partial v^2_j} = \Big(\frac{\partial L}{\partial v^2}\Big)_j$$

Our goal is to find the derivative of $L$ with respect to $v^1_i$. However, since all element in
both vectors
are related to each other, we need to consider every possible path from $v^1_i$ to $L$ and then sum
up the derivatives.

$$
\frac{\partial L}{\partial v^1_i} = \sum_{j = 0}^{n} \frac{\partial L}{\partial v^2_j} \frac{\partial
v^2_j}{\partial v^1_i}
$$

Since $ \partial L / \partial v^2_j$ are given, we only need to calculate $ \partial v^2_j / \partial
v^1_i$. There are two cases
that need to be considered:

$i = j:$

$$
\begin{align*}
\frac{\partial v^2_j}{\partial v^1_i} = \frac{\partial v^2_i}{\partial v^1_i}
&= \frac{\partial}{\partial v^1_i}\Big( \frac{ \exp {v^1_i}} {\sum_{k=0}^{n} \exp{v^1_k} } \Big)  \\

&= \frac{ \exp {v^1_i} \sum_{k=0}^{n} \exp{v^1_k} - \exp {v^1_i} \exp {v^1_i}} {(\sum_{k=0}^{n} \exp{v^1_k})^2}  \\

&= \exp {v^1_i} \frac{ \sum_{k=0}^{n} \exp{v^1_k} - \exp {v^1_i}} {(\sum_{k=0}^{n} \exp{v^1_k})^2}  \\

&= \frac{\exp{v^1_i}}{\sum_{k=0}^{n} \exp{v^1_k}} \frac{ \sum_{k=0}^{n} \exp{v^1_k} - \exp {v^1_i}} {\sum_{k=0}^{n} \exp{v^1_k}}  \\

&= \frac{\exp{v^1_i}}{\sum_{k=0}^{n} \exp{v^1_k}} \Big(1-\frac{\exp {v^1_i}} {\sum_{k=0}^{n} \exp{v^1_k}}\Big) \\
\end{align*}
$$

Where

$$
\begin{align*}
\sigma_i &= \sigma(v^1)_i = \frac{\exp{v^1_i}}{\sum_{k=0}^{n} \exp{v^1_k}} \\
\frac{\partial v^2_i}{\partial v^1_i}&= \sigma_i(1-\sigma_i)
\end{align*}
$$

$i \ne j:$

$$
\begin{align*}
\frac{\partial v^2_j}{\partial v^1_i}
&= \frac{\partial}{\partial v^1*i}\Big( \frac{ \exp {v^1_j}} {\sum*{k=0}^{n} \exp{v^1*k} } \Big)
\\
&= \frac{ 0 - \exp{v^1_j}\exp{v^1_i}} {(\sum*{k=0}^{n} \exp{v^1*k})^2}
\\
&= -\frac{\exp{v^1_j}}{\sum*{k=0}^{n} \exp{v^1*k}} \frac{\exp{v^1_i}} {\sum*{k=0}^{n} \exp{v^1_k}}
\\
&= -\sigma_j \sigma_i
\end{align*}
$$

To summarize:

$$
\frac{\partial v^2_j}{\partial v^1_i} =
\begin{equation}
\begin{cases}
\sigma_i(1-\sigma_i) && i = j \\\\
-\sigma_j \sigma_i && else \\\\
\end{cases}
\end{equation}
$$

Where

$$ \sigma*i=\frac{\exp{v^1_i}} {\sum*{k=0}^{n} \exp{v^1_k}} $$

Now, notice that the above equation for the total derivative, can be written as matrix multiplication:

$$
\frac{\partial L}{\partial v^1_i} = \sum_{j = 0}^{n} \frac{\partial L}{\partial v^2_j} \frac{\partial v^2_j}{\partial v^1_i} = \sum_{j = 0}^{n} \frac{\partial L}{\partial v^2_j} Q_{i,j}
$$

Such that:

$$
\frac{\partial L}{\partial v^1} = Q\frac{\partial L}{\partial v^2}
$$

$$
Q_{i, j} =
\begin{equation}
\begin{cases}
\sigma_i(1-\sigma_i) && i = j \\
-\sigma_j \sigma_i && else\\
\end{cases}
\end{equation}
$$

Notice that this matrix is symmetric $Q_{i, j} = Q_{j, i}$. This fact can be used to boost
performance.
