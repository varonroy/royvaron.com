---
layout: ../../../layouts/ArticleLayout.astro
title: Deriving Back Propagation in Matrix Form
sub_title: Article
author: Roy Varon Weinryb
email: varonroy@gmail.com
date: 15/12/2021
category: ai
---

## Introduction

The forward propagation process of neural networks can be very concisely written using a series of
Matrix-Vector multiplications.
The same can be said about the back propagation process. In this article
I will show step by step how derivatives propagate through Matrix-Vector multiplications and how to use
these results in
a neural network.

## Forward Propagation

Lets consider a portion of a neural network: two layers and the weights between, and an activation
function $f$.
Let the first layer have $n$ neurons and the second one $m$ neurons. Since these two layers are
nothing more than arrays of numbers, they can be represented
using vectors:

$$
\begin{gather}
\text{first layer}: & v^1 \in \mathbb{R}^n \\
\text{second layer}:&v^2 \in \mathbb{R}^m \\
\end{gather}
$$

Let $W_{i,j}$ be the weight connecting the $ith$ neuron of the second layer and the
$jth$ neuron of the first layer. This way, the forward propagation for each of the individual neurons
can be written as:
$$v^2_i = f\Big(\sum_{k = 1}^{n}W_{i, k} {v^1_k}\Big)$$
Or more compactly:
$$v^2 = f(W v^1) $$
To simplify things, lets split the above step into two:
$$s^2 = W v^1 $$
$$v^2 = f(s^2) $$

So far we have:
$$v^1 \in \mathbb{R}^n $$
$$s^2 \in \mathbb{R}^m $$
$$v^2 \in \mathbb{R}^m $$
$$W \in \mathbb{R}^{m \times n} $$
$$ $$
$$s^2 = W v^1 $$
$$v^2 = f(s^2) $$

## Back Propagation

Let the derivative of the second layer with respect to the loss function be given:
$$\frac{\partial L}{\partial v^2}$$
So the partial derivative of the loss with respect to the $ith$ neuron in the second layer will be
the $ith$ element of the above vector:
$$\frac{\partial L}{\partial v^2_i} = \Big(\frac{\partial L}{\partial v^2}\Big)_i$$

Going a step back, let's write how the $ith$ element of $s^2$ is propagation:
$$ v^2_i = f(s^2_i) $$

Then by taking the derivative:
$$ \frac{\partial v^2_i}{\partial s^2_i} = f'(s^2_i) $$
We can write the following vector:
$$ \frac{\partial v^2}{\partial s^2} = f'(s^2) $$
And apply the chain rule:

$$
\frac{\partial L}{\partial s^2_i} = \frac{\partial L}{\partial v^2_i} \frac{\partial v^2_i}{\partial
s^2_i} = \frac{\partial L}{\partial v^2_i} f'(s^2)
$$

And in vector form:
$$ \frac{\partial L}{\partial s^2} = \frac{\partial L}{\partial v^2} \circ f'(s^2) $$
(where $\circ$ represents element wise multiplication).

Next, the propagation towards $v^1$ will be more complicated, because unlike there is no longer a one
to one relation
between elements in $v^1$ and $s^2$. Each element of $ v^1 $ depends on each element of $ s^2 $ in
the following way:

$$ s^2_i = \sum*{k = 1}^{n} W*{i, k} {v^1_k} $$

So by taking the derivative of the $ith$ element of $s^2$ with respect to the $jth$ element of
$v^1$ we get:
$$\frac{\partial s^2_i}{\partial v^1_j} = W_{i, j}$$
This means that when we apply the chain rule to get $\partial L / \partial v^1_i $, we need to consider
how every element
of $s_2$ will impact the total derivative:

$$ \frac{\partial L}{\partial v^1*i} = \sum\_{j = 0}^{m}\frac{\partial L}{\partial s^2*j} \frac{\partial s^2*j}{\partial v^1_i} = \sum*{j = 0}^{m}\frac{\partial L}{\partial s^2*j} W*{j, i} $$

Notice how the indices in $W$ are flipped. We can flip the indices back by transposing the matrix

$$
\frac{\partial L}{\partial v^1_i} = \sum_{j = 0}^{m}\frac{\partial L}{\partial s^2_j} (W^T)_{i, j} = \sum_{j=0}^{m} (W^T)_{i, j} \frac{\partial L}{\partial s^2_j}
$$

So the $i^{th}$ element of $\partial L / \partial v^1_i$ equals to the dot product of the $ith$ row
of matrix $W^T$ with the vector $\partial L / \partial s^2_j$. This is exactly what matrix
multiplication does, so we can write:
$$ \frac{\partial L}{\partial v^1} = W^T \frac{\partial L}{\partial s^2}$$

Finally, we need to calculate the derivatives with respect to each of the weights in the matrix $W$.
Using the above forward propagation formula:
$$s^2_i = \sum_{k = 1}^{n}W_{i, k} {v^1_k} $$
We can find the partial derivative:
$$ \frac{\partial s^2*i}{\partial W*{i, j}} = v^1_j$$
Notice that the index of the element of $s^2$ has to match the index of the row of $W$. Since only
the $ith$ row has an effect on that element.

And now by applying the chain rule:

$$
\frac{\partial L}{\partial W_{i, j}} = \frac{\partial L}{\partial s^2_i} \frac{\partial
s^2_i}{\partial W_{i, j}} = \frac{\partial L}{\partial s^2_i} v^1_j
$$

It is a bit hard to see at first, but we can rewrite this expression as a multiplication of a column
vector by a row vector:
$$ \frac{\partial L}{\partial W} = \frac{\partial L}{\partial s^2} (v^1)^T $$

Here is a small example of how that multiplication would work:

$$
\begin{bmatrix} a_1 a_2 \end{bmatrix}
\begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix} = \begin{bmatrix}
a_1b_1 & a_1b_2 & a_1b_3 \\\\
a_2b_1 & a_2b_2 & a_2b_3 \\\\
\end{bmatrix}
$$

## Summary

To Summarize, given the following structure of a neural network:
$$v^1 \in \mathbb{R}^n $$
$$s^2 \in \mathbb{R}^m $$
$$v^2 \in \mathbb{R}^m $$
$$W \in \mathbb{R}^{m \times n} $$
$$ $$
$$s^2 = W v^1 $$
$$v^2 = f(s^2) $$

And the derivative of the loss function with respect to the last layer:
$$\frac{\partial L}{\partial v^2}$$

We can back propagate the derivative with the following formulas:
$$ \frac{\partial L}{\partial s^2} = \frac{\partial L}{\partial v^2} \circ f'(s^2) $$
$$ \frac{\partial L}{\partial v^1} = W^T \frac{\partial L}{\partial s^2}$$
$$ \frac{\partial L}{\partial W} = \frac{\partial L}{\partial s^2} (v^1)^T $$
