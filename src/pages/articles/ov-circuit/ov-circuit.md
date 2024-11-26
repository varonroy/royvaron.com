---
layout: ../../../layouts/ArticleLayout.astro
title: OV Circuits
sub_title: Understanding the OV circuit from the famous "A Mathematical Framework for Transformer Circuits" paper.
author: Roy Varon Weinryb
email: varonroy@gmail.com
date: 24/11/2024
category: ai
---

This article focuses on the OV circuits abstraction from the famous paper [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). It presents the neccessary background, fundamental assumptions and derivations.

### Motivation
The residual stream is linear. It simply accumulates the outputs of all the layers. Let $X_0$ be the token embeddings after the embedding layer and before the transformer blocks. Let `layer_i` denote the `ith` transformer block. In general, $X_i$ is going to denote the residual stream's value before the `ith` layer (that is, the input to the `ith` layer is `X_i`). We can also consider the output of each layer as adding some `delta` to the residual stream.

$$
X_{i+1} = X_i + \text{layer}(X_i) = X_{i} + \Delta_i
$$

Under this formalization, the residual stream becomes the sum of the original embeddings and the delta terms.

$$
X_i = X_0 + \Delta_0 + \cdots + \Delta_{i-1}
$$

However, notice that the operation that each layer performs is not linear, meaning that for any arbitrary layer and arbitrary vectors $X_A$ and $X_B$:
$$
\text{layer}(X_A + X_B) \ne \text{layer}(X_A) + \text{layer}(X_B)
$$

Therefore, we cannot "see" the effect that the `ith` layer has on the output:
$$
\text{layer}(X_0 + \Delta_0 + \cdots ) \ne \text{layer}(X_0) + \text{layer}(\Delta_0) + \cdots
$$

This is unfortunate since we cannot apply the circuit analysis from [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) on general transformers with an arbitrary number of layers. However, we can still apply their methodologies on a transformer with a single layer.

Consider A single matrix from the residual stream $X\in\mathbb{R}^{s\times d}$ (where $s$ is the number of tokens and $d$ is the dimension of the residual stream). Then, the layer will first apply the `QKV` matrices to calculate the queries, keys and values.
$$
\begin{gathered}
    X & \in \mathbb{R}^{s\times d} \\
    W_Q,W_K,W_V & \in \mathbb{R}^{d\times d_h} \\
    Q  = XW_Q & \in \mathbb{R}^{s\times d_h} \\
    K  = XW_K & \in \mathbb{R}^{s\times d_h} \\
    V  = XW_V & \in \mathbb{R}^{s\times d_h} \\
\end{gathered}
$$

> Note: $d_h$ is the dimension inside the head.

Then, the attention scores are calculated using the queries and keys.
$$
A=\text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}, \text{over the rows}\right) \in \mathbb{R}^{s\times s}
$$

$$
QK^T=
\begin{bmatrix}
    \text{---} & q_{1} & \text{---} \\
    \\
    \\
\end{bmatrix}
\left(
\begin{bmatrix}
    \\
    \text{---} & k_{2} & \text{---} \\
    \\
\end{bmatrix}
\right)^T
=
\begin{bmatrix}
     & q_1k_2^T &  \\
     \\
     \\
\end{bmatrix}\in \mathbb{R}^{s\times s}
$$

The element $a_{ij}$ in the attention matrix corresponds to query $i$ and key $j$. Or, from the value's point of view, the elements $a_{ij}$ specifiy the fraction of $V_{j:}$ (that is, the `jth` value) of the output of token $i$.

$$
AV=
\begin{bmatrix}
    \text{---} & a_{1} & \text{---} \\
    \\
    \\
\end{bmatrix}
\begin{bmatrix}
    \text{---} & v_{1} & \text{---} \\
    \text{---} & v_{2} & \text{---} \\
    \text{---} & v_{3} & \text{---} \\
\end{bmatrix}
=
\begin{bmatrix}
    \text{---} & o_{1} & \text{---} \\
    \\
    \\
\end{bmatrix}
=O \in\mathbb{R}^{s\times d_h}
$$

Then, the layer applies another linear transformation in order to map the dimension inside the head $d_h$ to the model's dimension $d$.

$$
\begin{aligned}
    W_O & \in \mathbb{R}^{d_h\times d} \\
    O W_O & = \text{Head Output} & \in \mathbb{R}^{s\times d} \\
\end{aligned}
$$

By looking at the above computations step by step, and putting aside the computation of the attention, we can describe the process by:
$$
\begin{align*}
    V & \gets XW_V \\
    O & \gets AV \\
    O_\text{layer} & \gets O W_O  \\
\end{align*}
$$

The first line applies the values transformation, the second applies the attention, and the third maps the head's output to the residual stream's dimension. Writing all the equations together:
$$
O_\text{layer} = (A(XW_V))W_O = AXW_VW_O
$$

The key insight here is that the attention mechanism (besides multiplying by the attention on the left), multiplies the tokens matrix on the right by $W_VW_O$. This is important, because multiplying two matrices one after another is the same as multiplying both matrices together and then applying the transformation with a single matrix.
$$
AXW_VW_O=AXW_{OV}
$$

### The Circuit
The OV circuit is an abstraction that helps us analyze the effect a single attention head has on the logits of a given token. It allows us to take the above analysis and extend to a full network. This is done by making two simplifying assumptions:
- The attention matrices $A$ are considered to be fixed. Mathematically, we will no longer consider $A$ to be a function of $X$, but a constant.
- The input (embedding) and output (un-embedding) layers are also assumed to be linear. Specifically, they are defined by two matrices $W_E$ (the embedding matrix) and $W_U$ (the un-embedding) matrix.

$$
\begin{aligned}
    X_0 & = EW_E \\
    Y & = X_{n}W_U \\
\end{aligned}
$$

Where $E\in\mathbb{R}^{s\times v}$, is a matrix containing one-hot vectors with length $v$ (the size of the vocabulary). $W_E\in\mathbb{R}^{v\times d}$ is the embedding matrix. $Y\in\mathbb{R}^{s\times v}$ is the matrix containing the output logits of the network. $W_U\in\mathbb{R}^{d\times v}$ is the un-embedding matrix.

By making these assumptions, the entire transformer can be described using the following two equations:

$$
\begin{aligned}
    Y & = (EW_E + \Delta_0+\cdots+\Delta_n)W_U \\
    \Delta_i & = A_i \left( EW_E + \sum_{j=0}^{i-1} \Delta_{j} \right) W_{{OV}_i} \\
\end{aligned}
$$

By expanding the parenthesis we get:
$$
\begin{aligned}
    Y & = EW_EW_U + \Delta_0W_U+\cdots+\Delta_nW_U \\
    \Delta_i & =
 A_i EW_EW_{{OV}_i} + A_i \Delta_{0}W_{{OV}_i} + A_i \Delta_{1}W_{{OV}_i} + \cdots + A_i \Delta_{i-1}W_{{OV}_i} \\
\end{aligned}
$$

The goal of the OV circuit is to analyze the effect the embedded tokens ($E$) have on the logits when passing through a specific atttention head. Mathematically, it means isolating all the terms with the embedding tokens and a specific layer.

$$
\begin{aligned}
    Y'_i & = \Delta_i'W_U \\
    \Delta_i' & = A_i EW_EW_{{OV}_i} \\
\end{aligned}
$$

Then, putting these two equations together, we get:

$$
\boxed{Y'_i = A_i EW_EW_{{OV}_i}W_U}
$$
- $A_i$ - the fixed attention scores of head $i$.
- $EW_E$ - the embedded input tokens.
- $W_{OV_i}$ - composition of the value and output projection matrices for head $i$.
- $W_U$ - the un-embedding matrix.

This equation describes the OV circuit.

> Note: In the literature, the OV circuit is commonly denoted as $W_UW_{OV}W_{E}$. However it is important to note that this is just a label, it doesn't literally mean the multiplication of three matrices. The equation for the circuit is the one presented above.
