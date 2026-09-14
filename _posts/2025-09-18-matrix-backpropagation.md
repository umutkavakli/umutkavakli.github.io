---
layout: distill
title: Backpropagation Through Matrix Multiplication
date: 2025-09-18 12:05:00-0100
description: A derivation of backpropagation through matrix multiplication, from scalar intuition to batch-wise gradient formulas.
toc:
  - name: Forward Propagation
  - name: Simple Computation Graph
  - name: Backward Propagation
  - name: The Pattern
---

Backpropagation appears quite straightforward when working with scalars or even simple vectors. However, once we step into the world of matrices, things quickly become more complex and difficult to follow. There are extra details and notations that make it less intuitive. 

Personally, although I managed to understand this concept while preparing for deep learning exams, I usually lose my intuition a few months later. Reviewing it again requires extra effort to bring everything back together. That's why I decided to write this technical blog post, both as a reminder for myself and as a guide for anyone trying to understand backpropagation in matrix form.

## Forward Propagation

When training a deep learning model, the first step is to compute a linear combination of input features. Given an input vector $x \in \mathbb{R}^M$ with $M$ features and a corresponding weight vector $w \in \mathbb{R}^M$, we calculate a weighted sum and add a bias term $b \in \mathbb{R}^1$ to produce a single output $z$:

$$
z = w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_M \cdot x_M + b
$$

This is simply the dot product between $w$ and $x$ with an added bias term:

$$
z = w^T x + b 
$$

To visualize this computation, we can represent it in matrix form:

$$
\begin{bmatrix}
z
\end{bmatrix}
=
\begin{bmatrix}
w_1 & w_2 & \cdots & w_M
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_M
\end{bmatrix}
+
\begin{bmatrix}
b
\end{bmatrix}
$$

or in image representation:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net1.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto;" alt="Net1">
</d-figure>

We can simplify our notation by incorporating the bias term directly into the weight vector. This involves adding the bias $b$ as the first element of the weight vector and including a constant input $x_0 = 1$:

$$
\begin{bmatrix}
z
\end{bmatrix}
=
\begin{bmatrix}
b & w_1 & w_2 & \cdots & w_M 
\end{bmatrix}
\cdot
\begin{bmatrix}
1 \\
x_1 \\
x_2 \\
\vdots \\
x_M
\end{bmatrix}
$$

For notational clarity, we'll denote the bias term as $w_0 = b$ and the constant input as $x_0 = 1$. Under this convention, both vectors have dimension $M+1$: $w \in \mathbb{R}^{M+1}$ and $x \in \mathbb{R}^{M+1}$. Our equation becomes:

$$
z = w^T x 
$$

or in matrix form:

$$
\begin{bmatrix}
z
\end{bmatrix}
=
\begin{bmatrix}
w_0 & w_1 & w_2 & \cdots & w_M 
\end{bmatrix}
\cdot
\begin{bmatrix}
x_0 \\
x_1 \\
x_2 \\
\vdots \\
x_M
\end{bmatrix}
$$

or in image representation:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net2.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto;" alt="Net2">
</d-figure>

### Adding Non-Linearity

The linear combination alone is insufficient for learning complex patterns. To add non-linearity, we apply an **activation function** $\sigma(\cdot)$ to the output $z$.

$$
a =  \sigma(z) =  \sigma(w^Tx)
$$

Although $\sigma$ often denotes the [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function, here it represents a general activation function. For this explanation, we'll use the [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) activation function in hidden layers due to its simplicity, computational efficiency, resistance to vanishing gradients, and widespread popularity:

$$
a = \sigma(z) = \max(0, z)
$$

we can show this with visual representation:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net3.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto;" alt="Net3">
</d-figure>

### Multi-class Classification

In practice, deep learning models often solve multi-class problems where we need to predict one of $C$ possible classes. This requires the output $a$ to be $C$-dimensional vector, with each dimension representing the score for a particular class. The predicted class corresponds to the entry with the highest activation value.

To generate $C$ outputs, we need $C$ distinct weight vectors, each of dimension $(M+1)$. To achieve this, we need a separate weight vector for each class. Stacking them gives a weight matrix $W \in \mathbb{R}^{C \times (M+1)}$:

$$
\underbrace{a}_{[C \times 1]} = \sigma \left( \underbrace{W}_{[C \times (M+1)]} \quad \underbrace{x}_{[(M+1) \times 1]} \right)
$$

or in expanded form:

$$
\begin{bmatrix} 
a_{1} \\
a_{2} \\ 
\vdots \\
a_{C} 
\end{bmatrix}
=
\sigma \left(
\begin{bmatrix}
W_{10} & W_{11} & W_{12} & \dots & W_{1M} \\ 
W_{20} & W_{21} & W_{22} & \dots & W_{2M} \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
W_{C0} & W_{C1} & W_{C2} & \dots & W_{CM} 
\end{bmatrix}

\begin{bmatrix}
x_{0} \\
x_{1} \\ 
x_{2} \\
\vdots \\ 
x_{M} 
\end{bmatrix}
\right)
$$

or in visual form:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net4.png" style="width: 100%; display: block; margin-left: auto; margin-right: auto;" alt="Net4">
</d-figure>

Each figure represents the same network, but highlights different output paths from the same inputs, and each path is computed by its own set of output weights.

### Batch Processing

The computations described so far process only a single input example (batch size = 1). In practice, we process multiple inputs in parallel to improve computational efficiency.

If we process batch size of $N$ inputs simultaneously, then our input becomes $X \in \mathbb{R}^{N \times (M+1)}$ (note the uppercase $X$ since we now have a matrix rather than a vector). To handle $N$ outputs while maintaining proper matrix dimensions, we transpose our weight matrix to $W \in \mathbb{R}^{(M+1) \times C}$:

$$
\underbrace{A}_{[N \times C]} = \sigma\left(\underbrace{X}_{[N \times (M+1)]} \quad \underbrace{W}_{[(M+1) \times C]}\right)
$$

or in matrix form:

$$
\begin{bmatrix}
A_{11} & A_{12} & \dots & A_{1C} \\ 
A_{21} & A_{22} & \dots & A_{2C} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
A_{N1} & A_{N2} & \dots & A_{NC} \\ 
\end{bmatrix}
=
\sigma\left(
\begin{bmatrix}
X_{10} & X_{11} & \dots & X_{1M} \\
X_{20} & X_{21} & \dots & X_{2M} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
X_{N0} & X_{N1} & \dots & X_{NM} \\ 
\end{bmatrix}

\begin{bmatrix}
W_{01} & W_{02} & \dots & W_{0C} \\ 
W_{11} & W_{12} & \dots & W_{1C} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
W_{M1} & W_{M2} & \dots & W_{MC} 
\end{bmatrix}
\right)
$$

When dealing with batches, it can sometimes be hard to picture how forward propagation works. We can visualize the process like this:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net5.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto;" alt="Net5">
</d-figure>

Note that in this illustration, the non-linearity (activation function) is not explicitly shown. The final output $A$ actually corresponds to the activated values. 

### Multi-layer Networks

The calculations presented so far describe a single layer. Deep learning models stack multiple layers to learn increasingly complex representations. We can denote each layer with a superscript $[l]$ to represent the layer number:

$$
\begin{align*} 
\underbrace{Z^{[l]}}_{[N \times H_{l}]} &= \underbrace{A^{[l-1]}}_{[N \times H_{l-1}]} \quad \underbrace{W^{[l]}}_{[H_{l-1} \times H_{l}]} \\ 
\underbrace{A^{[l]}}_{[N \times H_{l}]} &= \underbrace{\sigma(Z^{[l]})}_{[N \times H_{l}]}
\end{align*}
$$

Here $H_l$ represents the number of hidden units in layer $l$. The input to layer $l$ activated output from the previous layer $l-1$. For simplicity, we usually define $A^{[0]} = X$ (the input layer).

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net6.png" style="width: 95%; display: block; margin-left: auto; margin-right: auto;" alt="Net6">
</d-figure>

### Softmax for Classification

For classification problems, we apply the **softmax** activation function to the final layer's output to convert the raw scores into a probability distribution. The softmax function ensures that all outputs sum to 1, allowing us to interpret them as class probabilities.

The softmax function is applied row-wise (for each input example) across the $C$ classes:

$$
\hat{Y}_{ij} = \text{softmax}(Z_{ij}) = \frac{e^{Z_{ij}}}{\sum_{l=1}^{C} e^{Z_{il}}} \quad\quad i = 1,2,\dots,N \quad \text{ and } \quad j =1,2,\dots,C
$$

Intuitively, for input example $i$, this computes the probability of class $j$ by normalizing the exponential of its score by the sum of exponentials across all $C$ possible classes. 

The matrix representation shows this row-wise operation:

$$
\begin{bmatrix}
\hat{Y}_{11} & \hat{Y}_{12} & \dots & \hat{Y}_{1C} \\ 
\hat{Y}_{21} & \hat{Y}_{22} & \dots & \hat{Y}_{2C} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\hat{Y}_{N1} & \hat{Y}_{N2} & \dots & \hat{Y}_{NC} \\ 
\end{bmatrix}
=
\begin{bmatrix}
\text{softmax}\big( & Z_{11} & Z_{12} & \dots & Z_{1C} & \big) \\ 
\text{softmax}\big( & Z_{21} & Z_{22} & \dots & Z_{2C} & \big) \\ 
& \vdots & \vdots & \ddots & \vdots \\ 
\text{softmax}\big( & Z_{N1} & Z_{N2} & \dots & Z_{NC} & \big) \\ 
\end{bmatrix}
=
\begin{bmatrix}
\frac{e^{Z_{11}}}{\sum_{l=1}^{C} e^{Z_{1l}}} & \frac{e^{Z_{12}}}{\sum_{l=1}^{C} e^{Z_{1l}}} & \dots & \frac{e^{Z_{1C}}}{\sum_{l=1}^{C} e^{Z_{1l}}} \\
& & & \\
\frac{e^{Z_{21}}}{\sum_{l=1}^{C} e^{Z_{2l}}} & \frac{e^{Z_{22}}}{\sum_{l=1}^{C} e^{Z_{2l}}} & \dots & \frac{e^{Z_{2C}}}{\sum_{l=1}^{C} e^{Z_{2l}}} \\
\vdots & \vdots & \ddots & \vdots \\
& & & \\ 
\frac{e^{Z_{N1}}}{\sum_{l=1}^{C} e^{Z_{Nl}}} & \frac{e^{Z_{N2}}}{\sum_{l=1}^{C} e^{Z_{Nl}}} & \dots & \frac{e^{Z_{NC}}}{\sum_{l=1}^{C} e^{Z_{Nl}}} \\
\end{bmatrix}
$$

If we denote the last layer as $L$, then the model’s output for $N$ inputs and $C$ classes can be represented as follows: 

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net7.png" style="width: 95%; display: block; margin-left: auto; margin-right: auto;" alt="Net7">
</d-figure>

### Cross-Entropy Loss Function

After obtaining the predicted probabilities $\hat{Y}$, we measure the model's performance by comparing these predictions with the ground truth labels $Y \in \mathbb{R}^{N \times C}$ using a **loss function**. The ground truth is represented as one-hot encoded vectors, where each input example has exactly one correct class.

For multi-class classification, we typically use **Cross-Entropy (CE)** loss:

$$
\mathcal{L} = \text{CE}(Y, \hat{Y}) = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^C  Y_{ij} \log\hat{Y}_{ij} 
$$

This formula computes the element-wise product between the true labels $Y$ and the logarithm of predicted probabilities $\log\hat{Y}$ then averages across all examples to produce a single scalar loss value.

We can express this using matrix operations with the element-wise (Hadamard) product $\odot$:

$$
\underbrace{\mathcal{L}}_{[1 \times 1]} = \text{CE}(Y, \hat{Y}) = -\frac{1}{N} \quad \underbrace{1^{T}}_{[1 \times N]} \quad  (\underbrace{Y \odot \log\hat{Y}}_{[N \times C]}) \quad \underbrace{1}_{[C \times 1]}
$$

Expanding this matrix operation:

$$
\mathcal{L} =
-\frac{1}{N} 
\begin{bmatrix}
1_{1} & 1_{2} & \dots & 1_{N} 
\end{bmatrix}
\left(\begin{bmatrix}
Y_{11} & Y_{12} & \dots & Y_{1C} \\ 
Y_{21} & Y_{22} & \dots & Y_{2C} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
Y_{N1} & Y_{N2}& \dots & Y_{NC} \\ 
\end{bmatrix}
\odot
\begin{bmatrix}
\log\hat{Y}_{11} & \log\hat{Y}_{12} & \dots & \log\hat{Y}_{1C} \\ 
\log\hat{Y}_{21} & \log\hat{Y}_{22} & \dots & \log\hat{Y}_{2C} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\log\hat{Y}_{N1} & \log\hat{Y}_{N2} & \dots & \log\hat{Y}_{NC} \\ 
\end{bmatrix}\right)
\begin{bmatrix}
1_{1} \\ 
1_{2} \\ 
\vdots \\ 
1_{C} 
\end{bmatrix}
$$

We can simplify this by recognizing that multiplying by vectors of ones simply sums all elements in the matrix. Therefore:

$$
\mathcal{L} =
-\frac{1}{N} \ 
\text{sum}\left(\begin{bmatrix}
Y_{11}\log\hat{Y}_{11} & Y_{12}\log\hat{Y}_{12} & \dots & Y_{1C}\log\hat{Y}_{1C} \\ 
Y_{21}\log\hat{Y}_{21} & Y_{22}\log\hat{Y}_{22} & \dots & Y_{2C}\log\hat{Y}_{2C} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
Y_{N1}\log\hat{Y}_{N1} & Y_{N2}\log\hat{Y}_{N2} & \dots & Y_{NC}\log\hat{Y}_{NC} \\ 
\end{bmatrix}\right)
$$

We can show this summation with following image (excluding scaling factor $-\frac{1}{N}$):

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net8.png" style="width: 90%; display: block; margin-left: auto; margin-right: auto;" alt="Net8">
</d-figure>

Although this is a general representation, we denote the output cross-entropy loss as $-\log\hat{Y}$. Since the ground truth $Y$ is one-hot vector, it is 1 for true class and 0 for others in each input. Therefore, the assumption in this case can be: 

$$
\mathcal{L}_{ij} =
\begin{cases}
-\log\hat{Y}_{ij} & \text{if } \ Y_{ij} = 1 \\
0 & \text{otherwise}
\end{cases} 
\quad\quad\text{for } \ i=1,2,\dots, N
$$

This completes our mathematical framework for forward propagation in deep learning models, from single predictions to batch processing with multi-class classification and loss computation.

## Simple Computation Graph

Before diving lots of matrix gradient calculation, I would like to show a computation graph for forward and backward propagation so we can see the cases we need to be careful when we compute gradients. In this case, I want to only show scalar calculations first instead of thinking about matrices so we can adapt this into matrix backpropagation.

$$

\begin{array}{rcl}
a^{[0]} & = & x \\
z^{[1]} & = & a^{[0]} \cdot w^{[1]} & \\
a^{[1]} & = &  \sigma(z^{[1]}) \\
z^{[2]} & = & a^{[1]} \cdot w^{[2]} & \\
a^{[2]} & = &  \sigma(z^{[2]}) \\
\vdots \\
z^{[l]} & = & a^{[l-1]} \cdot w^{[l]} & \\
a^{[l]} & = &  \sigma(z^{[l]}) \\
\vdots \\
\\
z^{[L-1]} & = & a^{[L-2]} \cdot w^{[L-1]} & \\
a^{[L-1]} & = & \sigma(z^{[L-1]}) \\
z^{[L]} & = & a^{[L-1]} \cdot w^{[L]} & \\
\hat{y} & = &  \text{softmax}(z^{[L]}) \\
\mathcal{L} & = & \text{CE}(\hat{y}) & = -\sum^{C}_{i=1} y_i \log \hat{y}_i
\end{array}
$$

where $[l]$ and $[L]$ represents arbitrary layer $l$ and last layer $L$, respectively. $\sigma(\cdot)$ is activation function (ReLU in this case). In this representation, everything is scalar except $w^{[L]}$ because the output must be multi-class for softmax so I just played in the last layer to show a meaningful example. 

We can visualize the computation graph of this network as follows:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net9.png" style="width: 95%; display: block; margin-left: auto; margin-right: auto;" alt="Net9">
</d-figure>

### Chain Rule

Backpropagation relies on the chain rule of calculus to compute gradients efficiently. For a composite function $f(g(h(x)))$, the chain rule states:

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}
$$

In neural networks, we apply this principle to decompose the loss gradient into a product of simpler derivatives based on any parameter.

### Scalar Backward Propagation

After computing loss, we can start computing gradients with respect to it. Since there are $C$ classes, we need to calculate derivative of each prediction $\hat{y}_i$:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{\partial \left( -y_i \log\hat{y}_i \right)}{\partial \hat{y}_i}  = -\frac{y_i}{\hat{y}_i}
$$

We can visualize this gradient flow like this: 

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net10.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto;" alt="Net10">
</d-figure>

Since the predicted output $\hat{y}_i$ is computed using $z^{[L]}_i$ in both the numerator and the denominator

$$
y_i = \text{softmax}(z^{[L]}_i) = \frac{e^{z^{[L]}_i}}{\sum^{C}_{j=1} e^{z^{[L]}_j}},
$$

the derivative $\frac{\partial \mathcal{L}}{\partial z^{[L]}_k}$ depends not only on the numerator but also on the **denominator**:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}_k} = \sum_{i=1}^{C} \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z^{[L]}_k}
$$

It might be easier to understand when you see visual representation:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net11.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto;" alt="Net11">
</d-figure>

Since we already know $\frac{\partial \mathcal{L}}{\partial \hat{y}_i}$, the gradient expression simplifies to:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}_k} = \sum_{i=1}^{C} -\frac{y_i}{\hat{y}_i} \cdot \frac{\partial \hat{y_i}}{\partial z^{[L]}_k}
$$

The most confusing part begins here, because we now need to carefully compute the local gradient $\frac{\partial \hat{y}_i}{\partial z^{[L]}_k}$. To do this, we apply the quotient rule:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} = \frac{\partial \left(\frac{e^{z^{[L]}_i}}{\sum_{j=1}^C e^{z^{[L]}_j}} \right)}{\partial z^{[L]}_k} = \frac{\left(\frac{\partial e^{z^{[L]}_i}}{\partial z^{[L]}_k}\right) \cdot \sum_{j=1}^C e^{z^{[L]}_j} - e^{z^{[L]}_i} \cdot \left( \frac{\sum_{j=1}^C e^{z^{[L]}_j} }{\partial z^{[L]}_k}\right)} {\left(\sum_{j=1}^C e^{z^{[L]}_j} \right)^2}
$$

For this derivative, there are two distinct cases to consider:

$$
i = k \quad \text{ or } \quad i \neq k
$$

### Case 1: $i = k$

The derivative of numerator:

$$
\frac{\partial e^{z^{[L]}_i}}{\partial z^{[L]}_k} = e^{z^{[L]}_i}
$$

The derivative of the denominator is:

$$
\frac{\partial \left(\sum_{j=1}^C e^{z^{[L]}_j} \right)}{\partial z^{[L]}_k} = e^{z^{[L]}_k}
$$

Substituting these into the quotient rule:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} = \frac{z^{[L]}_i \cdot \sum_{j=1}^{C} z^{[L]}_j - z^{[L]}_i \cdot z^{[L]}_k}{\left( \sum_{j=1}^{C} z^{[L]}_j \right)^2}
$$

Factorizing into:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} = \frac{e^{z^{[L]}_i}}{\sum^{C}_{j=1} e^{z^{[L]}_j}} \cdot \left(1 - \frac{e^{z^{[L]}_k}}{\sum^{C}_{j=1} e^{z^{[L]}_j}} \right)
$$

Shortly, this is equivalent to:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} = \hat{y}_i \cdot (1 - \hat{y}_k)
$$

### Case 2: $i \neq k$

The derivative of the numerator is **zero**, because $e^{z_i}$ does not depend on $z_k$. The derivative of the denominator is:

$$
\frac{\partial \left(\sum_{j=1}^C e^{z^{[L]}_j} \right)}{\partial z^{[L]}_k} = e^{z^{[L]}_k}
$$

Thus, we have:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} = \frac{0 \cdot \sum_{j=1}^{C} z^{[L]}_j - z^{[L]}_i \cdot z^{[L]}_k}{\left( \sum_{j=1}^{C} z^{[L]}_j \right)^2}
$$

Simplifying:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} =  -\frac{e^{z^{[L]}_i}}{\sum^{C}_{j=1} e^{z^{[L]}_j}} \cdot \frac{e^{z^{[L]}_k}}{\sum^{C}_{j=1} e^{z^{[L]}_j}}
$$

Shortly, this is equivalent to:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} = - \hat{y}_i \cdot  \hat{y}_k
$$

Finally, we can summarize these two cases as

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} =
\begin{cases}
\hat{y}_i \cdot (1 - \hat{y}_k) & \text{ if } i = k \\
-\hat{y}_i \cdot \hat{y}_k & \text{ if } i \neq k
\end{cases}
$$

In literature, it is pretty common to use the following form:

$$
\frac{\partial \hat{y}_i}{\partial z^{[L]}_k} = \hat{y}_i \cdot (\delta_{ik} - \hat{y}_k)
$$

where $\delta_{ik}$ is 1 if $i = k$, 0 otherwise. When we combine the gradient of cross-entropy and local gradient, we have:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}_k} = \sum_{i=1}^{C} -\frac{y_i}{\hat{y}_i} \cdot \hat{y}_i \cdot (\delta_{ik} - \hat{y}_k)
$$

where $\hat{y}_i$ cancels each other:

$$
\begin{array}{rcl}
\frac{\partial \mathcal{L}}{\partial z^{[L]}_k} & = & \sum_{i=1}^{C} -y_i \cdot (\delta_{ik} - \hat{y}_k) \\
& = & -y_k + \hat{y}_k \sum_{i=1}^C y_i
\end{array}
$$

Since ground truth value $y_i$ is one-hot vector (1 for true class and 0 for others), equation simplifies to:

$$
\frac{\partial \mathcal{L}}{\partial z^{[L]}_k} = \hat{y}_k - y_k
$$

This compact form is what makes softmax combined with cross-entropy loss so useful. Instead of dealing with complicated fractions, we can directly use this result to propagate gradients to the next layers.

From this point, gradient calculations across layers will start to follow a recurring pattern. Each layer essentially repeats the same process: we compute gradients with respect to its **inputs** and **its weights**.

For the layer output $z^{[L]}$, we have two key variables to differentiate with respect to:

* **The weights $w^{[L]}$:** their gradients are crucial because they are the trainable parameters we want to optimize during learning.

* **The activations $a^{[L-1]}$:** while not parameters themselves, their gradients are equally important since they ensure the flow of gradients backward through the network, enabling earlier layers to update as well.

Thus, even though only the weight gradients directly influence optimization, the activation gradients play a important role in keeping backpropagation alive throughout the entire network:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net12.png" style="width: 80%; display: block; margin-left: auto; margin-right: auto;" alt="Net12">
</d-figure>

- If we calculate derivatives with respect to weights $w^{[l]}$:

$$
\frac{\partial \mathcal{L}}{\partial w^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \frac{\partial \left( a^{[l-1]} \cdot w^{[l]} \right)}{\partial w^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot a^{[l-1]} 
$$

- If we calculate derivatives with respect to weights $a^{[l-1]}$:

$$
\frac{\partial \mathcal{L}}{\partial a^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \frac{\partial \left( a^{[l-1]} \cdot w^{[l]} \right)}{\partial a^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot w^{[l]}
$$

Lastly, we need to calculate the derivative of $a^{[l]}$ with respect to $z^{[l]}$ in $\sigma(z^{[l]})$, assuming $\sigma(z^{[l]}) = \text{ReLU}(z^{[l]})$:

$$
\frac{\partial \mathcal{L}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[l]}} \cdot 1(z^{[l]} > 0)
$$

where $1(z^{[l]} > 0)$ outputs 1 if $(z^{[l]} > 0)$, 0 otherwise as a indicator function.

Therefore, we can summarize backpropagation for scalar values with following pattern:

$$
\begin{array}{rll}
\frac{\partial \mathcal{L}}{\partial z^{[L]}} &= \hat{y}-y \\
\vdots \\
\\
\frac{\partial \mathcal{L}}{\partial z^{[l]}} &= \frac{\partial \mathcal{L}}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l]}} &= \frac{\partial \mathcal{L}}{\partial a^{[l]}} \cdot 1(z^{[l]} > 0) \\
\frac{\partial \mathcal{L}}{\partial w^{[l]}} &= \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial w^{[l]}} &= \frac{\partial \mathcal{L}}{z^{[l]}} \cdot a^{[l-1]} \\
\frac{\partial \mathcal{L}}{\partial a^{[l-1]}} &= \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial a^{[l-1]}} &= \frac{\partial \mathcal{L}}{z^{[l]}} \cdot w^{[l]} \\
\frac{\partial \mathcal{L}}{\partial z^{[l-1]}} &= \frac{\partial \mathcal{L}}{\partial a^{[l-1]}} \cdot \frac{\partial a^{[l-1]}}{\partial z^{[l-1]}} &= \frac{\partial \mathcal{L}}{\partial a^{[l-1]}} \cdot 1(z^{[l-1]} > 0) \\
\vdots \\
\frac{\partial \mathcal{L}}{\partial z^{[1]}} &= \frac{\partial \mathcal{L}}{\partial a^{[1]}} \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} &= \frac{\partial \mathcal{L}}{\partial a^{[1]}} \cdot 1(z^{[1]} > 0) \\
\frac{\partial \mathcal{L}}{\partial w^{[1]}} &= \frac{\partial \mathcal{L}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial w^{[1]}} &= \frac{\partial \mathcal{L}}{z^{[1]}} \cdot a^{[0]} \\
\end{array}
$$

## Backward Propagation

After computing the loss through forward propagation, we need to update the model's weights to minimize this loss. Backpropagation is the algorithm that computes the gradients of the loss function with respect to each parameter in the network. We'll derive these gradients step by step, working backwards from the loss to the input layer.

### Gradient of Cross-Entropy with Softmax

Starting with loss function, we calculate relative gradients and go back step by step to calculate further gradients. Since the model has predictions with Softmax activation and the loss is calculated with cross-entropy, the combination of these methods has a nice property which simplifies the gradient:

$$
\frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \frac{1}{N} \ \underbrace{(\hat{Y}-Y)}_{[N \times C]}
$$

or in matrix representation:

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial Z^{[L]}} 
&= 
\frac{1}{N}
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial Z^{[L]}_{11}} & \frac{\partial \mathcal{L}}{\partial Z^{[L]}_{12}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[L]}_{1C}} \\
\frac{\partial \mathcal{L}}{\partial Z^{[L]}_{21}} & \frac{\partial \mathcal{L}}{\partial Z^{[L]}_{22}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[L]}_{2C}} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial \mathcal{L}}{\partial Z^{[L]}_{N1}} & \frac{\partial \mathcal{L}}{\partial Z^{[L]}_{N2}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[L]}_{NC}} \\ 
\end{bmatrix} \\\\
&=
\frac{1}{N}
\begin{bmatrix}
\hat{Y}_{11} - Y_{11} & \hat{Y}_{12} - Y_{12} & \dots & \hat{Y}_{1C}  - Y_{1C} \\ 
\hat{Y}_{21} - Y_{21} & \hat{Y}_{22} - Y_{22} & \dots & \hat{Y}_{2C} - Y_{2C} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\hat{Y}_{N1} - Y_{N1} & \hat{Y}_{N2} - Y_{N2} & \dots & \hat{Y}_{NC} - Y_{NC} \\ 
\end{bmatrix}
\end{align*}
$$ 

or in visual representation:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net13.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto;" alt="Net13">
</d-figure>

### Gradient of Arbitrary Layer $l$

At this stage, we can begin calculating the gradients of the weights in the corresponding layer. Since the forward propagation is computed as

$$
Z^{[l]} = A^{[l-1]} W^{[l]}
$$

There are two possible local gradients we might consider:

$$
\frac{\partial Z^{[l]}}{\partial W^{[l]}}
\quad \text{ or } \quad
\frac{\partial Z^{[l]}}{\partial A^{[l-1]}}
$$

You may wonder why we would need the gradient with respect to $A^{[l-1]}$ since our goal is to update the weight matrix $W^{[l]}$. The reason is that the derivative with respect to $A^{[l-1]}$ becomes necessary for updating the weights of the previous layer, because

$$
A^{[l-1]} = \sigma\left(A^{[l-2]} W^{[l-1]} \right)
$$

When we want to calculate the gradients of weight matrix $W^{[l]}$ with respect to the loss function $\mathcal{L}$, we use chain rule as follows:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{\text{Upstream Gradient }} \ \cdot \  \underbrace{\frac{\partial Z^{[l]}}{\partial W^{[l]}}}_{ \text{ Local Gradient}}
$$

The local gradient gives us:

$$
\frac{\partial Z^{[l]}}{\partial W^{[l]}} = \frac{\partial \left( A^{[l-1]} W^{[l]} \right)}{\partial W^{[l]}} = \underbrace{A^{[l-1]}}_{[N \times H_{l-1}]}
$$

Since there are $H_{l-1} \times H_l$ values in weight matrix $W^{[l]}$, we need to find each individual gradient so the shape of $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$ should be $H_{l-1} \times H_l$. However, the dimensions of matrices does not match when we try to calculate matrix multiplication: 

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{[l]}}}_{[H_{l-1} \times H_{l}]} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} \quad \underbrace{A^{[l-1]}}_{[N \times H_{l-1}]}
$$

If we take transpose of $A^{[l-1]}$ and place it to the left side of upstream gradient, we have the correct dimension matching:

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{[l]}}}_{[H_{l-1} \times H_{l}]} = \underbrace{(A^{[l-1]})^T}_{[H_{l-1} \times N]} \cdot \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} 
$$

At first glance, this transformation may seem arbitrary. Why do we transpose? How can we be sure this result is correct? To clarify, let’s check the matrix representation:

$$
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial W^{[l]}_{11}} & \frac{\partial \mathcal{L}}{\partial W^{[l]}_{12}} & \dots & \frac{\partial \mathcal{L}}{\partial W^{[l]}_{1H_{l-1}}} \\
\frac{\partial \mathcal{L}}{\partial W^{[l]}_{21}} & \frac{\partial \mathcal{L}}{\partial W^{[l]}_{22}} & \dots & \frac{\partial \mathcal{L}}{\partial W^{[l]}_{2H_{l-1}}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial \mathcal{L}}{\partial W^{[l]}_{H_{l}1}} & \frac{\partial \mathcal{L}}{\partial W^{[l]}_{H_{l}2}} & \dots & \frac{\partial \mathcal{L}}{\partial W^{[l]}_{H_{l}H_{l-1}}} \\
\end{bmatrix}
=
\begin{bmatrix}
{A^{[l-1]}_{11}}^T & {A^{[l-1]}_{12}}^T & \dots & {A^{[l-1]}_{1N}}^T \\
{A^{[l-1]}_{21}}^T & {A^{[l-1]}_{22}}^T & \dots & {A^{[l-1]}_{2N}}^T \\
\vdots & \vdots & \ddots & \vdots \\
{A^{[l-1]}_{H_{l-1}1}}^T & {A^{[l-1]}_{H_{l-1}2}}^T & \dots & {A^{[l-1]}_{H_{l-1}N}}^T \\
\end{bmatrix}

\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{11}} & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{12}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{1H_{l}}} \\
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{21}} & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{22}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{2H_{l}}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{N1}} & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{N2}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{NH_{l}}} \\
\end{bmatrix}
$$

If we look closely only one gradient calculation, such as $\frac{\partial \mathcal{L}}{\partial W^{[l]}_{11}}$:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}_{11}} 
=
\begin{bmatrix}
{A^{[l-1]}_{11}}^T & {A^{[l-1]}_{12}}^T & \dots & {A^{[l-1]}_{1N}}^T \\
\end{bmatrix}

\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{11}} \\
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{21}} \\
\vdots \\
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{N1}} \\
\end{bmatrix}
$$

Since single value of weight matrix is fixed for each input value, the gradient of this weight is just summation of all $N$ input derivatives:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}_{11}} = {A^{[l-1]}_{11}}^T \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{11}} + {A^{[l-1]}_{12}}^T \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{21}} + \dots + {A^{[l-1]}_{1N}}^T \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{N1}}
$$

In the image below, you can see the visualization of this calculation for 4 weights in the network:

<d-figure>
  <img src="/assets/img/posts/2025-09-18-matrix-backpropagation/net14.png" style="width: 95%; display: block; margin-left: auto; margin-right: auto;" alt="Net14">
</d-figure>

At the same time, we need to calculate the derivative of $A^{[l-1]}$, $\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}$, so upstream gradients can flow to previous layer and the gradient of weight matrix can be calculated in that layer: 

$$
\frac{\partial \mathcal{L}}{\partial A^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \frac{\partial Z^{[l]}}{\partial A^{[l-1]}} 
$$

The local gradient is:

$$
\frac{\partial Z^{[L]}}{\partial A^{[l-1]}} = \frac{\partial \left(A^{[l-1]} W^{[l]} \right)}{\partial A^{[l-1]}} = \underbrace{W^{[l]}}_{[H_{l-1} \times H_{l}]} 
$$

Once again, the dimensions of matrices do not match:

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}}_{[N \times H_{l-1}]} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} \ \cdot \ \underbrace{W^{[l]}}_{[H_{l-1} \times H_{l}]} 
$$

We can transpose the weight matrix to align matrices:

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}}_{[N \times H_{l-1}]} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} \quad \underbrace{W^{[l]^T}}_{[H_{l} \times  H_{l-1}]} 
$$

We can see the reasoning more easily if we check matrix representation again:

$$
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{11}} & \frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{12}} & \dots & \frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{1H_{l-1}}} \\
\frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{21}} & \frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{22}} & \dots & \frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{2H_{l-1}}} \\\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{N1}} & \frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{N2}} & \dots & \frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{NH_{l-1}}} \\
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{11}} & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{12}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{1H_{l}}} \\
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{21}} & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{22}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{2H_{l}}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{N1}} & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{N2}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{NH_{l}}} \\
\end{bmatrix}

\begin{bmatrix}
{W^{[l]^T}_{11}} & {W^{[l]^T}_{12}} & \dots & {W^{[l]^T}_{1H_{l-1}}} \\
{W^{[l]^T}_{21}} & {W^{[l]^T}_{22}} & \dots & {W^{[l]^T}_{2H_{l-1}}} \\
\vdots & \vdots & \ddots & \vdots \\
{W^{[l]^T}_{H_{l}1}} & {W^{[l]^T}_{H_{l}2}} & \dots & {W^{[l]^T}_{H_{l}H_{l-1}}} \\ 
\end{bmatrix}
$$

By focusing on only first value, $$\frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{11}}$$, the gradient of $$A^{l}_{11}$$ depends on the values that it affected during forward propagation (the same logic as $$\frac{\partial \mathcal{L}}{\partial W^{[l]}_{11}}$$):

$$
\frac{\partial \mathcal{L}}{\partial A^{[l-1]}_{11}} 
=
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial Z^{[l]}_{11}} & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{12}} & \dots & \frac{\partial \mathcal{L}}{\partial Z^{[l]}_{1H_{l}}} \\ 
\end{bmatrix}

\begin{bmatrix}
{W^{[l]^T}_{11}} \\
{W^{[l]^T}_{21}} \\
\vdots \\
{W^{[l]^T}_{H_{l}1}} \\
\end{bmatrix}
$$

Now we want to compute the gradient $\frac{\partial \mathcal{L}}{\partial Z^{[l-1]}}$. Recall that $A^{[l-1]}$ is obtained by applying an activation function to $Z^{[l-1]}$:

$$
\underbrace{A^{[l-1]}}_{[N \times H_{l-1}]} = \underbrace{\sigma(Z^{[l-1]})}_{[N \times H_{l-1}]}
$$

In our examples, we use the ReLU activation function. Its derivative with respect to **each element** is given by:

$$
\frac{\partial \ \text{ReLU}(Z^{[l-1]}_{ij})}{\partial Z^{[l-1]}_{ij}} = 
\begin{cases}
1 & \text{ if } \ Z^{[l]}_{ij} > 0, \\
0 & \text{ otherwise }
\end{cases}
$$

For $N \times H_{l-1}$ inputs, we must compute a gradient for each element, producing a matrix of 0s and 1s. The derivative of ReLU applied element-wise can be written as the indicator (mask) of positive entries:

$$
\frac{\partial \ \text{ReLU}(Z^{[l-1]})}{\partial Z^{[l-1]}} = 1\left(Z^{[l-1]} > 0\right)
$$

where $1(\cdot)$ is the element-wise indicator function (1 when the condition is true, 0 otherwise). In matrix form:

$$
\frac{\partial \ \text{ReLU}(Z^{[l-1]})}{\partial Z^{[l-1]}} =
\begin{bmatrix}
1(Z^{[l-1]}_{11} > 0) & 1(Z^{[l-1]}_{12} > 0) & \dots & 1(Z^{[l-1]}_{1H_{l-1}} > 0) \\
1(Z^{[l-1]}_{21} > 0) & 1(Z^{[l-1]}_{22} > 0) & \dots & 1(Z^{[l-1]}_{2H_{l-1}} > 0) \\
\vdots & \vdots & \ddots & \vdots \\ 
1(Z^{[l-1]}_{N1} > 0) & 1(Z^{[l-1]}_{N2} > 0) & \dots & 1(Z^{[l-1]}_{NH_{l-1}} > 0) \\
\end{bmatrix}
$$

Because the derivative is applied element-wise, the upstream gradient $\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}$ is combined with this mask element-wise, not by matrix multiplication:

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l-1]}}}_{[N \times H_{l-1}]} = \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}}_{[N \times H_{l-1}]} \odot \underbrace{\frac{\partial A^{[l-1]}}{\partial Z^{[l-1]}}}_{[N \times H_{l-1}]}
$$

or 

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l-1]}}}_{[N \times H_{l-1}]} = \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}}_{[N \times H_{l-1}]} \odot \underbrace{1\left(Z^{[l-1]} > 0\right)}_{[N \times H_{l-1}]}
$$

### The Pattern 

At this point, we completed the backpropagation using the matrix multiplication step. From here on, the process is essentially a **repetition**: each layer applies the same logic and propagates the gradients backwards through the weights (also biases implicitly) and activations.

The only real exception was the output layer, where we combined the softmax function with cross-entropy loss. This required a more detailed derivation, but the model is consistent for all hidden layers and is repeated across the entire network. You can see the pattern as follows:  

$$
\begin{array}{rll}
\underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[L]}}}_{[N \times C]} &= \frac{1}{N} \underbrace{\left(\hat{Y} - Y\right)}_{N \times C} \\
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{[L]}}}_{[H_{L-1} \times C]} &= \underbrace{\frac{\partial \left(A^{[L-1]} W^{[L]} \right)}{\partial W^{[L]}}}_{[H_{L-1} \times N]} \quad \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[L]}}}_{[N \times C]} &= \underbrace{A^{[L-1]^T}}_{[H_{L-1} \times N]} \quad \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[L]}}}_{[N \times C]} \\
\underbrace{\frac{\partial \mathcal{L}}{\partial A^{[L-1]}}}_{[N \times H_{L-1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[L]}}}_{[N \times C]} \quad \underbrace{\frac{\partial \left(A^{[L-1]} W^{[L]} \right)}{\partial A^{[L-1]}}}_{[C \times H_{L-1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[L]}}}_{[N \times C]} \quad \underbrace{W^{[L]^T}}_{[C \times H_{L-1}]} \\
\underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[L-1]}}}_{[N \times H_{L-1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[L-1]}}}_{[N \times H_{L-1}]} \odot \underbrace{\frac{\partial \left(\sigma\left(Z^{[L-1]}\right) \right)}{\partial Z^{[L-1]}}}_{[N \times H_{L-1}]} &=  \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[L-1]}}}_{[N \times H_{L-1}]} \odot \underbrace{1 \left( Z^{[L-1]} > 0 \right)}_{[N \times H_{L-1}]} \\
\vdots \\
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{[l]}}}_{[H_{l-1} \times H_{l}]} &=\underbrace{\frac{\partial \left(A^{[l-1]} W^{[l]} \right)}{\partial W^{[l]}}}_{[H_{l-1} \times N]} \quad \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} &= \underbrace{A^{[l-1]^T}}_{[H_{l-1} \times N]} \quad \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} \\
\underbrace{\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}}_{[N \times H_{l-1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} \quad \underbrace{\frac{\partial \left(A^{[l-1]} W^{[l]} \right)}{\partial A^{[l-1]}}}_{[H_{l} \times H_{l-1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{[N \times H_{l}]} \quad \underbrace{W^{[l]^T}}_{[H_{l} \times H_{l-1}]} \\
\underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l-1]}}}_{[N \times H_{l-1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}}_{[N \times H_{l-1}]} \odot \underbrace{\frac{\partial \left(\sigma\left(Z^{[l-1]}\right) \right)}{\partial Z^{[l-1]}}}_{[N \times H_{l-1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}}_{[N \times H_{l-1}]} \odot \underbrace{1 \left( Z^{[l-1]} > 0 \right)}_{[N \times H_{l-1}]} \\
\vdots \\
\underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[1]}}}_{[N \times H_{1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[1]}}}_{[N \times H_{1}]} \odot \underbrace{\frac{\partial \left(\sigma\left(Z^{[1]}\right) \right)}{\partial Z^{[1]}}}_{[N \times H_{1}]} &= \underbrace{\frac{\partial \mathcal{L}}{\partial A^{[1]}}}_{[N \times H_{1}]} \odot \underbrace{1 \left( Z^{[1]} > 0 \right)}_{[N \times H_{1}]} \\
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{[1]}}}_{[H_{0} \times H_{1}]} &= \underbrace{\frac{\partial \left(A^{[0]} W^{[1]} \right)}{\partial W^{[1]}}}_{[H_{0} \times N]} \quad \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[1]}}}_{[N \times H_{1}]} &= \underbrace{A^{[0]^T}}_{[H_{0} \times N]} \quad \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[1]}}}_{[N \times H_{1}]} \\
\end{array}
$$

I know it looks ugly but if you pay some attention, you will see the pattern for gradient calculation. There are just some rules that you need to follow for each block.