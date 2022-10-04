# Introduction
The implementation of LA-SiLU and LA-HardSiLU

# Front and Back-propagation
## Front-propagation
$$
\begin{aligned}
y_i^l &= {W_i^l}^{T}x_i^l \\
n_i^l &= \frac{y_i^l-\mu^l}{\sigma^l} \\
a_i^l &= y_i^l s(n_i^l) \\
\end{aligned}
$$

## Back-propagation
$$
\begin{aligned}
\frac{dL}{dy_i^l} &= \frac{dL}{da_i^l} s(n_i^l) + \sum_{j=1}^H{y_j^l \frac{\partial s(n_j^l)}{\partial n_j^l} \frac{\partial n_j^l}{\partial y_i^l}} \\
\end{aligned}
$$
