# Introduction
The implementation of LA-SiLU and LA-HardSiLU

# Front and Back-propagation
## Front-propagation
$$
\begin{aligned}
y_i^l &= {W_i^l}^{T}x_i^l \\
n^{l}_{i} &= \frac{y^{l}_{i}-\mu^{l}}{\sigma^{l}} \\
a^{l}_{i} &= y^{l}_{i} s\left(n^{l}_{i}\right) \\
\end{aligned}
$$

## Back-propagation
$$
\begin{aligned}
\frac{dL}{dy^{l}_{i}} &= \frac{dL}{da^{l}_{i}} s\left(n^{l}_{i}\right) + \sum^{H}_{j=1}{y^{l}_{j} \frac{\partial s\left(n^{l}_{j}\right)}{\partial n^{l}_{j}} \frac{\partial n^{l}_{j}}{\partial y^{l}_{j}} \\
\end{aligned}
$$
