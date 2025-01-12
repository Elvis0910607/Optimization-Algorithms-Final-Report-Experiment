# Optimization Algorithms Comparison: SGD, SVRG, and STORC
This repository contains an implementation and comparison of three optimization algorithms: Stochastic Gradient Descent (SGD), Stochastic Variance-Reduced Gradient (SVRG), and the STORC (STOchastic variance-Reduced Conditional gradient sliding) algorithm as introduced in the paper "Variance-Reduced and Projection-Free Stochastic Optimization" by Hazan and Luo.
### Overview
The focus of this project is to evaluate and compare the performance of these optimizers in terms of their convergence rates and computational efficiency.
#### Algorithms Included
1. SGD (Stochastic Gradient Descent):
  A widely used optimization algorithm that updates model parameters using a single random sample or mini-batch.
  No variance reduction is applied.
2. SVRG (Stochastic Variance-Reduced Gradient):
  An enhancement over SGD that leverages variance reduction techniques to improve convergence.
  Reduces the noise in gradient estimates.
3. STORC (STOchastic variance-Reduced Conditional gradient sliding):
  Combines variance reduction with the Frank-Wolfe optimization framework.
  Features a projection-free approach and an auxiliary sequence for improved convergence, particularly in structured constraint settings.
#### Reference
The STORC algorithm is based on:
Hazan, E., & Luo, H. (2016). Variance-Reduced and Projection-Free Stochastic Optimization. Proceedings of the 33rd International Conference on Machine Learning.
