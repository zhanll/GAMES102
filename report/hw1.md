- # 作业 1

  > 2020/10/18

  ## 问题

  **Input**：已知平面内 n 个点 $P_j(x_j,y_j), j=1,2,\dots,n$。

  **Output**: 拟合这些点的函数。

  要求：实现不同的拟合方法，并进行比较。输入点集可以进行交互式鼠标指定，或者其他方法生成。

  [![https://i.bmp.ovh/imgs/2020/10/c10f7ef259809aa0.png](https://camo.githubusercontent.com/9efb836ab79cfd5ba26bc142f27925fb1fbd9532/68747470733a2f2f692e626d702e6f76682f696d67732f323032302f31302f633130663765663235393830396161302e706e67)](https://camo.githubusercontent.com/9efb836ab79cfd5ba26bc142f27925fb1fbd9532/68747470733a2f2f692e626d702e6f76682f696d67732f323032302f31302f633130663765663235393830396161302e706e67)

  ## 一、插值型拟合方法

  1. 使用多项式函数（幂基函数的线性组合）$f(x)=\sum_{i=0}^{n-1}\alpha_i B_i(x)$ 插值 ${P_j}$，其中 $B_i(x)=x^i$
  2. 使用 Gauss 基函数的线性组合 $f(x)=b_0 + \sum_{i=1}^{n}b_i g_i(x)$ 插值 ${P_j}$，其中 $$ g_i(x)=\exp\left(-\frac{(x-x_i)^2}{2\sigma^2}\right) $$ 即对称轴在插值点上，$i=1,\dots,n$，缺省设 $\sigma =1$

1.Lagrange：

试验结果：

![lagrange](C:\Users\Administrator\Pictures\lagrange.png)

做法：

直接使用拉格朗日插值多项式：$L_n(x)=\sum\limits_{i=0}^n y_i * \prod\limits_{j=0,j\neq{i}} \frac{x - x_j}{x_i-x_j}$

2.Gauss:

试验结果：

![gauss](C:\Users\Administrator\Pictures\gauss.png)

  做法：

表达为$AX=b$，带入观察点求得矩阵A，再用A计算采样点对应的函数值



  ## 二、逼近型拟合方法

  - 固定幂基函数的最高次数m (m<n)，使用最小二乘法：$\min E$，其中 $E(x)=\sum_{i=0}^{n}(y_i-f(x_i))^2$ 拟合 ${P_j}$。
  - 岭回归（Ridge Regression）：对上述最小二乘法误差函数增加 $E_1$ 正则项，参数 $\lambda$，$\min (E+\lambda E_1)$，其中 $E_1=\sum_{i=1}^n\alpha_i^2$



试验结果：

1.Least Squares:

试验结果：

![LeastSquares](C:\Users\Administrator\Pictures\LeastSquares.png)

做法：

矩阵形式求解$\theta=(X^T*X)^{-1}*X^T*Y$,再用$\theta$计算采样点的函数值，手调多项式的最高次幂



2.Ridge Regression:

试验结果：

![RidgeRegression](C:\Users\Administrator\Pictures\RidgeRegression.png)

做法：

矩阵形式求解$\theta=(X^T*X+\lambda*I)^{-1}*X^T*Y$,再用$\theta$计算采样点的函数值，手调多项式的最高次幂，手调$\lambda$



四种对比：

![all](C:\Users\Administrator\Pictures\all.png)