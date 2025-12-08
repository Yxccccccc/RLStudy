# TRPO-Paper2Code

[TOC]

## 1. TRPO的优势或主要贡献

（1）置信域策略优化

​	在旧策略的**邻域**里优化策略

​	使用**KL散度**来衡量新旧策略的远近，将其控制在阈值内

（2）从理论上证明了策略是单调改进的

​	之前的policy-base的强化学习算法都是在改进策略，但是从未证明过策略是在**单调改进**的

​	eg：所有基于梯度上升的算法，如果设置的步长过大，可能步子会迈过头

​	**TRPO从理论上从数学的方式保证了策略一定是单调改进的**，换句话说，只要我们改进策略，那策略一定是在变得更好的

## 2. Paper精读

### 2.1 预备知识（Preliminaries）

​	**策略π的期望回报的定义为**：对discounted return求期望，如下式所示：
$$
\begin{equation}\eta(\pi)=\mathbb{E}_{s_0,a_0,\ldots}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t)\right]\end{equation}
$$
​	Kakade和Langford在前置工作中提出了：

​	新策略的期望汇报 = 旧策略的期望回报 + **新策略在旧策略优势函数上的累计期望**，如下式所示：
$$
\begin{equation}\eta(\tilde{\pi})=\eta(\pi)+\mathbb{E}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left[\sum_{t=0}^\infty\gamma^tA_\pi(s_t,a_t)\right]\end{equation}
$$
​	由于期望具有线性性质，故在上式中，若级数收敛，则**求期望和求无穷级数可以交换顺序**，可变为下式：
$$
 \begin{equation}\mathbb{E}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left[\sum_{t=0}^\infty\gamma^tA_\pi(s_t,a_t)\right]=\sum_{t=0}^\infty\gamma^t\cdot\mathbb{E}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left[A_\pi(s_t,a_t)\right]\end{equation}
$$
​	从而，可以引入**折扣访问频率（状态占据度量ρ）**：		
$$
\begin{equation}
\rho_\pi(s){=}P(s_0=s){+}\gamma P(s_1=s){+}\gamma^2P(s_2=s){+}\ldots
\end{equation}
$$

$$
\begin{equation}\rho_{\tilde{\pi}}(s)=\sum_{t=0}^\infty\gamma^t\cdot\mathbb{P}_{s_0\sim d,\tilde{\pi}}\left[s_t=s\right]\end{equation}
$$

​	上式的状态占据度量ρ表示的是：**状态s在策略π下的长期权重，是经过t步后处于状态s的折扣概率之和**

​	将$A_\pi(s_t,a_t)$视为随机变量，则其期望可以分解为：
$$
\begin{equation}\mathbb{E}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left[A_\pi(s_t,a_t)\right]=\sum_s\sum_aA_\pi(s,a)\cdot\mathbb{P}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left(s_t=s,a_t=a\right)\end{equation}
$$
​	其中$\mathbb{P}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left(s_t=s,a_t=a\right)$可以表示为：
$$
\begin{equation}\mathbb{P}\left[s_t=s,\mathrm{~}a_t=a\right]=\mathbb{P}\left[s_t=s\right]\cdot\mathbb{P}\left[a_t=a\mid s_t=s\right]\end{equation}
$$
​	由于其是基于马尔科夫决策过程（MDP）的，根据**马尔科夫性质：动作选择仅依赖当前状态**，即在状态s采取行动a的概率，仅依赖于当前状态，可得下式：	
$$
\begin{equation}\mathbb{P}\left[a_t=a\mid s_t=s\right]=\tilde{\pi}(a\mid s)\end{equation}
$$
​	将式（8）替换式（7）可得式（9）：
$$
\begin{equation}\mathbb{P}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left(s_t=s,a_t=a\right)=\mathbb{P}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left(s_t=s\right)\cdot\tilde{\pi}(a\mid s)\end{equation}
$$
​	由此，式（6）变为：
$$
\begin{equation}\mathbb{E}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left[A_\pi(s_t,a_t)\right]=\sum_s\sum_aA_\pi(s,a)\cdot\mathbb{P}_{s_0,a_0,\cdots\sim\tilde{\pi}}\left(s_t=s\right)\cdot\tilde{\pi}(a\mid s)\end{equation}
$$
​	但式（6）相较式（3）还少了γ，故由式（12）得到的式（3）如下：
$$
\begin{equation}\sum_{t=0}^\infty\sum_s\sum_a\gamma^t\cdot A_\pi(s,a)\cdot\mathbb{P}(s_t=s)\cdot\tilde{\pi}(a\mid s)=\sum_s\sum_a(\sum_{t=0}^\infty\gamma^t\cdot\mathbb{P}(s_t=s))\cdot\tilde{\pi}(a\mid s)\cdot A_\pi(s,a)\end{equation}
$$
​	其中，$\sum_{t=0}^{\infty}\gamma^t\cdot\mathbb{P}(s_t=s)$即为我们之前引入的**状态占据度量ρ**，最终得到**新策略期望回报：**
$$
\begin{equation}\eta(\tilde{\pi})=\eta(\pi)+\sum_s\rho_{\tilde{\pi}}(s)\sum_a\tilde{\pi}(a|s)A_\pi(s,a)\end{equation}
$$
​	至此，式（14）成功的将**策略改进和优势的期望联系起来**，但直接优化他的困难在于**$\rho^{\tilde{\pi}}$是依赖新策略**的

​	于是我们引入**代理函数（替代目标）**如下式所示：
$$
\begin{equation}L_\pi(\tilde{\pi})=\eta(\pi)+\sum_s\rho_\pi(s)\sum_a\tilde{\pi}(a|s)A_\pi(s,a)\end{equation}
$$
​	上式将$\rho_{\tilde{\pi}}(s)$近似为$\rho_{{\pi}}(s)$，我们可以看到：
$$
\begin{equation}\begin{aligned}
L_{\pi_{\theta_0}}(\pi_{\theta_0}) & =\eta(\pi_{\theta_0}), \\
\nabla_\theta L_{\pi_{\theta_0}}(\pi_\theta)|_{\theta=\theta_0} & =\nabla_\theta\eta(\pi_\theta)|_{\theta=\theta_0}
\end{aligned}\end{equation}
$$
​	**在旧策略$\pi$处，$L_\pi$与$\eta$的梯度相同**，所以在旧策略附近优化$L_\pi$，就近似于优化$\eta$

​	但该结论并没有告诉我们应该走**多大的步**进行策略更新

​	Kakade和Langford曾经提出了**保守策略迭代（CPI）**：
$$
\begin{equation}\pi_{\mathrm{new}}(a|s)=(1-\alpha)\pi_{\mathrm{old}}(a|s)+\alpha\pi^{\prime}(a|s).\end{equation}
$$
​	其中：$\pi^{\prime}=\arg\max_{\pi^{\prime}}L_{\pi_{\mathrm{old}}}(\pi^{\prime}).$

​	由上式可知，新策略的更新取的是**混合形式**的更新，由此，可以得到**性能改进下界**：
$$
\begin{equation}\begin{aligned}
\eta(\pi_{\mathrm{new}}) & \geq L_{\pi_{\mathrm{old}}}(\pi_{\mathrm{new}})-\frac{2\epsilon\gamma}{(1-\gamma)^{2}}\alpha^{2} \\
 & \mathrm{where~}\epsilon=\max_{s}|\mathbb{E}_{a\sim\pi^{\prime}(a|s)}\left[A_{\pi}(s,a)\right]|.
\end{aligned}\end{equation}
$$
​	不过该性能改进下界仅适用于式（15）生成的**混合策略**

​	TRPO的作者目标是将**更新法则推广到一般的随机策略**

> [!NOTE]
>
> **小结**
>
> 1.用 $\eta(\tilde\pi)=\eta(\pi)+\sum_s \rho^{\tilde\pi}(s)\sum_a \tilde\pi(a|s)A^\pi(s,a)$ 把**策略改进**和**优势的期望**联系起来；但直接优化它困难在于 $\rho^{\tilde\pi}$ 依赖新策略
>
> 2.引入**替代目标** $L_\pi(\tilde\pi)$：把 $\rho^{\tilde\pi}$ 近似成 $\rho^\pi$，从而得到对 $\eta$ 的**一阶一致**近似；**小步**提升 $L$ 会提升 $\eta$
>
> 3.CPI 给出一种**带下界**的更新（混合策略 + 步长 $\alpha$），说明**控制更新幅度**即可获得**单调改进**；但它局限于混合策略形式，激励我们在下一节把保证推广到**一般策略**，并最终导向 TRPO 的“KL 信赖域”思想

### 2.2 一般随机策略的单调改进保证（Monotonic Improvement Guarantee for General Stochastic Policies）

​	要将更新法则推广到更一般的随机策略，要将其中的步长$\alpha$替换为**新旧策略之间的一个距离度量**，从而引入**TV（全变差）散度**

​	两个离散的概率分布p,q的TV散度定义为：
$$
\begin{equation}D_{\mathrm{TV}}(P||Q)=\frac{1}{2}\Sigma_i|p_i-q_i|\end{equation}
$$
​	状态最大的TV距离为：
$$
\begin{equation}D_{\mathrm{TV}}^{\max}(\pi,\tilde{\pi})=\max_sD_{TV}(\pi(\cdot|s)\parallel\tilde{\pi}(\cdot|s))\end{equation}
$$
​	从而可以证明$L_\pi$和$\eta$的**误差下界**（具体证明在论文附录）：
$$
\begin{equation}\begin{aligned}
\eta(\pi_{\mathrm{new}}) & \geq L_{\pi_{\mathrm{old}}}(\pi_{\mathrm{new}})-\frac{4\epsilon\gamma}{(1-\gamma)^2}\alpha^2 \\
where & \epsilon=\max_{s,a}|A_\pi(s,a)|
\end{aligned}\end{equation}
$$
​	$\alpha$为新旧两个策略在所有状态下的最大总变差散度

​	接下来引入**KL散度**的概念：

​	两个离散的概率分布p,q的KL散度定义为：
$$
\begin{equation}D_{KL}(P||Q)=E_p\left[\log\frac{P(x)}{Q(x)}\right]=\Sigma_ip_i\log\frac{p_i}{q_i}\end{equation}
$$
​	其衡量的是**两个概率分布的远近**，**表示用q来近似p时的信息损失**

​	接着，利用TV散度和KL散度的关系**Pinsker不等式**：
$$
\begin{equation}D_{TV}(p\parallel q)^2\leq D_{\mathrm{KL}}(p\parallel q).\end{equation}
$$
​	令$D_{\mathrm{KL}}^{\max}(\pi,\tilde{\pi})=\max_sD_{\mathrm{KL}}(\pi(\cdot|s)\parallel\tilde{\pi}(\cdot|s))$，带入式（19）中的$\alpha$可得：
$$
\begin{equation}\begin{aligned}
\eta(\tilde{\pi}) & \geq L_\pi(\tilde{\pi})-CD_\mathrm{KL}^\mathrm{max}(\pi,\tilde{\pi}), \\
 & \mathrm{where~}C=\frac{4\epsilon\gamma}{(1-\gamma)^2}.
\end{aligned}\end{equation}
$$
​	**$\eta(\tilde{\pi})$是新策略$\tilde{\pi}$的真实期望回报，不等式的右边为新策略的性能下界**

​	基于式（22）即可得到原论文的算法1（带惩罚的近似策略迭代）：

<img src="https://gitee.com/yan-cuihua/typora-img/raw/master/img/image-20251208230648372.png" alt="image-20251208230648372" style="zoom:67%;" />

​	该过程可以视为一种**最小化-最大化（MM）**算法，每次迭代都找到目标函数的一个下界函数，只要让下界函数最大，就能保证目标$\eta$非减

​	尽管理论建议用**惩罚项**（系数 $C$）去限制步长，但按理论推荐的 $C$ 会让步子很小，为了在**稳健**的前提下允许更大的更新，下一节提出用**KL 约束**而不是惩罚项也就是对新旧策略施加一个**置信域**（trust region），这正是 TRPO 的核心思想

> [!NOTE]
>
> **小结**
>
> 1.把 CPI 的“单调改进下界”推广到了**一般随机策略**：用状态最大 TV（再借由 Pinsker 不等式转为KL）来度量新旧策略的“距离”。这样便得到 $\eta(\tilde\pi)\le L_\pi(\tilde\pi)+\tfrac{2\epsilon}{(1-\gamma)^2}\,D^{\max}_{\mathrm{KL}}(\pi,\tilde\pi)$
>
> 2.由此导出一个**带 KL 惩罚**的近似策略迭代（算法1），其每步都**保证不劣化**真实目标；也能从 **MM/近端**视角理解
>
> 3.然而用惩罚权重实现时步长偏小；为实现**大步且稳**的更新，TRPO 改为显式的**KL 信赖域约束**（下一节给出可实现形式）

### 2.3 参数化策略优化（Optimization of Parameterized Policies）

​	上一节独立于策略的参数化（且默认能在所有状态上评估策略）来讨论了策略优化问题，下面在**有限样本**和**任意参数化**的前提下推导出一个**可实践的算法**

​	将策略 $\pi_\theta(a|s)$参数化（参数向量为 $\theta$），并相应把此前的记号改写成关于 $\theta$ 的函数（例如 $\eta(\theta):=\eta(\pi_\theta)$），并用 $\theta_{\text{old}}$ 表示希望改进的旧参数

​	为在**稳健**前提下允许**更大步长**，可以用**约束**而非惩罚：
$$
\begin{equation}\max_\theta L_{\theta_{\mathrm{old}}}(\theta)\quad\mathrm{s.t.}\quad D_{\mathrm{KL}}^{\max}(\theta_{\mathrm{old}},\theta)\leq\delta\end{equation}
$$
​	这相当于在**每个状态**上都限制新旧策略之间的 KL 散度（最大 KL 约束）

​	尽管理论上动机充分，但它需要对**所有状态**施加约束，**不易求解**

​	因此给出一个**实用近似**：把“最大 KL”换成按旧策略访问分布 $\rho_{\theta_{\text{old}}}$ 加权的**平均 KL**：
$$
\begin{equation}\overline{D}_{\mathrm{KL}}^\rho(\theta_1,\theta_2):=\mathbb{E}_{s\thicksim\rho}\left[D_{\mathrm{KL}}(\pi_{\theta_1}(\cdot|s)\parallel\pi_{\theta_2}(\cdot|s))\right]\end{equation}
$$
​	即：
$$
\begin{equation}\max_\theta L_{\theta_{\mathrm{old}}}(\theta)\quad\mathrm{s.t.}\quad D_{\mathrm{KL}}^{\rho_{\theta_{\mathrm{old}}}}(\theta_{\mathrm{old}},\theta)\leq\delta\end{equation}
$$
​	实验也表明，这种**平均 KL 约束**与**最大 KL 约束**在经验上效果接近，但计算上更可行，且与以往的一些方法（如自然策略梯度等）相关

> [!NOTE]
>
> **小结**
>
> 1.把“单调改进”的理论**落地为可求解问题**：以**替代目标 $L$** 为优化对象，并通过对**策略变化施加 KL 信赖域**来控制步长与稳定性
>
> 2.理论上更严格的“**最大 KL 约束**”约束所有状态，但**约束数巨大**，难以直接求解；**平均 KL 约束**以旧策略的访问分布加权，更**实用**，且在实验中与最大 KL 表现相近

### 2.4 目标与约束的采样估计（Sample-Based Estimation of the Objective and Constraint）

​	本节说明在**有限样本**下，如何用**蒙特卡洛**近似目标与约束，从而在每次更新时可计算地进行求解

​	先将式（25）的求和改写为期望的形式：
$$
\begin{equation}\max_{\theta}\sum_{s}\rho_{\theta_{\mathrm{old}}}(s)\left(\sum_{a}\pi_{\theta}(a|s)A_{\theta_{\mathrm{old}}}(s,a)\right)\quad\mathrm{s.t.~}D_{\mathrm{KL}}^{\rho_{\theta_{\mathrm{old}}}}(\theta_{\mathrm{old}},\theta)\leq\delta\end{equation}
$$
​	① 用对 $s$ 的采样期望 $\mathbb{E}_{s\sim\rho_{\theta_{\text{old}}}}[\cdot]$ 代替对状态的求和

​	② 将优势 $A$ 替换为经验 $Q$ 值（因为向 $Q$ 加常数不改变梯度，相当于用了基线）

​	③ 用**重要性采样**把对动作的求和换成 $\mathbb{E}_{a\sim q(\cdot|s)}\big[\frac{\pi_\theta(a|s)}{q(a|s)}\hat Q(s,a)\big]$

​	从而得到与式（26）等价的期望形式：
$$
\begin{equation}\max_{\theta}\mathbb{E}_{s\sim\rho_{\theta_{\mathrm{old}}},a\sim q}\left[\frac{\pi_{\theta}(a|s)}{q(a|s)}\hat{Q}_{\theta_{\mathrm{dd}}}(s,a)\right]\quad\mathrm{s.t.~}\mathbb{E}_{s\sim\rho_{\theta_{\mathrm{old}}}}\left[D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{dd}}}(\cdot|s)\parallel\pi_{\theta}(\cdot|s))\right]\leq\delta\end{equation}
$$

> [!NOTE]
>
> **小结**
>
> 1.把“替代目标 + KL 约束”落地为**可采样**的期望形式，关键在于：对状态做经验平均、对动作用**重要性采样**，并用经验 $\hat Q$ 代替优势。
>
> 2.原论文中的Single Path和Vine采样方案不再此处展开

### 2.5 TRPO算法（TRPO Algorithm）

​	在上节的基础上，通过**两步近似**操作，得到**自然策略梯度**，其可以看做是式（25）的一种特例，具体实现如下：

​	①目标函数$L_{\theta_{old}}(\theta)$做**线性近似**，即在$\theta=\theta_{old}$**一阶泰勒展开**：
$$
\begin{equation}L_{\theta_{old}}(\theta)\approx L_{\theta_{old}}(\theta_{old})+\nabla_\theta L_{\theta_{old}}(\theta_{old})\cdot(\theta-\theta_{old})\end{equation}
$$
​	其中，$ L_{\theta_{old}}(\theta_{old})$是一个常数，所以在优化时，只需看$\nabla_\theta L_{\theta_{old}}(\theta_{old})\cdot(\theta-\theta_{old})$部分

​	②约束条件（平均KL散度）$D_{\mathrm{KL}}^{\rho_{\theta_{\mathrm{old}}}}$做**二次近似**，即在$\theta=\theta_{old}$做二阶**泰勒展开**：
$$
\begin{equation}\overline{D}_{KL}^{\rho_{\theta_{old}}}(\theta_{old},\theta)\approx\frac{1}{2}\Delta\theta^TA\Delta\theta\end{equation}
$$
​	其中A为平均KL散度在$\theta_{old}$处的Hessian矩阵（Hessian矩阵就是多元函数的二阶偏导数构成的对称矩阵）
$$
\begin{equation}A(\theta_{\mathrm{old}})_{ij}=\left.\frac{\partial}{\partial\theta_i}\frac{\partial}{\partial\theta_j}\mathbb{E}_{s\sim\rho_\pi}[D_{\mathrm{KL}}(\pi(\cdot|s,\theta_{\mathrm{old}})\parallel\pi(\cdot|s,\theta))\right]|_{\theta=\theta_{\mathrm{old}}}\end{equation}
$$
​	具体来讲，以**KL的Hessian**来构造 **Fisher 信息矩阵（FIM）**，而不是用“梯度外积的协方差”形式，这种**解析式**会在每个状态上**对动作积分**，因此**不依赖**当次采样到的动作 $a_n$（原论文附录C）

​	最终的约束形式为：
$$
\begin{equation}\max_{\theta}\left.\left|\nabla_{\theta}L_{\theta_{\mathrm{old}}}(\theta)\right|_{\theta=\theta_{\mathrm{old}}}\cdot(\theta-\theta_{\mathrm{old}})\right|\quad\mathrm{s.t.}\quad\frac{1}{2}(\theta_{\mathrm{old}}-\theta)^\top A(\theta_{\mathrm{old}})(\theta_{\mathrm{old}}-\theta)\leq\delta\end{equation}
$$
​	**TRPO自然策略梯度**更新为：
$$
\begin{equation}\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\lambda\left.A(\theta_{\mathrm{old}})^{-1}\nabla_{\theta}L(\theta)\right|_{\theta=\theta_{\mathrm{old}}}\end{equation}
$$
​	如果将约束从KL散度替换为 $\ell_2$ 范数，得到**标准策略梯度更新**：
$$
\begin{equation}\max_{\theta}\left[\nabla_{\theta}L_{\theta_{\mathrm{old}}}(\theta)|_{\theta=\theta_{\mathrm{old}}}\cdot(\theta-\theta_{\mathrm{old}})\right]\quad\mathrm{s.t.}\quad\frac{1}{2}\|\theta-\theta_{\mathrm{old}}\|_{2}^{2}\leq\delta\end{equation}
$$

> [!NOTE]
>
> **小结**
>
> 1.TRPO 的更新把多种方法串在一起：
>
> - 线性化 $L$ + 二次化 KL 约束 ⇒ **自然策略梯度**（用 Fisher 矩阵 $A$ 作为度量）
> - 把 KL 改为 $\ell_2$ 约束 ⇒ **标准策略梯度**
>
> 2.原论文中还提到了直接最小化 $L$ ⇒ **（近似）策略迭代**，本文中不做展开

## 3. 手撕代码

https://github.com/Yxccccccc/RLStudy/blob/main/Implements/TRPO.ipynb

## 4. References

[1]arXiv:Trust Region Policy Optimization

https://arxiv.org/abs/1502.05477

[2]Bilibili：【PPO的前身】【TRPO】第一部分 直观理解与算法理论——[东川路第一可爱猫猫虫](https://space.bilibili.com/675505667/?spm_id_from=333.788.upinfo.detail.click)

https://www.bilibili.com/video/BV1DcWozHEYi/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=51ce4eead5de12ddec5dda1b061728e6

