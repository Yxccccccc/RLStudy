# PPO-Paper2Code

[TOC]

## 1. PPO的优势或主要贡献

TRPO论文解读可以看我的上一篇文章[TRPO-Paper2Code](https://blog.csdn.net/weixin_60186742/article/details/155720422?spm=1001.2014.3001.5501 )

### 1.1 PPO主要解决了什么问题？

（1）**如何在不做复杂二阶优化的前提下，控制策略更新幅度?**

TRPO 用 KL 约束显式限制步长，PPO 用**剪切的概率比**（或自适应 KL 惩罚）把目标函数做成对原代理目标的**下界估计**，从而抑制过大更新，避免性能骤降

（2）**如何安全地对同一批轨迹进行多轮更新?**

标准 PG 多步会产生破坏性更新，PPO 的目标构造专门为此设计，可在每批数据上进行多 epoch 的小批量优化

### 1.2 PPO相较TRPO与其他算法有什么优势？

（1）TRPO 需要带约束的二阶近似（平均 KL 约束 + 共轭梯度求解），PPO 用**剪切的代理目标/或自适应 KL 惩罚**，在纯一阶优化框架里获得类似的稳健性，代码与调参更简单

（2）标准 PG 同一批数据做多轮优化易“把策略推太远”而崩溃，PPO 的**概率比剪切**在多轮小批量更新时限制策略漂移，提升样本效率同时保持稳定

（3） TRPO 对含噪声或**策略/价值共享参数**的架构不兼容（策略–价值共享网络就是“一个骨干 + 两个头”的 Actor–Critic 架构），PPO 在这些更通用的设置下工作良好

<u>总体来讲，PPO在**样本效率、简单性与训练时长**之间取得更好的平衡</u>

### 1.3 PPO主要做出了哪些贡献？

（1）提出**PPO-Clip**，但当更新过大时自动卡住，形成对性能的下界

（2）在PPO-Clip基础上提出**自适应 KL 惩罚变体PPO-Penalty**，给出 KL 罚项并**自适应**调节权重以追踪目标 KL

（3）在**同一批数据上多 epoch 的小批量 SGD**，配合**优势函数估计（GAE）**、熵正则和值函数损失等，给出可复现的默认超参与跨域评测

## 2. Paper精读

### 2.1 信赖域方法存在的问题（Problem of Trust Region Methods）

在TRPO中，同时对**策略更新的幅度**施加约束：
$$
\begin{equation}\begin{aligned}
 & \underset{\theta}{\operatorname*{\operatorname*{maximize}}}\quad\hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t\right] \\
 & \mathrm{subject~to}\quad\hat{\mathbb{E}}_t[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]]\leq\delta
\end{aligned}\end{equation}
$$
这里的 $\theta_{\mathrm{old}}$ 是旧策略的参数，TRPO的方法是将目标 $ L_{\theta_{\mathrm{old}}}(\theta)$ 线性化，将约束二次近似后，用**共轭梯度**高效近似求解

TRPO理论上建议将KL散度用作**惩罚而非硬性约束**，下式为惩罚形式：
$$
\begin{equation}\max_\theta\operatorname{imize}\hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t-\beta\operatorname{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]\right]\end{equation}
$$
但TRPO实际上采用硬约束而非惩罚，原因是**上式中的 $\beta$ 很难选定**，甚至同一问题的不同学习阶段都合适的单一 $\beta$

所以PPO首先做的工作就是：尝试用**一阶方法**去模拟 TRPO 的**单调改进**，实验表明**仅**用固定 $\beta$ 的带罚项目标并不足够，还需进一步修改。

> **小结**
>
> 1.TRPO的思路是：最大化**重要性采样**形式的替代目标，同时**约束平均 KL**，以防策略更新过大；理论上也可用**KL 罚项**，但 $\beta$ 很难稳妥设定
>
> 2.**PPO改进动机**：在仅用**一阶优化**的前提下，实现“既能多轮更新、又不走太远”的安全改进，这将引出后文的**截断比率**等设计
>
> 3.TRPO 与 PPO都在回答同一问题：**如何在不引发性能崩塌的前提下，尽可能迈出更大的改进步**。TRPO用二阶近似+KL约束；PPO用一阶方法+小技巧（如**截断**）让新旧策略“保持接近”。

### 2.2 截断替代目标（Clipped Surrogate Objective）

从TRPO的替代目标的出发进行优化

首先定义**概率比率$r_t(\theta)$：**
$$
\begin{equation}r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\mathrm{old}}(a_t\mid s_t)}\end{equation}
$$
由上式的定义可知，在旧策略处，概率比率$r(\theta_{\mathrm{old}})=1$

带入TRPO的保守策略迭代（CPI）替代目标$L^{CPI}(\theta)$为：
$$
\begin{equation}L^{CPI}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t\right]=\hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t\right]\end{equation}
$$
直接最大化$L^{CPI}(\theta)$会导致**过大的策略更新**，需要一种**靠近旧策略**的机制来抑制大更新

那接下来的优化目标就是考虑如何修改目标，以**惩罚**那些把 $r_t(\theta)$ 推离 1 的改变

由此，PPO提出了$L^{CLIP}(\theta)$，即PPO-Clip方法：
$$
\begin{equation}L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\left[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right]\end{equation}
$$
式中， $\epsilon$ 是超参数（原文中举例 $\epsilon=0.2$），表示允许偏离旧策略的幅度

多份材料都将该方法概括为：在**未截断项**与**截断项**之间取最小，形成**保守（悲观）估计**，避免为了增大回报而把 $r$ 推到极端

**更直观的角度**：

**正优势**：不鼓励把 $r$ 推到 $1+\epsilon$ 之外（防止“过分放大好动作”）

**负优势**：不鼓励把 $r$ 降到 $1-\epsilon$ 之外（防止“过分惩罚坏动作”）

通过**取最小**得到对 $L_{\text{CPI}}$ 的**下界**，既能多轮更新又不“走太远”

![image-20251209165538789](https://gitee.com/yan-cuihua/typora-img/raw/master/img/image-20251209165538789.png)

PPO-Clip 不显式用 KL 约束，而是靠**截断**在**目标层面**形成“软约束”

实现上，如Stable-Baselines3（SB3库）常配合**优势归一化**与**价值函数截断**以增加稳定性

> **小结**
>
> 1.$L^{CPI}$ 来自保守策略迭代；若无约束会诱发**过大**策略更新。PPO 通过**对比率截断并取最小**，在目标层面形成**保守下界**，实现“**能多轮更新**但**不走太远**”
>
> 2.$L^{CLIP}$ 确保了**稳定性**与**步长自抑制**——当更新过大时会被**惩罚**

### 2.3 自适应KL惩罚项系数（Adaptive KL Penalty Coefficient）

除了Clip方法外，还有另一种做法可以作为“截断替代目标”的**替代**或**补充**，即PPO-Penalty方法：

**在KL散度加入惩罚项**，同时自适应的调节惩罚项的系数，使每次策略更新的 KL 达到某个**目标值** $d_{\text{targ}}$

论文的实验中PPO-Penalty的效果不如PPO-Clip

Spinning Up 将 PPO 分为**PPO-Clip**与**PPO-Penalty**两类：前者直接在目标函数里**截断比率**，后者像 TRPO 一样**约束/惩罚 KL**并**自适应**系数

OpenAI 文档更推荐 Clip 作为默认实现

PPO-Penalty算法最简单的实现中，每次策略更新做以下步骤：

（1）用**多轮**小批量 SGD 优化**带 KL 罚项**的目标：
$$
\begin{equation}L^{KLPEN}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t-\beta\operatorname{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]\right]\end{equation}
$$
（2）计算每次更新后的$d$：
$$
\begin{equation}d=\hat{\mathbb{E}}_t[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]]\end{equation}
$$
（3）根据$d$与$d_{targ}$的比值更新$\beta$：
$$
\begin{equation}\begin{aligned}
 & -\mathrm{~If~}d<d_{\mathrm{targ}}/1.5,\beta\leftarrow\beta/2 \\
 & -\mathrm{~If~}d>d_{\mathrm{targ}}\times1.5,\beta\leftarrow\beta\times2
\end{aligned}\end{equation}
$$
（4）更新后的 $\beta$ 用于**下一次**策略更新

虽然偶尔会出现与 $d_{\text{targ}}$**偏差较大**的更新，但很少见，并且 $\beta$ 会很快**自我调整**

自适应 $\beta$ **收敛很快**，一般**不太挑**初值与阈值，但在大多数实际工程中仍建议**优先使用 Clip 版本**，Penalty 作为参考或在特定工程约束下选用

> **小结**
>
> 1.以**KL罚项**为核心，并用**自适应$\beta$把观测 KL 拉向目标 $d_{\text{targ}}$**
>
> 2.总体不如 **Clip** 好用，更多作为**BaseLine参照**

### 2.4 PPO算法实现（PPO Algorithm）

前述的两种替代目标（PPO-CLIP、PPO_Penalty）在常规策略梯度的实现基础上做以下几点改动即可求值、求导：

（1）用 $L_{\text{CLIP}}$ 或 $L_{\text{KLPEN}}$ 替换 $L_{\text{PG}}$，并对该目标做**多轮**随机梯度上升

（2）常见的**减方差优势估计**都会用到**价值函数 $V(s)$**（如 GAE 等）

（若策略与价值**共享网络**，则需要一个**把策略替代项与价值误差**合并的目标）

（3）可加**熵奖励**促进探索

综上，将策略损失、价值损失与熵奖励合成一个总目标，每轮迭代近似于：
$$
\begin{equation}L_t^{CLIP+VF+S}(\theta)=\hat{\mathbb{E}}_t\left[L_t^{CLIP}(\theta)-c_1L_t^{VF}(\theta)+c_2S[\pi_\theta](s_t)\right]\end{equation}
$$
式中， $c_1,c_2$ 为系数，$L^{t}_{\text{VF}}$ 是平方误差： $L^{t}_{\text{VF}} = (V_\theta(s_t)-V^{\text{targ}}_t)^2$

**训练目标=策略项+（−价值误差）+熵奖励**，实现上只需把**损失替换**并**多 epoch**优化

**A2C/A3C**的核心就是用**n 步回报**做优势，PPO也可以通过同样的**分段采样 + n 步/GAE 优势**，具体实现如下：

让每个Actor滚动T步，立即更新一次，接下来滚动下一段T步在更新（A3C的n-step return）

但在第 $t$ 步想给当前动作打分时，**看不到 $t$ 之后的奖励**，这时候就要**截断并“自举”（bootstrap）**只累计到这段末尾，然后用**价值函数 $V(s_T)$** 去“估”后面看不到的尾巴，如下式所示：
$$
\begin{equation}\hat{A}_t=-V(s_t)+r_t+\gamma r_{t+1}+\cdots+\gamma^{T-t+1}r_{T-1}+\gamma^{T-t}V(s_T)\end{equation}
$$
上式相当于：**（段内折现回报 + 末端价值）− 当前价值**，即n-step优势

**把这一小段能看到的奖励都折现相加**，在段尾接上**价值估计 $V(s_T)$** 当作“看不到的尾巴”，最后**减掉当前状态价值 $V(s_t)$**（这就是“价值基线”），得到“这步动作比平均水平好多少”

由此可以得到**截断版的GAE**：把本段内的 TD 残差**按 $(\gamma\lambda)$** 递减加权求和
$$
\begin{equation}\begin{aligned}
 & \hat{A}_t=\delta_t+(\gamma\lambda)\delta_{t+1}+\cdots+\cdots+(\gamma\lambda)^{T-t+1}\delta_{T-1} \\
 & \mathrm{where}\quad\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\end{aligned}\end{equation}
$$
式中：

$\lambda=1$：几乎等价于式 (10) 的 **“完整 n 步优势”**（段尾用 $V(s_T)$ 自举）；

$\lambda=0$：只用**一步 TD**（$\hat A_t \approx \delta_t$），**方差更小**但**偏差更大**；

$0<\lambda<1$：在**偏差–方差**之间平衡

综上可以得到**PPO-Actor-Critic风格的伪代码**（论文中的Algorithm 1）：

![image-20251209180548107](https://gitee.com/yan-cuihua/typora-img/raw/master/img/image-20251209180548107.png)

（1）每次迭代，让**$N$ 个并行 actor**各自收集**$T$**步数据

（2）把这 $NT$个样本上的替代损失构建好，用**小批量 SGD/Adam**训练**$K$** 个 epoch

（3）迭代结束把 $\theta_{\text{old}}\leftarrow\theta$，更新旧策略，进入下一轮

> **小结**
>
> 1.工程上把**策略损失、价值损失与熵奖励**合成总目标
>
> 2.用**截断 GAE**估计优势，训练采用**并行采样 + 小批量多 epoch 优化** + **滚动旧策略**

## 3. 手撕代码

[PPO代码实现](https://github.com/Yxccccccc/RLStudy/blob/main/Implements/PPO.ipynb)

## 4. References

[1]arXiv:[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)

[2]Spinning Up:[[Proximal Policy Optimization]](https://spinningup.openai.com/en/latest/algorithms/ppo.html?utm_source=chatgpt.com)

[3]Lil'Log:[Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/?utm_source=chatgpt.com)

[4]Stable-Baselines3:[PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html?utm_source=chatgpt.com)

[5]Github:[study_rlhf](https://github.com/wlll123456/study_rlhf)

[6]arXiv:[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438?utm_source=chatgpt.com)
