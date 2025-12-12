# GAE-Paper2Code

[TOC]

## 1. 广义优势估计（GAE）解决了什么问题？

策略梯度用优势函数 $A(s_t,a_t)$ 作为权重，但真实优势不可得，只能估计

直接用蒙特卡洛（整段回报）→ **无偏但高方差**

只用一步 TD 残差 → **低方差但有偏**

GAE 的出发点就是用一个时间尺度参数 $\lambda\in[0,1]$ 在两者之间连续插值。

## 2. Paper精读

### 2.1 预备知识

（1）策略梯度方法

策略梯度直接优化期望总回报 $J(\pi_\theta)=\mathbb{E}_{\pi_\theta}\big[\sum_{k=0}^{\infty}\gamma^k r_k\big]$，其中 $\theta$ 为策略参数 

常见的梯度形式为：
$$
\begin{equation}\nabla_\theta J(\pi_\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{k=0}^\infty\psi_k\nabla_\theta\log\pi_\theta(a_k|s_k)\right]\end{equation}
$$
其中 $\psi_k$ 可以是用于引导训练的信号，例如：折扣回报 $\sum_{k\ge0}\gamma^k r(s_k,a_k)$、折扣的状态—动作价值 $Q_{\gamma,\pi_\theta}(s_t,a_t)$、**优势函数** $A_{\gamma,\pi_\theta}(s_t,a_t)$ 等

若优势函数已知，则**优势函数能在这些选择中带来最低方差**

通常 $Q_{\gamma,\pi_\theta},V_{\gamma,\pi_\theta}$ 不可得，需要神经网络近似，故真实优势 $A_{\gamma,\pi_\theta}$ 也需近似

原论文中统一把优势估计写作 $\hat A(\cdot)$

（2）优势函数

有两种方式可以估计优势，分别为**蒙特卡洛方法（MC**）以及**时序差分方法（TD）**

蒙特卡洛估计优势通过对完整的轨迹进行采样，计算实际的回报来估计价值函数：
$$
\begin{equation}A(s,a)=R(s,a)-V(s)\end{equation}
$$
其中，$R(s,a)$ 是从状态 $s$ 采取动作 $a$ 后获得的实际总回报

通过这种方式，我们可以得到**无偏的优势估计**

但MC方法通常**无法用于在线场景**，只有在**整个**trajectory结束后才能得到估计结果，同时虽然得到的优势估计是无偏的，但**方差很大**

与蒙特卡洛方法不同，时序差分方法利用一步预测来估计价值函数，通过结合**当前时刻的估计值和下一个状态的估计值来更新当前的价值估计**：
$$
\begin{equation}A(s,a)=r+\gamma V(s^{\prime})-V(s)\end{equation}
$$
其中，$r$ 是当前状态 $s$ 采取动作 $a$ 所得到的即时奖励

时序差分方法相比蒙特卡洛方法有一些优点

首先，它可以在**不需要等待整个回合结束**的情况下进行更新，因此**适合在线学习**

其次，它的**估计结果方差较小**，因为它依赖于多个小步的估计，**平滑了每一步的波动**

然而，**时序差分方法也有局限性**

它虽然降低了方差，但由于使用了估计的价值来更新，可能会引入一些偏差

因此，如何在**偏差和方差之间进行权衡**，是估计价值函数时需要考虑的问题


> **小结**
>
> 给出策略梯度通式与“优势作权重”的低方差动机

### 2.2 广义优势估计（GAE）

GAE 通过在引入**可控偏差**的同时降低方差来缓解该问题 

它对传统优势估计进行**时间平滑**，使用**k 步估计器的指数加权平均**：
$$
\begin{equation}\hat{\mathcal{A}}_{\mathrm{GAE}}^{\gamma,\lambda}(s_t,a_t)=\sum_{k=0}^\infty(\gamma\lambda)^k\delta(s_{t+k},a_{t+k})\end{equation}
$$
其中：
$$
\begin{equation}\delta(s_t,a_t)=r_t(s_t,a_t)+\gamma V(s_{t+1})-V(s_t)\end{equation}
$$
$\delta(s_t,a_t)$为为 TD 误差，$\lambda\in(0,1)$ 为控制参数

参数 $\gamma,\lambda$ 共同决定偏差-方差平衡

考虑两种极端情况：

（1）当 $\lambda = 1$ 时：
$$
\begin{equation}\hat{A}_t^{(\lambda=1)}=\sum_{l=0}^\infty\gamma^l\delta_{t+l}^V=\sum_{l=0}^\infty\gamma^l(r_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l}))\end{equation}
$$
对右边做望远镜求和（把所有 $+\gamma^{l+1}V(s_{t+l+1})$ 与下一项的 $-\gamma^{l}V(s_{t+l})$ 抵消），得到：
$$
\begin{equation}\hat{A}_t^{(\lambda=1)}=\underbrace{\sum_{l=0}^\infty\gamma^lr_{t+l}}_{\text{MC 回报 }G_t}-V(s_t).\end{equation}
$$
也就是“**整段蒙特卡洛回报** $G_t$ **减去BaseLine** $V(s_t)$”

因此它是**无偏但高方差**的优势估计（偏差为 0 的原因是 MC 回报对期望回报是无偏估计）

GAE 原论文也明确说明：$\lambda=1$ 给出无偏估计，而 $\lambda<1$ 会有偏差”

（2）当 $\lambda=0$时：
$$
\begin{equation}\hat{A}_t^{(\lambda=0)}=\sum_{l=0}^\infty(\gamma\cdot0)^l\delta_{t+l}^V=\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t)\end{equation}
$$
这正是一**步 TD 误差**，也就是**TD(0)** 的目标

当把它当作优势去做策略梯度更新时，就是最经典的**一步 Actor–Critic**策略：

Actor用 $\delta_t^V$ 当权重更新，价值Critic拟合 $V$ 来提供这个 TD 目标

一步法**方差最低但偏差最大**（因为强烈依赖当前 $V$ 的近似）

（3）此时把两端连起来：**$\lambda$ 就是在MC 与 TD(0)之间打滑块**

一般的 $\lambda\in(0,1)$ 时，GAE 是对**各阶 $n$-步优势**做**指数加权平均**

相当于在“**高方差低偏差**”（$\lambda=1$ 的 MC 基线）与“**低方差高偏差**”（$\lambda=0$ 的一步 AC）之间**连续插值**

实际工程实现中，很多连续控制任务里 $\lambda\approx 0.9\sim0.99$ 往往更快、更稳

该**低偏差、低方差**的 $\hat A_{\gamma,\lambda}$ 可代入式(1) 引导策略更新

> **小结**
>
> 复述经典 GAE 的核心公式

### 2.3 实践

目前来讲，多数**On-Policy策略**算法（如 TRPO、PPO、A2C）默认启用 GAE 作为优势的估计方法

但 **Off-Policy** 算法（如 SAC、DQN）与 **IMPALA** 这类分布式方案通常不用 GAE

OpenAI Spinning Up 的官方 PPO 实现里，PPO 的经验缓冲区明确写着“用 GAE-Lambda 计算 advantage”

Stable-Baselines3（SB3） 的 PPO 参数里直接有 `gae_lambda`（“GAE 的偏差/方差权衡因子”），属于标准可调超参

RLlib（Ray）在 PPO 配置中提供 `use_gae`（并带 `lambda_`），默认按 GAE 算 advantage；还说明若不开 V-trace 就用 PPO 的 GAE

多数工程实现也把“**先用 GAE 算 advantage，再做归一化**”列为On-Policy算法的实现细节

**IMPALA/APPO** 等分布式 actor-learner 框架中，优势估计用 **V-trace** 重要性加权而非 GAE

**Off-Policy**策略算法核心是学习 **Q 函数/价值网络** 进行策略更新，不依赖 GAE 的优势估计

## 3. 手撕代码

```python
def compute_gae(
    rewards: torch.Tensor,            # [T, B] 或 [T]，float
    values: torch.Tensor,             # [T+1, B] 或 [T+1]，critic V(s)（含最后一步bootstrap）
    terminateds: torch.Tensor,        # [T, B] 或 [T]，bool/0-1：True=episode终止(Env.Terminated)
    truncateds: Optional[torch.Tensor]=None,  # [T, B] 或 [T]，bool/0-1：True=timeout(Env.Truncated)
    gamma: float = 0.99,
    lam: float = 0.95,
    advantage_norm: Literal["none","global","per_env"] = "global",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Generalized Advantage Estimation (GAE-Lambda)。

    约定/要点：
    - values 必须比 rewards 多 1 个时间步（最后一个是 bootstrap 的 V(s_{T})).
    - terminateds=True 表示**真正终止**（不可再bootstrap）；truncateds=True 表示**超时/截断**（允许bootstrap）。
    - 返回 (advantages, returns) 形状与 rewards 相同。
    - 返回张量均不带梯度（不反传到 critic），避免 value loss 的泄漏梯度问题。

    参考实现：
    - Spinning Up (PyTorch) 的 PPO 缓冲区：GAE-Lambda 回扫；SB3 的 returns/advantage 计算；CleanRL 的简洁写法。
      （本函数融合并拓展了 timeout 语义与归一化方式）.
    """
    # 统一成二维 [T, B]
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
        values = values.unsqueeze(1)
        terminateds = terminateds.unsqueeze(1)
        if truncateds is not None:
            truncateds = truncateds.unsqueeze(1)

    T, B = rewards.shape
    assert values.shape[0] == T + 1 and values.shape[1] == B, \
        f"values 应为 [T+1,B]，但得到 {tuple(values.shape)} 与 T={T}, B={B}"

    # 转 float/bool
    v = values.detach()
    r = rewards.detach()
    term = terminateds.to(dtype=torch.bool)
    if truncateds is None:
        trunc = torch.zeros_like(term)
    else:
        trunc = truncateds.to(dtype=torch.bool)

    adv = torch.zeros(T, B, device=rewards.device)
    last_gae = torch.zeros(B, device=rewards.device)

    # 反向时间回扫（标准 GAE-Lambda）
    for t in reversed(range(T)):
        # “下一步可bootstrap”的掩码：真正终止 -> 0；timeout(截断) -> 1
        next_nonterminal = (~term[t]).float()  # True->1., False(done)->0.
        # 若该步是 timeout/truncated，则按“非终止”处理（仍允许 bootstrap）
        next_nonterminal = torch.where(trunc[t], torch.ones_like(next_nonterminal), next_nonterminal)

        delta = r[t] + gamma * v[t + 1] * next_nonterminal - v[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae

    # 优势 + baseline 得到回报目标（value target）
    returns = adv + v[:-1]

    # 可选：优势标准化（实践中极其常见，提升数值稳定性）
    if advantage_norm != "none":
        if advantage_norm == "global":
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + eps)
        elif advantage_norm == "per_env":
            mean = adv.mean(dim=0, keepdim=True)
            std = adv.std(dim=0, keepdim=True, unbiased=False)
            adv = (adv - mean) / (std + eps)
        else:
            raise ValueError(f"advantage_norm={advantage_norm} 非法")

    return adv, returns
```

## 4. References

[1] arXiv: [Generalized Advantage Estimation for Distributional Policy Gradients](https://arxiv.org/abs/2507.17530)

[2] 知乎：[广义优势估计](https://zhuanlan.zhihu.com/p/717883634)
