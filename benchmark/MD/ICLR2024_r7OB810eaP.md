# NON-ERGODICITY IN REINFORCEMENT LEARNING: ROBUSTNESS VIA ERGODICITY TRANSFORMATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Envisioned application areas for reinforcement learning (RL) include autonomous driving, precision agriculture, and finance, which all require RL agents to make decisions in the real world. A significant challenge hindering the adoption of RL methods in these domains is the non-robustness of conventional algorithms. In this paper, we argue that a fundamental issue contributing to this lack of robustness lies in the focus on the expected value of the return as the sole "correct" optimization objective. The expected value is the average over the statistical ensemble of infinitely many trajectories. For non-ergodic returns, this average differs from the average over a single but infinitely long trajectory. Consequently, optimizing the expected value can lead to policies that yield exceptionally high returns with probability zero but almost surely result in catastrophic outcomes. This problem can be circumvented by transforming the time series of collected returns into one with ergodic increments. This transformation enables learning robust policies by optimizing the long-term return for individual agents rather than the average across infinitely many trajectories. We propose an algorithm for learning ergodicity transformations from data and demonstrate its effectiveness in an instructive, non-ergodic environment and on standard RL benchmarks.

# 1 INTRODUCTION

Reinforcement learning (RL) has experienced remarkable progress in recent years, particularly within virtual environments (Mnih et al., 2015; Silver et al., 2017; Duan et al., 2016; Vinyals et al., 2019). However, the seamless transition of RL methods to real-world applications lags behind, primarily due to the non-robust nature of conventional RL approaches (Amodei et al., 2016; Leike et al., 2017; Russell et al., 2015). In addressing this issue, researchers have explored a spectrum of methods from risk-sensitive RL (Prashanth et al., 2022) to robust (worst-case) RL (Pinto et al., 2017). In this paper, we take a step back and look at the optimization objective in RL and how it may, by design, result in non-robust policies. Traditional RL literature, including influential references and introductory textbooks such as Sutton & Barto (2018); Bertsekas (2019); Powell (2021), typically frames the RL problem as maximizing the expected return, i.e., the expected value of the sum of rewards collected throughout a trajectory. Intuitively, at each time step, an agent shall choose an action that maximizes the return it can expect when choosing this action and following the optimal policy from then onward. While this indeed seems intuitive, it becomes problematic when the returns are non-ergodic. When the returns are non-ergodic, the average over many trajectories—which resembles an expected value—differs from the average along one long trajectory. We find non-ergodic returns in various contexts, as we discuss in more detail in section 6. One example are settings in which we have "absorbing barriers," i.e., states, from which there is no return. Such as when an autonomous car crashes in an accident. Suppose an autonomous car learns a driving policy through RL. At deployment time, when we have a passenger in the car, it does not matter to the passenger whether the policy of the autonomous car receives a high return when averaging over multiple trajectories—a high ensemble-average return could also result from half of the journeys reaching the destination very fast and half crashing and never reaching it. The return in a single instance of a long journey would be negligible if a crash occurred somewhere along the way—and this is the return that would matter to the individual. Thus, the time average would be the better choice for an optimization objective in such scenarios.

Optimizing the time average might require developing entirely new RL algorithms. Nevertheless, existing RL algorithms have demonstrated remarkable performance by optimizing expected returns. An alternative is to find a suitable transformation. This is related to human decision-making. In economics and game theory, researchers have found that humans typically do not optimize expected monetary returns (Bernoulli, 1954), which would correspond to optimizing across a statistical ensemble. Instead, they seem to optimize along individual time trajectories, corresponding to different behavioral protocols unless monetary returns are state-independent, i.e., independent of the current wealth level. Optimization along time trajectories can be implemented by a state-dependent transformation of monetary returns chosen so as to make changes in the transformed quantity ergodic. Optimizing expected values of these changes then also optimizes long-term growth along an individual trajectory. As for the autonomous car, so for the human, it appears more natural to care about long-term performance. For the individual person, it typically does not matter whether a particular investment pays off when averaged over a statistical ensemble—instead, what matters is whether or not investing according to some protocol pays off in the long run in the single trajectory.

Motivated by economics, in particular, by utility (Bernoulli, 1954) and prospect (Kahneman & Tversky, 1997) theory, the field of risk-sensitive RL (Prashanth et al., 2022) has emerged. In most of risk-sensitive RL, e.g., algorithms using an entropic risk measure, the agents try to optimize the expected value of transformed returns. By learning with transformed returns, the agents can achieve higher performance with lower variance. Utility and prospect theory do not consider potential nonergodicity. Instead, these theories rely on psychological arguments to argue that some humans are more "risk-averse" than others. Peters & Adamou (2018) have shown how acknowledging nonergodicity and that humans are more likely to optimize the long-term return than an average over an ensemble of infinitely many trajectories can recover widespread transformations used in utility theory. Empirical research (Meder et al., 2021; Vanhoyweghen et al., 2022) has further shown that this treatment can better predict actual human behavior. The ergodicity perspective does not rely on psychology as an explanation; instead, it explains psychological observations. It is, in this sense, more fundamental and, as a result, more general, namely applicable to cases where psychology cannot be invoked, particularly to inanimate optimizers such as machines devoid of a psyche.

Inspired by Peters & Adamou (2018), we analyze for which dynamics a popular transformation from risk-sensitive RL optimizes the long-term return. Further, we propose an algorithm for learning a suitable transformation when the reward function is unknown, which is the typical setting in RL.

Contributions. In this paper, we make the following contributions:

- We illustrate and assess the impact of non-ergodic returns on RL algorithm policies through an intuitive example. This showcases the implications of optimizing for the expected value in non-ergodic settings—which we commonly encounter in RL problems—and it makes a case for the need for an ergodicity transformation.  
- We propose a transformation that can convert a trajectory of returns into a trajectory with ergodic increments. This enables off-the-shelf RL algorithms to optimize their long-term return instead of the conventional expected value, resulting in more robust policies without the need to develop novel RL algorithms.  
- We demonstrate the performance of this transformation in an intuitive example and, as a proof-of-concept, on standard RL benchmarks. In particular, we show that our transformation indeed yields more robust policies when returns are non-ergodic.

# 2 PROBLEM SETTING

We consider a standard RL setting in which an agent with states  $s \in S \subseteq \mathbb{R}^n$  in the state space  $S$  and actions  $a \in A \subseteq \mathbb{R}^m$  in the action space  $A$  shall learn a policy  $\pi: S \to A$ . Its performance is measured by an unknown reward function  $r: S \times A \to \mathbb{R}$ . The agent's goal is to maximize the accumulated rewards  $r(t_k)$  it receives during a trajectory, i.e., the return  $R(T)$  at  $t_k = T$ ,

$$
R (T) = \sum_ {\tau_ {k} = 0} ^ {T} r (\tau_ {k}), \tag {1}
$$

where  $r(t_k) \coloneqq r(s(t_k), a(t_k))$ . For this, the agent interacts with its environment by selecting actions, receiving rewards, and utilizing this feedback to learn an optimal policy. The RL problem is in-

herently stochastic, as it involves learning from finite samples, often within stochastic environments and with potentially stochastic policies. In standard RL, we, therefore, typically aim at maximizing the expected value of equation 1 (cf. the "reward hypothesis" stated by (Sutton & Barto, 2018, p. 53))

$$
\mathbb {E} \left[ \sum_ {\tau_ {k} = 0} ^ {T} r \left(\tau_ {k}\right) \right]. \tag {2}
$$

Nonetheless, this conventional approach may encounter challenges when the dynamics are nonergodic. To illustrate this point, we consider an instructive example introduced by Peters (2019).

# 2.1 ILLUSTRATIVE EXAMPLE

Imagine an agent starting with an initial reward of  $r(t_0) = 100$  is offered the following game. We toss a (fair) coin. If it comes up heads, the agent wins  $50\%$  of its current return. If it comes up tails, the agent loses  $40\%$ . Mathematically, this translates to

$$
r (t _ {k}) = \left\{ \begin{array}{l l} 0. 5 R (t _ {k} - 1) & \text {i f} \eta = 1, \\ - 0. 4 R (t _ {k} - 1) & \text {o t h e r w i s e}, \end{array} \right.
$$

where  $\eta$  is a Bernoulli random variable with equal probability for both outcomes.

When analyzing the game dynamics, we find that the agent receives an expected reward  $\mathrm{r(t_k)}$  equal to  $5\%$  of its current return. Consequently, the expected return for any trajectory length  $T$  appears favorable, growing exponentially with  $T$ :

$$
\mathbb {E} [ R (T) ] = 1 0 0 \cdot 1. 0 5 ^ {\mathrm {T}}.
$$

However, when we simulate the game for ten agents and 1000 time steps, we find that all of them end up having a return of almost zero (see figure 1). The reason is that the coin toss game is non-ergodic. If

the dynamics of a stochastic process are non-ergodic, the average over infinitely many samples may be arbitrarily different from the average over a single but infinitely long trajectory. Translated to the coin toss example, if we simulate infinitely many trajectories of the game, each of finite duration  $T$ , we obtain a small set of agents that end up exponentially "rich" so that averaging over all of them, i.e., taking the expected value, yields  $100 \cdot 1.05^T$ . However, if we increase the duration,  $T \to \infty$ , the set of agents ending up exponentially rich shrinks exponentially to measure zero. That is, if we only simulate one agent for  $T \to \infty$  and average over time, we receive a time average  $\lim_{T \to \infty} \frac{1}{T} \sum_{\tau_k = 0}^{T} r(\tau_k) = 0$  almost surely. A summary of the statistical properties of the coin-toss game can be found in (Hulme et al., 2023, Appendix).

To define ergodicity properly and connect it explicitly to RL, let us abstract from the coin-toss example and consider an arbitrary discrete-time stochastic process  $X$ . We can now generate multiple realizations of this process, in the example, by playing the game multiple times. Let  $X^{(j)}(t_k)$  denote the value of realization  $j$  at time step  $t_k$ . The process  $X$  is ergodic if, for any time step  $t_k$  and realization  $i$ ,

$$
\lim  _ {N \rightarrow \infty} \frac {1}{N} \sum_ {j = 1} ^ {N} X ^ {(j)} \left(t _ {k}\right) = \lim  _ {T \rightarrow \infty} \frac {1}{T} \sum_ {\tau_ {k} = 1} ^ {T} X ^ {(i)} \left(\tau_ {k}\right) \quad \text {a l m o s t s u r e l y}. \tag {3}
$$

The left hand side is  $\mathbb{E}[X(t_k)]$ , the expected value of  $X$  at time  $t_k$ . The right-hand side is the time average of realization  $i$ . For an ergodic process, these averages are equal. In the RL setting, we are interested in whether or not the rewards  $r(t_k)$  are ergodic:

$$
\mathbb {E} [ r (t _ {k}) ] = \lim  _ {T \rightarrow \infty} \frac {1}{T} \sum_ {\tau_ {k} = 1} ^ {T} r (\tau_ {k}) = \lim  _ {T \rightarrow \infty} \frac {R (T)}{T} \quad \text {a l m o s t s u r e l y}. \tag {4}
$$

![](images/84659b5c29dbd3c62f20ea33e3e0d9a649bbed11cfb621793ed7af89a0748278.jpg)  
Figure 1: Simulation of the coin toss experiment. We simulate the game for 1000 time steps and 10 agents. The dashed red horizontal line marks the initial reward of 100, and the dashed blue ascending line the expected value. After 1000 time steps, all agents end up with a lower return than they started with (note the logarithmic scaling of the y-axis).

For ergodic rewards, maximizing the expected value at each step corresponds to maximizing the long-term growth rate of the return for any given realization. However, as the coin-toss example demonstrates, when rewards are non-ergodic, optimizing the expected value may yield policies with negative long-term growth rate.

# 2.2 SOLVING THE ERGODICITY PROBLEM

Redefining the optimization objective of RL algorithms may require a complete redesign. Alternatively, we can take existing algorithms and modify the returns to make their increments ergodic. Peters & Adamou (2018) have shown, in a continuous-time setting, that for a broad class of stochastic processes, we can find transformations  $\mathsf{h}(\mathsf{R})$  such that their increments  $\Delta \mathsf{h}$  are ergodic and follow a standard Brownian motion. In our discrete-time setting, this translates to

$$
h \left(R \left(t _ {k} + 1\right)\right) = h \left(R \left(t _ {k}\right)\right) + \mu + \sigma v \left(t _ {k}\right), \tag {5}
$$

with drift  $\mu$ , volatility  $\sigma$ , and standard normal random variable  $\nu(t_k)$ . For our purposes, where we want to learn a transformation  $h$  from data instead of deriving it analytically as Peters & Adamou (2018), it even suffices if  $\nu(t_k)$  has finite variance, i.e., it does not have to be normally distributed.

In the following, we assess the performance of standard RL algorithms in the coin toss game, with and without a transformation  $h$ . We then propose an algorithm for learning a transformation  $h$  with ergodic increments and relate our findings to risk-sensitive RL.

# 3 RL WITH NON-ERGODIC DYNAMICS

For the coin toss example, we can easily verify that the dynamics are non-ergodic. Optimizing the expected value then yields a "policy" in which the agent decides to play the game, leading to ruin in the long run almost surely. While standard RL algorithms aim at optimizing the expected value, they need to approximate it from finitely many samples. Thus, we evaluate whether a standard RL algorithm indeed proposes a detrimental policy in this section and discuss how we can transform the returns to prevent this. In the version presented in the previous section, the coin toss game offers the agent a binary decision: either play or not. Here, we make the game slightly more challenging by letting the agent decide how much of its current return ("wealth") it invests at each time step. Thus, we have a continuous variable  $F \in [0,1]$  and the dynamics for the reward are

$$
r (t _ {k}) = \left\{ \begin{array}{l l} 0. 5 F R (t _ {k} - 1) & \text {i f} \eta = 1, \\ - 0. 4 F R (t _ {k} - 1) & \text {o t h e r w i s e .} \end{array} \right.
$$

We use the popular proximal policy optimization (PPO) algorithm (Schulman et al., 2017), leveraging the implementation provided by Raffin et al. (2021) without changing any hyperparameters, to learn a policy. Having trained a policy for  $1 \times 10^{5}$  episodes, we execute it 100 times for 1000 time steps and show the first ten trajectories in figure 2a. We see that all ten agents end up with a return lower than the initial reward of 100. While this could still be caused by a bad choice of agents, it is confirmed by computing statistics over all 100 trajectories. When computing the median of the return after 1000 time steps, we obtain  $2.5 \times 10^{-4}$ , i.e., the average agent ends up with a return close to zero. The mean over all agents yields 115. That is, a small subset of agents obtains a high return. This confirms the discussion from the previous section. Even if it only approximates the expected value, PPO does learn a policy that leads to ruin for most agents.

One possibility for coping with non-ergodic dynamics is finding a suitable transformation. For the coin toss game, where the dynamics are relatively straightforward and the outcomes are fully known, we can analytically identify an appropriate transformation: the logarithm (Hulme et al., 2023, Appendix). We subsequently train the PPO algorithm once more with the logarithmic transformation. Specifically, we redefine the rewards as  $\tilde{\mathbf{r}}(t_k) = \log(\mathsf{R}(t_k)) - \log(\mathsf{R}(t_{k-1}))$ . As before, we run 100 experiments for 1000 time steps each and show the first ten trajectories in figure 2b. We see that all agents end up with a significantly higher return compared to the initial reward. A statistical analysis confirms this observation, yielding a median return of 5645 and a mean of 15883. Both values substantially surpass those obtained by the agents trained with untransformed returns.

This evaluation underscores that standard RL algorithms may inadvertently learn policies leading to unfavorable outcomes for most agents when dealing with non-ergodic dynamics. Furthermore, it demonstrates that an appropriate transformation can mitigate this.

![](images/8ccc29217bf2cfd6feb2874039d6502aa677deeed29e196b9a4b2dea86cf6f28.jpg)  
(a) Without transformation.

![](images/8289e7891bed0e8e89e093d6f7a3d3679eb4c64f709828258225ed60ba1bf5ae.jpg)  
Figure 2: Learning bet strategies for the adapted coin toss game. Without transformation, most agents end up losing, while they end up winning with transformation.  
(b) With transformation.

Remark 1. The quantitative results clearly differ between runs, as environment and training process are stochastic. Nevertheless, the qualitative results are consistent: the training with transformed returns results in better performance. With transformed returns, the agents sometimes get trapped in local optima with  $\mathsf{F} = 0$ , which still results in significantly higher returns for the average agent.

# 4 LEARNING AN ERGODICITY TRANSFORMATION

In scenarios like the coin toss game, due to the perfect information of future returns, it is possible to derive a suitable transformation analytically—Peters & Adamou (2018) provide a more detailed discussion for more general dynamics. However, the true power of reinforcement learning (RL) lies in its ability to handle complex environments for which we lack accurate analytical expressions. Therefore, it is desirable to learn transformations directly from data.

The central characteristic of the transformation is that it should render the increments of the transformed return ergodic. Ideally, we aim for a transformation whose increments are independent and identically distributed (i.i.d.). However, determining this i.i.d. property with a high degree of accuracy, especially from real-world data, can be challenging. Instead, we approximate the behavior of the transform to that of a variance-stabilizing transform.

Definition 1 (Bartlett (1947)). A variance stabilizing transform is defined as

$$
h (x) = \int_ {0} ^ {x} \frac {1}{\sqrt {v (u)}} d u,
$$

with variance function  $\nu(\mathfrak{u})$  describing the variance of a random variable as a function of its mean.

A variance stabilizing transform aims to transform a given time series into one with constant variance, independent of the mean (Bartlett, 1947). This is a generalization of our desired i.i.d. property as if the transformation  $\mathsf{h}(\mathsf{R}(\mathsf{t}_{\mathsf{k}}))$  has i.i.d. increments, then the increments also have constant variance, independent of the mean. Thus, our objective becomes finding a variance stabilizing transform following definition 1. In our case, as we want to stabilize the variance of the increments, we adapt the original definition of the variance function  $\nu (\mathfrak{u})$  in definition 1 to

$$
v (u) = \operatorname {V a r} \left[ R \left(t _ {k + 1}\right) - R \left(t _ {k}\right) \mid R \left(t _ {k}\right) = u \right].
$$

This variance function represents the variance of the following increment as a function of the current transformed return.

The approach for estimating  $\nu(u)$  from data is inspired by the additivity and variance stabilization method for regression (Tibshirani, 1988). Estimating  $\nu(u)$  first involves plotting  $\mathsf{R}(t_k)$  against  $\log((\mathsf{R}(t_{k+1}) - \mathsf{R}(t_k) - \hat{\mu})^2)$ , with  $\hat{\mu}$  the empirical mean of the increments. In our setting, the mean of the increments of the original untransformed process may not be constant throughout a trajectory. Hence, assuming a constant  $\hat{\mu}$  results in small values having an over-estimated variance and large values having an under-estimated variance. The straightforward way to fix this would be to estimate  $\mu(u)$  as a function of  $u$ ; however, this introduces a further estimation problem. Instead, we can

estimate the second moment function and use this as a proxy for the variance function,

$$
\mu^ {2} (u) = \mathbb {E} \left[ \left(\mathrm {R} \left(t _ {k + 1}\right) - \mathrm {R} \left(t _ {k}\right)\right) ^ {2} \mid \mathrm {R} \left(t _ {k}\right) = u \right].
$$

In the supplementary material, we show that  $\mu^2(u) \propto \nu(u)$ , which is satisfactory for our needs as if the process  $\mathsf{R}(\mathfrak{t}_k)$  has i.i.d. increments, then so will the process  $a \cdot \mathsf{R}(\mathfrak{t}_k)$  for any  $a \in \mathbb{R}$ .

To estimate the function  $\log (\mu^2 (u))$  we plot  $\mathsf{R}(\mathsf{t}_k)$  against  $\log ((\mathsf{R}(\mathsf{t}_{k + 1}) - \mathsf{R}(\mathsf{t}_k))^2)$ . Then, fitting a curve represents taking the expected value. We use the locally estimated scatter-plot smoothing (LOESS) method (Cleveland, 1979). The reason behind estimating  $\log (\mu^2 (u))$  is that this guarantees  $\mu^2 (u)$  always to be positive, which is vital as the variance stabilizing transform requires us to take the square root. This approach follows the reasoning by Tibshirani (1988). We provide a Python implementation of the transformation and the coin toss example in the supplementary material.

Having derived this transformation, we apply it to the coin toss game. We first collect a return trajectory with  $\mathsf{F} = 1$ . Based on this trajectory, we learn an ergodicity transformation following the steps described in this section. Then, we again train a PPO agent but feed it the increments of transformed returns as previously with the logarithmic transformation. As before, we execute the learned policy 100 times for 1000 time steps each and show rollouts for the first ten agents in figure 3. Also with this transformation, most agents end up learning winning strategies. The statistics confirm this: across all 100 agents, we have a median return of around 17517 and an average return of around 956884. Thus, we conclude that we can learn a suitable transformation from data, enabling PPO to learn a policy that benefits individual agents in the long run.

![](images/a69005e1a2e23d0f2289677fd1264593fddfbe32d3a81a449cebcab8d817baec.jpg)  
Figure 3: Learning bet strategies for the adapted coin toss game with learned transformation. Similar to the logarithm, also with the learned transformation, the majority of the agents ends up winning.

# 5 RISK-SENSITIVE RL

The ergodicity transformation serves as a means for RL agents to optimize the long-term performance of individual returns, enabling the learning of robust policies, as demonstrated in figure 3. Another approach to improving the robustness of RL algorithms is through risk-sensitive RL. While risk-sensitive RL is not motivated by ergodicity, it also proposes transforming returns. Inspired by Peters & Adamou (2018), we can analyze these transformations and determine under which dynamics they yield transformed returns with ergodic increments. This analysis allows us to gain insights into which type of transformation may offer robust performance in which settings.

Here, we focus on the exponential transformation,

$$
h _ {\mathrm {r s}} (R) := \beta \exp (\beta R),
$$

where  $\beta \in \mathbb{R} \setminus \{0\}$  is a hyperparameter with  $\beta < 0$  the "risk-averse", and  $\beta > 0$  "risk-seeking" case.

For the sake of clarity, we perform our analysis in continuous time. We assume that the return follows an arbitrary Ito process

$$
\mathrm {d} R = f (R) \mathrm {d} t + g (R) \mathrm {d} W (t), \tag {6}
$$

where  $f(R)$  and  $g(R)$  are arbitrary functions of  $R$  and  $W(t)$  is a Wiener process. This captures a large class of stochastic processes, as both  $f$  and  $g$  can be nonlinear and even stochastic. We further assume that the risk-sensitive transformation  $h_{\mathrm{rs}}$  extracts an ergodic observable from equation 6 such that its increments follow a Brownian motion, i.e., the continuous-time version of equation 5:

$$
\mathrm {d h} _ {\mathrm {r s}} = \mu \mathrm {d t} + \sigma \mathrm {d W} (\mathrm {t}). \tag {7}
$$

As we know  $h_{\mathrm{rs}}$ , we now seek to find  $f$  and  $g$  for which equation 7 holds.

Following Itô's lemma (Itô, 1944), we can write dR as

$$
d R = \left(\frac {\partial R}{\partial t} + \mu \frac {\partial R}{\partial h _ {r s}} + \frac {1}{2} \sigma^ {2} \frac {\partial^ {2} R}{\partial h _ {r s} ^ {2}}\right) d t + \sigma \frac {\partial R}{\partial h _ {r s}} d W (t). \tag {8}
$$

As we can invert  $h_{\mathrm{rs}}(R)$  such that  $R(h_{\mathrm{rs}}) = \frac{\ln\left(\frac{h_{\mathrm{rs}}}{\beta}\right)}{\beta}$  and since the inverse is twice differentiable, we can insert it into equation 8 and obtain

$$
\mathrm {d} R = \left(\frac {\mu}{\beta h _ {\mathrm {r s}}} - \frac {1}{2} \frac {\sigma^ {2}}{\beta h _ {\mathrm {r s}} ^ {2}}\right) \mathrm {d} t + \frac {\sigma}{\beta h _ {\mathrm {r s}}} \mathrm {d} W (t)
$$

$$
\mathrm {d} R = \left(\frac {\mu}{\beta^ {2} \exp (\beta R)} - \frac {1}{2} \frac {\sigma^ {2}}{\beta^ {3} \exp (2 \beta R)}\right) \mathrm {d} t + \frac {\sigma}{\beta^ {2} \exp (\beta R)} \mathrm {d} W (t). \tag {9}
$$

This equation provides valuable insights into the role of  $\beta$ . Specifically, it highlights that the volatility term (the coefficient of  $\mathrm{d}W(t)$ ) is always positive, regardless of the sign of  $\beta$ . However, the drift term (the coefficient of  $\mathrm{dt}$ ) depends on the sign of  $\beta$ . For  $\beta < 0$ , the drift term is positive, while for  $\beta > 0$ , it starts negative when  $\beta$  is small and then turns positive as  $\beta$  increases.

From an ergodicity perspective, the risk-averse variant with  $\beta < 0$  is suitable when equation 9 exhibits a positive drift, while the risk-seeking variant with  $\beta > 0$  is more appropriate when equation 9 has a negative drift. This aligns with intuitive reasoning: when the drift is negative, there is limited gain from caution, and one might choose to go all in and hope for luck. This is also the case when the drift is too small to outweigh the volatility.

The differential dynamics in equation 9 have a closed-form solution. As the derivations are relatively lengthy, we defer them to the appendix, and here provide directly the solution:

$$
R _ {t} = \frac {1}{\beta} \ln \left| \frac {\sigma}{\beta} \right| + \frac {1}{\beta} \ln \left| \frac {\mu}{\sigma} t + W _ {t} + \frac {\beta}{\sigma} \right|. \tag {10}
$$

The obtained return dynamics are logarithmic in time. Logarithmic returns (or regrets) are common in the RL literature. Consider a scenario where a robot arm must reach a set point, and the reward is defined as the negative distance to that set point. Initially, rapid progress can be made by moving quickly in the roughly correct direction. As the robot gets closer, the movement becomes more fine-grained and slower, resulting in slower progress. By using an exponential transformation, we counteract this phenomenon, ensuring that all time steps contribute equally to the return.

We next apply the exponential transformation to the coin- toss game and test both the "risk-averse" and the "risk-seeking" setting. For the risk-seeking setting  $(\beta >0)$ , we quickly run into numerical problems. The coin-oss problem has itself exponential dynamics, and thus, returns can get large. Exponentiating those again lets us reach the limits of machine precision. For the risk-averse setting  $(\beta < 0)$ , we consistently learn constant policies with  $F = 0$ . While this is still better than the policies standard PPO learned, it cannot compete with the results from figure 3.

This outcome is not surprising. From an ergodicity perspective, the exponential transformation is only suitable if the dynamics are logarithmic. The dynamics of the coin-toss game are exponential, which is precisely the inverse behavior. Thus, we would not expect the transformation to yield good policies, as is confirmed by our experiments.

# 6 ERGODICITY IN RL AND RELATED WORK

The coin-toss game is an excellent example to illustrate the problem of maximizing the expected value of non-ergodic rewards. When maximizing non-ergodic rewards, we may end up with a policy that receives an arbitrarily high return with probability zero but leads to failure almost surely. Also in less extreme cases, the expected value prefers risky policies if their return in case of success outweighs the failure cases. This results in learning non-robust policies, a behavior frequently observed in standard RL algorithms (Amodei et al., 2016; Leike et al., 2017; Russell et al., 2015).

Non-ergodicity is not unique to the coin-toss game. Peters & Klein (2013) have shown that geometric Brownian motion (GBM) is a non-ergodic stochastic process. GBM is commonly used to model economic processes, a domain where RL algorithms are increasingly applied (Charpentier et al., 2021; Zheng et al., 2022). Thus, especially in economics, ergodicity should not simply be assumed. Nevertheless, the example of GBM is also informative for other applications. Generally, RL is most interesting when the environment dynamics are too complex to model, i.e., we usually deal with nonlinear dynamics. If already a linear stochastic process such as GBM is non-ergodic, we cannot assume ergodicity for the general dynamics we typically consider in RL.

Another way of "ergodicity-breaking" is often motivated using the example of Russian roulette (Ornstein, 1973). When multiple people play Russian roulette for one round each, and their average outcome is considered, the probability of death is one in six. However, if a single person plays the game infinitely many times, that person will eventually die with probability one. In the context of RL, this is akin to the presence of absorbing barriers or safety thresholds that an agent must not cross. Particularly in RL applications where the consequences of failure can be catastrophic, such as in autonomous driving (Brunke et al., 2022), these safety thresholds become vital.

Consequently, in the literature on Markov decision processes (MDPs), we find work that argues about the (non-)ergodicity of MDPs, see, for instance, (Sutton & Barto, 2018, Ch. 10) or (Puterman, 2014, Ch. 8). Therein, the notion of ergodicity is mainly used to describe MDPs in which every state will be visited eventually. Following this notion, there has been work within the RL community that provides guarantees while explicitly assuming ergodicity (Pesquerel & Maillard, 2022; Ok et al., 2018; Agarwal et al., 2022) or by guaranteeing to avoid any states within an "absorbing" barrier, i.e., only exploring an ergodic sub-MDP (Turchetta et al., 2016; Heim et al., 2020). For Q-learning, Majeed & Hutter (2018) have shown convergence even for non-ergodic and non-MDP processes. Nevertheless, none of these works, as a consequence of non-ergodicity, question the use of the expectation operator in the objective function.

In this paper, we have proposed to transform returns to deal with non-ergodic rewards. In the previous section, we have shown how a popular transformation from risk-sensitive RL (Mihatsch & Neuneier, 2002; Shen et al., 2014; Fei et al., 2021; Noorani & Baras, 2021; Noorani et al., 2022; Prashanth et al., 2022) can be motivated from an ergodicity perspective. Reward-weighted regression (Peters & Schaal, 2007; 2008; Wierstra et al., 2008; Abdelmaleki et al., 2018; Peng et al., 2019) also proposes to use transformations, but the transformations are typically justified using intuitive arguments instead of from an ergodicity perspective. Interestingly, most existing work also uses an exponential transformation, which is the cornerstone of risk-sensitive control. Thus, the analysis we have done for risk-sensitive RL also applies to reward-weighted regression.

Another approach that optimizes transformed returns is Bayesian optimization for iterative learning (BOIL) (Nguyen et al., 2020). BOIL is developed for hyperparameter optimization. While this setting is different from the one we consider, we show in the supplementary material that the transformation used in BOIL can be replaced with ours, leading to similar or better results.

Through the ergodicity transformation, we seek to optimize the long-term performance of RL agents. Improving the long-term performance of RL agents in continuous tasks is also the goal of average reward RL. The idea of optimizing the average reward criterion originated in dynamic programming (Howard, 1960; Blackwell, 1962; Veinott, 1966), and has already in the early days of RL been taken up to develop various algorithms, see, for instance, the survey by Mahadevan (1996). Also in recent years, the average reward criterion has been used for novel RL algorithms (Zhang & Ross, 2021; Wei et al., 2020; 2022). In average reward RL, we still take the expected value of the reward function and let time go to infinity. Were the reward function ergodic, it would not matter whether we first take the expected value or first let time go to infinity. However, for a non-ergodic function, it does. In average reward RL, we first take the expected value. For the coin-toss game, that would yield an optimization criterion that grows exponentially while the set of agents that obtain a return larger than zero shrinks to a set of measure zero as time goes to infinity. Thus, average reward RL may fall into the same trap as conventional RL when dealing with non-ergodic reward functions.

# 7 PROOF-OF-CONCEPT

The coin-toss game, while illustrative, represents a simplified scenario. To establish the broader applicability of the ergodicity perspective and associated transformations in RL, we conducted experiments on two classical RL benchmarks: the cart-pole system and the reacher, using the implementations provided by Brockman et al. (2016). Both environments feature discrete action spaces. Thus, instead of PPO, which is designed for continuous action spaces, we use the REINFORCE algorithm (Williams, 1992). The REINFORCE algorithm is a Monte-Carlo algorithm. It always collects a return trajectory and then uses this trajectory to update its weights. In our setting, this is advantageous as it allows us to learn a transformation using the collected trajectory.

![](images/ef53bdbfb03df59a9ea058d40ccdaec442b3e52e75b98ef16b6a955e8d414dab.jpg)  
Figure 4: Ergodic vs. standard REINFORCE on common benchmarks. For the cart-pole, we see slight improvements when using the ergodicity transformation, while for the reacher, only ergodic REINFORCE learns a successful policy.

We here compare two settings. First, we train the algorithm in the standard way. Second, after collecting a return trajectory, we first derive the transformation, transform the returns, and then use the transformed returns to update the REINFORCE algorithm. In the plots, we always show the untransformed returns. In both settings, we change the length of pole and links for cart-pole and reacher, respectively, during testing to evaluate the robustness of the learned policies. Further details on hyperparameter choices are provided in the supplementary material.

Cart-pole. In the cart-pole environment, the objective is to maintain the pole in an upright position for as long as possible. To evaluate the long-term performance of the ergodicity transformation, we train the algorithm using episode lengths of 100 time steps but test it with episodes lasting 200 time steps. Thus, as we see in figure 4a, the return during testing is higher than during training. We also see that with transformation, the return is slightly higher than with the standard training procedure. While the differences are not dramatic in this relatively simple environment, they demonstrate the potential benefits of our approach.

Reacher. In the reacher environment, we aim to track a set point with the end of the last link. Thus, extending the episode length does not make sense in this setting. However, this is unnecessary to demonstrate the advantage of using the ergodicity transformation. As we see in figure 4b, only the ergodic REINFORCE algorithm learns a reasonable policy in this more challenging environment.

# 8 CONCLUSIONS AND LIMITATIONS

This paper discussed the impact of ergodicity on the choice of the optimization criterion in RL. If the rewards are non-ergodic, focusing on the expected return yields non-robust policies that we currently find with conventional RL algorithms. An alternative to changing the objective and, with this, having to come up with entirely new RL algorithms is trying to find an ergodicity transformation. We presented a method for learning an ergodicity transformation that converts a time series of returns into a time series with ergodic increments. We further showed how the ergodicity perspective provides a theoretical foundation for transformations used in risk-sensitive RL. We demonstrated the effectiveness of the proposed transformation on standard RL benchmark environments.

This paper is the first step toward acknowledging non-ergodicity of reward functions and focusing on the long-term return and, with that, robustness in RL. This opens various directions for future research. Firstly, addressing the challenge of transforming returns in RL algorithms that update weights incrementally rather than relying on episodic data remains an open question. Secondly, our transformation currently focuses solely on the current return, but returns may also depend on the current state of the system, suggesting the possibility of state-dependent transformations. Thirdly, extending this research to multi-agent RL could be promising, building on insights from Fant et al. (2023) and Peters & Adamou (2022) regarding the impact of non-ergodicity on the emergence of cooperation in biological multi-agent systems. Finally, investigating the connection between optimizing time-average growth rates instead of expected values and discount factors, as explored by Adamou et al. (2021), could make the discount factor as a hyperparameter in RL dispensable.

# REPRODUCIBILITY STATEMENT

The algorithm for learning an ergodicity transformation introduced in section 4 is contained in the supplementary material. The supplementary material also contains an implementation of the coin-toss game and code for training a standard PPO agent, a PPO agent with the logarithmic transformation, and a PPO agent with the ergodicity transformation from section 4 on the game.

# REFERENCES

Abbas Abdelmaleki, Jost Tobias Springenberg, Jonas Degrave, Steven Bohez, Yuval Tassa, Dan Belov, Nicolas Heess, and Martin Riedmiller. Relative entropy regularized policy iteration. arXiv preprint arXiv:1812.02256, 2018.  
Alexander Adamou, Yonatan Berman, Diomides Mavroyiannis, and Ole Peters. Microfoundations of discounting. Decision Analysis, 18(4):257-272, 2021.  
Mridul Agarwal, Qinbo Bai, and Vaneet Aggarwal. Regret guarantees for model-based reinforcement learning with long-term average constraints. In Uncertainty in Artificial Intelligence, pp. 22-31, 2022.  
Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. arXiv preprint arXiv:1606.06565, 2016.  
Maurice S Bartlett. The use of transformations. Biometrics, 3(1):39-52, 1947.  
Daniel Bernoulli. Exposition of a new theory on the measurement of risk. *Econometrica*, 22(1): 23-36, 1954.  
Dimitri Bertsekas. Reinforcement Learning and Optimal Control. Athena Scientific, 2019.  
David Blackwell. Discrete dynamic programming. The Annals of Mathematical Statistics, pp. 719-726, 1962.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI Gym. arXiv preprint arXiv:1606.01540, 2016.  
Lukas Brunke, Melissa Greeff, Adam W Hall, Zhaocong Yuan, Siqi Zhou, Jacopo Panerati, and Angela P Schoellig. Safe learning in robotics: From learning-based control to safe reinforcement learning. Annual Review of Control, Robotics, and Autonomous Systems, 5:411-444, 2022.  
Arthur Charpentier, Romuald Elie, and Carl Remlinger. Reinforcement learning in economics and finance. Computational Economics, pp. 1-38, 2021.  
William S Cleveland. Robust locally weighted regression and smoothing scatterplots. Journal of the American Statistical Association, 74(368):829-836, 1979.  
Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning, pp. 1329-1338, 2016.  
Lorenzo Fant, Onofrio Mazzarisi, Emanuele Panizon, and Jacopo Grilli. Stable cooperation emerges in stochastic multiplicative growth. Physical Review E, 108(1):L012401, 2023.  
Yingjie Fei, Zhuoran Yang, and Zhaoran Wang. Risk-sensitive reinforcement learning with function approximation: A debiasing approach. In International Conference on Machine Learning, pp. 3198-3207. PMLR, 2021.  
Steve Heim, Alexander von Rohr, Sebastian Trimpe, and Alexander Badri-Spröwitz. A learnable safety measure. In Conference on Robot Learning, pp. 627-639, 2020.  
Ronald A Howard. Dynamic Programming and Markov Processes. John Wiley, 1960.  
Oliver Hulme, Arne Vanhoyweghen, Colm Connaughton, Ole Peters, Simon Steinkamp, Alexander Adamou, Dominik Baumann, Vincent Ginis, Bert Verbruggen, James Price, and Benjamin Skjold. Reply to" the limitations of growth-optimal approaches to decision making under uncertainty". Econ Journal Watch, 20(2):335-348, 2023.  
Kiyosi Ito. Stochastic integral. Proceedings of the Imperial Academy, 20(8):519-524, 1944.  
Donald R Jones, Matthias Schonlau, and William J Welch. Efficient global optimization of expensive black-box functions. Journal of Global optimization, 13(4):455, 1998.

Daniel Kahneman and Amos Tversky. Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2):263-292, 1997.  
Peter E. Kloeden and Eckhard Platen. Numerical Solution of Stochastic Differential Equations. Springer Berlin, Heidelberg, 1992.  
Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A Ortega, Tom Everitt, Andrew Lefrancq, Laurent Orseau, and Shane Legg. AI safety gridworlds. arXiv preprint arXiv:1711.09883, 2017.  
Sridhar Mahadevan. Average reward reinforcement learning: Foundations, algorithms, and empirical results. Machine Learning, 22:159-195, 1996.  
Sultan Javed Majeed and Marcus Hutter. On Q-learning convergence for non-Markov decision processes. In International Joint Conference on Artificial Intelligence, pp. 2546-2552, 2018.  
David Meder, Finn Rabe, Tobias Morville, Kristoffer H Madsen, Magnus T Koudahl, Ray J Dolan, Hartwig R Siebner, and Oliver J Hulme. Ergodicity-breaking reveals time optimal decision making in humans. PLoS Computational Biology, 17(9):e1009217, 2021.  
Oliver Mihatsch and Ralph Neuneier. Risk-sensitive reinforcement learning. Machine Learning, 49 (2):267-290, 2002.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, pp. 1928-1937, 2016.  
Vu Nguyen, Sebastian Schulze, and Michael Osborne. Bayesian optimization for iterative learning. Advances in Neural Information Processing Systems, pp. 9361-9371, 2020.  
Erfaun Noorani and John S Baras. Risk-sensitive reinforce: A Monte Carlo policy gradient algorithm for exponential performance criteria. In IEEE Conference on Decision and Control, pp. 1522-1527, 2021.  
Erfaun Noorani, Christos Mavridis, and John Baras. Risk-sensitive reinforcement learning with exponential criteria. arXiv preprint arXiv:2212.09010, 2022.  
Jungseul Ok, Alexandre Proutiere, and Damianos Tranos. Exploration in structured reinforcement learning. Advances in Neural Information Processing Systems, 2018.  
Donald S Ornstein. An application of ergodic theory to probability theory. The Annals of Probability, 1(1):43-58, 1973.  
Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.  
Fabien Pesquerel and Odalric-Ambrym Maillard. IMED-RL: Regret optimal learning of ergodic Markov decision processes. Advances in Neural Information Processing Systems, pp. 26363-26374, 2022.  
Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression for operational space control. In International Conference on Machine Learning, pp. 745-750, 2007.  
Jan Peters and Stefan Schaal. Learning to control in operational space. The International Journal of Robotics Research, 27(2):197-212, 2008.  
Ole Peters. The ergodicity problem in economics. Nature Physics, 15(12):1216-1221, 2019.  
Ole Peters and Alexander Adamou. The time interpretation of expected utility theory. arXiv preprint arXiv:1801.03680, 2018.

Ole Peters and Alexander Adamou. The ergodicity solution of the cooperation puzzle. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 380 (2227):20200425, 2022.  
Ole Peters and William Klein. Ergodicity breaking in geometric Brownian motion. Physical Review Letters, 110(10):100603, 2013.  
Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust adversarial reinforcement learning. In International Conference on Machine Learning, pp. 2817-2826. PMLR, 2017.  
Warren B Powell. Reinforcement Learning and Stochastic Optimization. John Wiley & Sons, 2021.  
LA Prashanth, Michael C Fu, et al. Risk-sensitive reinforcement learning via policy gradient search. Foundations and Trends® in Machine Learning, 15(5):537-693, 2022.  
Martin L Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, 2014.  
Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1-8, 2021.  
Stuart Russell, Daniel Dewey, and Max Tegmark. Research priorities for robust and beneficial artificial intelligence. AI Magazine, 36(4):105-114, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Yun Shen, Michael J Tobia, Tobias Sommer, and Klaus Obermayer. Risk-sensitive reinforcement learning. Neural Computation, 26(7):1298-1328, 2014.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of Go without human knowledge. Nature, 550(7676):354-359, 2017.  
Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction. MIT Press, 2018.  
Robert Tibshirani. Estimating transformations for regression via additivity and variance stabilization. Journal of the American Statistical Association, 83(402):394-405, 1988.  
Matteo Turchetta, Felix Berkenkamp, and Andreas Krause. Safe exploration in finite Markov decision processes with Gaussian processes. Advances in Neural Information Processing Systems, 2016.  
Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double Q-learning. In AAAI Conference on Artificial Intelligence, pp. 2094-2100, 2016.  
Arne Vanhoyweghen, Brecht Verbeken, Cathy Macharis, and Vincent Ginis. The influence of ergodicity on risk affinity of timed and non-timed respondents. Scientific Reports, 12(1):1-9, 2022.  
Arthur F Veinott. On finding optimal policies in discrete dynamic programming with no discounting. The Annals of Mathematical Statistics, 37(5):1284-1294, 1966.  
Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature, 575(7782):350-354, 2019.  
Ziyu Wang and Nando de Freitas. Theoretical analysis of bayesian optimisation with unknown gaussian process hyper-parameters. arXiv preprint arXiv:1406.7758, 2014.  
Chen-Yu Wei, Mehdi Jafarnia Jahromi, Haipeng Luo, Hiteshi Sharma, and Rahul Jain. Model-free reinforcement learning in infinite-horizon average-reward Markov decision processes. In International Conference on Machine Learning, pp. 10170-10180, 2020.

Honghao Wei, Xin Liu, and Lei Ying. A provably-efficient model-free algorithm for infinite-horizon average-reward constrained markov decision processes. In AAAI Conference on Artificial Intelligence, pp. 3868-3876, 2022.  
Daan Wierstra, Tom Schaul, Jan Peters, and Juergen Schmidhuber. Episodic reinforcement learning by logistic reward-weighted regression. In International Conference on Artificial Neural Networks, pp. 407-416, 2008.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229-256, 1992.  
Yiming Zhang and Keith W Ross. On-policy deep reinforcement learning for the average-reward criterion. In International Conference on Machine Learning, pp. 12535-12545, 2021.  
Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C Parkes, and Richard Socher. The AI economist: Taxation policy design via two-level deep multiagent reinforcement learning. Science Advances, 8(18):eabk2607, 2022.
