# MULTI-ARMED BANDITS WITH ABSTENTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce a novel extension of the canonical multi-armed bandit problem that incorporates an additional strategic element: abstention. In this enhanced framework, the agent is not only tasked with selecting an arm at each time step, but also has the option to abstain from accepting the stochastic instantaneous reward before observing it. When opting for abstention, the agent either suffers a fixed regret or gains a guaranteed reward. Given this added layer of complexity, we ask whether we can develop efficient algorithms that are both asymptotically and minimax optimal. We answer this question affirmatively by designing and analyzing algorithms whose regrets meet their corresponding information-theoretic lower bounds. Our results offer valuable quantitative insights into the benefits of the abstention option, laying the groundwork for further exploration in other online decision-making problems with such an option. Numerical results further corroborate our theoretical findings.

# 1 INTRODUCTION

In the realm of online decision-making, the multi-armed bandit model, originally introduced by Thompson (1933), has long served as a quintessential benchmark for capturing the delicate interplay between exploration and exploitation. In stochastic multi-armed bandit problems, the agent sequentially selects an arm from the given set at each time step and subsequently observes a random reward associated with the chosen arm. To maximize cumulative rewards, the agent must strike a balance between the persistent pursuit of the arm with the highest expected reward (exploitation) and the adventurous exploration of other arms to gain a deeper understanding of their potential (exploration). This fundamental challenge finds applications across a wide array of domains, ranging from optimizing advertising campaigns to fine-tuning recommendation systems.

However, real-world scenarios often come fraught with complexities that challenge the simplicity of the canonical bandit model. One notable complexity arises when the agent is equipped with an additional option to abstain from accepting the stochastic instantaneous reward before actually observing it. This added layer of decision-making considerably enriches the strategic landscape, altering how the agent optimally navigates the trade-off between exploration and exploitation.

Consider, for example, the domain of clinical trials. When evaluating potentially hazardous medical treatments, researchers can proactively deploy safeguards such as preemptive medications or consider purchasing specialized insurance packages to shield against possible negative consequences. However, these protective measures come with costs, which may be modeled as either fixed regrets or fixed rewards in the context of the clinical study's cumulative regret. In these scenarios, researchers have the option to observe the outcomes of a treatment while abstaining from incurring the associated random regret through these costly prearranged measures. Opting for abstinence can promote more responsible decision-making and reduce the overall cumulative regret of the study.

Building upon this challenge, we introduce an innovative extension to the canonical multi-armed bandit model that incorporates abstention as a legitimate strategic option. At each time step, the agent not only selects which arm to pull but also decides whether to abstain. Depending on how the abstention option impacts the cumulative regret, which is the agent's primary optimization objective, our abstention model offers two complementary settings, namely, the fixed-regret setting where abstention results in a constant regret, and the fixed-reward setting where abstention yields a deterministic reward. Collectively, these settings provide the agent with a comprehensive toolkit for adeptly navigating the complicated landscape of online decision-making.

Main contributions. Our main results and contributions are summarized as follows:

(i) In Section 2, we provide a rigorous mathematical formulation of the multi-armed bandit model with abstention. Our focus is on cumulative regret minimization across two distinct yet complementary settings: fixed-regret and fixed-reward. These settings give rise to divergent performance metrics, each offering unique analytical insights. Importantly, both settings encompass the canonical bandit model as a particular case.  
(ii) In the fixed-regret setting, we judiciously integrate two abstention criteria into a Thompson Sampling-based algorithm proposed by Jin et al. (2023). This integration ensures compatibility with the abstention option, as elaborated in Algorithm 1. The first abstention criterion employs a carefully constructed lower confidence bound, while the second is tailored to mitigate worst-case scenarios. We establish both asymptotic and minimax upper bounds on the cumulative regret. Furthermore, we derive corresponding lower bounds, thereby demonstrating that our algorithm attains asymptotic and minimax optimality simultaneously.  
(iii) In the fixed-reward setting, we introduce a general strategy, outlined in Algorithm 2. This method is capable of transforming any algorithm that is both asymptotically and minimax optimal in the canonical model to one that also accommodates the abstention option. Remarkably, this strategy maintains its universal applicability and straightforward implementation while provably achieving both forms of optimality—asymptotic and minimax.  
(iv) To empirically corroborate our theoretical contributions, we conduct a series of numerical experiments in Section 5. These experiments substantiate the effectiveness of our algorithms and highlight the performance gains achieved through the inclusion of the abstention option.

# 1.1 RELATED WORK

Canonical multi-armed bandits. The study of cumulative regret minimization in canonical multi-armed bandits has attracted considerable scholarly focus. Two dominant paradigms for evaluating optimality metrics emerge: asymptotic optimality and minimax optimality. Briefly, the former considers the behavior of algorithms as the time horizon approaches infinity for a specific problem instance, while the latter seeks to minimize the worst-case regret over all possible instances. A diverse array of policies have been rigorously established to achieve asymptotic optimality across various settings. Notable examples include UCB2 (Auer et al., 2002), DMED (Honda & Takemura, 2010), KL-UCB (Cappé et al., 2013), and Thompson Sampling (Agrawal & Goyal, 2012; Kaufmann et al., 2012). In the context of the worst-case regret, MOSS (Audibert & Bubeck, 2009) stands out as the pioneering method that has been verified to be minimax optimal. Remarkably,  $\mathrm{KL - UCB^{+ + }}$  (Ménard & Garivier, 2017) became the first algorithm proved to achieve both asymptotic and minimax optimality. Very recently, Jin et al. (2023) introduced Less-Exploring Thompson Sampling, an innovation that boosts computational efficiency compared to classical Thompson Sampling while concurrently achieving asymptotic and minimax optimality. For a comprehensive survey of bandit algorithms, we refer to Lattimore & Szeptsvári (2020).

Machine learning with abstention. Starting with the seminal works of Chow (1957; 1970), the concept of learning with abstention (also referred to as rejection) has been extensively explored in various machine learning paradigms. These include, but are not limited to, classification (Herbei & Wegkamp, 2006; Bartlett & Wegkamp, 2008; Cortes et al., 2016), ranking (Cheng et al., 2010; Mao et al., 2023), and regression (Wiener & El-Yaniv, 2012; Zaoui et al., 2020; Kalai & Kanade, 2021).

Within this broad spectrum of research, our work is most directly related to those that explore the role of abstention in the context of online learning. To the best of our knowledge, Cortes et al. (2018) firstly incorporated the abstention option into the problem of online prediction with expert advice (Littlestone & Warmuth, 1994). In their model, at each time step, each expert has the option to either make a prediction based on the given input or abstain from doing so. When the agent follows the advice of an expert who chooses to abstain, the true label of the input remains undisclosed, and the learner incurs a known fixed loss. Subsequently, Neu & Zhivotovskiy (2020) introduced a different abstention model, which is more similar to ours. Here, the abstention option is only available to the agent. Crucially, the true label is always revealed to the agent after the decision has been made, regardless of whether the agent opts to abstain. Their findings suggest that equipping the agent with an abstention option can significantly improve the guarantees on the worst-case regret.

![](images/9dac2907f9c3e9c3ed7b5ed6ab38a8b089bbabff6e44c8a2d3e09b01278a69af.jpg)  
Figure 1: Interaction protocol for multi-armed bandits with fixed-regret and fixed-reward abstention.

Although set in different contexts, these existing works consistently demonstrate the value of incorporating abstention into online decision-making processes, underscoring the urgent need to analyze and quantify its benefits in the field of multi-armed bandits.

# 2 PROBLEM SETUP

Multi-armed bandits with abstention. We consider a  $K$ -armed bandit model, enhanced with an additional option to abstain from accepting the stochastic instantaneous reward prior to its observation. Let  $\mu \in \mathcal{U} \coloneqq \mathbb{R}^K$  denote a specific bandit instance, where  $\mu_i$  represents the unknown mean reward associated with pulling arm  $i \in [K]$ . For simplicity, we assume that arm 1 is the unique optimal arm, i.e.,  $1 = \arg \max_{i \in [K]} \mu_i$ , and we define  $\Delta_i \coloneqq \mu_1 - \mu_i$  as the suboptimality gap for each arm  $i \in [K]$ .

At each time step  $t \in \mathbb{N}$ , the agent chooses an arm  $A_{t}$  from the given arm set  $[K]$ , and, simultaneously, decides whether or not to abstain, indicated by a binary variable  $B_{t}$ . Regardless of the decision to abstain, the agent observes a random variable  $X_{t}$  from the selected arm  $A_{t}$ , which is drawn from a Gaussian distribution  $\mathcal{N}(\mu_{A_t}, 1)$  and independent of observations obtained from the previous time steps. Notably, the selection of both  $A_{t}$  and  $B_{t}$  might depend on the previous decisions and observations, as well as on each other. More formally, let  $\mathcal{F}_t \coloneqq \sigma (A_1,B_1,X_1,\dots ,A_t,B_t,X_t)$  denote the  $\sigma$ -field generated by the cumulative interaction history up to and including time  $t$ . It follows that the pair of random variables  $(A_{t},B_{t})$  is  $\mathcal{F}_{t - 1}$ -measurable.

The instantaneous regret at time  $t$  is determined by both the binary abstention variable  $B_{t}$  and the observation  $X_{t}$ . Based on the outcome of the abstention option, we now discuss two complementary settings. In the fixed-regret setting, the abstention option incurs a constant regret. Opting for abstention  $(B_{t} = 1)$  leads to a deterministic regret of  $c > 0$ , in contrast to the initial regret linked to arm  $A_{t}$  when not selecting abstention  $(B_{t} = 0)$ , which is given by  $\mu_{1} - X_{t}$ .

Alternatively, in the fixed-reward setting, the reward of the abstention option is predetermined to be  $c \in \mathbb{R}$ . Since the abstention reward  $c$  may potentially surpass  $\mu_1$ , the best possible expected reward at a single time step is  $\mu_1 \lor c := \max \{\mu_1, c\}$ . If the agent decides to abstain  $(B_t = 1)$ , it guarantees a deterministic reward of  $c$ , leading to a regret of  $\mu_1 \lor c - c$ . Conversely, if  $B_t = 0$ , the agent receives a per-time reward  $X_t$ , resulting in a regret of  $\mu_1 \lor c - X_t$ .

See Figure 1 for a schematic of our model in the two settings.

Regret minimization. Our overarching goal is to design and analyze online algorithms  $\pi$  that minimize their expected cumulative regrets up to and including the time horizon  $T$ . The regrets are formally defined for the two distinct settings as follows:

- Fixed-regret setting:

$$
R _ {\mu , c} ^ {\mathrm {R G}} (T, \pi) := \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \left(\left(\mu_ {1} - X _ {t}\right) \cdot \mathbb {1} \left\{B _ {t} = 0 \right\} + c \cdot \mathbb {1} \left\{B _ {t} = 1 \right\}\right) \right]. \tag {1}
$$

- Fixed-reward setting:

$$
R _ {\mu , c} ^ {\mathrm {R W}} (T, \pi) := T \cdot (\mu_ {1} \vee c) - \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \left(X _ {t} \cdot \mathbb {1} \{B _ {t} = 0 \} + c \cdot \mathbb {1} \{B _ {t} = 1 \}\right) \right]. \tag {2}
$$

An online algorithm  $\pi$  consists of two interrelated components: the arm sampling rule that selects  $A_{t}$ , and the abstention decision rule that determines  $B_{t}$  at each time step  $t\in [T]$ . Additionally, we use  $\Pi^{\mathrm{RG}}$  and  $\Pi^{\mathrm{RW}}$  to denote the collections of all online policies for the fixed-regret and fixed-reward settings, respectively. For the sake of analytical convenience, we also introduce the canonical regret  $R_{\mu}^{\mathrm{CA}}(T,\pi)\coloneqq T\mu_1 - \mathbb{E}\big[\sum_{t = 1}^{T}X_{t}\big]$ , which disregards the abstention option and remains well-defined within our abstention model. Furthermore, when there is no ambiguity, we will omit the dependence of the regret on the policy. For example, we often abbreviate  $R_{\mu ,c}^{\mathrm{RG}}(T,\pi)$  as  $R_{\mu ,c}^{\mathrm{RG}}(T)$ .

Remark 1. It is worth mentioning that our model is a strict generalization of the canonical multi-armed bandit model (without the abstention option). Specifically, it particularizes to the canonical model as the abstention regret  $c$  tends to positive infinity in the fixed-regret setting and as the abstention reward  $c$  tends to negative infinity in the fixed-reward setting. Nevertheless, the incorporation of an extra challenge, the abstention decision (denoted as  $B_{t}$ ), offers the agent the potential opportunity to achieve superior performance in terms of either regret.

Other notations. For  $x, y \in \mathbb{R}$ , we denote  $x \wedge y \coloneqq \min \{x, y\}$  and  $x \vee y \coloneqq \max \{x, y\}$ . For any arm  $i \in [K]$ , let  $N_i(t) \coloneqq \sum_{s=1}^{t} \mathbb{1}\{A_s = i\}$  and  $\hat{\mu}_i(t) \coloneqq \sum_{s=1}^{t} X_s \mathbb{1}\{A_s = i\} / N_i(t)$  denote its total number of pulls and empirical estimate of the mean up to time  $t$ , respectively. In particular, we set  $\hat{\mu}_i(t) = +\infty$  if  $N_i(t) = 0$ . To count abstention records, we also use  $N_i^{(0)}(t)$  and  $N_i^{(1)}(t)$  to denote its number of pulls without and with abstention up to time  $t$ , respectively. That is,  $N_i^{(0)}(t) \coloneqq \sum_{s=1}^{t} \mathbb{1}\{A_s = i$  and  $B_s = 0\}$  and  $N_i^{(1)}(t) \coloneqq \sum_{s=1}^{t} \mathbb{1}\{A_s = i$  and  $B_s = 1\}$ . Additionally, we define  $\hat{\mu}_{is}$  as the empirical mean of arm  $i$  based on its first  $s$  pulls. Furthermore, we use  $\alpha, \alpha_1$ , and so forth to represent universal constants that do not depend on the problem instances (including  $\mu$ ,  $c$ ,  $T$ ,  $K$ ), with possibly different values in different contexts.

# 3 FIXED-REGRET SETTING

In this section, we focus on the fixed-regret setting. Specifically, we design a conceptually simple and computationally efficient algorithm, namely Fixed-Regret Thompson Sampling with Abstention (or FRG-TSwA), to minimize the cumulative regret while incorporating fixed-regret abstinence. To evaluate the performance of our algorithm from a theoretical standpoint, we establish both instance-dependent asymptotic and instance-independent minimax upper bounds on the cumulative regret, as elaborated upon in Section 3.1. Furthermore, in Section 3.2, we provide lower bounds for the problem of regret minimization in multi-armed bandits with fixed-regret abstinence. These findings substantiate that our algorithm achieves both asymptotic and minimax optimality simultaneously. The pseudocode for FRG-TSwA is presented in Algorithm 1 and elucidated in the following.

In terms of the arm sampling rule, our algorithm is built upon Less-Exploring Thompson Sampling (Jin et al., 2023), a minimax optimal enhancement of the celebrated Thompson Sampling algorithm (Thompson, 1933). We refer to Remark 3 for the reason behind this choice. During the initialization phase, each arm is sampled exactly once. Following that, at each time  $t$ , an estimated reward  $a_{i}(t)$  is constructed for each arm  $i \in [K]$ , which is either drawn from the posterior distribution  $\mathcal{N}(\hat{\mu}_i(t - 1), 1 / N_i(t - 1))$  with probability  $1 / K$  or set to be the empirical mean  $\hat{\mu}_i(t - 1)$  otherwise. Subsequently, the algorithm consistently pulls the arm  $A_{t}$  with the highest estimated reward.

With regard to the abstention decision rule, we propose two abstention criteria that work in tandem (as detailed in Step 5 of Algorithm 1). The first criterion is gap-dependent in nature. In particular, we choose to abstain if there exists an arm  $i \in [K] \setminus \{A_t\}$  for which the difference between its lower confidence bound and the empirical mean of the arm  $A_t$  exceeds  $c$ . This condition signifies that the suboptimality gap  $\Delta_{A_t}$  is at least  $c$  with high probability. The second abstention criterion is gap-independent and more straightforward. It is motivated from the construction of worst-case scenarios as detailed in the proof of our lower bound. Under this criterion, we opt for the abstention option if  $c \leq \sqrt{K / t}$ , which implies that the abstention regret remains acceptably low at time  $t$  in view of the worst-case scenarios.

# Algorithm 1 Fixed-Regret Thompson Sampling with Abstention (or FRG-TSWA)

Input: Arm set  $[K]$  and abstention regret  $c > 0$

1: Sample each arm once, and choose to abstain  $(B_{t} = 1)$  if and only if  $\sqrt{\frac{K}{t}} \geq c$ .  
2: Initialize  $\hat{\mu}_i(K)$  and  $N_{i}(K) = 1$  for all  $i\in [K]$ .  
3: for  $t = K + 1, \dots, T$  do  
4: For each arm  $i\in [K]$  , sample  $\theta_{i}(t)\sim \mathcal{N}(\hat{\mu}_{i}(t - 1),1 / N_{i}(t - 1))$  and set

$$
a _ {i} (t) = \left\{ \begin{array}{l l} \theta_ {i} (t) & \text {w i t h p r o b a b i l i t y 1 / K} \\ \hat {\mu} _ {i} (t - 1) & \text {w i t h p r o b a b i l i t y 1 - 1 / K}. \end{array} \right.
$$

5: Pull the arm  $A_{t} = \arg \max_{i\in [K]}a_{i}(t)$ , and choose to abstain  $(B_{t} = 1)$  if and only if

$$
\max  _ {i \in [ K ] \backslash \{A _ {t} \}} \left(\hat {\mu} _ {i} (t - 1) - \sqrt {\frac {6 \log t + 2 \log (c \vee 1)}{N _ {i} (t - 1)}}\right) - \hat {\mu} _ {A _ {t}} (t - 1) \geq c \text {o r} \sqrt {\frac {K}{t}} \geq c.
$$

6: Observe  $X_{t}$  from the arm  $A_{t}$ , and update  $\hat{\mu}_i(t)$  and  $N_{i}(t)$  for all  $i\in [K]$ .

7: end for

# 3.1 UPPER BOUNDS

Theorem 1 below provides two distinct types of theoretical guarantees pertaining to our algorithm's performance on the cumulative regret  $R_{\mu,c}^{\mathrm{RG}}(T)$ , which is defined in Equation (1) for the fixed-regret setting. The complete proof of Theorem 1 is deferred to Appendix C.1.

Theorem 1. For all abstention regrets  $c > 0$  and bandit instances  $\mu \in \mathcal{U}$ , Algorithm 1 guarantees that

$$
\limsup_{T\to \infty}\frac{R_{\mu,c}^{\mathrm{RG}}(T)}{\log T}\leq 2\sum_{i > 1}\frac{\Delta_{i}\wedge c}{\Delta_{i}^{2}}.
$$

Furthermore, there exists a universal constant  $\alpha > 0$  such that

$$
R _ {\mu , c} ^ {\mathrm {R G}} (T) \leq \left\{ \begin{array}{l l} c T & \text {i f} c \leq \sqrt {K / T} \\ \alpha (\sqrt {K T} + \sum_ {i > 1} \Delta_ {i}) & \text {i f} c > \sqrt {K / T}. \end{array} \right.
$$

Remark 2. The theoretical challenges associated with Theorem 1 revolve around quantifying the regret that results from inaccurately estimating the suboptimality gaps associated to the abstention criteria. More precisely, from both asymptotic and worst-case perspectives, it is crucial to establish upper bounds on  $\mathbb{E}[N_i^{(1)}(T)]$  for arms  $i$  with  $\Delta_{i} < c$  (which, by definition, includes the best arm), and on  $\mathbb{E}[N_i^{(0)}(T)]$  for arms  $i$  with  $\Delta_{i} > c$ . These complexities necessitate a deeper exploration into the arm sampling dynamics inherent to Less-Exploring Thompson Sampling, and preclude us from formulating a generalized strategy akin to the upcoming Algorithm 2 for the fixed-reward setting.

Remark 3. As previously highlighted, our model in the fixed-regret setting particularizes to the canonical multi-armed bandit model as the abstention regret  $c$  approaches infinity. Similarly, when  $c$  tends towards infinity, the two abstention criteria are never satisfied, and the procedure of Algorithm 1 simplifies to that of Less-Exploring Thompson Sampling. It is worth noting that this latter algorithm is not only asymptotically optimal but also minimax optimal for the canonical model. This is precisely why we base our algorithm upon it, rather than the conventional Thompson Sampling algorithm, which has been shown not to be minimax optimal (Agrawal & Goyal, 2017).

# 3.2 LOWER BOUNDS

In order to establish the asymptotic lower bound, we need to introduce the concept of  $R^{\mathrm{RG}}$ -consistency, which rules out overly specialized algorithms that are tailored exclusively to specific problem instances. Roughly speaking, a  $R^{\mathrm{RG}}$ -consistent algorithm guarantees a subpolynomial cumulative regret for any given problem instance.

Definition 1 ( $R^{\mathrm{RG}}$ -consistency). An algorithm  $\pi \in \Pi^{\mathrm{RG}}$  is said to be  $R^{\mathrm{RG}}$ -consistent if for all abstention regrets  $c > 0$ , bandit instances  $\mu \in \mathcal{U}$ , and  $a > 0$ ,  $R_{\mu, c}^{\mathrm{RG}}(T, \pi) = o(T^{a})$ .

Now we present both asymptotic and minimax lower bounds on the cumulative regret in Theorem 2, which is proved in Appendix C.2.

Theorem 2. For any abstention regret  $c > 0$ , bandit instance  $\mu \in \mathcal{U}$  and  $R^{\mathrm{RG}}$ -consistent algorithm  $\pi$ , it holds that

$$
\liminf_{T\to \infty}\frac{R_{\mu,c}^{\mathrm{RG}}(T,\pi)}{\log T}\geq 2\sum_{i > 1}\frac{\Delta_{i}\wedge c}{\Delta_{i}^{2}}.
$$

For any abstention regret  $c > 0$  and time horizon  $T \geq K$ , there exists a universal constant  $\alpha > 0$  such that

$$
\inf_{\pi \in \Pi^{\mathrm{RG}}}\sup_{\mu \in \mathcal{U}}R^{ \mathrm{RG}}_{\mu ,c}(T,\pi)\geq \alpha (\sqrt{KT}\wedge cT).
$$

Comparing the upper bounds on the cumulative regret of our algorithm FRG-TSwA in Theorem 1 with the corresponding lower bounds in Theorem 2, it becomes evident that our algorithm exhibits both asymptotic and minimax optimality.

Asymptotic optimality. For any abstention regret  $c > 0$  and bandit instance  $\mu \in \mathcal{U}$ , the regret of our algorithm satisfies the following limiting behaviour:

$$
\lim  _ {T \to \infty} \frac {R _ {\mu , c} ^ {\mathrm {R G}} (T)}{\log T} = 2 \sum_ {i > 1} \frac {\Delta_ {i} \wedge c}{\Delta_ {i} ^ {2}}.
$$

The above asymptotically optimal result yields several intriguing implications. First, the inclusion of the additional fixed-regret abstention option does not obviate the necessity of differentiating between suboptimal arms and the optimal one, and the exploration-exploitation trade-off remains crucial. In fact, to avoid the case in which the cumulative regret grows polynomially, the agent must still asymptotically allocate the same proportion of pulls to each suboptimal arm, as in the canonical model. This assertion is rigorously demonstrated in the proof of the lower bound (refer to Appendix C.2 for comprehensive details). Nevertheless, the abstention option does indeed reduce the exploration cost for the agent. Specifically, when exploring any suboptimal arm with a suboptimality gap larger than  $c$ , our algorithm leans towards employing the abstention option to minimize the instantaneous regret. This aspect is formally established in the proof of the asymptotic upper bound (see Appendix C.1 for further details).

Minimax optimality. In the context of worst-case guarantees for the cumulative regret, we focus on the dependence on the problem parameters:  $c$ ,  $K$  and  $T$ . Notably, the  $\sum_{i > 1} \Delta_i$  term<sup>3</sup> is typically considered as negligible in the literature (Audibert & Bubeck, 2009; Agrawal & Goyal, 2017; Lattimore & Szeptsváři, 2020). Therefore, Theorem 1 demonstrates that our algorithm attains a worst-case regret of  $O(\sqrt{KT} \wedge cT)$ , which is minimax optimal in light of Theorem 2.

A phase transition phenomenon can be clearly observed from the worst-case guarantees, which dovetails with our intuitive understanding of the fixed-regret abstention setting. When the abstention regret  $c$  is sufficiently low, it becomes advantageous to consistently opt for abstention to avoid the worst-case scenarios. On the contrary, when the abstention regret  $c$  exceeds a certain threshold, the abstention option proves to be inadequate in alleviating the worst-case regret, as compared to the canonical model.

Remark 4. Although our model allows for the selected arm  $A_{t}$  and the abstention option  $B_{t}$  to depend on each other, the procedure used in both algorithms within this work is to first determine  $A_{t}$  before  $B_{t}$ ; this successfully achieves both forms of optimality. Nevertheless, this approach might no longer be optimal beyond the canonical  $K$ -armed bandit setting. In  $K$ -armed bandits, each arm operates independently. Conversely, in models like linear bandits, pulling one arm can indirectly reveal information about other arms. Policies based on the principle of optimism in the face of uncertainty, as well as Thompson Sampling, fall short of achieving asymptotic optimality in the context of linear bandits (Lattimore & Szeptsvári, 2017). Therefore, the abstention option becomes particularly attractive if there exists an arm that incurs a substantial regret but offers significant insights into the broader bandit instance.

Algorithm 2 Fixed-Reward Algorithm with Abstention (or FRW-ALGwA)  
Input: Arm set  $[K]$ , abstention reward  $c\in \mathbb{R}$ , and a base algorithm ALG that is both asymptotically and minimax optimal for the canonical multi-armed bandit model.  
1: Initialize  $\hat{\mu}_i(0) = +\infty$  for all arms  $i\in [K]$ .  
2: for  $t = 1,2,\ldots ,T$  do  
3: Pull the arm  $A_{t}$  chosen by the base algorithm ALG.  
4: Choose to abstain  $(B_{t} = 1)$  if and only if  $\hat{\mu}_{A_t}(t - 1)\leq c$   
5: Observe  $X_{t}$  from the arm  $A_{t}$ , and update  $\hat{\mu}_i(t)$  for all  $i\in [K]$ .  
6: end for

# 4 FIXED-REWARD SETTING

In this section, we investigate the fixed-reward setting. Here, the reward associated with the abstention option remains consistently fixed at  $c \in \mathbb{R}$ . When exploring a specific arm, the agent has the capability to determine whether selecting the abstention option yields a higher reward (or equivalently, a lower regret) solely based on its own estimated mean reward. However, in the fixed-regret setting, this decision can only be made by taking into account both its own estimated mean reward and the estimated mean reward of the potentially best arm. In this regard, the fixed-reward setting is inherently less complex than the fixed-regret setting. As a result, it becomes possible for us to design a more general strategy Fixed-Reward Algorithm with Abstention (or FRW-ALGwA), whose pseudocode is presented in Algorithm 2. Despite the straightforward nature of our algorithm, we demonstrate its dual attainment of both asymptotic and minimax optimality through an exhaustive theoretical examination in Sections 4.1 and 4.2.

As its name suggests, our algorithm FRW-ALGwA leverages a base algorithm ALG that is asymptotically and minimax optimal for canonical multi-armed bandits as its input. For comprehensive definitions of asymptotic and minimax optimality within the canonical model, we refer the reader to Appendix A. Notably, eligible candidate algorithms include  $\mathrm{KL - UCB^{+ + }}$  (Ménard & Garivier, 2017), ADA-UCB (Lattimore, 2018), MOTS-J (Jin et al., 2021) and Less-Exploring Thompson Sampling (Jin et al., 2023). In the operation of our algorithm, at each time step  $t$ , the base algorithm determines the selected arm  $A_{t}$  according to the partial interaction historical information  $(A_{1},X_{1},A_{2},X_{2},\ldots ,A_{t - 1},X_{t - 1})$ . Subsequently, the algorithm decides whether or not to abstain, indicated by the binary random variable  $B_{t}$ , by comparing the empirical mean of the arm  $A_{t}$ , denoted as  $\hat{\mu}_{A_t}(t - 1)$ , to the abstention reward  $c$ .

# 4.1 UPPER BOUNDS

Recall the definition of the cumulative regret  $R_{\mu, c}^{\mathrm{RW}}(T)$ , as presented in Equation (2) for the fixed-reward setting. Theorem 3 establishes both the instance-dependent asymptotic and instance-independent minimax upper bounds for Algorithm 2; see Appendix D.1 for the proof.

Theorem 3. For all abstention rewards  $c \in \mathbb{R}$  and bandit instances  $\mu \in \mathcal{U}$ , Algorithm 2 guarantees that

$$
\limsup_{T\to \infty}\frac{R^{\mathrm{RW}}_{\mu,c}(T)}{\log T}\leq 2\sum_{i > 1}\frac{\mu_{1} \lor c - \mu_{i} \lor c}{\Delta_{i}^{2}}.
$$

Furthermore, there exists a universal constant  $\alpha > 0$  such that

$$
R _ {\mu , c} ^ {\mathrm {R W}} (T) \leq \alpha \left(\sqrt {K T} + \sum_ {i \in [ K ]} \left(\mu_ {1} \vee c - \mu_ {i}\right)\right).
$$

Remark 5. It is worth considering the special case where  $c \geq \mu_1$ , where opting for abstention results in a reward even greater than, or equal to, the mean reward of the best arm. For this particular case, as per Theorem 3, since  $\mu_1 \lor c - \mu_i \lor c = 0$  for all  $i > 1$ , our algorithm achieves a regret of  $o(\log T)$ . This result, in fact, is not surprising. In contrast to the fixed-regret setting where the regret associated with the abstention option is strictly positive, in this specific scenario of the fixed-reward setting, selecting the abstention option is indeed the optimal action at a single time step, regardless of the arm pulled. Therefore, there is no necessity to distinguish between suboptimal arms and the optimal one, and the

exploration-exploitation trade-off becomes inconsequential. However, when the abstention reward is below the mean reward of the best arm, i.e.,  $c < \mu_1$ , maintaining a subpolynomial cumulative regret still hinges on the delicate balance between exploration and exploitation, as evidenced by the forthcoming exposition of the asymptotic lower bound.

# 4.2 LOWER BOUNDS

We hereby introduce the concept of  $R^{\mathrm{RW}}$ -consistency for the fixed-reward setting, in a manner analogous to the fixed-regret setting. Following this, we present two distinct lower bounds for the problem of regret minimization in multi-armed bandits with fixed-reward abstention in Theorem 4. The proof for Theorem 4 is postponed to Appendix D.2.

Definition 2 ( $R^{\mathrm{RW}}$ -consistency). An algorithm  $\pi \in \Pi^{\mathrm{RW}}$  is said to be  $R^{\mathrm{RW}}$ -consistent if for all abstention rewards  $c \in \mathbb{R}$ , bandit instances  $\mu \in \mathcal{U}$ , and  $a > 0$ ,  $R_{\mu, c}^{\mathrm{RW}}(T, \pi) = o(T^{a})$ .

Theorem 4. For any abstention reward  $c \in \mathbb{R}$ , bandit instance  $\mu \in \mathcal{U}$  and  $R^{\mathrm{RW}}$ -consistent algorithm  $\pi$ , it holds that

$$
\liminf_{T\to \infty}\frac{R_{\mu,c}^{\mathrm{RW}}(T,\pi)}{\log T}\geq 2\sum_{i > 1}\frac{\mu_{1}\lor c - \mu_{i}\lor c}{\Delta_{i}^{2}}.
$$

For any abstention reward  $c \in \mathbb{R}$  and time horizon  $T \geq K$ , there exists a universal constant  $\alpha > 0$  such that

$$
\inf_{\pi \in \Pi^{\mathrm{RW}}}\sup_{\mu \in \mathcal{U}}R^{ \mathrm{RW}}_{\mu ,c}(T,\pi)\geq \alpha \sqrt{KT}.
$$

By comparing the upper bounds in Theorem 3 with the lower bounds in Theorem 4, it is firmly confirmed that Algorithm 2 is both asymptotically and minimax optimal in the fixed-reward setting.

Asymptotic optimality. For any abstention reward  $c \in \mathbb{R}$  and bandit instance  $\mu \in \mathcal{U}$ , our algorithm ensures the following optimal asymptotic behavior for the cumulative regret:

$$
\lim  _ {T \rightarrow \infty} \frac {R _ {\mu , c} ^ {\mathrm {R W}} (T)}{\log T} = 2 \sum_ {i > 1} \frac {\mu_ {1} \lor c - \mu_ {i} \lor c}{\Delta_ {i} ^ {2}}.
$$

Since it holds generally that  $\mu_1 \vee c - \mu_i \vee c \leq \Delta_i$  for each arm  $i > 1$ , our algorithm effectively reduces the cumulative regret in the asymptotic regime through the incorporation of the fixed-reward abstention option.

Minimax optimality. As for the worst-case performance of our algorithm, disregarding the additive term  $\sum_{i\in [K]}\left(\mu_1\vee c - \mu_i\right)$ , it achieves an optimal worst-case regret of  $O(\sqrt{KT})$ . While this worst-case regret aligns with that in the canonical multi-armed bandit model, it is noteworthy that this achievement is non-trivial, demanding meticulous management of the asymptotic regret performance in parallel.

Moreover, there is no occurrence of the phase transition phenomenon in the fixed-reward setting. This absence can be attributed to the intrinsic nature of the fixed-reward abstention option. For any abstention reward  $c \in \mathbb{R}$  and online algorithm, we can always construct a challenging bandit instance that leads to a cumulative regret of  $\Omega(\sqrt{KT})$ , as demonstrated in the proof of the minimax lower bound in Appendix D.2.

# 5 NUMERICAL EXPERIMENTS

In this section, we conduct numerical experiments to empirically validate our theoretical insights. Due to space limitations, we report our results only for the fixed-regret setting here. Results pertaining to the fixed-reward setting are available in Appendix E. In each experiment, the reported cumulative regrets are averaged over 2,000 independent trials and the corresponding standard deviations are displayed as error bars in the figures.

To confirm the benefits of incorporating the abstention option, we compare the performance of our proposed algorithm FRG-TSwA (Algorithm 1) with that of Less-Exploring Thompson Sampling

(Jin et al., 2023), which serves as a baseline algorithm without the abstention option. We consider two synthetic bandit instances. The first instance  $\mu^{\dagger}$  with  $K = 7$  has uniform suboptimality gaps:  $\mu_1^\dagger = 1$  and  $\mu_i^\dagger = 0.7$  for all  $i\in [K]\setminus \{1\}$ . For the second instance  $\mu^{\ddagger}$  with  $K = 10$ , the suboptimality gaps are more diverse:  $\mu_1^{\ddagger} = 1$ ,  $\mu_i^{\ddagger} = 0.7$  for  $i\in \{2,3,4\}$ ,  $\mu_i^{\ddagger} = 0.5$  for  $i\in \{5,6,7\}$  and  $\mu_i^{\ddagger} = 0.3$  for  $i\in \{8,9,10\}$ . The empirical averaged cumulative regrets of both methods with

abstention regret  $c = 0.1$  for different time horizons  $T$  are presented in Figure 2. To demonstrate their asymptotic behavior, we also plot the instance-dependent asymptotic lower bound on the cumulative regret (see Theorem 2) in each sub-figure. It can be observed that FRG-TSWA is clearly superior compared to the non-abstaining baseline, especially for large values of  $T$ . This demonstrates the advantage of the abstention mechanism. With regard to the growth trend, as the time horizon  $T$  increases, the curve corresponding to FRG-TSWA closely approximates that of the asymptotic lower bound. This suggests that the expected cumulative regret of FRG-TSWA matches the lower bound asymptotically, thereby substantiating the theoretical results presented in Section 3.

![](images/55d64ad626405d931d7eb39c0881eb726c31dc434d3c255f2574ed27a965d542.jpg)  
(a) Instance  $\mu^{\dagger}$  
Figure 2: Empirical regrets with abstention regret  $c = 0.1$  for different time horizons  $T$ .

![](images/59dacc1055b8991a403707941f0a0ab2e65160bd424c750d9e91de5c860ee38e.jpg)  
(b) Instance  $\mu^{\dagger}$

To illustrate the effect of the abstention regret  $c$ , we evaluate the performance of FRG-TSwA for varying values of  $c$  while keeping the time horizon  $T$  fixed at 10,000. The experimental results for both bandit instances  $\mu^{\dagger}$  and  $\mu^{\ddagger}$  are presented in Figure 3. Within each sub-figure, we observe that as  $c$  increases, the empirical averaged cumulative regret initially increases but eventually saturates beyond a certain threshold value of  $c$ . These empirical observations align well with our expectations. Indeed, when provided with com

plete information about the bandit instance, if the abstention regret  $c$  exceeds the largest suboptimality gap, the agent gains no advantage in choosing the abstention option when selecting any arm. However, we remark that the agent lacks this oracle-like knowledge of the suboptimality gaps and must estimate them on the fly. Consequently, this results in the inevitable selection of the abstention option, even when the abstention regret  $c$  is large.

![](images/06bc34c7da8b32cfdba683cd3cf8d63a9ff963cb44fe06ae91475458083162ce.jpg)  
Figure 3: Empirical regrets with time horizon  $T = 10,000$  for different abstention regrets  $c$ .  
(a) Instance  $\mu^{\dagger}$

![](images/b40693c7f6308463518bc5bda6ecd1adf87f70503c2c41b3f2e59fd754dfb561.jpg)  
(b) Instance  $\mu^{\dagger}$

# 6 CONCLUSIONS AND FUTURE WORK

In this paper, we consider, for the first time, a multi-armed bandit model that allows for the possibility of abstaining from accepting the stochastic rewards, alongside the conventional arm selection. This innovative framework is motivated by real-world scenarios where decision-makers may wish to hedge against highly uncertain or risky actions, as exemplified in clinical trials. Within this enriched paradigm, we address both the fixed-regret and fixed-reward settings, providing tight upper and lower bounds on asymptotic and minimax regrets for each scenario. For the fixed-regret setting, we thoughtfully adapt a recently developed asymptotically and minimax optimal algorithm by Jin et al. (2023) to accommodate the abstention option while preserving its attractive optimality characteristics. For the fixed-reward setting, we convert any asymptotically and minimax optimal algorithm for the canonical model into one that retains these optimality properties when the abstention option is present. Finally, experiments on synthetic datasets validate our theoretical results and clearly demonstrate the advantage of incorporating the abstention option.

As highlighted in Remark 4, a fruitful avenue for future research lies in expanding the abstention model from  $K$ -armed bandits to linear bandits. An intriguing inquiry is whether the inclusion of the abstention feature can lead to enhanced asymptotic and minimax theoretical guarantees.

# REFERENCES

Shipra Agrawal and Navin Goyal. Analysis of Thompson sampling for the multi-armed bandit problem. In Conference on Learning Theory (COLT), pp. 39-1. JMLR Workshop and Conference Proceedings, 2012.  
Shipra Agrawal and Navin Goyal. Near-optimal regret bounds for Thompson sampling. Journal of the ACM (JACM), 64(5):1-24, 2017.  
Jean-Yves Audibert and Sébastien Bubeck. Minimax policies for adversarial and stochastic bandits. In Conference on Learning Theory (COLT), volume 7, pp. 1-122, 2009.  
Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. Gambling in a rigged casino: The adversarial multi-armed bandit problem. In Proceedings of IEEE 36th annual foundations of computer science, pp. 322-331. IEEE, 1995.  
Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2):235-256, 2002.  
Peter L Bartlett and Marten H Wegkamp. Classification with a reject option using a hinge loss. Journal of Machine Learning Research, 9(8), 2008.  
Olivier Cappé, Aurélien Garivier, Odalric-Ambrym Maillard, Rémi Munos, and Gilles Stoltz. Kullback-Leibler upper confidence bounds for optimal sequential allocation. Annals of Statistics, pp. 1516-1541, 2013.  
Weiwei Cheng, Michael Rademaker, Bernard De Baets, and Eyke Hüllermeier. Predicting partial orders: Ranking with abstention. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2010, pp. 215-230. Springer, 2010.  
Chi-Keung Chow. An optimum character recognition system using decision functions. IRE Transactions on Electronic Computers, pp. 247-254, 1957.  
Chi-Keung Chow. On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory, 16(1):41-46, 1970.  
Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri. Learning with rejection. In Algorithmic Learning Theory: 27th International Conference, ALT 2016, Bari, Italy, October 19-21, 2016, Proceedings 27, pp. 67-82. Springer, 2016.  
Corinna Cortes, Giulia DeSalvo, Claudio Gentile, Mehryar Mohri, and Scott Yang. Online learning with abstention. In International Conference on Machine Learning, pp. 1059-1067. PMLR, 2018.  
Aurelien Garivier, Pierre Ménard, and Gilles Stoltz. Explore first, exploit next: The true shape of regret in bandit problems. Mathematics of Operations Research, 44(2):377-399, 2019.  
Radu Herbei and Marten H Wegkamp. Classification with reject option. The Canadian Journal of Statistics / La Revue Canadienne de Statistique, pp. 709-721, 2006.  
Junya Honda and Akimichi Takemura. An asymptotically optimal bandit algorithm for bounded support models. In Conference on Learning Theory, pp. 67-79. CiteSeer, 2010.  
Tianyuan Jin, Pan Xu, Jieming Shi, Xiaokui Xiao, and Quanquan Gu. Mots: Minimax optimal Thompson sampling. In International Conference on Machine Learning, pp. 5074-5083. PMLR, 2021.  
Tianyuan Jin, Xianglin Yang, Xiaokui Xiao, and Pan Xu. Thompson sampling with less exploration is fast and optimal. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 15239-15261. PMLR, 2023.  
Adam Kalai and Varun Kanade. Towards optimally abstaining from prediction with ood test examples. Advances in Neural Information Processing Systems, 34:12774-12785, 2021.

Emilie Kaufmann, Nathaniel Korda, and Rémi Munos. Thompson sampling: An asymptotically optimal finite-time analysis. In Algorithmic Learning Theory: 23rd International Conference, ALT 2012, Lyon, France, October 29-31, 2012. Proceedings 23, pp. 199-213. Springer, 2012.  
Nathaniel Korda, Emilie Kaufmann, and Rémi Munos. Thompson sampling for 1-dimensional exponential family bandits. Advances in Neural Information Processing Systems, 26, 2013.  
Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4-22, 1985.  
Tor Lattimore. Refining the confidence level for optimistic bandit strategies. Journal of Machine Learning Research, 19(1):765-796, 2018.  
Tor Lattimore and Csaba Szepesvári. The end of optimism? An asymptotic analysis of finite-armed linear bandits. In Artificial Intelligence and Statistics, pp. 728-737. PMLR, 2017.  
Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.  
Nick Littlestone and Manfred K Warmuth. The weighted majority algorithm. Information and Computation, 108(2):212-261, 1994.  
Anqi Mao, Mehryar Mohri, and Yutao Zhong. Ranking with abstention. arXiv preprint arXiv:2307.02035, 2023.  
Pierre Ménard and Aurélien Garivier. A minimax and asymptotically optimal algorithm for stochastic bandits. In International Conference on Algorithmic Learning Theory, pp. 223-237. PMLR, 2017.  
Gergely Neu and Nikita Zhivotovsky. Fast rates for online prediction with abstention. In Conference on Learning Theory, pp. 3030-3048. PMLR, 2020.  
William R. Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3-4):285-294, 1933.  
Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, 2009.  
Yair Wiener and Ran El-Yaniv. Pointwise tracking the optimal regression function. Advances in Neural Information Processing Systems, 25, 2012.  
Ahmed Zaoui, Christophe Denis, and Mohamed Hebiri. Regression with reject option and application to knn. Advances in Neural Information Processing Systems, 33:20073-20082, 2020.
