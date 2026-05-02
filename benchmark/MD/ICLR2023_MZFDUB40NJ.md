# UNCERTAINTY-AWARE OFF POLICY LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Off-policy learning, referring to the procedure of policy optimization with access only to logged feedback data, has shown importance in various real-world applications, such as search engines, recommender systems, etc. While the ground-truth logging policy, which generates the logged data, is usually unknown, previous work directly takes its estimated value in off-policy learning, resulting in a biased estimator. This estimator has both high bias and variance on samples with small and inaccurate estimated logging probabilities. In this work, we explicitly model the uncertainty in the estimated logging policy and propose a novel Uncertainty-aware Inverse Propensity Score estimator (UIPS) for improved off-policy learning. Experiment results on synthetic and three real-world recommendation datasets demonstrate the advantageous sample efficiency of the proposed UIPS estimator.

# 1 INTRODUCTION

In many real-world applications, including search engines (Agarwal et al. (2019)), online advertisements (Strehl et al. (2010)), recommender systems (Chen et al. (2019); Liu et al. (2022)), only logged feedback data is available for subsequent policy optimization. For example, in recommender systems, various complicated recommendation models (i.e., policies) (Zhou et al. (2018); Guo et al. (2017)) were optimized with the logged user interactions (e.g., clicks or staytime) to items recommended by previous recommendation policies. However, such logged data is known to be biased, since one does not know the feedback on items that previous policy (which is generally referred as the logging policy) did not take. This inevitably distorts the evaluation and optimization of a new policy when it tends to select items not in the logged data.

Off-policy learning emerges as a favorable way to learn an improved policy only from logged data by addressing the mismatch between the learning policy and the logging policy. One of the most commonly used off-policy learning methods is Inverse Propensity Scoring (IPS) (Chen et al. (2019); Munos et al. (2016)), which assigns per-sample importance weight to the training objective on the logged data, so as to get an unbiased optimization objective in expectation. The importance weight in IPS is the probability ratio between the learning policy and the logging policy.

However, the ground-truth logging policy is unavailable to the learner, e.g., it is not recorded in the data. One common treatment taken by previous work (Strehl et al. (2010); Liu et al. (2022); Chen et al. (2019); Ma et al. (2020)) is to first employ a supervised learning method (e.g., logistic regression, neural networks, etc.) to estimate the logging policy, and then take the estimated logging policy for off-policy learning. We theoretically show that such an approximation results in a biased estimator which is sensitive to those inaccurate and small estimated logging probabilities. Worse still, the small values of the estimated logging probabilities usually mean that there are fewer related samples in the logged data, so its estimation usually has high uncertainties, i.e., inaccurate estimation with high probability. Figure 1 shows a piece of empirical evidence from a large-scale recommendation benchmark KuaiRec dataset (Gao et al. (2022)), where items with lower frequencies in the logged dataset have lower estimated logging probabilities and higher uncertainties concurrently. The high bias and variance caused by these samples greatly hinder the performance of off-policy learning.

In this work, we explicitly take the uncertainty of the estimated logging policy into consideration and design a novel Uncertainty-aware Inverse Propensity Score estimator (UIPS) as the optimization objective for policy learning. UIPS introduces an additional weight to approach the ground-truth propensity from the estimated one, and learns an improved policy by alternating: (1) Find the optimal weight that makes the estimator as accurate as possible, taking into consideration the uncertainty

![](images/3ace53d4c46c980f107fcf9dfd3c0a8c180727013a5bc7cefd29499dfe661e35.jpg)  
(a) Estimated Logging Probability

![](images/adbc0323cb1254174207d97c9fdef5bb784959013c5e4fe0733145a2761eb24a.jpg)  
Figure 1: Estimated logging policy and its uncertainty under different item frequency on KuaiRec.  
(b) Uncertainty of Estimation

of estimated logging policy; (2) Improve the policy by optimizing the resulting estimator. We further find a closed-form solution for the optimal weight by deriving an upper bound on the mean squared error (MSE) to the ground-truth policy value. The optimal weight adjusts sample weights considering both the uncertainty of estimated logging probabilities and the propensity scores, rather than simply boosting or penalizing samples with high uncertain logging probabilities. Experiment results on synthetic and three real-world recommendation datasets demonstrate the sample efficiency of UIPS. All data and code can be found in supplementary materials for reproducibility.

To summarize, our contribution in this work is as follows:

- We point out that directly using the estimated logging policy leads to sub-optimal off-policy learning, since the resulting biased estimator is greatly distorted by samples with inaccurate and small estimated logging probabilities.  
- We take the uncertainty of estimated logging policy into consideration and propose UIPS for more accurate off-policy learning.  
- Experiments on synthetic and three real-world recommendation datasets demonstrate UIPS's strong advantage in sample efficiency in off-policy learning.

# 2 PRELIMINARY: OFF-POLICY LEARNING

We focus on the standard contextual bandit setup to explain the key concepts. Following convention (Joachims et al. (2018); Saito & Joachims (2022); Su et al. (2020)), let  $\pmb{x} \in \mathcal{X} \subseteq R^d$  be a  $d$ -dimensional context vector drawn from an unknown probability distribution  $p(\pmb{x})$ . Each context is associated with a finite set of actions denoted by  $\mathcal{A}$ , where  $|\mathcal{A}| < \infty$ . Let  $\pi: \mathcal{A} \times \mathcal{X} \to [0,1]$  denote a stochastic policy, such that  $\pi(a|x)$  is the probability of selecting action  $a$  under context  $x$  and  $\sum_{a \in \mathcal{A}} \pi(a|x) = 1$ . Under a given context, reward  $r_{x,a}$  is observed when action  $a$  is chosen. Take news recommendation for example,  $x$  represents the state of a user, summarizing his/her interaction history with the recommender system, each action  $a$  is a candidate news article, the policy is a recommendation algorithm, and the reward  $r_{x,a}$  denotes the user feedback on article  $a$ , e.g., whether the user clicks the article. Let  $V(\pi)$  denote the expected reward or value of the policy  $\pi$ :

$$
V (\pi) = \mathbb {E} _ {\boldsymbol {x} \sim p (\boldsymbol {x}), a \sim \pi (a | \boldsymbol {x})} [ r _ {\boldsymbol {x}, a} ]. \tag {1}
$$

We look for a policy  $\pi(a|\pmb{x})$  to maximize  $V(\pi)$ . In the rest of the paper, we denote  $\mathbb{E}_{\pmb{x} \sim p(\pmb{x}), a \sim \pi(a|\pmb{x})[\cdot]}$  as  $\mathbb{E}_{\pi}[\cdot]$  for simplicity.

In contrast to performing online updates by following the learning policy  $\pi(a|\pmb{x})$ , in off-policy learning we can only access a set of logged feedback data denoted by  $D := \{(\pmb{x}_n, a_n, r_{\pmb{x}_n, a_n}) | n \in [N]\}$ , where  $[N] := \{1, \dots, N\}$ . Given  $\pmb{x}_n$ , the action  $a_n$  was generated by a stochastic logging policy  $\beta^*$ , i.e., the probability action  $a_n$  was selected is  $\beta^*(a_n | \pmb{x}_n)$ . The actions  $\{a_1, \dots, a_N\}$  and their corresponding rewards  $\{r_{\pmb{x}_1, a_1}, \dots, r_{\pmb{x}_N, a_N}\}$  are generated independently given  $\beta^*$ . Due to the nature of policy optimization, the learning policy  $\pi(a|\pmb{x})$  is expected to be different from  $\beta^*(a|\pmb{x})$ , unless  $\beta^*(a|\pmb{x})$  is already optimal. Moreover, in practice the situation could be further complicated. Again, consider the news recommendation scenario. Due to the scalability requirement, industrial

recommender systems usually adopt a two-stage framework (Ma et al. (2020)), where one or several candidate generation models first produce a candidate set and a separate ranking model reranks candidate items to present top-K item to users. While  $\beta^{*}(a|x)$  depicts the whole two-stage process, the learning policy  $\pi(a|x)$  is usually employed in one particular stage (e.g., the reranking stage), implying drastic differences between the logging and learning policies. The main challenge of off-policy learning is then to address the distribution discrepancy between  $\beta^{*}(a|x)$  and  $\pi(a|x)$ , and learn a policy  $\pi(a|x)$  to maximize  $V(\pi)$  with access only to the logged dataset  $D$ .

One of most widely used methods to address the distribution shift between  $\pi(a|\pmb{x})$  and  $\beta^{*}(a|\pmb{x})$  is the Inverse Propensity Score (IPS) (Chen et al. (2019); Munos et al. (2016)). One can easily get that:

$$
V (\pi) = \mathbb {E} _ {\beta^ {*}} \left[ \frac {\pi (a | \boldsymbol {x})}{\beta^ {*} (a | \boldsymbol {x})} r _ {\boldsymbol {x}, a} \right],
$$

yielding the following empirical estimator of  $V(\pi)$ :

$$
\hat {V} _ {\mathrm {I P S}} (\pi) = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {\pi \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)}{\beta^ {*} \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)} r _ {\boldsymbol {x} _ {n}, a _ {n}}, \tag {2}
$$

where  $\pi (a_n|\pmb {x}_n) / \beta^* (a_n|\pmb {x}_n)$  is referred to as the propensity score. In the rest of paper, without further specification, we use the empirical estimation of expectation in our practical calculation. Various algorithms can be readily used for policy optimization under  $\hat{V}_{\mathrm{IPS}}(\pi)$ , including value-based methods (Silver et al. (2016)), policy-based methods (Levine & Koltun (2013); Schulman et al. (2015); Williams (1992)). In this work, we adopt a well-known policy gradient algorithm, REINFORCE (Williams (1992)). Assume policy  $\pi (a|x)$  is parameterized by  $\vartheta$ , via the "log-trick", the gradient of  $\hat{V}_{\mathrm{IPS}}(\pi_{\vartheta})$  with respect to  $\vartheta$  can be readily derived as follows:

$$
\nabla_ {\boldsymbol {\vartheta}} \hat {V} _ {\mathrm {I P S}} \left(\pi_ {\boldsymbol {\vartheta}}\right) = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {\pi \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)}{\beta^ {*} \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)} r _ {\boldsymbol {x} _ {n}, a _ {n}} \nabla_ {\boldsymbol {\vartheta}} \log \left(\pi_ {\boldsymbol {\vartheta}} \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)\right). \tag {3}
$$

Approximation with unknown logging policy. In many real-world applications, the logging policy, i.e., the  $\beta^{*}(a|\pmb{x})$  of each observation  $(\pmb{x}, a)$ , is unknown. One reason is the legacy issue, i.e., the probabilities were not logged when collecting data. Another reason is that the exact value of  $\beta^{*}(a|\pmb{x})$  is intrinsically unavailable such as in the two-stage recommender systems. As the solution, previous work employs various supervised learning methods (e.g., logistic regression (Schnabel et al. (2016)), neural networks (Chen et al. (2019), etc.) to estimate the logging policy, and replaces  $\beta^{*}(a|\pmb{x})$  with its estimated value  $\hat{\beta}(a|\pmb{x})$  to get the following estimator for policy learning:

$$
\hat {V} _ {\mathrm {B I P S}} \left(\pi_ {\boldsymbol {\vartheta}}\right) = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {\pi_ {\boldsymbol {\vartheta}} \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)}{\hat {\beta} \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)} r _ {\boldsymbol {x} _ {n}, a _ {n}}. \tag {4}
$$

However, as shown in the following proposition, inaccurate  $\hat{\beta}(a|\pmb{x})$  leads to high bias and variance of  $\hat{V}_{\mathrm{BIPS}}(\pi_{\vartheta})$ . Worse still, smaller inaccurate  $\hat{\beta}(a|\pmb{x})$  further enlarges the this bias and variance.

Proposition 1. The bias and variance of  $\hat{V}_{\mathrm{BIPS}}(\pi_{\vartheta})$  can be derived as follows:

$$
\operatorname {B i a s} \left(\hat {\mathrm {V}} _ {\mathrm {B I P S}} (\pi_ {\boldsymbol {\vartheta}})\right) = \mathbb {E} _ {D} \left[ \hat {V} _ {\mathrm {B I P S}} (\pi_ {\boldsymbol {\vartheta}}) - V (\pi_ {\boldsymbol {\vartheta}}) \right] = \mathbb {E} _ {\pi_ {\boldsymbol {\vartheta}}} \left[ r _ {\boldsymbol {x}, a} \left(\frac {\beta^ {*} (a | \boldsymbol {x})}{\hat {\beta} (a | \boldsymbol {x})} - 1\right) \right]
$$

$$
N \cdot \mathrm {V a r} _ {D} \left(\hat {V} _ {\mathrm {B I P S}} (\pi_ {\vartheta})\right) = \mathrm {V a r} _ {\pi_ {\vartheta}} \left(\frac {\beta^ {*} (a | \boldsymbol {x})}{\hat {\beta} (a | \boldsymbol {x})} r _ {\boldsymbol {x}, a}\right) + \mathbb {E} _ {\pi_ {\vartheta}} \left[ \left(\frac {\pi_ {\vartheta} (a | \boldsymbol {x})}{\beta^ {*} (a | \boldsymbol {x})} - 1\right) \cdot \frac {\beta^ {*} (a | \boldsymbol {x}) ^ {2}}{\hat {\beta} (a | \boldsymbol {x}) ^ {2}} r _ {\boldsymbol {x}, a} ^ {2} \right]
$$

Smaller  $\hat{\beta}(a|\pmb{x})$  usually implies fewer related training samples in the logged data, and thus  $\hat{\beta}(a|\pmb{x})$  will be inaccurate with a higher probability. To make it more explicit, we take KuaiRec dataset (Gao et al. (2022)) as an example and estimate the logging policy following (Chen et al. (2019)). Figure 1 shows the estimated  $\hat{\beta}(a|\pmb{x})$  and its corresponding uncertainties in items of different observation frequencies in the logged dataset. As uncertainty measures how large the confidence interval is about the current estimation, higher uncertainty implies that the true value may be away from the empirical mean estimate with a high probability. We defer the discussion about our detailed uncertainty calculation in Section 3. We can observe from Figure 1 that as item frequency decreases, the estimated logging probability also decreases, but the estimation uncertainty increases. This implies that smaller  $\hat{\beta}(a|\pmb{x})$  is usually 1) more inaccurate and 2) associated with high uncertainty.

As a result, with high bias and variance caused by inaccurate  $\hat{\beta}(a|\pmb{x})$ , it is erroneous to improve  $\pi_{\vartheta}(a|\pmb{x})$  by simply optimizing  $\hat{V}_{\mathrm{BIPS}}(\pi_{\vartheta})$ . We propose uncertainty-aware off-policy learning to address this challenge.

# 3 UNCERTAINTY-AWARE OFF-POLICY LEARNING

Our idea is incorporating the uncertainty of the logging policy estimation into policy learning. Observing that

$$
V (\pi_ {\boldsymbol {\vartheta}}) = \mathbb {E} _ {\beta^ {*}} \left[ \frac {\pi_ {\boldsymbol {\vartheta}} (a | \boldsymbol {x})}{\hat {\beta} (a | \boldsymbol {x})} \cdot \frac {\hat {\beta} (a | \boldsymbol {x})}{\beta^ {*} (a | \boldsymbol {x})} \cdot r _ {\boldsymbol {x}, a} \right],
$$

we propose to learn the optimal policy by optimizing the following empirical estimator:

$$
\hat {V} _ {\mathrm {U I P S}} \left(\pi_ {\boldsymbol {\vartheta}}\right) = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {\pi_ {\boldsymbol {\vartheta}} \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)}{\hat {\beta} \left(a _ {n} \mid \boldsymbol {x} _ {n}\right)} \cdot \phi_ {\boldsymbol {x} _ {n}, a _ {n}} \cdot r _ {\boldsymbol {x} _ {n}, a _ {n}} \tag {5}
$$

where  $\phi_{\pmb{x}_n,a_n}$  is a weight, which reflects  $\hat{\beta} (a_n|\pmb {x}_n) / \beta^* (a_n|\pmb {x}_n)$ , to be selected to make  $\hat{V}_{\mathrm{UIPS}}(\pi_{\vartheta})$  as close to  $V(\pi_{\vartheta})$  as possible. Intuitively, one should give small weights to samples whose  $\hat{\beta} (a|\pmb {x})$  is far below the ground-truth  $\beta^{*}(a|\pmb {x})$ . Thus, we divide offline policy improvement into two steps, and repeat them until certain convergence condition is met:

- Uncertainty aware policy evaluation: Derive the optimal uncertainty aware  $\phi_{\mathbf{x},a}$  to make  $\hat{V}_{\mathrm{UIPS}}(\pi_{\vartheta})$  as accurate as possible.  
- Policy Improvement: Learn an improved policy  $\pi_{\vartheta}(a|x)$  by optimizing  $\hat{V}_{\mathrm{UIPS}}(\pi_{\vartheta})$

# 3.1 Uncertainty Aware Policy Evaluation

Optimal uncertainty aware weight  $\phi_{\pmb{x},a}$ . We measure the accuracy of  $\hat{V}_{\mathrm{UIPS}}(\pi_{\pmb{\vartheta}})$  by its mean squared error (MSE) to  $V(\pi_{\pmb{\vartheta}})$  following previous work (Su et al. (2020); Saito & Joachims (2022)). MSE captures both the bias and variance of an estimator, since it is the summation of squared bias and variance. We then locate the  $\phi_{\pmb{x},a}$  that can minimize the MSE. In particular, we demonstrate the optimal  $\phi_{\pmb{x},a}$  has a closed-form formula which relates to both the value of  $\pi_{\pmb{\vartheta}}(a|\pmb{x}) / \hat{\beta}(a|\pmb{x})$  and the estimation uncertainty of  $\hat{\beta}(a|\pmb{x})$ .

More specifically, instead of directly minimizing the MSE, which is intractable, we find the desirable  $\phi_{\pmb{x},a}$  by minimizing the upper bound of MSE in the following theorem.

Theorem 1. Assume  $r_{\pmb{x},a} \in [0,1]$ , the mean squared error (MSE) between  $\hat{V}_{\mathrm{UIPS}}(\pi_{\vartheta})$  and ground-truth estimator  $V(\pi_{\vartheta})$  is upper bounded as follows:

$$
\begin{array}{l} \mathrm {M S E} \left(\hat {V} _ {\mathrm {U I P S}} (\pi_ {\boldsymbol {\vartheta}})\right) = \mathbb {E} _ {D} \left[ \left(\hat {V} _ {\mathrm {U I P S}} (\pi_ {\boldsymbol {\vartheta}}) - V (\pi_ {\boldsymbol {\vartheta}})\right) ^ {2} \right] = \mathrm {B i a s} \left(\hat {V} _ {\mathrm {U I P S}} (\pi_ {\boldsymbol {\vartheta}})\right) ^ {2} + \mathrm {V a r} \left(\hat {V} _ {\mathrm {U I P S}} (\pi_ {\boldsymbol {\vartheta}})\right) \\ \leq \mathbb {E} _ {\boldsymbol {\pi} _ {\boldsymbol {\vartheta}}} \left[ r _ {\boldsymbol {x}, a} ^ {2} \frac {\boldsymbol {\pi} _ {\boldsymbol {\vartheta}} (a | \boldsymbol {x})}{\beta^ {*} (a | \boldsymbol {x})} \right] \cdot \mathbb {E} _ {\beta^ {*}} \left[ \left(\frac {\beta^ {*} (a | \boldsymbol {x})}{\hat {\beta} (a | \boldsymbol {x})} \phi_ {\boldsymbol {x}, a} - 1\right) ^ {2} \right] + \mathbb {E} _ {\beta^ {*}} \left[ \frac {\boldsymbol {\pi} _ {\boldsymbol {\vartheta}} (a | \boldsymbol {x}) ^ {2}}{\hat {\beta} (a | \boldsymbol {x}) ^ {2}} \phi_ {\boldsymbol {x}, a} ^ {2} \right] \\ \end{array}
$$

The upper bound in Theorem 1 strictly increases with the two expectations related to  $\phi_{\pmb{x},a}$ , which implies that for some choice  $\lambda \in [0,\infty]$ , the MSE-optimizing  $\phi_{\pmb{x},a}$  can be derived by minimizing:

$$
\lambda \mathbb {E} _ {\beta^ {*}} \left[ \left(\frac {\beta^ {*} (a | \boldsymbol {x})}{\hat {\beta} (a | \boldsymbol {x})} \phi_ {\boldsymbol {x}, a} - 1\right) ^ {2} \right] + \mathbb {E} _ {\beta^ {*}} \left[ \frac {\pi_ {\vartheta} (a | \boldsymbol {x}) ^ {2}}{\hat {\beta} (a | \boldsymbol {x}) ^ {2}} \phi_ {\boldsymbol {x}, a} ^ {2} \right]. \tag {6}
$$

We cannot directly minimize Eq.(6) since the unknown  $\beta^{*}(a|\pmb{x})$  is involved. However, various ways (Gal & Ghahramani (2016); Xu et al. (2021)) can be employed to get the confidence interval which will contain  $\beta^{*}(a|\pmb{x})$  with high probability. More specifically, following previous work (Joachims et al. (2018)), we assume  $\beta^{*}(a|\pmb{x})$  can be modelled by a softmax function on top of an unknown function  $f_{\theta^{*}}(\pmb{x}, a)$ , i.e., the realizable assumption. Then we can get:

$$
\beta^ {*} (a | \boldsymbol {x}) = \frac {\exp \left(f _ {\boldsymbol {\theta} ^ {*}} (\boldsymbol {x} , a)\right)}{\sum_ {a ^ {\prime}} \exp \left(f _ {\boldsymbol {\theta} ^ {*}} \left(\boldsymbol {x} , a ^ {\prime}\right)\right)}, \quad \hat {\beta} (a | \boldsymbol {x}) = \frac {\exp \left(f _ {\boldsymbol {\theta}} (\boldsymbol {x} , a)\right)}{\sum_ {a ^ {\prime}} \exp \left(f _ {\boldsymbol {\theta}} \left(\boldsymbol {x} , a ^ {\prime}\right)\right)}, \tag {7}
$$

where  $f_{\theta}(\pmb{x}, a)$  is an estimate of  $f_{\theta^*}(\pmb{x}, a)$ . Following the conventional definition of confidence interval (Abbasi-Yadkori et al. (2011)), we define  $\gamma$  and  $U_{\pmb{x},a}$  such that  $|f_{\theta^*}(\pmb{x}, a) - f_{\theta}(\pmb{x}, a)| \leq \gamma U_{\pmb{x},a}$  holds with probability at least 1-δ, where  $\gamma$  is a function of  $\delta$  (typically the smaller  $\delta$  is, the larger  $\gamma$  is). Then  $\gamma U_{\pmb{x},a}$  measures the width of confidence interval of  $f_{\theta}(\pmb{x}, a)$  against its ground-truth  $f_{\theta^*}(\pmb{x}, a)$ . This implies that  $\beta^*(a|\pmb{x}) \in B_{\pmb{x},a}$  with probability at least 1-δ, where:

$$
\pmb {B} _ {\pmb {x}, a} = \left[ \frac {\hat {Z} \exp (- \gamma U _ {\pmb {x} , a})}{Z ^ {*}} \hat {\beta} (a | \pmb {x}), \frac {\hat {Z} \exp (\gamma U _ {\pmb {x} , a})}{Z ^ {*}} \hat {\beta} (a | \pmb {x}) \right], Z ^ {*} = \sum_ {a ^ {\prime}} \exp (f _ {\theta^ {*}} (a ^ {\prime} | \pmb {x})), \hat {Z} = \sum_ {a ^ {\prime}} \exp (f _ {\theta} (a ^ {\prime} | \pmb {x})).
$$

Since  $\beta^{*}(a|\pmb{x})$  can be any value in  $B_{\pmb{x},a}$ , we adopt the idea of robust optimization (Chen et al. (2020)) and find the optimal  $\phi_{\pmb{x},a}$  by solving the following optimization problem:

$$
\min  _ {\boldsymbol {\phi} _ {\boldsymbol {x}, a}} \max  _ {\beta_ {\boldsymbol {x}, a} \in \boldsymbol {B} _ {\boldsymbol {x}, a}} \quad \lambda \mathbb {E} _ {\beta^ {*}} \left[ \left(\frac {\beta_ {\boldsymbol {x} , a}}{\hat {\beta} (a | \boldsymbol {x})} \phi_ {\boldsymbol {x}, a} - 1\right) ^ {2} \right] + \mathbb {E} _ {\beta^ {*}} \left[ \frac {\pi_ {\boldsymbol {\theta}} (a | \boldsymbol {x}) ^ {2}}{\hat {\beta} (a | \boldsymbol {x}) ^ {2}} \phi_ {\boldsymbol {x}, a} ^ {2} \right]. \tag {8}
$$

The following theorem derives a closed-form formula for the optimal solution of (8).

Theorem 2. Let  $\eta_1, \eta_2 \in [\exp(-\gamma U_x^{\max}), \exp(\gamma U_x^{\max})]$ , where  $U_x^{\max} = \max_a U_{x,a}$ . The optimization problem in Eq.(8) has a closed-form solution as follows:

$$
\phi_ {\boldsymbol {x}, a} ^ {*} = \min  \left(\lambda / \left[ \frac {\lambda}{\eta_ {1}} \exp (- \gamma U _ {\boldsymbol {x}, a}) + \frac {\eta_ {1} \pi_ {\vartheta} (a | \boldsymbol {x}) ^ {2}}{\hat {\beta} (a | \boldsymbol {x}) ^ {2} \exp (- \gamma U _ {\boldsymbol {x} , a})} \right], 2 \eta_ {2} / [ \exp (\gamma U _ {\boldsymbol {x}, a}) + \exp (- \gamma U _ {\boldsymbol {x}, a}) ]\right)
$$

Insights on  $\phi_{\pmb{x},a}^{*}$ . The second term of  $\phi_{\pmb{x},a}^{*}$  (i.e.,  $2\eta_{2} / [\exp (\gamma U_{\pmb{x},a}) + \exp (-\gamma U_{\pmb{x},a})]$ ) acts like a capping threshold to ensure  $\phi_{\pmb{x},a}^{*} \leq 2\eta_{2}$  holds even with small  $\pi_{\vartheta}(a|x) / \hat{\beta}(a|x)$  (as shown in Lemma 1 in Appendix A.1). The key component is the first term, and Lemma 1 implies that:

- If the propensity score  $\pi_{\vartheta}(a|x) / \hat{\beta}(a|x)$  is above the threshold  $\sqrt{\lambda} / \eta_1$ , UISP will assign a smaller weight to samples with more uncertain/inaccurate  $\hat{\beta}(a|x)$  to prevent the distortions from those with a large propensity score but an inaccurate logging probability.  
- If the propensity score  $\pi_{\vartheta}(a|x) / \hat{\beta}(a|x)$  is below the threshold (but not small enough to activate the second term), then the propensity score at the worse case (i.e., taking  $B_{x,a}^{-} = \hat{\beta}(a|x)\hat{Z}\exp(-\gamma U_{x,a}) / Z^{*}$  as denominator) matters. If the propensity score at the worse case is under control, i.e.,  $\pi_{\vartheta}(a|x) / B_{x,a}^{-} < \sqrt{\lambda}$ , a larger  $U_{x,a}$  implies a small propensity score  $\pi_{\vartheta}(a|x) / \hat{\beta}(a|x)$ , and UISP tends to boost this safe sample with a higher  $\phi_{x,a}^{*}$ . Otherwise  $\phi_{x,a}^{*}$  still decreases as  $U_{x,a}$  becomes higher.

Uncertainty estimation. Now we describe how to calculate  $U_{\boldsymbol{x},a}$ , i.e., the uncertainty of the estimated  $\hat{\beta}(a|\boldsymbol{x})$ . In this work, we choose to estimate  $\beta^{*}(a|\boldsymbol{x})$  using a neural network, due to its encouraging representation learning capacity. And various ways (Gal & Ghahramani (2016); Xu et al. (2021)) can be leveraged to perform the uncertainty estimation in a neural network. For example, (Gal & Ghahramani, 2016) proposed to estimate uncertainty using dropout; and (Xu et al., 2021) provided a theoretical bound. Here we adopt the result in (Xu et al. (2021)) due to its computational efficiency and theoretical soundness. Following the proof of Theorem 4.4 in (Xu et al. (2021)), given the logged dataset  $D$ , we can get with high probability  $\exists \gamma$ :

$$
| f _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {n}, a _ {n}) - f _ {\boldsymbol {\theta} ^ {*}} (\boldsymbol {x} _ {n}, a _ {n})) | \leq \gamma \sqrt {\boldsymbol {g} (\boldsymbol {x} _ {n} , a _ {n}) ^ {T} \boldsymbol {M} _ {D} ^ {- 1} \boldsymbol {g} (\boldsymbol {x} _ {n} , a _ {n})}
$$

where  $\pmb{g}(\pmb{x}_n, a_n)$  is the gradient of function  $f_{\pmb{\theta}}(\pmb{x}_n, a_n)$ , i.e.,  $\pmb{g}(\pmb{x}_n, a_n) = \nabla_{\pmb{\theta}} f_{\pmb{\theta}}(\pmb{x}_n, a_n)$ , and  $M_D = \sum_{n=1}^N \pmb{g}(\pmb{x}_n, a_n) \pmb{g}(\pmb{x}_n, a_n)^T$ , implying  $U_{\pmb{x}_n, a_n} = \sqrt{\pmb{g}(\pmb{x}_n, a_n)^T \pmb{M}_D^{-1} \pmb{g}(\pmb{x}_n, a_n)}$ .

# 3.2 Policy Improvement

After getting the optimal  $\phi_{\pmb{x},a}^{*}$  as in Theorem 2, the policy  $\pi_{\vartheta}(a|\pmb{x})$  can be updated by the following REINFORCE gradient:

$$
\nabla_ {\vartheta} V _ {\mathrm {U I P S}} \left(\pi_ {\vartheta}\right) = \mathbb {E} _ {\beta^ {*}} \left[ \frac {\pi_ {\vartheta} (a | \boldsymbol {x})}{\hat {\beta} (a | \boldsymbol {x})} \cdot \phi_ {\boldsymbol {x}, a} ^ {*} \cdot r _ {\boldsymbol {x}, a} \nabla_ {\vartheta} \log \left(\pi_ {\vartheta} (a | \boldsymbol {x})\right) \right]. \tag {9}
$$

UIPS then iterates policy evaluation and policy improvement for policy learning until converge.

# 4 EMPIRICAL EVALUATION

In this section, we evaluate UIPS on both synthetic datasets and three real-world datasets with unbiased data. We compare UIPS with the following baselines, which can be grouped into five categories:

- Cross-Entropy (CE): A supervised learning method with the cross-entropy loss as its objective, which is the commonly used learning approach for a model with softmax output. No off-policy correction is performed in this method.

- IPS-Cap (Chen et al. (2019)): The standard IPS based off-policy learning, which prunes propensity scores to control variance, i.e., taking  $\min(c, \frac{\pi_{\phi}(a|x)}{\hat{\beta}(a|x)})$  as the propensity score. Setting  $c$  to a small value can reduce variance, but introduces bias.  
- MinVar & stableVar (Zhan et al. (2021)), Shrinkage (Su et al. (2020)): This line of work improves off-policy evaluation estimators by reweighing each sample. For example, MinVar and stableVar reweigh each sample by  $\frac{h_{\boldsymbol{x},a}}{\sum_{a'} h_{\boldsymbol{x},a'}}$  with  $h_{\boldsymbol{x},a} = \frac{\hat{\beta}(a|\boldsymbol{x})}{\pi_{\vartheta}(a|\boldsymbol{x})^2}$  and  $h_{\boldsymbol{x},a} = \frac{\sqrt{\hat{\beta}(a|\boldsymbol{x})}}{\pi_{\vartheta}(a|\boldsymbol{x})}$  respectively, since they find that  $\pi_{\vartheta}(a|\boldsymbol{x})^2 / \hat{\beta}(a|\boldsymbol{x})$  is directly related to variance. Su et al. (2020) proposes to shrink the propensity scores by multiplying a weight  $\lambda / (\lambda + \frac{\pi_{\vartheta}(a|\boldsymbol{x})^2}{\hat{\beta}(a|\boldsymbol{x})^2})$ , which is a special case of the proposed UIPS with  $U_{\boldsymbol{x},a} = 0$  and  $\eta_1 = 1$ . All these work simply treats  $\hat{\beta}(a|\boldsymbol{x})$  as  $\beta^*(a|\boldsymbol{x})$ , and none of them consider the accuracy or uncertainty of  $\hat{\beta}(a|\boldsymbol{x})$ .  
- SNIPS (Swaminathan & Joachims (2015c)), BanditNet (Joachims et al. (2018)), POEM (Swaminathan & Joachims (2015b)), POXM (Lopez et al. (2021)), Adaptive (Liu et al. (2022)): This line of work aims for more stable and accurate policy learning. For example, SNIPS normalizes the estimator by the sum of propensity scores in each batch. BanditNet extends SNIPS and leverages an additional Lagrangian term to normalize the estimator by an approximated sum of propensity scores of all samples. POEM jointly optimizes the estimator and its variance. POXM controls estimation variance by pruning samples with small logging probabilities. Adaptive proposes a new formulation to utilize negative samples.  
- UIPS-P and UIPS-O: Two variants of our proposed UIPS with different ways of leveraging uncertainties. UIPS-P directly penalizes samples whose estimated logging probabilities are of high uncertainties, i.e., taking  $\phi_{\pmb{x},a} = 1.0 / \exp (\gamma U_{\pmb{x},a})$ , which follows previous work on offline reinforcement learning (Wu et al. (2021); An et al. (2021)). UIPS-O adversarially uses the worst propensity scores  $(\pi_{\vartheta}(a|x) / B_{x,a}^{-})$  for policy learning, i.e.,  $\phi_{\pmb{x},a} = 1.0 / \exp (-\gamma U_{\pmb{x},a})$ .

# 4.1 Synthetic Data

Data generation. Following previous work (Ma et al. (2020); Lopez et al. (2021)), we generate a synthetic dataset by a supervision-to-bandit conversion on Wiki10-31K dataset (Bhatia et al. (2016)), which is an extreme multi-label classification dataset. The Wiki10-31K dataset contains approximately 20K samples. Each sample is a associated with a feature vector  $\tilde{\pmb{x}}$  of 101,938 dimensions and a label vector  $\pmb{y}_{\tilde{\pmb{x}}}$  of 31K classes with more than one positive class. Let  $\pmb{y}_{\tilde{\pmb{x}},a}$  denote the label of class  $a$  under  $\tilde{\pmb{x}}$  and we take each class as an action. We adopt the Wiki10-31K dataset rather than ones in the UCI machine learning repository (Swaminathan & Joachims (2015a)), since it will be much harder with such a large action space.

We then split the dataset into train, validation, test sets with size 11K:3K:6K. The test set is from the official split. Since the original feature vector  $\tilde{\pmb{x}}$  is too sparse, for ease of learning, we first embed it to dimension  $d$  by  $\pmb {x} = \pmb {W}\tilde{\pmb{x}}$ , and synthesize the ground-truth logging policy  $\beta^{*}(a|x)$  by:

$$
\beta^ {*} (a | \boldsymbol {x}) = \frac {\exp \left(\boldsymbol {x} ^ {T} \boldsymbol {\theta} _ {a} ^ {*} / \tau\right)}{\sum_ {a ^ {\prime}} \exp \left(\boldsymbol {x} ^ {T} \boldsymbol {\theta} _ {a ^ {\prime}} ^ {*} / \tau\right)}, \tag {10}
$$

where  $\mathbf{W}$  and  $\{\theta_{a}^{*}\}$  are pre-learned parameters by applying a logistic regression model on the train set,  $\tau$  is a hyper-parameter that controls the skewness of logging distribution. A small value of  $\tau$  leads to a near-deterministic policy, while a larger  $\tau$  makes logging policy smoother. For each sample in train set, given  $\mathbf{x}$ , we sample an action  $a$  according to  $\beta^{*}(a|\mathbf{x})$ , and obtain the reward  $r_{\mathbf{x},a} = \mathbf{y}_{\tilde{\mathbf{x}},a}$ , resulting bandit feedback  $(\mathbf{x},a,r_{\mathbf{x},a})$ . We repeat above process  $N$  times to collect the logged dataset. In our experiments, we take  $d = 64$ ,  $N = 100$ . Due to space limit, more implementation details can be found in Appendix A.2.

Evaluation metrics. To evaluate the learned policy  $\pi_{\vartheta}(a|x)$ , we calculate Precision@K (P@K), Recall@K (R@K) and NDCG@K as in previous work (Lopez et al. (2021); Ma et al. (2020)). Higher P@K, R@K and NDCG@K implies a better policy.

Table 1 shows the experiment results on three synthetic datasets generated under different  $\tau$ . We can first observe that as  $\tau$  increases, i.e., the probability of selecting positive actions decreases, the performance of most algorithms drop, including CE, IPS-Cap, UIS, Shrinkage, POEM, Adaptive, etc. However, UIS still achieves the best performance on all three datasets under all three metrics. SNIPS, BanditNet, POXM are more robust to small logging probabilities of positive actions,

<table><tr><td></td><td colspan="3">τ = 0.5</td><td colspan="3">τ = 1</td><td colspan="3">τ = 2</td></tr><tr><td>Algorithm</td><td>P@5</td><td>R@5</td><td>NDCG@5</td><td>P@5</td><td>R@5</td><td>NDCG@5</td><td>P@5</td><td>R@5</td><td>NDCG@5</td></tr><tr><td>CE</td><td>0.5559</td><td>0.1575</td><td>0.6048</td><td>0.5508</td><td>0.1561</td><td>0.5999</td><td>0.5447</td><td>0.1544</td><td>0.5933</td></tr><tr><td>IPS-Cap</td><td>0.5583</td><td>0.1576</td><td>0.6093</td><td>0.5554</td><td>0.1573</td><td>0.6028</td><td>0.5471</td><td>0.1549</td><td>0.5946</td></tr><tr><td>MinVar</td><td>0.5372</td><td>0.1521</td><td>0.5905</td><td>0.5342</td><td>0.1508</td><td>0.5855</td><td>0.5478</td><td>0.1551</td><td>0.5962</td></tr><tr><td>StableVar</td><td>0.4430</td><td>0.1269</td><td>0.5073</td><td>0.5450</td><td>0.1546</td><td>0.5959</td><td>0.5349</td><td>0.1512</td><td>0.5851</td></tr><tr><td>Shrinkage</td><td>0.5573</td><td>0.1577</td><td>0.6059</td><td>0.5608</td><td>0.1580</td><td>0.6122</td><td>0.5465</td><td>0.1545</td><td>0.5963</td></tr><tr><td>SNIPS</td><td>0.4287</td><td>0.1225</td><td>0.4994</td><td>0.4714</td><td>0.1327</td><td>0.5418</td><td>0.5065</td><td>0.1420</td><td>0.5748</td></tr><tr><td>BanditNet</td><td>0.4711</td><td>0.1329</td><td>0.5393</td><td>0.4811</td><td>0.1351</td><td>0.5498</td><td>0.4876</td><td>0.1363</td><td>0.5590</td></tr><tr><td>POEM</td><td>0.5541</td><td>0.1562</td><td>0.6046</td><td>0.5540</td><td>0.1565</td><td>0.6039</td><td>0.5459</td><td>0.1545</td><td>0.5946</td></tr><tr><td>POXM</td><td>0.4707</td><td>0.1328</td><td>0.5392</td><td>0.4646</td><td>0.1302</td><td>0.5380</td><td>0.4870</td><td>0.1361</td><td>0.5584</td></tr><tr><td>Adaptive</td><td>0.4203</td><td>0.1165</td><td>0.4909</td><td>0.4160</td><td>0.1154</td><td>0.4912</td><td>0.3939</td><td>0.1097</td><td>0.4704</td></tr><tr><td>UIPS-P</td><td>0.4864</td><td>0.1366</td><td>0.5566</td><td>0.4827</td><td>0.1354</td><td>0.5528</td><td>0.4852</td><td>0.1355</td><td>0.5566</td></tr><tr><td>UIPS-O</td><td>0.4871</td><td>0.1382</td><td>0.5528</td><td>0.4797</td><td>0.1348</td><td>0.5479</td><td>0.5124</td><td>0.1441</td><td>0.5796</td></tr><tr><td>UIPS</td><td>0.5666</td><td>0.1606</td><td>0.6169</td><td>0.5624</td><td>0.1583</td><td>0.6154</td><td>0.5497</td><td>0.1553</td><td>0.6005</td></tr></table>

Table 1: Experimental results on synthetic datasets.  
Table 2: Performance under different uncertainties.  

<table><tr><td></td><td colspan="3">Low Frequent Action Related Samples
(High Uncertainty)</td><td colspan="3">High Frequent Action Related Samples
(Low Uncertainty)</td></tr><tr><td>Algorithm</td><td>P@5(RI)</td><td>R@5(RI)</td><td>NDCG@5(RI)</td><td>P@5(RI)</td><td>R@5(RI)</td><td>NDCG@5(RI)</td></tr><tr><td>CE</td><td>0.5186</td><td>0.1228</td><td>0.5521</td><td>0.5931</td><td>0.1921</td><td>0.6575</td></tr><tr><td>IPS-Cap</td><td>0.5170(-0.31%)</td><td>0.1218(-0.81%)</td><td>0.5539(+0.32%)</td><td>0.5996(+1.10%)</td><td>0.1935(+0.73%)</td><td>0.6647(+1.10%)</td></tr><tr><td>Shrinkage</td><td>0.5145(-0.79%)</td><td>0.1212(-1.30%)</td><td>0.5519(-0.04%)</td><td>0.5982(+0.86%)</td><td>0.1931(+0.52%)</td><td>0.6628(+0.81%)</td></tr><tr><td>UIPS</td><td>0.5276(+1.74%)</td><td>0.1250(+1.79%)</td><td>0.5623(+1.85%)</td><td>0.6055(+2.09%)</td><td>0.1961(+2.08%)</td><td>0.6715(+2.13%)</td></tr></table>

![](images/3b4cc2cbf939d93750362053944d43aae426489d8477bad2a58d1fb189ac420b.jpg)  
Figure 2: Effect of  $\lambda$  and  $\gamma$  on NDCG@5.

Figure 3: MSE of different off-policy evaluation estimators.  

<table><tr><td>Algorithm</td><td>MSE</td></tr><tr><td>IPS-CaP</td><td>0.4953</td></tr><tr><td>minVar</td><td>0.8928</td></tr><tr><td>stableVar</td><td>0.8112</td></tr><tr><td>Shrinage</td><td>0.5125</td></tr><tr><td>UIPS</td><td>0.4516</td></tr></table>

since their performance increase as  $\tau$  increases. Moreover, when it is harder to accurately estimate the logging probabilities, for example, the logging policy is skewed to part of positive actions (i.e.,  $\tau = 0.5$ ), or the logging policy is too smooth to sample enough positive actions (i.e.,  $\tau = 2$ ), considering the estimation uncertainty generally leads to larger improvements. For example, compared to Shrinkage (a special case of UIS with uncertainties always being zero), UIS achieves  $1.8\%$  ( $0.7\%$ ) improvements on NDCG@5 when  $\tau = 0.5$  ( $\tau = 2.0$ ), which is larger than  $0.5\%$  with  $\tau = 1.0$ . Finally, regardless of the scale of propensity scores, blindly reweighing through uncertainties also leads to poor performance, as shown by UIS-P and UIS-O.

Performance under different uncertainty levels. As shown in Figure 1, low frequency actions in the logged dataset suffer higher uncertainties in their propensity estimation. Thus, we divide the test set into two subsets according to the average frequency of associated actions, where the uncertainty in the subset associated with low-frequency actions is on average  $9\%$  higher than that in the subset associated with high-frequency actions. Table 2 shows the results on these two subsets when  $\tau = 0.5$ . We only report the results of the best three baselines due to space limit. One can clearly observe that only UISs performed better than CE on the test set associated with low frequency actions, implying the advantage of UISs in dealing with the inaccurately estimated logging probabilities.

Ablation Study. In this experiment, we aim to answer two questions: (1) Can  $\hat{V}_{\mathrm{UIPS}}(\pi_{\vartheta})$  in Eq. (5) lead to more accurate off-policy evaluation? (2) How will UISP perform with different hyperparameters. Due to space limit, we report results on synthetic dataset with  $\tau = 0.5$ .

To answer the first question, we evaluate the following  $\epsilon$ -greedy policy:  $\pi(a|\boldsymbol{x}) = \frac{1 - \epsilon}{|M_x|} \cdot \mathbb{I}\{a \in M_x\} + \epsilon / |\mathcal{A}|$ , where  $M_x$  contains all positive actions associated with feature vector  $\boldsymbol{x}$ . Then for

Table 3: Experimental results on real-world unbiased datasets.  

<table><tr><td></td><td colspan="3">Yahoo</td><td colspan="3">Coat</td><td colspan="3">KuaiRec</td></tr><tr><td>Algorithm</td><td>P@5</td><td>R@5</td><td>NDCG@5</td><td>P@5</td><td>R@5</td><td>NDCG@5</td><td>P@50</td><td>R@50</td><td>NDCG@50</td></tr><tr><td>CE</td><td>0.2840</td><td>0.7648</td><td>0.6153</td><td>0.2766</td><td>0.4548</td><td>0.4493</td><td>0.8769</td><td>0.0239</td><td>0.8769</td></tr><tr><td>IPS-Cap</td><td>0.2718</td><td>0.7323</td><td>0.5801</td><td>0.2841</td><td>0.4741</td><td>0.4509</td><td>0.8746</td><td>0.0238</td><td>0.8791</td></tr><tr><td>MinVar</td><td>0.2878</td><td>0.7783</td><td>0.6296</td><td>0.2862</td><td>0.4803</td><td>0.4456</td><td>0.8811</td><td>0.0240</td><td>0.8864</td></tr><tr><td>StableVar</td><td>0.2803</td><td>0.7499</td><td>0.5973</td><td>0.2834</td><td>0.4667</td><td>0.4433</td><td>0.8516</td><td>0.0231</td><td>0.8631</td></tr><tr><td>Shrinkage</td><td>0.2842</td><td>0.7670</td><td>0.6247</td><td>0.2828</td><td>0.4656</td><td>0.4535</td><td>0.8759</td><td>0.0238</td><td>0.8695</td></tr><tr><td>SNIPS</td><td>0.2265</td><td>0.5979</td><td>0.4512</td><td>0.2717</td><td>0.4381</td><td>0.4139</td><td>0.8460</td><td>0.0230</td><td>0.8455</td></tr><tr><td>BanditNet</td><td>0.2474</td><td>0.6632</td><td>0.5172</td><td>0.2890</td><td>0.4674</td><td>0.4298</td><td>0.8869</td><td>0.0242</td><td>0.8918</td></tr><tr><td>POEM</td><td>0.2703</td><td>0.7280</td><td>0.5770</td><td>0.2834</td><td>0.4636</td><td>0.4479</td><td>0.8768</td><td>0.0239</td><td>0.8825</td></tr><tr><td>POXM</td><td>0.2277</td><td>0.6040</td><td>0.4738</td><td>0.2703</td><td>0.4374</td><td>0.4065</td><td>0.9059</td><td>0.0248</td><td>0.9133</td></tr><tr><td>Adaptive</td><td>0.2846</td><td>0.7634</td><td>0.6048</td><td>0.2848</td><td>0.4672</td><td>0.4259</td><td>0.8382</td><td>0.0227</td><td>0.8505</td></tr><tr><td>UIPS-P</td><td>0.2004</td><td>0.5188</td><td>0.3714</td><td>0.2786</td><td>0.4518</td><td>0.4118</td><td>0.8811</td><td>0.0240</td><td>0.8749</td></tr><tr><td>UIPS-O</td><td>0.1987</td><td>0.5142</td><td>0.3705</td><td>0.271</td><td>0.4341</td><td>0.4205</td><td>0.8819</td><td>0.0240</td><td>0.8764</td></tr><tr><td>UIPS</td><td>0.2878</td><td>0.7805</td><td>0.6329</td><td>0.2890</td><td>0.4821</td><td>0.4654</td><td>0.9124</td><td>0.0250</td><td>0.9180</td></tr></table>

each  $\pmb{x}$  in the test set, we sample 1K data points in a similar way as discussed previously to calculate the value of estimators. Table 3 shows the average MSE of the estimators to ground-truth policy value under 20 different random seeds. We only compared with baselines on off-policy evaluation estimator, i.e., IPS-Cap, MinVar, stableVar and Shrinkage. One can observe from Table 3 that UIS does lead to the smallest MSE, implying the most accurate off-policy evaluation.

To answer the second question, we fixed  $\eta_1, \eta_2$ , and vary  $\lambda$  and  $\gamma$  to track the performance of UIPS. Recall that a larger  $\gamma$  implies a higher chance the derived interval contains  $\beta^*(a|x)$ , while  $\sqrt{\lambda} / \eta_1$  is the threshold determining the value of  $\phi_{x,a}^*$  as shown in Lemma 1. Figure 2 reports NDCG@5 under different  $\gamma$  and  $\lambda$ . Results on P@5 and R@5 can be found in Appendix A.2. We can observe that to make UIPS perform,  $B_{x,a}$  needs to be of high confidence, e.g.,  $\gamma = 25$  performed the best when  $\tau = 0.5$ . Moreover, the threshold  $\sqrt{\lambda} / \eta_1$  cannot be too small or too large.

# 4.2 Real-World Data

Off-policy learning has been its utility in recommendation scenarios (Chen et al. (2019); Ma et al. (2020)), where context vector  $\pmb{x}$  denotes the state of a user and each candidate item is taken as an action. To further demonstrate the efficiency of UIPS in real-world scenarios, we evaluate it on three recommendation datasets with unbiased logging policies: (1) Yahoo!R3 $^1$ ; (2) Coat $^2$ ; (3) KuaiRec (Gao et al. (2022)), from music, fashion and micro-video recommendation scenario respectively. All these datasets contain a set of biased data collected from users' interactions on the platform, and a set of unbiased data collected from a randomized controlled trial where items are randomly selected. The statistics of the three datasets are summarized in Table 4 in Appendix A.2. As in (Ding et al. (2022)), the biased data is used for training, and the unbiased data is for testing, with a small part of unbiased data split for validation purpose (5% on Yahoo and Coat, and 15% on KuaiRec). We take the reward as 1 if: (1) the rating is larger than 3 in Yahoo!R3 and Coat datasets; (2) the user watched more than 70% of the video in KuaiRec. Otherwise, the reward is labeled as 0. Due to space limit, one can refer to Appendix A.2 for more implementation details, e.g., model architectures, etc.

We still adopt P@K, R@K and NDCG@K as our evaluation metrics. Following (Ding et al. (2022)), we take  $K = 5$  on Yahoo!R3 and Coat datasets, and  $K = 50$  on KuaiRec dataset. From Table 3, we can observe that on all three datasets, the proposed UIPS achieves the highest precision, recall and NDCG. IPS-Cap cannot consistently outperform CE: it outperforms CE on Coat and KuaiRec, but fails on the Yahoo dataset. One possible reason is due to the inaccuracy of the estimated logging probabilities, since number of logged samples per user is much smaller on Yahoo dataset. For example,  $70\%$  of testing users have fewer than 24 logged interactions in Yahoo dataset, while each user in Coat has 24 recorded interactions and the number is even larger on KuaiRec dataset. BanditNet, POEM and POXM tend to perform better with a larger action space, while MinVar, StableVar and Shrinkage as well as Adaptive are more suitable for scenarios with small action size. UIPS still outperforms Shrinkage, highlighting the importance of modeling uncertainty in the estimated logging policy. However, reweighing based solely on uncertainties, ignoring the corresponding propensity scores, will also lead to poor performance, as shown by UIPS-P and UIPS-O.

# 5 RELATED WORK

This work is the first of its kind to take into consideration the uncertainty of estimated logging policy for improved policy learning. The following two lines of work are related to this paper.

Off-policy learning. In many real-world applications, such as search engines, recommender systems, etc., interactive online model update is expensive and risky (Jiang & Li (2016)). Off-policy learning has therefore attracted increasing interest, since it can leverage the already logged feedback data (Agarwal et al. (2019); Chen et al. (2019); Liu et al. (2022)). The main challenge in off-policy learning is how to address the mismatch between the logging policy and the learning policy. One line of work (Achiam et al. (2017); Schulman et al. (2015)) circumvents this by constraining the learning policy not too far from the logging policy. However, such constraint is too restrictive thus not applicable in some scenarios such as recommender systems where user behaviors and items change rapidly. Another more common and widely-applied approach is to leverage Inverse Propensity Score (IPS) method to correct the discrepancy between two policies. And various methods are proposed for stabilized learning (Swaminathan & Joachims (2015c;a;b)) and variance control (Lopez et al. (2021); Liu et al. (2022)) on top of IPS. However, all these work directly use the estimated logging policy for off-policy correction, leading to sub-optimal performance as shown in our experiments. Some other work further extend IPS-based off-policy learning for more complex problems, such as slate recommendation (Swaminathan et al. (2017)), two-stage recommender systems (Ma et al. (2020)), etc. But they still fail to realize the effect of accuracy of the estimated logging policy. A recent work (Ding et al. (2022)) on causal recommendation also argues that propensity scores may not be correct due to unobserved confounders. However, they assume the effect of the unobserved confounder for any sample can be bounded by a pre-defined hyper-parameter, and adversarially search for the worst-case propensity to update model parameters. Adapting to off-policy learning, it is a special case of our UIS-P variant with uncertainty as a pre-defined constant.

Off-policy learning can be directly built on off-policy evaluation. In this line of research, several work (Su et al. (2020); Zhan et al. (2021)) also propose to control the high variance of learning caused by small logging probabilities by instance reweighing. However, they directly take the estimated logging policy as true logging policy for correction, thus worse than UIPS as shown in experiments. A recent work (Saito & Joachims (2022)) assumes additional structure in action space and proposes the marginalized IPS. Instead, our work considers the uncertainty when estimating the logging policy and thus does not add new assumptions about the problem space.

Uncertainty-aware Learning. Estimation uncertainty has been extensively used for making trade-offs between exploration and exploitation in online learning (Xu et al. (2021); Zhou et al. (2020); Abbasi-Yadkori et al. (2011)). Recently, several work on offline reinforcement learning (Wu et al. (2021); An et al. (2021); Bai et al. (2022)) penalizes the value function of out-of-distribution states and actions by directly subtracting uncertainty to tackle the extrapolating error. However, blindly penalizing samples of high uncertainty (i.e., UISP-P) is problematic, as we showed in our experiments. Proper correction depends on both uncertainty in logging policy estimation and the actual value of estimated logging probabilities.

# 6 CONCLUSION

In this paper, we propose a novel Uncertainty-aware Inverse Propensity Score estimator (UIPS) to explicitly model the uncertainty about estimated logging policy for improved off-policy learning. UIPS weighs each logged instance to approach the ground-truth estimator and a closed-form solution of the optimal weight is derived by minimizing the upper bound of the mean squared error (MSE). An improved policy can be obtained by optimizing the resulting estimator. Extensive experiments on synthetic datasets and three real-world datasets demonstrate the efficiency of UIPS.

As demonstrated in this work, explicitly modeling the uncertainty of estimated logging policy is crucial for effective off-policy learning; but the best use of this uncertainty is not to simply down-weigh or drop instances with uncertain estimations, but to balance it with the actually estimated logging probabilities in a per-instance basis. As our future work, it is promising to investigate how UIPS can be extended to value-based learning methods, e.g., actor-critics. And on the other hand, it is also important to analyze how tight our upper bound analysis of MSE is; and if possible, find new ways to tighten it for improvements.

# REFERENCES

Yasin Abbasi-Yadkori, David Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems, pp. 2312-2320, 2011.  
Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pp. 22-31. PMLR, 2017.  
Aman Agarwal, Ivan Zaitsev, Xuanhui Wang, Cheng Li, Marc Najork, and Thorsten Joachims. Estimating position bias without intrusive interventions. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, pp. 474-482, 2019.  
Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline reinforcement learning with diversified q-ensemble. Advances in neural information processing systems, 34:7436-7447, 2021.  
Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhihong Deng, Animesh Garg, Peng Liu, and Zhaoran Wang. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. arXiv preprint arXiv:2202.11566, 2022.  
K. Bhatia, K. Dahiya, H. Jain, P. Kar, A. Mittal, Y. Prabhu, and M. Varma. The extreme classification repository: Multi-label datasets and code, 2016. URL http://manikvarma.org/downloads/XC/XMLRepository.html.  
Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, and Ed H Chi. Top-k off-policy correction for a reinforce recommender system. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, pp. 456-464, 2019.  
Ruidi Chen, Ioannis Ch Paschalidis, et al. Distributionally robust learning. Foundations and Trends® in Optimization, 4(1-2):1-243, 2020.  
Sihao Ding, Peng Wu, Fuli Feng, Yitong Wang, Xiangnan He, Yong Liao, and Yongdong Zhang. Addressing unmeasured confounder for recommendation with sensitivity analysis. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 305-315, 2022.  
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050-1059. PMLR, 2016.  
Chongming Gao, Shijun Li, Wenqiang Lei, Jiawei Chen, Biao Li, Peng Jiang, Xiangnan He, Jiaxin Mao, and Tat-Seng Chua. Kuairec: A fully-observed dataset and insights for evaluating recommender systems. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management, CIKM '22, 2022. doi: 10.1145/3511808.3557220. URL https://doi.org/10.1145/3511808.3557220.  
Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorization-machine based neural network for ctr prediction. arXiv preprint arXiv:1703.04247, 2017.  
Nan Jiang and Lihong Li. Doubly robust off-policy value evaluation for reinforcement learning. In International Conference on Machine Learning, pp. 652-661. PMLR, 2016.  
Thorsten Joachims, Adith Swaminathan, and Maarten De Rijke. Deep learning with logged bandit feedback. In International Conference on Learning Representations, 2018.  
Sergey Levine and Vladlen Koltun. Guided policy search. In International conference on machine learning, pp. 1-9. PMLR, 2013.  
Yaxu Liu, Jui-Nan Yen, Bowen Yuan, Rundong Shi, Peng Yan, and Chih-Jen Lin. Practical counterfactual policy learning for top-k recommendations. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1141-1151, 2022.  
Romain Lopez, Inderjit S Dhillon, and Michael I Jordan. Learning from extreme bandit feedback. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 8732-8740, 2021.

Jiaqi Ma, Zhe Zhao, Xinyang Yi, Ji Yang, Minmin Chen, Jiaxi Tang, Lichan Hong, and Ed H Chi. Off-policy learning in two-stage recommender systems. In Proceedings of The Web Conference 2020, pp. 463-473, 2020.  
Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. Advances in neural information processing systems, 29, 2016.  
Yuta Saito and Thorsten Joachims. Off-policy evaluation for large action spaces via embeddings. arXiv preprint arXiv:2202.06317, 2022.  
Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. Recommendations as treatments: Debiasing learning and evaluation. In international conference on machine learning, pp. 1670-1679. PMLR, 2016.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pp. 1889-1897. PMLR, 2015.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484-489, 2016.  
Alex Strehl, John Langford, Lihong Li, and Sham M Kakade. Learning from logged implicit exploration data. Advances in neural information processing systems, 23, 2010.  
Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík. Doubly robust off-policy evaluation with shrinkage. In International Conference on Machine Learning, pp. 9167-9176. PMLR, 2020.  
Adith Swaminathan and Thorsten Joachims. Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research, 16(1):1731-1755, 2015a.  
Adith Swaminathan and Thorsten Joachims. Counterfactual risk minimization: Learning from logged bandit feedback. In International Conference on Machine Learning, pp. 814-823. PMLR, 2015b.  
Adith Swaminathan and Thorsten Joachims. The self-normalized estimator for counterfactual learning. advances in neural information processing systems, 28, 2015c.  
Adith Swaminathan, Akshay Krishnamurthy, Alekh Agarwal, Miro Dudik, John Langford, Damien Jose, and Imed Zitouni. Off-policy evaluation for slate recommendation. Advances in Neural Information Processing Systems, 30, 2017.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229-256, 1992.  
Yue Wu, Shuangfei Zhai, Nitish Srivastava, Joshua Susskind, Jian Zhang, Ruslan Salakhutdinov, and Hanlin Goh. Uncertainty weighted actor-critic for offline reinforcement learning. arXiv preprint arXiv:2105.08140, 2021.  
Pan Xu, Zheng Wen, Handong Zhao, and Quanquan Gu. Neural contextual bandits with deep representation and shallow exploration. In International Conference on Learning Representations, 2021.  
Ruohan Zhan, Vitor Hadad, David A Hirshberg, and Susan Athey. Off-policy evaluation via adaptive weighting with data from contextual bandits. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 2125-2135, 2021.  
Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural contextual bandits with ucb-based exploration. In International Conference on Machine Learning, pp. 11492-11502. PMLR, 2020.  
Guorui Zhou, Xiaogiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1059-1068, 2018.
