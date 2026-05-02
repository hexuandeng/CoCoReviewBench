# DEEP LIFETIME CLUSTERING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The goal of lifetime clustering is to develop an inductive model that maps subjects into  $K$  clusters according to their underlying (unobserved) lifetime distribution. We introduce a neural-network based lifetime clustering model that can find cluster assignments by directly maximizing the divergence between the empirical lifetime distributions of the clusters. Accordingly, we define a novel clustering loss function over the lifetime distributions (of entire clusters) based on a tight upper bound of the two-sample Kuiper test p-value. The resultant model is robust to the modeling issues associated with the unobservability of termination signals, and does not assume proportional hazards. Our results in real and synthetic datasets show significantly better lifetime clusters (as evaluated by C-index, Brier Score, Logrank score and adjusted Rand index) as compared to competing approaches.

# 1 INTRODUCTION

Survival analysis is widely used to model the relationship between subject covariates and the time until a particular terminal event of interest (e.g., death, or quitting of social media) that marks the end of all activities (or measurements) of that subject (known as the lifetime of the subject). For instance, a subject's logins to a social network are her activities and the time until she quits the social network permanently is her lifetime.

The lifetime of a subject can be unobserved for two possible reasons: (a) the terminal event was right-censored, i.e., the subject did not have a terminal event within the finite data-collection period, or (b) the terminal events are inherently unobservable. Right-censoring happens for instance when a patient is still alive at the time of data-collection. Unobservability happens for instance, in social networks, when a subject simply stops using the service but does not provide a clear termination signal by deleting her account. In such a scenario, the terminal events remain unobserved for most if not all subjects, even if the subjects quit the service within the data-collection period.

Numerous survival methods have been proposed (Witten and Tibshirani, 2010a; Hothorn et al., 2006; Ishwaran et al., 2008) to predict the lifetime of a subject given her covariates and her activities/measurements for a brief initial period of time, while also accounting for right-censoring. More recent deep learning models for lifetime prediction (Lee et al., 2018; Ren et al., 2018; Chapfuwa et al., 2018) have achieved much success due to their flexibility to model complex relationships, and by avoiding limiting assumptions like parametric lifetime distributions (Ranganath et al., 2016) and proportional hazards (Katzman et al., 2018). In scenarios where terminal events are never observed (unobservability), it is a standard practice to introduce artificial termination signals through a predefined "timeout" for the period of inactivity, i.e., a social network user inactive for  $m$  months has her last observed activity declared a terminal event. Such a specification is typically arbitrary and can adversely affect the analysis depending on the "timeout" value used.

Notwithstanding the fact that lifetimes are hard to predict without termination signals in the training data, we are generally interested in clustering subjects based on their underlying lifetime distribution to improve decision-making. Applications include identifying disease subtypes (Gan et al., 2018), understanding the implications of distinct manufacturing processes on machine parts, and qualitatively analyzing different survival groups in a social network. Although accurately predicting time-to-terminal-event for an individual is important in a variety of applications, lifetime clustering plays a complementary role of providing a more holistic picture.

Lifetime clustering remains a relatively unexplored topic despite being an important tool. Although traditional unsupervised clustering methods such as  $k$ -means and hierarchical clustering are popular

for this task (Bhattacharjee et al., 2001; Sørlie et al., 2001; Bullinger et al., 2004), they may produce clusters that are entirely uncorrelated with lifetimes (Gaynor and Bair, 2013). Semi-supervised clustering (Bair and Tibshirani, 2004) and supervised sparse clustering (Witten and Tibshirani, 2010b) employ a two-stage lifetime clustering process: (i) identify covariates associated with lifetime using Cox scores (Cox, 1992), and (ii) treat these covariates differently while performing  $k$ -means clustering. They assume proportional hazards (i.e., constant hazard ratios over time) and require the presence of termination signals. Furthermore, a decoupled two-stage process such as the above is not guaranteed to obtain clusters with maximally distinct lifetime distributions; rather, we require an end-to-end learning framework that prescribes a loss function specifically over the lifetime distributions of different clusters.

In this work we tackle the important task of inductive lifetime clustering without assuming proportional hazards, while also smoothly handling the unobservability of termination signals.

Contributions. (i) We introduce DeepCLife, an inductive neural-network based lifetime clustering model that finds cluster assignments by maximizing the divergence between empirical (non-parametric) lifetime distributions of different clusters. Whereas the subjects of different clusters have distinct lifetime distributions, within a cluster all subjects share the same lifetime distribution even if they have different lifetimes. Our model is robust to the modeling issues associated with the unobservability of termination signals and does not assume proportional hazards.

(ii) We define a novel clustering loss function over empirical lifetime distributions (of entire clusters) based on the Kuiper two-sample test. We provide a tight upper bound of the Kuiper p-value with easy-to-compute gradients facilitating its use as a loss function, which until now was prohibitively expensive due to the test's infinite sum.  
(iii) Finally, our results on real and synthetic datasets show that the proposed lifetime clustering approach produces significantly better clusters with distinct lifetime distributions (as evaluated by Logrank score, C-index, Brier Score and Rand index) as compared to competing approaches.

# 2 FORMAL FRAMEWORK

In this section, we formally define the statistical framework underlying the clustering approach introduced later in the paper. Notation remark: We use superscript  $(u)$  to refer to variables indexed by a subject  $u$  and use subscript  $k$  to refer to random variables indexed by a random subject of cluster  $k$ . See Table 3 in the supplementary material for a complete list of all the variables and their meanings.

We assume that there are distinct underlying clusters with different event processes describing the activity events of a random subject (e.g., logins to a social network, measurements of a patient) in the respective clusters. To make these notions more formal, we introduce our event process.

Definition 1 (Abstract Event Process). Consider the  $k$ -th cluster. The Random Marked Point Process (RMPP) for the activity events is  $\Phi_k = \{X_k, \{(A_{k,i}, M_{k,i}, Y_{k,i})\}_{i \in \mathbb{N}}, \Theta_k\}$  over discrete times  $t = 0, 1, \ldots$ , where  $X_k$  is the random variable representing the covariates of a random subject in cluster  $k$ ,  $\Theta_k$  is the time to the zeroth activity event (joining),  $M_{k,i}$  represent covariates of event  $i$ ,  $Y_{k,i}$  is the inter-event time between the  $i$ -th and the  $(i-1)$ -st activity events (e.g., logins), and  $A_{k,i} = 1$  indicates an event with a termination signal (death), otherwise  $A_{k,i} = 0$ . All these variables may be arbitrarily dependent. This definition is model-free, i.e., we will not prescribe a model for  $\Phi_k$ .

We observe the RMPP over a time window  $[0, t_{\mathrm{m}}]$ . Using the above formalism, we define the true lifetime of a subject in cluster  $k$  as the sum of all inter-event times until termination signal  $A_{k,i} = 1$  is observed.

Definition 2 (True lifetime). The random variable that defines the true lifetime until the terminal event of a subject in cluster  $k$  is  $T_{k} \coloneqq \max_{i}\left(\sum_{i^{\prime}\leq i}Y_{k,i^{\prime}}\prod_{i^{\prime \prime} < i}(1 - A_{k,i^{\prime \prime}})\right)$ .

The true lifetime  $T_{k}$  is unobserved if: (a) the terminal event  $A_{k,i} = 1$  does not occur within the observation time period  $[0, t_{\mathrm{m}}]$ , or (b) the termination signals are inherently unobservable. Now, the true lifetime distribution of a subject in cluster  $k$  is defined as the probability that the subject has at least one more activity event after time  $t$ , and is given by  $S_{k}(t) \coloneqq P[T_{k} > t] = 1 - F_{k}(t)$ ,  $t \in \mathbb{N} \cup \{0\}$ , where  $F_{k}(t)$  is the underlying cumulative distribution function (CDF) of  $T_{k}$ .

Training data. Our training data  $\mathcal{D}$  consists of subjects from  $K$  underlying (hidden) clusters with distinct lifetime distributions  $S_{k}$  for  $k\in \{1,\ldots ,K\}$ . For a subject  $u\in \mathcal{D}$ , we observe the following quantities  $\{X^{(u)},\{(M_i^{(u)},Y_i^{(u)})\}_{i = 1}^{Q^{(u)}(t_m)},\Theta^{(u)}\}$ , where  $X^{(u)},Y_{i}^{(u)},M_{i}^{(u)}$  and  $\Theta^{(u)}$  are analogously

![](images/61fa91fc5f79f36da836d55296640f75a93e61d8dbf6fbba86fa95dcdb3e54a6.jpg)  
Figure 1: Depicting two clusters (low-risk and high-risk; as shown by the true lifetime distributions) following different RMPPs, each with two subjects.  $T^{(u)}$  is the true lifetime of subject  $u$  (may be unobserved due to right censoring or due to unobservability of termination signals).  $H^{(u)}$  is her observed lifetime, the period between the first and the last observed event.  $\chi^{(u)}$  is the time between the last observed event and  $t_{\mathrm{m}}$ , and  $Q^{(u)}(t_{\mathrm{m}})$  is the number of events of  $u$  after her joining and before  $t_{\mathrm{m}}$ .

defined as in Definition 1, but for a given subject  $u$ .  $Q^{(u)}(t)$  is the number of observed activity events of  $u$  after her joining and before time  $t$ . The training data may or may not contain the termination signals,  $\{A_i^{(u)}\}_{i=1}^{Q^{(u)}(t_m)}$ , where  $A_i^{(u)} = 1$  indicates that event  $i$  was a terminal event for subject  $u$  (e.g., death). Termination signals are typically available in healthcare applications, whereas in the case of social networks, we usually do not observe the termination signal (i.e., account deletion) for any subject.

We define the observed lifetime of a subject  $u$  as  $H^{(u)} \coloneqq \sum_{i=1}^{Q^{(u)}(t_m)} Y_i^{(u)}$ , i.e., the sum of all the inter-event times within the observation period  $[0, t_m]$ . In the absence of termination signals, we additionally define inactive period as the time elapsed since the last observed event of subject  $u$ , given by  $\chi^{(u)} \coloneqq t_m - \Theta^{(u)} - H^{(u)}$ . Figure 1 shows the true lifetime, the observed lifetime, and the inactive period for four users of two different clusters. The events are observed (denoted by solid circles) only till the time of measurement  $t_m$ , whereas the rest of the events are right-censored (denoted by solid diamonds). The termination signal (denoted by dotted diamonds) may be unobserved even if it occurs before  $t_m$  (eg.,  $u_3$ ).

Lastly, we formally define our clustering problem.

Definition 3 (Clustering problem). Consider a dataset  $\mathcal{D}$  with  $N$  subjects. Our goal is to find a mapping  $\kappa : \left(X^{(u)}, \{(M_i^{(u)}, Y_i^{(u)})\}_{i=1}^{Q^{(u)}(\Theta^{(u)} + \tau)}\right) \to \{1, \dots, K\}$ , that inductively maps subject covariates and observed activity events for a brief initial period of time  $\tau$  into clusters, such that the divergence  $\Delta$  between the empirical lifetime distributions of these clusters is maximized, i.e.,

$$
\kappa^ {\star} = \underset {\kappa \in \mathcal {K}} {\arg \max } \underset { \begin{array}{c} i, j \in \{1 \dots K \}, \\ i \neq j \end{array} } {\min } \Delta (\hat {S} _ {i} (\kappa), \hat {S} _ {j} (\kappa)), \tag {1}
$$

where  $\mathcal{K}$  is a set of all mappings,  $\hat{S}_k(\kappa)$  is the empirical lifetime distribution of subjects in  $\mathcal{D}$  mapped to cluster  $k$  through  $\kappa$ , and  $\Delta$  is an empirical distribution divergence measure.

$\kappa^{*}$  optimized in this fashion guarantees that subjects in different clusters have different lifetime distributions. For a new unseen subject in the test data,  $\kappa^{*}$  would be able to inductively assign a cluster within  $\tau$  time of her joining.

# 3 THE DEEPLIFE MODEL

In this section, we propose a practical lifetime clustering approach using neural networks that optimizes Equation (1). Let  $\mathcal{D}$  be the training data as defined in Section 2. Since we want to maximize divergence between empirical lifetime distributions, we assume discrete times (relative to subject joining),  $t\in \{0,1,\dots ,t_{\mathrm{max}}\}$ , where  $t_\mathrm{max} = \max_{u\in \mathcal{D}}H^{(u)}$  is the maximum observed lifetime of any subject  $u\in \mathcal{D}$ . Note that it is sufficient to define the empirical distribution till  $t_\mathrm{max}$ , since we have not observed any subject with lifetime greater than  $t_\mathrm{max}$ .

![](images/44bdb643fb130106761391901642fe9f0410e0164786cf39555f287488993ec7.jpg)  
(a) Model architecture

![](images/ce8997b657b8f39e9bc53e2454795756f25ea52b45c955b7bbbe3d3649d205a9.jpg)  
(b) Overlapping lifetime distributions invalidates proportional hazard assumption

![](images/f20bfe1a38bbcb33ca2216faffbafbf2b3b858dc0b85500454b6f9ef1bd68407.jpg)  
(c) True lifetime distributions (dashed) vs empirical distributions (solid)

![](images/b36f584b848c82a70815a2e3c74f06a150bdf2a492cfafa8528fcde52479bfa4.jpg)  
(d) Samples  $= 100$  
Figure 2: (a) Feedforward neural network  $g(\cdot; W_1)$  outputs the cluster assignments for a batch of users. The cluster assignments along with the probability of termination is used to obtain lifetime distributions of each cluster using Kaplan-Meier estimator. Finally, logarithm of Kuiper p-value upper bound is used as the divergence loss  $\Delta$ . (b) Lifetime distributions can have different shapes and can cross each other, violating proportional hazards assumptions. (c) Divergence metric must account for the uncertainty in the distributions, otherwise divergence maximization leads to imbalanced clusters. (d-e) Upper and lower bounds of the logarithm of Kuiper p-value when varying the Kuiper statistic  $D^{+} + D^{-}$ .

![](images/bd6f1788623c6c2e22ae7284c35ecf9bdc268c8955171854a3ab417aa0dfa184.jpg)  
(e) Samples  $= 1000$

# 3.1 CLUSTER ASSIGNMENTS:  $\alpha_{k}^{(u)}(W_{1})$

We define a neural network  $g$  that takes user covariates and the event data for a subject  $u$  during a brief initial period  $\tau$  after her joining as input, and outputs her cluster assignment probabilities,  $\alpha_{k}^{(u)}(W_{1})$  for all  $k \in \{1, \dots, K\}$ ,

$$
\vec {\alpha} ^ {(u)} \left(W _ {1}\right) = g \left(X ^ {(u)}, \left\{M _ {i} ^ {(u)}, Y _ {i} ^ {(u)} \right\} _ {i = 1} ^ {Q ^ {(u)} \left(\Theta^ {(u)} + \tau\right)}; W _ {1}\right), \tag {2}
$$

where  $W_{1}$  are the weights of the neural network. The final layer of  $g$  is a softmax layer with  $K$  units. Figure 2a depicts  $g$  as a feedforward neural network with  $L - 1$  hidden layers, although our model is not restricted to a feedforward architecture. In our experiments, we compute summary statistics over the observed events  $\{M_i^{(u)}, Y_i^{(u)}\}_i$  in order to make it compatible with the feedforward architecture.

# 3.2 PROBABILITY OF TERMINATION:  $\beta^{(u)}(W_2)$

During the training of our model, we require termination signals or a probabilistic estimation of the termination signals in order to write the likelihood of the model. Given the model parameter  $W_{2}$ , we define  $\beta^{(u)}(W_2)$  as the probability that the last observed event of  $u$  was terminal, i.e.,  $u$  will have no future activity events after the last observed event.

If the termination signals  $A_{i}^{(u)}$  are observed in the training data  $\mathcal{D}$  (e.g., healthcare), clearly  $\beta^{(u)}(W_2) \coloneqq A_{Q^{(u)}(t_m)}^{(u)}$  (i.e.,  $W_2$  is ignored).

When such termination signals are unobservable (e.g., social network), existing survival methods commonly use a timeout window of predefined size  $W_{\mathrm{fixed}}$  over the inactive period  $\chi^{(u)}$  to specify the probability of termination as  $\beta^{(u)}(W_{\mathrm{fixed}}) \coloneqq 1[\chi^{(u)} > W_{\mathrm{fixed}}]$ . However, such specification is arbitrary and precludes any learning of the window size parameter  $W_{\mathrm{fixed}}$ . Instead, we model the latent termination probabilities  $\beta^{(u)}(W_2)$  using a smooth non-decreasing function of  $\chi^{(u)}$ , i.e., higher the period of inactivity, higher the probability that the last observed event was terminal.

If the termination signals  $A_{i}^{(u)}$  are unobservable in the training data  $\mathcal{D}$  (e.g., social network), we use  $\beta^{(u)}(W_2) \coloneqq 1 - e^{-\xi^{(u)}\cdot \chi^{(u)}}$  with a shared rate parameter  $\xi^{(u)} = W_2 > 0$ . Practitioners can use a more flexible model by using a neural network parameterized by  $W_{2}$  to describe the rate parameter  $\xi^{(u)}$ .

# 3.3 EMPIRICAL LIFETIME DISTRIBUTION OF CLUSTER  $k: \hat{S}_k(W_1, W_2; \mathcal{D})$

Given the training data  $\mathcal{D}$  and model parameters  $W_{1}$  and  $W_{2}$ , we can obtain the soft cluster assignments  $\alpha_{k}^{(u)}(W_{1})$  and the probability of termination  $\beta^{(u)}(W_2)$  for all subjects  $u\in \mathcal{D}$  and clusters  $k\in \{1\dots K\}$  as shown in Section 3.1 and Section 3.2. In this subsection, we obtain the empirical lifetime distribution of all the clusters  $k = 1\dots K$ , using the Kaplan-Meier estimates (Kaplan and Meier, 1958). We do not assume a parametric form for the lifetime distribution, and rather use empirical distributions in our optimization to allow lifetime curves of any shape (Figure 2b). Kaplan-Meier estimates are a maximum likelihood estimate of the lifetime distribution of a set of subjects assuming (a) hard memberships (each subject entirely belongs to the set) and (b) the presence of termination signals. We modify the estimates to account for partial memberships and probability of termination instead.

Proposition 1. Given the training data  $\mathcal{D}$ , a cluster  $k$ , the cluster assignment probabilities  $\{\alpha_k^{(u)}(W_1)\}_{u\in \mathcal{D}}$ , and the probabilities of termination  $\{\beta^{(u)}(W_2)\}_{u\in \mathcal{D}}$ , the maximum likelihood estimate of the empirical lifetime distribution of cluster  $k$  is given by,

$$
\hat {S} _ {k} \left(W _ {1}, W _ {2}; \mathcal {D}\right) [ t ] = \prod_ {j = 0} ^ {t} \frac {s _ {k} \left(W _ {1} ; \mathcal {D}\right) [ j ] - d _ {k} \left(W _ {1} , W _ {2} ; \mathcal {D}\right) [ j ]}{s _ {k} \left(W _ {1} ; \mathcal {D}\right) [ j ]}, \tag {3}
$$

for all  $t \in \{0,1,\ldots,t_{\text{max}}\}$ , where,  $s_k(W_1;\mathcal{D})[j] = \sum_{u \in \mathcal{D}} \mathbf{1}[H^{(u)} \geq j] \cdot \alpha_k^{(u)}(W_1)$ , is the expected number of subjects in cluster  $k$  who are at risk (of termination) at time  $j$ , and,  $d_k(W_1,W_2;\mathcal{D})[j] = \sum_{u \in \mathcal{D}} \mathbf{1}[H^{(u)} = j] \cdot \beta^{(u)}(W_2) \cdot \alpha_k^{(u)}(W_1)$ , is the expected number of subjects that are predicted to have had a terminal event at time  $j$ .

The proof is presented in the supplementary material.

# 3.4 EMPIRICAL DISTRIBUTION DIVERGENCE LOSS:  $\Delta (\hat{S}_a,\hat{S}_b)$

We rewrite the objective function of lifetime clustering from Definition 3 with respect to the model parameters  $W_{1}$  and  $W_{2}$  as follows,

$$
W _ {1} ^ {*}, W _ {2} ^ {*} = \underset {W _ {1}, W _ {2}} {\arg \max } \underset {i, j \in \{1 \dots K \},} {\min } \Delta \left(\hat {S} _ {i}, \hat {S} _ {j}\right)  , \tag {4}
$$

where  $\hat{S}_i$  is the empirical distribution, a shorthand for the vector  $\hat{S}_i(W_1,W_2;\mathcal{D})$ , and  $\Delta$  is a divergence measure between two empirical distributions.

We note the following essential requirements for the divergence measure  $\Delta$ : (a)  $\Delta$  defined over empirical distributions must take into account sample sizes, (b)  $\Delta$  should have easy-to-compute gradients since it is used as an objective function to train neural networks, and (c)  $\Delta$  should not assume proportional hazards, and should allow for crossing lifetime curves (see Figure 2b).

Divergence measures such as Kullback-Leibler (Kullback and Leibler, 1951) and MMD (Gretton et al., 2012) fulfill  $(\mathbf{b},\mathbf{c})$  but not (a), and will result in sample anomalies as depicted in Figure 2c (also seen in our experiments: Figure 3d). Logrank test (Mantel, 1966), commonly used for comparing lifetime distributions, fulfills (a, b), but has low statistical power when the proportional hazards assumption is not met (Peto and Peto, 1972; Bland and Altman, 2004) (e.g., Figure 2b). Finally, p-value from two-sample tests such as the Kolmogorov-Smirnov (K-S) test (Massey Jr, 1951) fulfill (a, c), but not (b) as they require the computation of an infinite sum, resulting in an impractical objective function unless heuristic approximations are made.

We propose to use the Kuiper test (Kuiper, 1960), a two-sample test closely related to the K-S test with increased statistical power in distinguishing distribution tails (Tygert, 2010). Specifically, we define  $\Delta (\hat{S}_a,\hat{S}_b):= -\log (\mathrm{KD}(\hat{S}_a,\hat{S}_b))$ , where KD is the p-value from the Kuiper test between  $\hat{S}_a$  and  $\hat{S}_b$ . Our choice of the Kuiper test is because it is amenable to upper and lower bounds as shown next, thus avoiding the prohibitive heuristic approximations of infinite sums.

Proposition 2. (Bounds for Kuiper  $p$ -value.) Given two empirical lifetime distributions  $\hat{S}_a$  and  $\hat{S}_b$  with discrete support and sample sizes  $n_a$  and  $n_b$  respectively, define the maximum positive and

negative separations between them,

$$
\hat {D} _ {a, b} ^ {+} = \sup  _ {t \in \{0, \dots t _ {m a x} \}} \left(\hat {S} _ {a} [ t ] - \hat {S} _ {b} [ t ]\right), \quad \hat {D} _ {a, b} ^ {-} = \sup  _ {t \in \{0, \dots t _ {m a x} \}} \left(\hat {S} _ {b} [ t ] - \hat {S} _ {a} [ t ]\right).
$$

The Kuiper test  $p$ -value Kuiper (1960) gives the probability that  $\Lambda$ , the empirical deviation for  $n_a$  and  $n_b$  observations under the null hypothesis  $S_a = S_b$ , exceeds the observed value  $V = \hat{D}_{a,b}^{+} + \hat{D}_{a,b}^{-}$ :

$$
K D \left(\hat {S} _ {a}, \hat {S} _ {b}\right) \equiv P [ \Lambda > V ] = 2 \sum_ {j = 1} ^ {\infty} \left(4 j ^ {2} \lambda_ {a, b} ^ {2} - 1\right) e ^ {- 2 j ^ {2} \lambda_ {a, b} ^ {2}}, \tag {5}
$$

$\lambda_{a,b} = (\sqrt{M_{a,b}} + 0.155 + \frac{0.24}{\sqrt{M_{a,b}}})V$  and  $M_{a,b} = \frac{n_a n_b}{n_a + n_b}$  is the effective sample size. Then, the upper bound  $^2$  for the Kuiper  $p$ -value is,

$$
\begin{array}{l} K D (\hat {S} _ {a}, \hat {S} _ {b}) \leq \min  \left(1, 2 \cdot \mathbf {1} [ r _ {a, b} ^ {(l o)} \geq 1 ] \cdot \left(w (r _ {a, b} ^ {(l o)}, \lambda_ {a, b}) - w (1, \lambda_ {a, b}) + v (r _ {a, b} ^ {(l o)}, \lambda_ {a, b})\right) \right. \\ \left. + v \left(r _ {a, b} ^ {(u p)}, \lambda_ {a, b}\right) - w \left(r _ {a, b} ^ {(u p)}, \lambda_ {a, b}\right)\right), \tag {6} \\ \end{array}
$$

where  $v(r,\lambda) = (4r^2\lambda^2 - 1)e^{-2r^2\lambda^2}$ ,  $w(r,\lambda) = -re^{-2r^2\lambda^2}$ ,  $r_{a,b}^{(lo)} = \left\lfloor \frac{1}{\sqrt{2}\lambda_{a,b}} \right\rfloor$ , and,  $r_{a,b}^{(up)} = \left\lceil \frac{1}{\sqrt{2}\lambda_{a,b}} \right\rceil$ . The proof is presented in the supplementary material.

Figures 2d and 2e show empirical results of the upper and lower bounds as a function of  $V = D^{+} + D^{-}$  against the expensive heuristic computation where the infinite sum in Equation (5) is approximated with  $10^{4}$  terms (Press et al., 1996). We optimize Equation (4) with  $\Delta (\hat{S}_a, \hat{S}_b) \coloneqq -\log (\mathrm{KD}^{(\mathrm{up})}(\hat{S}_a, \hat{S}_b))$  where  $\mathrm{KD}^{(\mathrm{up})}$  denotes the Kuiper p-value upper bound in Equation (6).  $\Delta$  defined this way satisfies all three requirements we described in the beginning of Section 3.4: takes sample sizes into account, has closed form expression with easy-to-compute gradients, and does not assume proportional hazards.

Implementation. We implement our clustering procedure as a feedforward neural network in Pytorch (Paszke et al., 2017) and use ADAM (Kingma and Ba, 2014) to optimize Equation (4). Each iteration of the optimization takes as input a batch of subjects, generates a single value for the set loss, calculates the gradients, and updates the parameters  $W_{1}$  and  $W_{2}$ . We show in the supplementary material that the time and space complexity per iteration of the proposed approach are  $O(KB + K^2 t_{\mathrm{max}})$  and  $O(Kt_{\mathrm{max}})$  respectively, where  $K$  is the number of clusters and  $B$  is the batch size. For large values of  $K$ , we achieve tractability by sampling  $K$  random pairs of clusters every iteration instead of all  $\binom{K}{2}$  possible pairs.

# 4 RELATED WORK

Majority of the work in survival analysis has dealt with the task of predicting the time to an observable terminal event (e.g., death), especially when the number of features is much larger than the number of subjects (Witten and Tibshirani, 2010a; pre; Hothorn et al., 2006; Shivaswamy et al., 2007). Recently, many deep learning approaches (Luck et al., 2017; Katzman et al., 2018; Lee et al., 2018; Ren et al., 2018) have been proposed for predicting the lifetime distribution of a subject given her covariates, while effectively handling censored data that typically arise in survival tasks. DeepHit (Lee et al., 2018) introduced a novel architecture and a ranking loss function in addition to the log-likelihood loss for lifetime prediction in the presence of multiple competing risks. Using a log-likelihood loss similar to DeepHit, Ren et al. (2018) propose a recurrent architecture to predict the survival distribution that captures sequential dependencies between neighbouring time points. Finally, Chapfuwa et al. (2018) introduce an adversarial learning framework to model the lifetime given the subject covariates. In contrast to these works on predicting lifetimes, our task is to cluster the subjects based on their underlying lifetime distributions.

There are relatively fewer works that perform lifetime clustering. Many unsupervised approaches have been proposed to identify cancer subtypes in gene expression data but do not consider the lifetime (Eisen et al., 1998; Alizadeh et al., 2000; Bhattacharjee et al., 2001; Sørlie et al., 2001; Bullinger et al., 2004), and may produce clusters that are entirely independent of the lifetimes. Semi-supervised clustering (Bair and Tibshirani, 2004) and supervised sparse clustering (Witten and Tibshirani, 2010b) use Cox scores (Cox, 1992) to identify features associated with the lifetime and treat these features

Table 1: (Synthetic) C-index (\%) and Adjusted Rand index (\%) for clusters with standard errors in parentheses for different methods  ${}^{3}$  .  

<table><tr><td rowspan="2">Method</td><td colspan="2">D{C1,C2}</td><td colspan="2">D{C1,C3}</td><td colspan="2">D{C1,C2,C3}</td></tr><tr><td>C-index ↑ (%)</td><td>Adj. Rand Index ↑ (%)</td><td>C-index ↑ (%)</td><td>Adj. Rand Index ↑ (%)</td><td>C-index ↑ (%)</td><td>Adj. Rand Index ↑ (%)</td></tr><tr><td>SSC-Bair</td><td>62.75 (0.35)</td><td>74.66 (0.48)</td><td>62.99 (0.26)</td><td>56.86 (0.63)</td><td>63.77 (0.24)</td><td>47.67 (0.24)</td></tr><tr><td>SSC-Gaynor</td><td>56.34 (0.50)</td><td>19.88 (0.51)</td><td>57.21 (0.43)</td><td>16.60 (0.56)</td><td>56.75 (0.32)</td><td>5.84 (0.11)</td></tr><tr><td>DeepHit+GMM</td><td>63.31 (0.32)</td><td>85.05 (1.01)</td><td>65.23 (0.21)</td><td>78.47 (0.94)</td><td>59.59 (1.98)</td><td>38.77 (7.16)</td></tr><tr><td>DeepCLife-MMD</td><td>64.35 (0.33)</td><td>98.47 (0.28)</td><td>67.25 (0.26)</td><td>99.68 (0.16)</td><td>62.17 (0.71)</td><td>36.75 (1.10)</td></tr><tr><td>DeepCLife-KuiperUB</td><td>64.37 (0.32)</td><td>99.02 (0.14)</td><td>67.24 (0.26)</td><td>99.94 (0.06)</td><td>68.96 (0.38)</td><td>73.61 (0.62)</td></tr></table>

differently while using k-means to perform the final clustering. Unlike these lifetime clustering methods, DeepCLife does not assume proportional hazards, and can smoothly handle the absence of termination signals. Our supplementary material has a more in-depth discussion of related work.

# 5 RESULTS

Baselines. We perform experiments on one synthetic dataset and two real-world datasets - Friend-ster social network and MIMIC III healthcare dataset (Johnson et al., 2016).

We compare the following lifetime clustering approaches: (a) SSC-Bair, a semi-supervised clustering method (Bair and Tibshirani, 2004) that performs k-means clustering on selected covariates that have high Cox scores (Cox, 1992); (b) SSC-Gaynor or supervised sparse clustering (Gaynor and Bair, 2013), a modification of sparse clustering (Witten and Tibshirani, 2010b) that weights the covariates based on their Cox scores; (c) DeepHit+GMM, a Gaussian mixture model applied over last layer embeddings learnt by DeepHit (Lee et al., 2018); (d) DeepCLife-MMD, the DeepCLife model with Maximum Mean Discrepancy (Gretton et al., 2012) as the divergence measure; (e) DeepCLife-KuiperUB, the DeepCLife model with the proposed divergence measure based on the Kuiper p-value upper bound (Equation (6)).

Termination signals for evaluation and baselines. (Timeout.) The competing methods and most evaluation metrics for survival applications require clear termination signals. In scenarios where we do not observe termination signals (e.g., Friendster experiments), we specify termination signals artificially when training the baselines using a pre-defined "timeout", i.e.,  $\beta^{(u)}(W_{\mathrm{fixed}}) = 1[\chi^{(u)} > W_{\mathrm{fixed}}]$ . During evaluation, we use the same  $W_{\mathrm{fixed}}$  to specify the termination signals and compute the metrics. This helps the competing methods since they are trained and evaluated using termination signals with the same  $W_{\mathrm{fixed}}$ , whereas our approach is not trained with these termination signals.

Metrics. We use the following metrics for evaluating the clusters obtained from the methods.

- Logrank Score  $\uparrow$ . Logrank test (Mantel, 1966) statistic is a non-parametric test that outputs high values when it is unlikely for the  $K$  groups to have the same lifetime distribution.  
- Adjusted Rand index  $\uparrow$ . The Adjusted Rand index (Hubert and Arabie, 1985) is a measurement of cluster agreement, compared to the ground truth clustering (if available). ARI is 0.0 for random cluster assignments and 1.0 for perfect assignments.  
- C-Index  $\uparrow$ . Concordance index (Harrell et al., 1982) is a commonly used metric that calculates the fraction of pairs of subjects for which the model predicts the correct order of survival while also incorporating censoring. C-index is 1.0 for perfect predictions and 0.5 for random predictions.  
- Integrated Brier Score  $\downarrow$ . Integrated Brier score (Brier, 1950; Graf et al., 1999) computes mean squared difference between the survival probabilities and the actual outcome over  $[0, t_{\max}]$ . It ranges from 0.25 for random predictions to 0 for perfect predictions.

Although cluster evaluation metrics like Logrank score are more suitable for the lifetime clustering task, we also use predictive measures like C-index following prior work (Gaynor and Bair, 2013) to further validate the clusters.

Evaluation. We evaluate the models using 5-fold cross validation. We use the  $i$ th fold for testing and sample  $N^{(\mathrm{tr})}$  subjects from the remaining 4 folds for training. We use  $20\%$  of the  $N^{(\mathrm{tr})}$  subjects as validation for early stopping and hyperparameter tuning for the different approaches.

![](images/0da28f5caae40e5d3b261508fd5fb4a30cd09e26092a35147b50065a959ac856.jpg)  
(a) SSC-Bair

![](images/9325fbb01b27a8274d1354907e6f8f951c16939b24b29d8c8ed3723e8c8c8dd8.jpg)  
(b) SSC-Gaynor

![](images/da0a0043ad228a3d89a79698e4cd257e90331ffa1486c0ecbb440119521764a7.jpg)  
(c) DeepHit+GMM

![](images/49289d24df490a79a8004f929f50f20b17f2659b7cc4b08bba095fab4e9d3f43.jpg)  
(d) DeepCLife-MMD

![](images/8d0200be38b33c0ab44af551674040355e57fc85a2c47dee47ae973ff57ab745.jpg)  
(e) DeepCLife-KuiperUB  
Figure 3: (Friendster) Empirical lifetime distributions of clusters obtained from different methods for  $K = 3$  (legend shows cluster sizes  $n_1, n_2, n_3$ ). Baseline methods (a-c) employ a two-stage clustering process and do not guarantee clusters with maximally different lifetime distributions. (d) DeepCLife-MMD suffers from sample anomalies ( $n_2 = 0$ ). (e) DeepCLife-KuiperUB obtains clusters with significantly different lifetime distributions (best Logrank scores).

Table 2: (Friendster) C-index (\%), Integrated Brier Score (\%) and Logrank score with standard errors in parentheses for different methods $^3$  and  $K = 2,4$  clusters with number of training examples  $N^{(\mathrm{tr})} = 10^5$ .  

<table><tr><td rowspan="2">Method</td><td colspan="3">K=2</td><td colspan="3">K=4</td></tr><tr><td>C-index ↑ (%)</td><td>I.B.S ↓ (%)</td><td>Logrank Score ↑</td><td>C-index ↑ (%)</td><td>I.B.S ↓ (%)</td><td>Logrank Score ↑</td></tr><tr><td>SSC-Bair</td><td>64.42 (0.15)</td><td>22.18 (0.02)</td><td>5479.27 (38.06)</td><td>67.18 (0.13)</td><td>21.55 (0.03)</td><td>13013.86 (182.07)</td></tr><tr><td>SSC-Gaynor</td><td>64.42 (0.18)</td><td>22.17 (0.02)</td><td>5557.41 (38.43)</td><td>69.99 (0.28)</td><td>21.62 (0.01)</td><td>15204.81 (41.29)</td></tr><tr><td>DeepHit+GMM</td><td>64.33 (1.80)</td><td>22.04 (0.23)</td><td>9207.46 (5031.48)</td><td>76.55 (0.12)</td><td>20.64 (0.02)</td><td>40703.01 (477.25)</td></tr><tr><td>DeepCLife-MMD</td><td>67.49 (0.11)</td><td>22.07 (0.02)</td><td>27642.60 (1301.85)</td><td>70.93 (1.80)</td><td>22.40 (0.07)</td><td>33030.93 (4277.24)</td></tr><tr><td>DeepCLife-KuiperUB</td><td>75.58 (0.15)</td><td>20.13 (0.02)</td><td>47837.25 (297.63)</td><td>77.04 (0.88)</td><td>18.99 (0.20)</td><td>59236.36 (2126.55)</td></tr></table>

# 5.1 EXPERIMENTS

Synthetic experiment. We test our method with a synthetic dataset for which we have the true clusters as ground truth. We generate 3 clusters  $C_1, C_2$  and  $C_3$  with different lifetime distributions such that  $C_2$  and  $C_3$  have proportional hazards, but lifetime distribution of  $C_1$  crosses the other two curves (shown in Figure 2b). We choose an arbitrary time of measurement,  $t_m = 150$ , to imitate right censoring. We sample  $10^4$  subjects for each cluster, their features drawn from mixture of Gaussians. The lifetime  $T^{(u)}$  of a subject  $u$  is randomly sampled from the lifetime distribution of her ground truth cluster. Table 1 shows performance of the methods on three synthetic datasets:  $\mathcal{D}_{\{C_1,C_2\}}$ ,  $\mathcal{D}_{\{C_1,C_3\}}$ ,  $\mathcal{D}_{\{C_1,C_2,C_3\}}$ . DeepCLife-KuiperUB and DeepCLife-MMD were able to recover perfect ground truth cluster assignments on  $\mathcal{D}_{\{C_1,C_2\}}$  and  $\mathcal{D}_{\{C_1,C_3\}}$ , whereas the baseline methods performed invariably worse. On the harder dataset  $\mathcal{D}_{\{C_1,C_2,C_3\}}$ , DeepCLife-KuiperUB recovered the ground truth clusters almost twice as better than any other method.

Friendster experiment. Friendster dataset consists of 15 million users in the Friendster online social network along with the comments sent and received by the users. We consider a subset with 1.1 million users who had participated in at least one comment. We use each user's profile information (like age, gender, location, etc.) as covariates, and define activity events as the comments sent or received by the user. The task is to cluster new test users using their covariates and their event information for  $\tau = 5$  months from joining. Note that we do not observe termination signals (i.e., account deletion) for any subject in the data. We use an arbitrary window of  $W_{\mathrm{fixed}} = 10$  months over the inactivity period to obtain termination signals ( $\approx 65\%$  assumed to have quit) for

![](images/3f0a104b0d12df4743faec26c55f6dced5f9466eac80e5a275fe598e0b24f934.jpg)  
Figure 4: (MIMIC III) (a) Empirical lifetime distributions obtained by DeepCLife-KuiperUB for  $K = 2$ . (b) C-index (\%), Brier Score (\%) and Logrank score with standard errors for different methods on healthcare data for  $K = 2$ .

$K = 2$  

<table><tr><td>Method</td><td>C-index ↑ (%)</td><td>Brier Score ↓ (%)</td><td>Logrank ↑ (%)</td></tr><tr><td>SSC-Bair</td><td>52.78 (0.76)</td><td>15.89 (0.20)</td><td>1177.52 (419.61)</td></tr><tr><td>SSC-Gaynor</td><td>52.32 (0.96)</td><td>15.87 (0.15)</td><td>12070.97 (5933.58)</td></tr><tr><td>DeepHit+GMM</td><td>64.93 (0.54)</td><td>15.70 (0.15)</td><td>17193.38 (1115.70)</td></tr><tr><td>DeepCLife-MMD</td><td>58.85 (0.60)</td><td>15.03 (0.35)</td><td>15783.84 (1101.45)</td></tr><tr><td>DeepCLife-KuiperUB</td><td>66.30 (1.09)</td><td>15.50 (0.23)</td><td>20525.26 (1145.00)</td></tr></table>

(b)

# (a) DeepCLife-KuiperUB

the competing methods and for computing the evaluation metrics; DeepCLife does not require such arbitrary specification but learns a smooth timeout window during training.

Table 2 shows results for  $K = 2,4$  clusters. We note that the proposed method obtains higher C-index values and Brier scores compared to the baselines even without termination signals. DeepCLife-KuiperUB achieves a significant improvement in Logrank scores compared to the baselines because its loss specifically maximizes for differences in empirical distributions, while also taking sample sizes into account. DeepCLife-MMD on the other hand does not account for sample sizes, and hence performs worse. The empirical lifetime distributions of the clusters obtained from different methods for  $K = 3$  are shown in Figure 3. The clusters obtained from the baselines (a-c) are not substantially different from each other. Although DeepCLife-MMD obtains clusters with distinct lifetime distributions, it suffers from sample anomalies i.e., outputs clusters with very few or no subjects (e.g.,  $\hat{S}_2$  in Figure 3d). DeepCLife-KuiperUB outputs clusters that have significantly different lifetime distributions with the best Logrank scores. Corresponding plots of empirical lifetime distributions for  $K = 2,4,5$  are presented in the supplementary material. For  $K = 4$ , we observe that DeepCLife-KuiperUB finds crossing yet distinct lifetime distributions.

Qualitative Analysis: In the clusters found by DeepCLife-KuiperUB in Friendster, we see that a user in a low-risk cluster has on average 7.76 friends, sends 5.06 comments with an average response time of 20 days. On the other hand, a user in a high-risk cluster has just 1.56 friends on average and sends far fewer comments, around 1.07, but with a fast response time of 1.32 days. Interestingly, users that stayed longer in the system had lower activity rate in the beginning.

MIMIC III experiment. MIMIC III dataset<sup>5</sup> (Johnson et al., 2016) consists of around 46500 patients admitted to the Intensive Care Unit (ICU). The task is to cluster the patients based on their mortality within 30 days of admission to the ICU (Purushotham et al., 2017). Unlike Friendster experiments, we can observe terminal events. Lifetime of a patient is right-censored if she was discharged within the 30-day period ( $\approx$ 84% right-censored). We use only the initial  $\tau = 24$  hours of patient measurements (e.g. heart rate, respiratory rate, etc.) to perform the clustering. Results for  $K = 2$  in Table 4b show that DeepCLife-KuiperUB achieves significantly better C-index and Logrank scores, followed by DeepHit+GMM. The corresponding empirical lifetime distributions of the clusters are shown in Figure 4a.

# 6 CONCLUSION

In this work we introduced Kuiper-based nonparametric loss function to maximize the divergence between empirical distributions, and a corresponding upper bound with easy-to-compute gradients. The loss function is then used to train a feedforward neural network to inductively map subjects into  $K$  lifetime-based clusters without requiring termination signals. We show that this approach produces clusters with better C-index values and Logrank scores than competing methods.

# REFERENCES

Charu C Aggarwal, Stephen C Gates, and Philip S Yu. On using partial supervision for text categorization. TKDE, 2004.

Ahmed M Alaa and Mihaela van der Schaar. Deep multi-task gaussian processes for survival analysis with competing risks. In NIPS, 2017.  
Ash A Alizadeh, Michael B Eisen, R Eric Davis, Chi Ma, Izidore S Lossos, Andreas Rosenwald, Jennifer C Boldrick, Hajeer Sabet, Truc Tran, Xin Yu, et al. Distinct types of diffuse large b-cell lymphoma identified by gene expression profiling. Nature, 403(6769):503-511, 2000.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein Generative Adversarial Networks. In ICML, 2017.  
Eric Bair and Robert Tibshirani. Semi-supervised methods to predict patient survival from gene expression data. PLoS Biol, 2(4):e108, 2004.  
Sugato Basu, Arindam Banerjee, and Raymond Mooney. Semi-supervised clustering by seeding. In ICML, 2002.  
Sugato Basu, Mikhail Bilenko, and Raymond J Mooney. A probabilistic framework for semi-supervised clustering. In SIGKDD, 2004.  
Arindam Bhattacharjee, William G Richards, Jane Staunton, Cheng Li, Stefano Monti, Priya Vasa, Christine Ladd, Javad Beheshti, Raphael Bueno, Michael Gillette, et al. Classification of human lung carcinomas by mRNA expression profiling reveals distinct adenocarcinoma subclasses. Proceedings of the National Academy of Sciences, 98(24):13790-13795, 2001.  
J Martin Bland and Douglas G Altman. The logrank test. *Bmj*, 328(7447):1073, 2004.  
Glenn W Brier. Verification of forecasts expressed in terms of probability. *Monthey Weather Review*, 78(1):1-3, 1950.  
Lars Bullinger, Konstanze Döhner, Eric Bair, Stefan Fröhling, Richard F Schlenk, Robert Tibshirani, Hartmut Döhner, and Jonathan R Pollack. Use of gene-expression profiling to identify prognostic subclasses in adult acute myeloid leukemia. New England Journal of Medicine, 350(16):1605-1616, 2004.  
Paidamoyo Chapfuwa, Chenyang Tao, Chunyuan Li, Courtney Page, Benjamin Goldstein, Lawrence Carin, and Ricardo Henao. Adversarial time-to-event modeling. arXiv preprint arXiv:1804.03184, 2018.  
SK Chuang, T Cai, CW Douglass, LJ Wei, and TB Dodson. Frailty approach for the analysis of clustered failure time observations in dental research. Journal of dental research, 84(1):54-58, 2005.  
David R Cox. Regression models and life-tables. In *Breakthroughs in statistics*, pages 527-541. Springer, 1992.  
Richard M Dudley. Real analysis and probability, volume 74. Cambridge University Press, 2002. ISBN 0521007542.  
Michael B Eisen, Paul T Spellman, Patrick O Brown, and David Botstein. Cluster analysis and display of genome-wide expression patterns. PNAS, 1998.  
K. Fukumizu, A. Gretton, X. Sun, and B. Schölkopf. Kernel Measures of Conditional Dependence. In NIPS, 2008.  
Yanglan Gan, Ning Li, Guobing Zou, Yongchang Xin, and Jihong Guan. Identification of cancer subtypes from single-cell rna-seq data using a consensus clustering method. BMC medical genomics, 11(6):117, 2018.  
Sheila Gaynor and Eric Bair. Identification of biologically relevant subtypes via preweighted sparse clustering. *Biostatistics*, pages 1-33, 2013.  
Erika Graf, Claudia Schmoor, Willi Sauerbrei, and Martin Schumacher. Assessment and comparison of prognostic classification schemes for survival data. Statistics in medicine, 18(17-18):2529-2545, 1999.

Arthur Gretton, Kenji Fukumizu, Zaid Harchaoui, and Bharath K. Sriperumbudur. A Fast, Consistent Kernel Two-Sample Test. In Advances in Neural Information Processing Systems, 2009.  
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(Mar):723-773, 2012.  
Frank E Harrell, Robert M Califf, David B Pryor, Kerry L Lee, and Robert A Rosati. Evaluating the yield of medical tests. Jama, 247(18):2543-2546, 1982.  
Torsten Hothorn, Peter Buhlmann, Sandrine Dudoit, Annette Molinaro, and Mark J Van Der Laan. Survival ensembles. Biostatistics, 7(3):355-373, 2006.  
Philip Hougaard. Frailty models for survival data. *Lifetime data analysis*, 1(3):255-273, 1995.  
Xuelin Huang and Robert A Wolfe. A frailty model for informative censoring. Biometrics, 58(3): 510-520, 2002.  
Lawrence Hubert and Phipps Arabie. Comparing partitions. Journal of classification, 2(1):193-218, 1985.  
Hemant Ishwaran, Udaya B Kogalur, Eugene H Blackstone, Michael S Lauer, et al. Random survival forests. The annals of applied statistics, 2(3):841-860, 2008.  
Hemant Ishwaran, Udaya B Kogalur, Eiran Z Gorodeski, Andy J Minn, and Michael S Lauer. High-dimensional variable selection for survival data. Journal of the American Statistical Association, 105(489):205-217, 2010.  
Alistair EW Johnson, Tom J Pollard, Lu Shen, H Lehman Li-wei, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. Scientific data, 3:160035, 2016.  
Edward L Kaplan and Paul Meier. Nonparametric estimation from incomplete observations. Journal of the American statistical association, 53(282):457-481, 1958.  
Jared L Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a cox proportional hazards deep neural network. BMC medical research methodology, 18(1):24, 2018.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Nicolaas H Kuiper. Tests concerning random points on a circle. In Indagationes Mathematicae (Proceedings), 1960.  
Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics, 22(1):79-86, 1951.  
Vincenzo Lagani and Ioannis Tsamardinos. Structure-based variable selection for survival data. Bioinformatics, 26(15):1887-1894, 2010.  
Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning approach to survival analysis with competing risks. AAAI, 2018.  
Huimin Li, Dong Han, Yawen Hou, Huilin Chen, and Zheng Chen. Statistical inference methods for two crossing survival curves: a comparison of methods. PLoS One, 10(1):e0116774, 2015a.  
Yujia Li, Kevin Swersky, and Rich Zemel. Generative moment matching networks. In ICML, 2015b.  
Margaux Luck, Tristan Sylvain, Héloise Cardinal, Andrea Lodi, and Yoshua Bengio. Deep learning for patient-specific kidney graft survival analysis. arXiv preprint arXiv:1705.10245, 2017.  
Nathan Mantel. Evaluation of survival data and two new rank order statistics arising in its consideration. Cancer Chemother Rep, 50:163-170, 1966.

Frank J Massey Jr. The kolmogorov-smirnov test for goodness of fit. Journal of the American statistical Association, 46(253):68-78, 1951.  
Kamal Nigam, Andrew McCallum, Sebastian Thrun, Tom Mitchell, et al. Learning to classify text from labeled and unlabeled documents. AAAI/IAAI, 792, 1998.  
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Richard Peto and Julian Peto. Asymptotically efficient rank invariant test procedures. Journal of the Royal Statistical Society. Series A (General), pages 185-207, 1972.  
William H Press, Saul A Teukolsky, William T Vetterling, and Brian P Flannery. Numerical recipes in  $C$ , volume 2. Cambridge university press Cambridge, 1996.  
Sanjay Purushotham, Chuizheng Meng, Zhengping Che, and Yan Liu. Benchmark of deep learning models on large healthcare mimic datasets. arXiv preprint arXiv:1710.08531, 2017.  
William M Rand. Objective criteria for the evaluation of clustering methods. Journal of the American Statistical association, 66(336):846-850, 1971.  
Rajesh Ranganath, Adler Perotte, Noémie Elhadad, and David Blei. Deep survival analysis. arXiv preprint arXiv:1608.02158, 2016.  
Kan Ren, Jiarui Qin, Lei Zheng, Zhengyu Yang, Weinan Zhang, Lin Qiu, and Yong Yu. Deep recurrent survival analysis. arXiv preprint arXiv:1809.02403, 2018.  
Bruno Ribeiro and Christos Faloutsos. Modeling website popularity competition in the attention-activity marketplace. In WSDM, 2015.  
Pannagadatta K Shivaswamy, Wei Chu, and Martin Jansche. A support vector approach to censored targets. In ICDM, 2007.  
Therese Sørlie, Charles M Perou, Robert Tibshirani, Turid Aas, Stephanie Geisler, Hilde Johnsen, Trevor Hastie, Michael B Eisen, Matt Van De Rijn, Stefanie S Jeffrey, et al. Gene expression patterns of breast carcinomas distinguish tumor subclasses with clinical implications. Proceedings of the National Academy of Sciences, 98(19):10869-10874, 2001.  
Yizhou Sun, Jiawei Han, Charu C Aggarwal, and Nitesh V Chawla. When will it happen?: relationship prediction in heterogeneous information networks. In WSDM, 2012.  
Mark Tygert. Statistical tests for whether a given set of independent, identically distributed draws comes from a specified probability density. PNAS, 2010.  
Daniela M Witten and Robert Tibshirani. Survival analysis with high-dimensional covariates. Statistical methods in medical research, 19(1):29-51, 2010a.  
Daniela M Witten and Robert Tibshirani. A framework for feature selection in clustering. Journal of the American Statistical Association, 105(490):713-726, 2010b.
