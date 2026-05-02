# A PROBABILISTIC FRAMEWORK FOR TASK-ALIGNED INTRA- AND INTER-AREA NEURAL MANIFOLD ESTIMATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Latent manifolds provide a compact characterization of neural population activity and of shared co-variability across brain areas. Nonetheless, existing statistical tools for extracting neural manifolds face limitations in terms of interpretability of latents with respect to task variables, and can be hard to apply to datasets with no trial repeats. Here we propose a novel probabilistic framework that allows for interpretable partitioning of population variability within and across areas in the context of naturalistic behavior. Our approach for task aligned manifold estimation (TAME-GP) explicitly partitions variability into private and shared sources which can themselves be subdivided in task-relevant and task irrelevant components, uses a realistic Poisson noise model, and introduces temporal smoothing of latent trajectories in the form of a Gaussian Process prior. This TAME-GP graphical model allows for robust estimation of task-relevant variability in local population responses, and of shared co-variability between brain areas. We demonstrate the efficiency of our estimator on within model and biologically motivated simulated data. We also apply it to several datasets of neural population recordings during behavior. Overall, our results demonstrate the capacity of TAME-GP to capture meaningful intra- and inter-area neural variability with single trial resolution.

# 1 INTRODUCTION

Systems neuroscience is gradually shifting from relatively simple and controlled tasks, to studying naturalistic closed-loop behaviors where no two observations (i.e., "trials") are alike (Michaiel et al., 2020; Shamash et al., 2021; Noel et al., 2021). Concurrently, neurophysiological techniques are advancing rapidly (Stevenson & Kording, 2011; Jun et al., 2017; Angotzi et al., 2019; Boi et al., 2020; Steinmetz et al., 2021) to allow recording from an ever-increasing number of simultaneous neurons (i.e., "neural populations") and across multiple brain areas. These trends lead to a pressing need for statistical tools that compactly characterize the statistics of neural activity within and across brain regions. Dimensionality reduction techniques are a popular tool for interrogating the structure of neural responses (Cunningham & Byron, 2014). However, as neural responses are driven by increasingly complex task features, the main axes of variability extracted using these techniques often intermix task and nuisance variables, making them hard to interpret. Alternatively, dimensionality reduction techniques that do allow for estimating task-aligned axes of variability (Brendel et al., 2011; Semedo et al., 2019; Keeley et al., 2020; Glaser et al., 2020), do not apply to communication between brain areas, and/or necessitate trial repeat structure that does not occur in natural behavior.

Here, we introduce a probabilistic approach for learning interpretable task-relevant neural manifolds that capture both intra- and inter-area neural variability with single trial resolution. Task Aligned Manifold Estimation with Gaussian Process priors (TAME-GP) incorporates elements of demixed PCA (dPCA; Machens (2010); Brendel et al. (2011); Kobak et al. (2016)) and probabilistic canonical correlation analysis (pCCA; Bach & Jordan (2005)) into a graphical model that additionally includes biologically relevant Poisson noise. The model uses a Gaussian Process (GP) prior to enforce temporal smoothness, which allows for robust reconstruction of single-trial latent dynamics (see Damianou et al. (2016) for a similar approach using Gaussian observation noise). We demonstrate the

robustness and flexibility of TAME-GP in comparison to alternative approaches using synthetic data and neural recordings from rodents and primates during naturalistic tasks. This reveals TAME-GP as a valuable tool for dissecting sources of variability within and across brain areas during behavior.

Related work. Dimensionality reduction is usually achieved by unsupervised methods that identify axes of maximal variability in the data, such as PCA. In neuroscience, this is often accompanied by additional smoothing over time reflecting the underlying neural dynamics (e.g., Gaussian process factor analysis (GPFA) (Yu et al., 2008); see GP-LVM (Ek & Lawrence, 2009) for similar approaches outside of neuroscience). This low dimensional projection is followed by a post hoc interpretation of latents in the context of behavioral variables, often by visualization. Alternative approaches such as dPCA (Machens, 2010; Brendel et al., 2011; Kobak et al., 2016) explicitly look for axes of neural variability that correlate with task variables of interest. However, these require partitioning trials into relatively few categories, based on experimental conditions or behavioral choices and averaging within conditions. This makes them unusable in naturalistic tasks where a single trial treatment is needed. Similarly, SNP-GPFA (Keeley et al., 2020) can partition (multi-region) neural activity into 'shared signal' and 'private noise' components, but only using data with stimulus repeats. Under 'no-repeat' conditions, pCCA (Bach & Jordan, 2005) can find subspaces of maximal cross-correlation between linear projections of task variables and neural responses (under gaussian noise assumptions), without the need for a priori grouping of trials by experimental condition or choice. This approach can also be applied for determining shared axes of co-variability across areas, an analog for communication subspaces (Semedo et al., 2019). Nonetheless, its noise model assumptions are mismatched to neural data. More fundamentally, pCCA only considers pairwise relationships, preventing a joint multi-area and task variables analysis. Overall, existing approaches come with practical limitations and do not directly address the routing of task-relevant information across brain areas.

# 2 TASK-ALIGNED MANIFOLD ESTIMATION WITH GP PRIORS (TAME-GP)

In its most general form, the graphical model of TAME-GP models a set of spike-count population responses  $\mathbf{x}^{(j)}$  from up to  $n$  different areas, $^2$  together with task variable of interest  $\mathbf{y}$  (Fig. 1A). The neural responses are driven by a set of  $n + 1$  low-dimensional latent variables  $\mathbf{z}^{(j)}$ . Specifically, the responses in area  $j$  arise as a linear combination of private latent variability  $\mathbf{z}^{(j)}$  and shared latents  $\mathbf{z}^{(0)}$ , which reflect task interpretable aspects of the underlying dynamics, with Poisson noise and an exponential link function:

$$
p \left(\mathbf {x} _ {i} ^ {(j)} | \mathbf {z} ^ {(0: n)}\right) = \text {P o i s s o n} \left(\exp \left(W _ {i} ^ {(0, j)} \mathbf {z} ^ {(0)} + W _ {i} ^ {(j, j)} \mathbf {z} ^ {(j)} + h _ {i} ^ {(j)}\right)\right), \tag {1}
$$

with parameters  $\mathbf{W}^{(0 / j,j)}$  and  $\mathbf{h}^{(j)}$ .

To make latents interpretable with respect to task variables  $\mathbf{y}$ , we adapt a probabilistic framing of CCA (Bach & Jordan, 2005) to introduces dependencies between any of the latents  $\mathbf{z}^{(k)}$ , which could be private or shared across areas, and  $\mathbf{y}$ :

$$
p \left(\mathbf {y} | \mathbf {z} ^ {(k)}\right) = \mathcal {N} \left(\mathbf {y}; \mathbf {C z} ^ {(k)} + \mathbf {d}, \boldsymbol {\Psi}\right), \text {w i t h p a r a m e t e r s} \mathbf {C}, \mathbf {d}, \boldsymbol {\Psi}. \tag {2}
$$

Finally, we regularize all latents to be smooth over time, through the introduction of a Gaussian Process prior, as in GPFA (Yu et al., 2008),  $z^{(j)}\sim \mathrm{GP}\left(\mathbf{0},k_{j}(\cdot ,\cdot)\right)$  , with area and dimension specific hyperparameters  $\tau$ $k_{j}\left(z_{t,i}^{(j)},z_{t^{\prime},i^{\prime}}^{(j)}\right) = \delta_{ii^{\prime}}\exp \left(-\frac{(t - t^{\prime})^{2}}{2\tau_{i}^{(j)}}\right)$  , where  $z_{t,i}^{(j)}$  is the  $i$  -th component of the  $j$  -th latent at time  $t$  , and  $\delta_{ii^{\prime}}$  is the Kronecker delta.

Putting these elements together results in a factorization of the joint distribution of the form:

$$
p \left(\mathbf {x} ^ {(1: n)}, \mathbf {y}, \mathbf {z} ^ {(0: n)}\right) = \prod_ {j = 0} ^ {n} p \left(\mathbf {z} ^ {(j)}\right) p \left(\mathbf {y} | \mathbf {z} ^ {(0)}\right) \prod_ {i, j} p \left(x _ {i} ^ {(j)} | \mathbf {z} ^ {(0)}, \mathbf {z} ^ {(j)}\right). \tag {3}
$$

![](images/2b2358a6bb74a606a37628a4c330ac9e5129be4240bfc3e7e6b3e4ea7e0f2a4a.jpg)

![](images/669a838cbd647ee529b02e5ed95b56174248f335851231db51c11bdc197e0b5a.jpg)

![](images/0de0530475a47bc38abbfc5777511e58d65a478a4b0b4b5ef83a4db2e5214eea.jpg)  
Figure 1: A. TAME-GP generative model. B. Example draws of spiking activity and a task variable from the TAME-GP graphical model. C. Model log-likelihood as a function of the EM iteration (left) and cross-validated leave-one-neuron-out marginal likelihood as a function of  $\mathbf{z}^{(0)}$  dimension (right). D-F. Latent variables estimation for within model simulated data: ground truth latent factors and model posterior mean  $\pm 95\%$  CI for three latent dimensions.

![](images/b785662f4a3a8c51830f65181f5a8ddb7173a7cb783250b5b0bcc241a9245a23.jpg)

This general form allows for a unified mathematical treatment of several estimation tasks of interest. We will detail key instances of this class that have practical relevance for neuroscience when presenting our numerical results below.

# 3 EM-BASED PARAMETER LEARNING

E-step Since a closed form solution of the posterior is not available (due the Poisson noise), we construct a Laplace approximation of the posterior  $^3$ ,  $p(\mathbf{z}|\mathbf{x},\mathbf{y},\boldsymbol {\theta})\approx q(\mathbf{z}|\mathbf{x},\mathbf{y},\boldsymbol {\theta}) = \mathcal{N}\left(\mathbf{z};\hat{\mathbf{z}}, - \mathbf{H}^{-1}\right)$  where  $\hat{\mathbf{z}}$  is the MAP of the joint log-likelihood and  $\mathbf{H}$  is its corresponding Hessian. Both of these quantities are estimated numerically.

The MAP estimate is obtained by gradient descent on the joint log likelihood. Using Eq. (3), the gradient of the joint log likelihood w.r.t. the latents can be written as

$$
\begin{array}{l} \nabla_ {\mathbf {z} ^ {(j)}} \log p (\mathbf {z}, \mathbf {x}, \mathbf {y}) = \sum_ {l} \left(\sum_ {j \geq 0} \nabla_ {\mathbf {z} ^ {(j)}} \log p (\mathbf {z} ^ {(j)}) + \sum_ {t > 0} \nabla_ {\mathbf {z} ^ {(j)}} \log p (\mathbf {y} _ {t} | \mathbf {z} _ {t} ^ {(0)})\right) \\ \left. + \sum_ {t > 0} \sum_ {j > 0} \nabla_ {\mathbf {z} ^ {(j)}} \log p \left(\mathbf {x} _ {t} ^ {(j)} | \mathbf {z} _ {t} ^ {(0)}, \mathbf {z} _ {t} ^ {(j)}\right)\right), \\ \end{array}
$$

where  $l \in (1:M)$  refers to the trial number, explicit index omitted for brevity. For a given trial, expanding one term at the time we have

$$
\nabla_ {\mathbf {z} ^ {(j)}} \log p (\mathbf {z} ^ {(j)}) = - \mathbf {K} ^ {(j)} \mathbf {z} ^ {(j)}
$$

$$
\nabla_ {\mathbf {z} _ {t} ^ {(0)}} \log p (\mathbf {y} | \mathbf {z} _ {t} ^ {(0)}) = \mathbf {C} ^ {\top} \Psi^ {- 1} (\mathbf {y} _ {t} - \mathbf {C} \mathbf {z} _ {t} ^ {(0)} - \mathbf {d})
$$

$$
\nabla_ {\mathbf {z} _ {t} ^ {(k)}} \log p (\mathbf {x} _ {t} ^ {(j)} | \mathbf {z} _ {t} ^ {(0)}, \mathbf {z} _ {t} ^ {(j)}) = \mathbf {W} ^ {(k, j) \top} (\mathbf {x} _ {t} - \exp (\mathbf {W} ^ {(0, j)} \mathbf {z} _ {t} ^ {(0)} + \mathbf {W} ^ {(j, j)} \mathbf {z} _ {t} ^ {(j)} + \mathbf {h} ^ {(j)})),
$$

where  $j > 0$ ,  $k \in \{0, j\}$ . The corresponding second moments are

$$
\nabla_ {\mathbf {z} ^ {(j)}} ^ {2} \log p \left(\mathbf {z} ^ {(j)}\right) = - \mathbf {K} ^ {(j)} j \in (0: n)
$$

$$
\nabla_ {\mathbf {z} _ {t} ^ {(0)}} ^ {2} \log p (\mathbf {y} | \mathbf {z} _ {t} ^ {(0)}) = - \mathbf {C} ^ {\top} \boldsymbol {\Psi} ^ {- 1} \mathbf {C}
$$

$$
\nabla_ {\mathbf {z} _ {t} ^ {(h)}} \nabla_ {\mathbf {z} _ {t} ^ {(k)}} \log p (\mathbf {x} _ {t} ^ {(j)} | \mathbf {z} _ {t} ^ {(0)}, \mathbf {z} _ {t} ^ {(j)}) = - \mathbf {W} ^ {(k, j) \top} \mathrm {d i a g} (\exp (\mathbf {W} ^ {(0, j)} \mathbf {z} _ {t} ^ {(0)} + \mathbf {W} ^ {(j, j)} \mathbf {z} _ {t} ^ {(j)} + \mathbf {h} ^ {(j)})) \mathbf {W} ^ {(h, j)}.
$$

with  $h, k \in \{0, j\}$ . Inverting the  $D \times D$  dimensional Hessian matrix is cubic in  $D = T \sum_{j} d_{j}$ , where  $T$  is the trial length and  $d_{j}$  denotes the dimensionality of latent  $\mathbf{z}^{(j)}$ , which restricts the number and

dimensionality of latents in practice. The Hessian of the log likelihood is sparse but does not have a factorized structure. Nonetheless, we can take advantage of the block matrix inversion theorem, to speed up the computation to  $\mathcal{O}(T^3\sum_j d_j^3)$  (see Appendix A.2), with additional improvements based on sparse GP methods (Wilson & Nickisch, 2015; Gardner et al., 2018) left for future work.

M-step Given the approximate posterior  $q$  found in the E-step, the parameters updates can be derived analytically for a few parameters, and numerically for the rest. Introducing the notation  $\pmb{\mu}_{t}^{(k)} = \mathbb{E}_{q}[\pmb{z}_{t}^{k}]$  and  $\pmb{\Sigma}_{t}^{(k,h)} = \mathbb{E}_{q}[\pmb{z}_{t}^{(k)}\pmb{z}_{t}^{(h)\top}] - \pmb{\mu}_{t}^{(k)}\pmb{\mu}_{t}^{(h)\top}$ , we have

$$
\bar {\mathbf {C}} = \left[ \sum_ {l, t} \mathbf {y} _ {t} \boldsymbol {\mu} _ {t} ^ {(0) \top} - \frac {1}{T M} \sum_ {l, t} \mathbf {y} _ {t} \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0) \top} \right] \left[ \sum_ {l, t} \boldsymbol {\Sigma} _ {t} ^ {(0, 0)} + \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0)} \boldsymbol {\mu} _ {t} ^ {(0) \top} - \frac {1}{T M} \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0)} \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0) \top} \right] ^ {- 1}
$$

$$
\bar {\mathbf {d}} = \frac {1}{T M} \left(\sum_ {l, t} \mathbf {y} _ {t} - \bar {\mathbf {C}} \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0)}\right)
$$

$$
\begin{array}{l} \bar {\Psi} = \frac {1}{T M} \left[ \sum_ {l, t} \mathbf {y} _ {t} \mathbf {y} _ {t} ^ {\top} - \left(\sum_ {l, t} \mathbf {y} _ {t} \boldsymbol {\mu} _ {t} ^ {(0) \top} \bar {\mathbf {C}} ^ {\top} + \bar {\mathbf {C}} \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0)} \mathbf {y} _ {t} ^ {\top}\right) - \left(\sum_ {l, t} \mathbf {y} _ {t} \bar {\mathbf {d}} ^ {\top} + \bar {\mathbf {d}} \sum_ {l, t} \mathbf {y} _ {t} ^ {\top}\right) \right. \\ + \bar {\mathbf {C}} \left(\sum_ {l, t} \left(\boldsymbol {\Sigma} _ {t} ^ {(0, 0)} + \boldsymbol {\mu} _ {t} \boldsymbol {\mu} _ {t} ^ {(0)}\right)\right) \bar {\mathbf {C}} ^ {\top} + \left(\bar {\mathbf {C}} \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0)} \bar {\mathbf {d}} ^ {\top} + \bar {\mathbf {d}} \sum_ {l, t} \boldsymbol {\mu} _ {t} ^ {(0) \top} \bar {\mathbf {C}} ^ {\top}\right) + T M \bar {\mathbf {d}} \bar {\mathbf {d}} ^ {\top} \Bigg ] \\ \end{array}
$$

where  $l = 1:M$  and  $t = 1:T$  are trial and time within trial indices.

The other observation model parameters are computed numerically by optimizing the expected log-likelihood under the posterior. In particular, for neuron  $i$  in population  $j$  we have

$$
\begin{array}{l} \mathcal {L} \left(W _ {i} ^ {(0, j)}, W _ {i} ^ {(j, j)}, h _ {i}\right) = \sum_ {t, l} x _ {t i} \left(h _ {i} + \left[ \begin{array}{c c} W _ {i} ^ {(0, j)} & W _ {i} ^ {(j, j)} \end{array} \right] \left[ \begin{array}{c} \boldsymbol {\mu} _ {t} ^ {(0)} \\ \boldsymbol {\mu} _ {t} ^ {(j)} \end{array} \right]\right) \\ - \exp \left(h _ {i} + \left[ \begin{array}{c c} W _ {i} ^ {(0, j)} & W _ {i} ^ {(j, j)} \end{array} \right] \left[ \begin{array}{c} \boldsymbol {\mu} _ {t} ^ {(0)} \\ \boldsymbol {\mu} _ {t} ^ {(j)} \end{array} \right] \right. \\ + \frac {1}{2} \left[ W _ {i} ^ {(0, j)} \quad W _ {i} ^ {(j, j)} \right] \left[ \begin{array}{c c} \boldsymbol {\Sigma} _ {t} ^ {(0, 0)} & \boldsymbol {\Sigma} _ {t} ^ {(0, j)} \\ \boldsymbol {\Sigma} _ {t} ^ {(0, j) \top} & \boldsymbol {\Sigma} _ {t} ^ {(j, j)} \end{array} \right] \left[ \begin{array}{c} W _ {i} ^ {(0, j) \top} \\ W _ {i} ^ {(j, j) \top} \end{array} \right]). \tag {4} \\ \end{array}
$$

For each neural population, we jointly optimized the projection weights and the intercept of all neurons with a full Newton scheme by storing the inverse Hessian in compressed sparse row (CSR) format (see Appendix A.3 for the gradient and Hessian of  $\mathcal{L}$ ).

The GP-prior parameters were also learned from data by gradient based optimization (using the limited-memory Broyden-Fletcher-Goldfarb-Shanno scheme (Virtanen et al., 2020)). First, we set  $\lambda_i^{(j)} = -\log (2\tau_i^{(j)})$ , and optimize for  $\lambda_i^{(j)}$  to enforce a positive time constant. We define  $\pmb{K}_i^{(j)} \in \mathbb{R}^{T \times T}$ , such that  $\left[\pmb{K}_i^{(j)}\right]_{ts} = \exp \left(-e^{\lambda_i^{(j)}}(t - s)^2\right)$ . The resulting objective function will take the form,  $\mathcal{L}\left(\lambda_i^{(j)}\right) = -\mathrm{trace}\left(\pmb{K}_i^{(j)-1}\mathbb{E}_q[\pmb{z}_i^{(j)}\pmb{z}_i^{(j)\top}]\right) - \log |\pmb{K}_i^{(j)}|$ . Gradients are provided in Appendix A.4, together with the procedure for parameter initialization (Appendix A.5).

# 4 RESULTS

Latent reconstruction for within model data. To validate the estimation procedure, we first used a simulated dataset sampled from the TAME-GP graphical model, with predefined parameters. Specifically, we simulated two neural populations  $\mathbf{x}^{(1)}$  and  $\mathbf{x}^{(2)}$ , each with 50 units and a one-dimensional task relevant variable  $y$ . We fixed the private latent factors  $\mathbf{z}^{(1)}$  and  $\mathbf{z}^{(2)}$  to two dimensions, and that of the shared factor  $\mathbf{z}^{(0)}$  to one. The projection weights  $\mathbf{W}^{(j)}$  and  $\mathbf{C}$ , the intercept terms  $\mathbf{d}$  and  $\mathbf{h}^{(j)}$ , the observation variance matrix  $\Phi$ , and the GP time constants of the factors were randomly

assigned. The parameters were chosen such that the overall mean firing rate was about  $20\mathrm{Hz}$  in both areas. We simulated spike counts at 50ms resolution for 200 draws from the process (which we will refer to as 'trials' in analogy to experiments), each lasting 2.5 seconds (see example trial in Fig. 1B). Given this data, we assessed the ability of our EM-based estimator to recover its true latent structure. $^4$  The marginal log likelihood saturated after a relatively small number of EM iterations (Fig. 1C). As a basic test of our ability to determine the dimensionality of latents, we systematically varied the dimensionality of the shared latent, while fixing the dimensions of  $\mathbf{z}^{(1)}$  and  $\mathbf{z}^{(2)}$  to their ground truth value of 2. We found that the best model fit was achieved at the ground truth task dimension 1, demonstrating that we are able to infer true latent dimensionality from data (Fig.1D).

Finally, we assessed the quality of the recovered latents in individual test trials. Due to known degeneracies, originally documented in linear gaussian latent models (Roweis & Ghahramani, 1999), the latent factors in TAME-GP are identifiable up to an affine transformation of the latent space. To address this, we used Procustes (Schönemann, 1966) to realign the latent axes back to the original space. The resulting posterior mean estimate of the latents show an excellent agreement with the ground truth factors (cross-validated linear regression  $R^2$  of 0.99 between the MAP estimate of latents and ground truth, Fig. 1 D-F), while the model predicted rates explained  $98\%$  of the ground truth firing rate variance. Overall, these numerical tests confirm that EM provides a veridical estimation of ground truth latent structure for within distribution data.

Task-aligned latent reconstruction for simulated latent dynamical systems models. The simple graphical model of TAME-GP captures axes of neural variability of scientific interest, but is far from an accurate generative model for neural dynamics during behavior. To assess the ability of TAME-GP to extract underlying structure from complex and out-of-distribution neural data, we used latent dynamical systems models in which we can explicitly define the flow of information from external stimuli and between areas, in several scenarios of practical interest.

The first in silico experiment focuses on identifying axes of task-relevant variability in neural responses. As a simple test case, we modeled a single neural population with a 6d latent structure (Fig. 2A). Two of the latent dimensions were task-relevant, driven by an observed temporally smooth external input  $\mathbf{y}_t$ , while the other four dimensions were intrinsic to the circuit. The key distinction between this process and the TAME-GP model assumptions is that the observed task variable acts as an input drive to the underlying latent dynamics rather than mapping to the latents directly. The latent dynamics take the form of a multivariate AR(1),

$$
\left\{ \begin{array}{l l} \mathbf {z} _ {\operatorname {p r}, t + 1} & = A _ {\operatorname {p r}} \left(\mathbf {z} _ {\operatorname {p r}, t} - \mu_ {t}\right) \Delta t + \sqrt {2 \Delta t} \mathrm {d} \mathbf {w} _ {t} ^ {(0)} \\ \mathbf {z} _ {\operatorname {t r}, t + 1} & = A _ {\operatorname {t r}} \left(\mathbf {z} _ {\operatorname {t r}, t} - \mathbf {y} _ {t}\right) \Delta t + \sqrt {2 \Delta t} \mathrm {d} \mathbf {w} _ {t} ^ {(1)}, \end{array} \right. \tag {5}
$$

where  $A_{\mathrm{pr}} \in \mathbb{R}^{4 \times 4}$  and  $A_{\mathrm{tr}} \in \mathbb{R}^{2 \times 2}$  the private and task relevant dynamics,  $\mathbf{y}_t \in \mathbb{R}^2$  and  $\mu_t \in \mathbb{R}^4$  inputs drawn from a factorized RBF kernel, and  $w_t^{(i)}$  is independent white noise for  $i = 0,1$ . Given these latent dynamics, spikes are generated as described by the TAME-GP observation model with  $\mathbf{W} \in \mathbb{R}^{100 \times 6}$ , and  $\mathbf{d} \in \mathbb{R}^{100}$ . We adjusted the parameters as to cover several average population firing rates by regulating  $\mathbf{d}$ , for a fixed number of trials (200) and a fixed trial duration (5 seconds). For simplicity, we circumvent the hyperparameter selection step by assuming that all estimators have access to the ground truth latent dimensionality: TAME-GP assumed 2 shared and 4 private latents. Unsupervised methods (pPCA, P-GPFA) were tasked with extracting the main two axes of neural variability in the data, while the supervised methods (pCCA) estimated 2d latents that correlate with task variable  $\mathbf{y}$ ; the same alignment procedure was used in all cases.

Fig. 2B illustrates the latent dynamics as estimated by TAME-GP, pPCA (Tipping & Bishop, 1999), P-GPFA (Hooram, 2015), and pCCA (Bach & Jordan, 2005). We quantify the latent space estimation accuracy by mean squared error, demonstrating that TAME-GP captured the stimulus driven dynamics better than other methods (Fig. 2C, see Suppl. Info. Fig. S1). P-GPFA showed a tendency to over-smooth, which obscured most of the underlying fine timescale latent structure. PCA failed by focusing on main axes of variability irrespective of task relevance, while CCA estimates were visually less interpretable. Only pCCA and TAME-GP found projections that selectively encoded for  $\mathbf{z}_{tr}$  with TAME-GP outperforming pCCA across conditions.

![](images/2e45fbd79070a931d4bde2098b063a69f22a07bbf7c88a357b80b69665b44c04.jpg)  
A

![](images/582db249a2d50e1431c3640b0146a7e932e854029db8933c236980d797903d9e.jpg)  
B

![](images/4cc0885972850f0b076b5dba697e6e13c5036a09a9a0d462c6c9acee08938658.jpg)

![](images/28cfa6ac4963669e13efbd4e5d0bf424bb2b2415a675245caa5caa2ff920e391.jpg)

![](images/11ec92484136c108600fe901802a416aeceed09961cff1ff93b5560dfab297dc.jpg)

![](images/ddfa766b2424d9473d80382a34bfb1a6c228147a485ecc70aab8a62913303c13.jpg)

![](images/14c12111f9a2f254c24edd7f398422a2dde50e2858f7db30e81684d2f2ffda0b.jpg)  
Figure 2: Methods comparison for single area task manifold alignment. A. TAME-GP graphical model for single area (top) and schematic for data generating process (bottom). B. Ground truth task relevant dynamics (green) and estimated low dimensional projection for TAME-GP (purple), P-GPFA (blue), pPCA (dark gray) and pCCA (light gray). C Mean squared error between the true shared dynamics and the model reconstruction, mean  $\pm$  s.d. over 10-fold cross-validation. D. Example single trial firing rate reconstruction. E. Mean squared error between the true and reconstructed firing rate across conditions, mean  $\pm$  s.d. over 10-folds of cross-validation.  
C

![](images/4c18a323946dc8f07c9a5d3a56ddb59fa137652aa7554fb8eabf863130c37517.jpg)

![](images/ee18953edf1679dda47d610d8d90eba34e66a6394fdfb57a62a3d5f094c98ca5.jpg)  
E

We also compared these methods in terms of their ability to predict the ground truth firing rate generating the observed spiking responses (total dimensions matching the ground truth of 6). Both TAME-GP and P-GPFA showed a stable and accurate firing rate reconstruction error across conditions (Fig. 2D,E), while the factorized linear gaussian methods (pPCA, pCCA) performed poorly. This may be due to the larger model mismatch, while additionally suffering from the lack of temporal smoothing, especially for low firing rates. Overall, TAME-GP was the only procedure that both captured the overall data statistics well and extracted accurate task-interpretable latents.

Assessing inter-area communication in simulated latent dynamical systems. In the second set of numerical experiments, we focused on estimating low-dimensional communication sub-spaces across neural populations (Fig. 3A). The ground truth data was again constructed using latent dynamical systems models, which now included two populations (Fig. 3B), where a low dimensional projection of the dynamics in one area, the sender, drive the dynamics of the other area, the receiver:

$$
\left\{ \begin{array}{l} \mathbf {z} _ {\mathrm {S}, t + 1} = A _ {S} \left(\mathbf {z} _ {\mathrm {S}, t} - \mathbf {y} _ {t}\right) \Delta t + \sqrt {2 \Delta t} \mathbf {w} _ {t} ^ {(0)} \\ \mathbf {z} _ {\mathrm {s h}} = P \cdot \mathbf {z} _ {\mathrm {S}} \\ \mathbf {z} _ {\mathrm {R}, t + 1} = A _ {R} \left(\mathbf {z} _ {\mathrm {R}, t} - \lambda_ {t} - \mathbf {z} _ {\mathrm {s h}, t}\right) \Delta t + \sqrt {2 \Delta t} \mathbf {w} _ {t} ^ {(1)}, \end{array} \right. \tag {6}
$$

where  $A_{S} \in \mathbb{R}^{4 \times 4}$  and  $A_{R} \in \mathbb{R}^{4 \times 4}$  are the sender and receiver dynamics,  $\mathbf{y}_{t}$  and  $\lambda_{t}$  are temporally smooth inputs drawn from independent GPs with factorized RBF kernels,  $P \in \mathbb{R}^{2 \times 4}$  defines the shared submanifold projection, and  $w_{t}^{(i)}$  is independent white noise. These latents map into spikes as above. We simulated three average firing rate conditions and varied the ground truth number of shared dimensions, from one to three. We compared our method with the two most commonly used approaches to communication manifold: pCCA and Semedo's reduced-rank regression procedure for communication manifold estimation (Semedo et al., 2019) (Fig. 3C), as well as SNP-GPFA (Keeley et al., 2020) (both with and without trial repeats, see Appendix A.6 and Suppl. Fig. 6).

TAME-GP (without task alignment) outperformed alternative approaches in terms of the reconstruction error of both ground truth firing rates (Fig. 3D, F) and shared latent dynamics (Fig. 3E). Furthermore, when testing the ability of different approaches to infer the dimensionality of the shared manifold through model comparison, the leave-one-out likelihood saturated at the ground truth dimension for all simulations (Fig. 3I), and peaked at the correct dimension  $75\%$  of the times (Fig. 3G,H). In contrast, the Semedo estimator tends to systematically overestimate the dimensionality of the shared manifold in this dataset.

Finally, we tested the general case in which we search for a communication subspace that aligns to task variable  $\mathbf{y}$ . To do so, we fit TAME-GP to the same dataset but assuming that  $\mathbf{y}_t$  is observed. We found again that TAME-GP has the best reconstruction accuracy, which saturates at the ground truth dimensionality  $(d = 2)$ . These observations are consistent across firing rate levels (see Suppl.

![](images/ac1c2de2f8d793858c35bddcb122131af5deab07fa19b8e52ba0f5cd64ec0a61.jpg)  
Figure 3: A. Schematic of communication subspace (left) and associated TAME-GP graphical model versions (right). B. Ground truth spike count generation process. C. Example shared latent reconstruction for TAME-GP (purple), PCCA (light grey) and, reduced rank regression (dark grey); ground truth in orange. D. Statistics for firing rate prediction quality. E. Statistics of shared dynamics reconstruction. F. Example reconstructions of the receiver firing rates compared to the ground truth (green). G. TAME-GP leave-one-neuron-out log-likelihood for different ground truth shared manifold dimensionality  $(d = 1,2,3)$  and increasing population rate from 5.1, 10.7,  $15.9\mathrm{Hz}$  (respectively, dashed, dashed-dotted and continuous lines). Lines styles show different average firing rate conditions. H. Difference between estimated and true  $\mathbf{z}_{\mathrm{sh}}$  dimensionality for TAME-GP (purple) and reduced rank regression (grey). I. Model fit quality as a function of latent dimensionality for all estimators. Ground truth dimension  $d = 2$  (dashed line). Error bars show mean  $\pm$  s.d. over 10-folds of cross-validation.  
Fig. 7). When fitting SNP-GPFA to simulated data in the case of precise stimulus repetitions and comparing it to TAME-GP, we find that both models are able to capture the latent space factorization. However, only TAME-GP works well in the case when latent dynamics vary across episodes, as would be the case during natural behavior (i.e. without stimulus repeats, see Suppl. Fig. 6, Table 1 and Appendix A.6 for details). Overall, these results suggest that TAME-GP can robustly recover meaningful sources of co-variability across areas in a range of experimentally relevant setups.

Mouse neural recordings during open-field exploration. As a first validation of the method, we estimated the manifold formed by simultaneously recorded head direction cells  $(n = 33)$  in the anterodorsal thalamic nuclei (ADN) of a mouse exploring a circular open field. These neurons form a well understood to code for heading Taube (1995) spanning a circular manifold Chaudhuri et al. (2019), and thus provide a close to ground truth setup for testing the interpretability of extracted latent structure. Recorded responses were segmented in 10sec time series, discretized in 20ms bins, and fit with a either a head-direction aligned 2d latent manifold (Fig.4A); private noise dimension  $d = 5$ ), or with two unsupervised methods pPCA and PGPFA, each with latent dimensionality  $d = 2$ . All methods recovered the underlying circular structure of the heading representation to some degree (Fig.4B). We decoded head direction from the extracted 2d latents and confirmed that TAME-GP preserved more information than pPCA, and comparable to P-GPFA (Fig.4C), with an overall superior data fit quality relative to pPCA (Fig.4D), as assessed by the  $R^2$  between model leave-one-neuron-out firing rate predictions and the raw spike counts Yu et al. (2008). Overall, these results confirm that the TAME-GP estimator can extract sensible coding structure from real neural data that does not exactly match the assumptions of the model.

Multi-area neural recordings in monkeys during VR spatial navigation Finally, we tested the ability of TAME-GP to find task aligned neural manifolds in a challenging dataset characterized by

![](images/e3bad3e5ff0b1413931c334113b99707f7a0a6e357fc8dd5fe5416416659b7dc.jpg)

![](images/798bf5dbfa7a47f4a6768e84a735b687339404912eefb72f9e4281a620a3359a.jpg)

![](images/7e34aab52d8f3f405b6afcbd880a5a21184aeb9e365b6c169404afaa4d4d13ba.jpg)

![](images/5225cc37b07ab4c3d959c3569d52b0beb9b7ffcaab090ddcafdb252312077a47.jpg)

![](images/897c2d81d40f03a563ccbf9e9824c2c1c6f019e52fde81a7fb5d943f71a156bb.jpg)

![](images/acb78bfa20194cf744c63370aedfcebfd9e0b82e0404336ed9e915bbb00e9b49.jpg)

![](images/089730a1debc28871c1531092aafcdeeba020ef10e1c8c33576ad555cb177a50.jpg)

![](images/69ff473dfb0f65431d28a6cf1275056a9be1ef634db107947b15ccd94b4ab955.jpg)

![](images/7745455120753ae4c85a96e394c3d4162b798e93099f370b25b76d94ee8a0154.jpg)

![](images/cddb3c046f4af4da6ed3e1fc7a17ed4edd0e71fe43f6f8477e659c75bf4314c0.jpg)

![](images/21706fc435a23884903fe0767a6973554cf64d969a9ace7b51157010ada2d99a.jpg)

![](images/a3ea158db23bd0632a659cdf5ab1349a66c79694c28bacc8a0e75b22dd7b9e32.jpg)  
Figure 4: Fitting TAME-GP to neural data. A. Graphical model for heading aligned mouse population responses in area ADN. B. Latent population dynamics, colored by time-varying heading, for various manifold estimators. C. Head direction decoding from 2d latents extracted with each method (by Lasso regression). Mean  $\pm$  standard deviation over a 5-fold cross-validation. D. Scatter plot of leave-one-neuron-out spike count variance explained for dimension matched TAME-GP and pPCA. Dots represent individual neurons. E. Schematic of the firefly task. Initial target location is randomized and remains visible for  $300\mathrm{ms}$ . The monkey has to use the joystick to navigate to the internally maintained target position. F. Top view of example monkey trajectories; increasing contrast marks initial location of the target (right, center, left). G. Within-area TAME-GP estimation aligned a latent task variable: the distance travelled.H. Scatter plot of leave-one-neuron-out spike count variance explained for dimension-matched TAME-GP and pPCA. Dots represent individual neurons. I. Single trial TAME-GP estimates of the task relevant dynamics, compared to J. those of P-GPFA. Trajectories are color-graded according to the initial angular target location (as in B). Lasso regression decoding of K. and L. linear distance travelled. TAME-GP decoding  $R^2$  (purple) is based on a 2d task relevant latent. P-GPFA  $R^2$  (blue) estimates were obtained for a range of latent dimensions (1-10). M. Communication subspace estimation between MSTd and dlPFC. N. As H, for shared latent space. O. Lasso regression decoding of task relevant variables (sorted by their shared subspace information content) from the shared (orange) and private latents (green, red) estimated by TAME-GP. Mean  $R^2 \pm$  s.e.m. were computed with by 10-fold cross-validation.

![](images/962ebf558f0c6d885d7f4e987749bed22dec4c9715671a170e14391ccad52b9f.jpg)

![](images/ad4c5785f99f218c69109ca327e21cfd21588837a78f8fa1c041b0643f62b29c.jpg)

![](images/b073ae6b9624dd0f3886f793db9d0b09a13200d818d2f4fbf034a8616e4ed6b2.jpg)

![](images/34d5f2c674d032b80231d3624ba1a623ddebe03d8c33a212ff02732c634aafcc.jpg)

a high-dimensional task space and lack of trial repeats. Specifically, monkeys navigate in virtual reality by using a joystick controlling their linear and angular velocity to "catch fireflies" (Fig.4E, F) (Lakshminarasimhan et al., 2018). Spiking activity was measured (binned in 6ms windows, sessions lasting over  $90\mathrm{min}$ ) and neurons in the two recorded brain areas (MSTd and dlPFC) showed mixed selectivity, encoding a multitude of task relevant variables (Noel et al., 2021). As a result, responses are high dimensional and unsupervised dimensionality reduction methods inevitably capture an uninterpretable mixture of task relevant signals in their first few latent dimensions.

We used TAME-GP to extract latent projections that align with the ongoing distance from the origin, decomposed in an angular and a radial component (Fig. 4G). We set the task relevant latent  $\mathbf{z}^{(0)}$  dimensions to two, matching the number of task variables. We verified the accuracy of the model by computing leave-one-neuron-out firing rate predictions and calculating the  $R^2$  between model predictions and raw spike counts. The TAME-GP estimator systematically outperformed pPCA with matched number of latents by this metric (Fig. 4H). We also compared the latent factors found by TAME-GP to those obtained by P-GPFA (Fig. 4I, J). For both variables, we found that the task variables were better accounted for by a two-dimensional TAME-GP estimated latent than by up to 10 dimensional latent spaces extracted with P-GPFA (Fig. 4K, L). A similar compression of the manifold was achieved in a separate dataset of monkey (pre-)motor responses during sequential

reaches (see Appendix A.8 and associated Fig. 9). This confirms that TAME-GP provides a compact low dimensional account of neural variability with respect to task variables of interest.

Lastly, we probed the model's ability to learn a communication subspace (Fig. 4M) between MSTd and dlPFC, brain areas that are known to interact during this task (Noel et al., 2021). In this instance, we selected the number of shared and private latent dimensions by maximizing the leave-one-neuron-out spike counts variance explained over a grid of candidate values (see Suppl. Fig. 8 and Appendix A.7). As before, we find that the TAME-GP reconstruction accuracy surpasses that of dimensionality-matched pPCA, for both MSTd and dlPFC (Fig. 4N). Since the shared manifold estimation was agnostic to task variables in this case, we used decoding from latent spaces to ask if the shared variability between these areas carried information about task variables known to drive single neuron responses in these areas. We found that the monkey's horizontal eye position, as well as latent task variables such as the travelled distance or the distance still remaining to target were mostly accounted for in shared, as opposed to private, axes of variability (Fig. 4O). This recapitulates prior observations made at the single-cell level (Noel et al., 2021). Overall, the results demonstrate that TAME-GP can extract interpretable low-dimensional latents and shared neural subspaces from complex and high-dimensional datasets.

# 5 DISCUSSION

Technological advances in systems neuroscience place an ever-increasing premium on the ability to concisely describe high-dimensional task-relevant neural responses. Here we introduce TAME-GP, a flexible statistical framework for partitioning neural variability in terms of private or shared (i.e., inter-area) sources, aligned to task variables of interest, and with single trial resolution. Our method was shown to provide compact latent manifold descriptions that better capture neural variability than any of the standard approaches we compared it against.

An important nuance that distinguishes various neural dimensionality reduction methods is whether the covariability being modeled is that of trial-averaged responses (i.e. stimulus correlations), residual fluctuations around mean responses (i.e. noise correlations) or a combination of the two (total correlations). Since isolating either the signal or the noise correlations alone would require across trial averages, our approach models total correlations, time resolved within individual trials. This differentiates our shared variability estimates from the traditional definition of a communication subspace (Semedo et al., 2019), which uses noise correlations alone, while keeping some of its spirit. It also makes it applicable to datasets without trial repeats.

The model adapts the approach of pCCA as a way of ensuring that the extracted latents reflect axes of neural variability that carry specific task relevant information. This choice has appealing mathematical properties in terms of unifying the problems of finding interpretable axes and communication subspaces, but is not the most natural one in terms of the true generative process of the data. While behavioral outputs are causal outcomes of the neural activity as described by the TAME-GP graphical model, sensory variables act as drivers for the neural responses and should causally affect the latent dynamics, not the other way around. Hence a natural next step will be to incorporate in the framework explicit stimulus responses, perhaps by taking advantage of recent advances in estimating complex tuning functions during naturalistic behavior (Balzani et al., 2020).

It would be interesting to explore the use temporal priors with more interesting structure, for instance spectral mixture kernels (Wilson & Adams, 2013), introducing prior dependencies across latent dimensions (de Wolff et al., 2021), or using non-reversible GP priors that better capture the causal structure of neural dynamics (Rutten et al., 2020). More generally, the probabilistic formulation allows the ideas formalized by TAME-GP to be combined with other probabilistic approaches for describing stimulus tuning and explicit latent neural dynamics (Zhao & Park, 2017; Nassar et al., 2018; Duncker et al., 2019; Glaser et al., 2020; Duncker & Sahani, 2021). Hence, this work adds yet another building block in our statistical arsenal for tackling questions about neural population activity as substrate for brain computation.

Broader impact We do not foresee any negative consequences to society from our work. Task aligned manifold extraction may prove useful in clinical applications, specifically for increasing robustness of BMI decoders by exploiting the intrinsic structure of the neural responses. Code implementing the TAME-GP estimator and associated demos is available at: <link to github>

# REFERENCES

Gian Nicola Angotzi, Fabio Boi, Aziliz Lecomte, Ermanno Miele, Mario Malerba, Stefano Zucca, Antonino Casile, and Luca Berdondini. Sinaps: An implantable active pixel sensor cmos-probe for simultaneous large-scale neural recordings. *Biosensors and Bioelectronics*, 126:355-364, 2019.  
Francis R Bach and Michael I Jordan. A probabilistic interpretation of canonical correlation analysis. Technical report, 2005.  
Edoardo Balzani, Kaushik Lakshminarasimhan, Dora Angelaki, and Cristina Savin. Efficient estimation of neural tuning during naturalistic behavior. Advances in Neural Information Processing Systems, 33:12604-12614, 2020.  
Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning, volume 4. Springer, 2006.  
Fabio Boi, Nikolas Perentos, Aziliz Lecomte, Gerrit Schwesig, Stefano Zordan, Anton Sirota, Luca Berdondini, and Gian Nicola Angotzi. Multi-shanks sinaps active pixel sensor cmos probe: 1024 simultaneously recording channels for high-density intracortical brain mapping. bioRxiv, pp. 749911, 2020.  
Wieland Brendel, Ranulfo Romo, and Christian K Machens. Demixed principal component analysis. Advances in neural information processing systems, 24, 2011.  
Rishidev Chaudhuri, Berk Gerçek, Biraj Pandey, Adrien Peyrache, and Ila Fiete. The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep. Nature neuroscience, 22(9):1512-1520, 2019.  
John P Cunningham and M Yu Byron. Dimensionality reduction for large-scale neural recordings. Nature neuroscience, 17(11):1500-1509, 2014.  
Andreas Damianou, Neil D Lawrence, and Carl Henrik Ek. Multi-view learning as a nonparametric nonlinear inter-battery factor analysis. arXiv preprint arXiv:1604.04939, 2016.  
Taco de Wolff, Alejandro Cuevas, and Felipe Tobar. Mogptk: The multi-output gaussian process toolkit. Neurocomputing, 424:49-53, 2021.  
Lea Duncker and Maneesh Sahani. Dynamics on the manifold: Identifying computational dynamical activity from neural population recordings. Current opinion in neurobiology, 70:163-170, 2021.  
Lea Duncker, Gergo Bohner, Julien Boussard, and Maneesh Sahani. Learning interpretable continuous-time models of latent stochastic dynamical systems. In International Conference on Machine Learning, pp. 1726-1734. PMLR, 2019.  
Carl Henrik Ek and PHTND Lawrence. Shared Gaussian process latent variable models. PhD thesis, CiteSeer, 2009.  
Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David Bindel, and Andrew G Wilson. Gpytorch: Blackbox matrix-matrix gaussian process inference withgpu acceleration. Advances in neural information processing systems, 31, 2018.  
Joshua Glaser, Matthew Whiteway, John P Cunningham, Liam Paninski, and Scott Linderman. Recurrent switching dynamical systems models for multiple interacting neural populations. Advances in neural information processing systems, 33:14867-14878, 2020.  
Nam Hooram. Poisson extension of gaussian process factor analysis for modeling spiking neural populations master's thesis. Department of Neural Computation and Behaviour, Max Planck Institute for Biological Cybernetics, Tubingen, 8, 2015.  
James J Jun, Nicholas A Steinmetz, Joshua H Siegle, Daniel J Denman, Marius Bauza, Brian Barbarits, Albert K Lee, Costas A Anastassiou, Alexandru Andrei, Căgatay Aydin, et al. Fully integrated silicon probes for high-density recording of neural activity. Nature, 551(7679):232-236, 2017.

S.L. Keeley, M.C. Aoi, Y. Yu, S.L. Smith, and Pillow J.W. Identifying signal and noise structure in neural population activity with gaussian process factor models. NeurIPS, 34, 2020.  
Dmitry Kobak, Wieland Brendel, Christos Constantinidis, Claudia E Feierstein, Adam Kepecs, Zachary F Mainen, Xue-Lian Qi, Ranulfo Romo, Naoshige Uchida, and Christian K Machens. Demixed principal component analysis of neural population data. *Elife*, 5:e10989, 2016.  
Kaushik J Lakshminarasimhan, Marina Petsalis, Hyeshin Park, Gregory C DeAngelis, Xaq Pitkow, and Dora E Angelaki. A dynamic bayesian observer model reveals origins of bias in visual path integration. Neuron, 99(1):194-206, 2018.  
Christian K Machens. Demixing population activity in higher cortical areas. Frontiers in computational neuroscience, 4:126, 2010.  
Angie M Michaiel, Elliott TT Abe, and Christopher M Niell. Dynamics of gaze control during prey capture in freely moving mice. *Elife*, 9:e57458, 2020.  
Josue Nassar, Scott W Linderman, Yuan Zhao, Mónica Bugallo, and Il Memming Park. Learning structured neural dynamics from single trial population recording. In 2018 52nd Asilomar Conference on Signals, Systems, and Computers, pp. 666-670. IEEE, 2018.  
Jean-Paul Noel, Edoardo Balzani, Eric Avila, Kaushik Lakshminarasimhan, Stefania Bruni, Panos Alefantis, Cristina Savin, and Dora E Angelaki. Flexible neural coding in sensory, parietal, and frontal cortices during goal-directed virtual navigation. bioRxiv, 2021.  
Matthew G Perich, Patrick N Lawlor, Konrad P Kording, and Lee E Miller. Extracellular neural recordings from macaque primary and dorsal premotor motor cortex during a sequential reaching task. https://crcs.org/, 2018.  
Sam Roweis and Zoubin Ghahramani. A unifying review of linear gaussian models. Neural computation, 11(2):305-345, 1999.  
Virginia Rutten, Alberto Bernacchia, Maneesh Sahani, and Guillaume Hennequin. Non-reversible gaussian processes for identifying latent dynamical structure in neural data. Advances in neural information processing systems, 33:9622-9632, 2020.  
Peter H Schonemann. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31(1):1-10, 1966.  
João D Semento, Amin Zandvakili, Christian K Machens, M Yu Byron, and Adam Kohn. Cortical areas interact through a communication subspace. Neuron, 102(1):249-259, 2019.  
Philip Shamash, Sarah F Olesen, Panagiota Iordanidou, Dario Campagner, Nabhojit Banerjee, and Tiago Branco. Mice learn multi-step routes by memorizing subgoal locations. Nature Neuroscience, 24(9):1270-1279, 2021.  
Nicholas A Steinmetz, Cagatay Aydin, Anna Lebedeva, Michael Okun, Marius Pachitariu, Marius Bauza, Maxime Beau, Jai Bhagat, Claudia Böhm, Martijn Broux, et al. Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain recordings. Science, 372(6539): eabf4588, 2021.  
Ian H Stevenson and Konrad P Kording. How advances in neural recording affect data analysis. Nature neuroscience, 14(2):139-142, 2011.  
JS Taube. Head direction cells recorded in the anterior thalamic nuclei of freely moving rats. 15(1): 70-86, 1995. doi: 10.1523/JNEUROSCI.15-01-00070.1995.  
Michael E Tipping and Christopher M Bishop. Probabilistic principal component analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61(3):611-622, 1999.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, PEARU Peterson, Warren Weckesser, Jonathan Bright, Stefan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cirmrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261-272, 2020. doi: 10.1038/s41592-019-0686-2.  
Andrew Wilson and Ryan Adams. Gaussian process kernels for pattern discovery and extrapolation. In International conference on machine learning, pp. 1067-1075. PMLR, 2013.  
Andrew Wilson and Hannes Nickisch. Kernel interpolation for scalable structured gaussian processes (kiss-gp). In International conference on machine learning, pp. 1775-1784. PMLR, 2015.  
Byron M Yu, John P Cunningham, Gopal Santhanam, Stephen Ryu, Krishna V Shenoy, and Maneesh Sahani. Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity. Advances in neural information processing systems, 21, 2008.  
Yuan Zhao and Il Memming Park. Variational latent gaussian process for recovering single-trial dynamics from population spike trains. Neural computation, 29(5):1293-1316, 2017.
