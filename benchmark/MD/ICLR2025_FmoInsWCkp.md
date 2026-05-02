# PREDICTING TIME-VARYING FLUX AND BALANCE IN METABOLIC SYSTEMS USING STRUCTURED NEURAL ODE PROCESSES

Anonymous authors

Paper under double-blind review

# ABSTRACT

We develop a novel data-driven framework as an alternative to dynamic flux balance analysis, bypassing the demand for deep domain knowledge and manual efforts to formulate the optimization problem. The proposed framework is end-to-end, which trains a structured neural ODE process (SNODEP) model to estimate flux and balance samples using gene-expression time-series data. SNODEP is designed to circumvent the limitations of the standard neural ODE process model, including restricting the latent and decoder sampling distributions to be normal and lacking structure between context points for calculating the latent, thus more suitable for modeling the underlying dynamics of a metabolic system. Through comprehensive experiments (156 in total), we demonstrate that SNODEP not only predicts the unseen time points of real-world gene-expression data and the flux and balance estimates well but can even generalize to more challenging unseen knockout configurations and irregular data sampling scenarios, all essential for metabolic pathway analysis. We hope our work can serve as a catalyst for building more scalable and powerful models for genome-scale metabolic analysis.

# 1 INTRODUCTION

A distinctive characteristic of deep neural networks is their capability to implicitly learn complicated features and dynamics from data, significantly saving human effort in composing those handcrafted features and devising complex models. Therefore, there has been a growing interest in using them in a variety of scientific contexts, such as quantum chemistry (von Glehn et al., 2022), tokamak controller design (Degrave et al., 2022), climate sciences (Lam et al., 2022; Nguyen et al., 2023), molecule generation (Hoogeboom et al., 2022) and drug discovery (Askr et al., 2023), to name a few. For drug discovery problems in particular, it is essential to answer the questions of where and how the drug should be targeted. The machine learning community has attracted increased attention in molecular design to address the latter question (Luo et al., 2022; Corso et al., 2022). On the other hand, metabolic pathway analysis techniques, such as flux balance analysis (FBA) (Orth et al., 2010) and dynamic FBA (Mahadevan et al., 2002), have been shown highly effective in finding drug targets (Sen & Oresic, 2023). These methods are widely used to study the effect of drugs or environmental stress simulated by gene knockouts on unwanted cells, such as cancer cells, by curbing their metabolism (Raskevicius et al., 2018). Nevertheless, several key parameters, including the optimization objective and constraints for the reaction flux in their linear programming (LP) formulation, must be determined using domain expertise for each case, largely limiting their generality and scalability. In this work, we aim to develop scalable data-driven methods that can directly predict the behavior of metabolic systems with time-varying flux, thus avoiding the manual effort required to build FBA models.

More specifically, we achieve this by leveraging single-cell RNA sequencing (scRNA-seq) time-series data (Chen et al., 2019) and using single-cell flux estimation analysis (scFEA) technique from Alghamdi et al. (2021) to estimate flux and balance of the metabolic system, because scRNA-seq can churn out data in bulk, and getting time-series single-cell gene-expression data is much less labor intensive than getting actual flux-balance time-series data. The challenge, however, lies in that gene expression trajectories for individual cells cannot be tracked over time since cells die once their gene expression is read. Instead, we only have gene expression samples from different cells at each

timestep, which can be viewed as samples from a time-varying distribution resembling a random process. In fact, it's well known that gene transcription is stochastic, especially when considered at the single cell level (Thattai & Van Oudenaarden, 2001). Thus, the amounts of molecules produced, or the chemical concentration, from a collection of cells can be considered to be sampled from some distribution, with the amounts of mRNA molecules showing a Poisson-like behavior in a steady state as shown in Thattai & Van Oudenaarden (2001).

Since the time-varying metabolic concentrations are known to follow a non-linear ordinary differential equation (ODE), we propose a novel Structured Neural ODE Process (SNODEP) architecture that is built on top of the standard neural ODE processes (Norcliffe et al., 2021) to predict the underlying dynamics of the metabolic system. We note that standard neural ODE processes have several design choices that might not help to model the ODE dynamics in our case, like lack of structure in the encoder to get the latent distribution from the context points and the use of Gaussian parametric family for latent posterior and decoder distributions. Consequently, we design the architecture of SNODEP to bypass these shortcomings, showing improved performance in tasks such as predicting gene-expression distributions on unseen timesteps, predicting metabolic-flux and metabolic-balance distribution on unseen timesteps, and predicting the corresponding distributions for gene-knockout cases, considering both regularly and irregularly sampled data, all for several metabolic pathways.

Contributions. We formulate the prediction problem of metabolic flux and balance as a stochastic neural processing task, where the goal is to learn the underlying dynamics by predicting their time-varying distributions under different configurations (Section 2). We propose an end-to-end training framework, which first defines the intermediary steps required to estimate metabolic flux and balance from scRNA-seq data and then learns a novel SNODEP model that can predict the unseen time points of flux and balance and their dynamics under gene-knockout configurations (Section 3.2). The proposed SNODEP architecture is designed by addressing a few limitations of the standard architecture of neural ODE processes (Section 3.1); thus, it is more suitable to model the time-varying distributions from metabolic systems. Comprehensive experiments on real-world datasets and various metabolic pathways demonstrate that SNODEP is highly effective in modeling the dynamics of gene expressions and predicting metabolic flux and balance, consistently outperforming alternative models such as standard neural ODE processes (Sections 4.2-4.4). We also showcase the superiority of SNODEP under gene-knockout variations and scenarios with irregularly sampled data (Section 4.5), suggesting its versatility and strong potential in solving challenges in biomedical domains.

# 1.1 RELATED WORK

Metabolic Pathway Analysis. Genome-scale metabolic models (GSMMs) have proven to be powerful tools in the design of therapeutic treatments. For instance, Raškevičius et al. (2018) employed GSMMs to identify therapeutic windows for cancer treatment, while Larsson et al. (2020) used them to simulate gene knockouts in a Glioblastoma cancer cell model, identifying potential therapeutic targets and predicting side effects in healthy brain tissue. Despite their importance, GSMMs are time-consuming and require significant domain expertise to build. Recent studies have explored integrating machine learning techniques with GSMMs, as reviewed in Sahu et al. (2021). From a dynamical standpoint, Costello & Martin (2018) framed pathway dynamics prediction as a machine learning problem, using XGBoost models to predict such dynamics, but their framework is not end-to-end. More recently, Aghaee et al. (2024) introduced a graph neural network model to simulate the dynamic behavior of metabolites in oxidative stress pathways in bacterial cell cultures for synthetic data. In addition, RNA velocity (La Manno et al., 2018) estimates the time derivative of gene expressions but needs spliced and unspliced mRNA counts, usually not reported in the experiments. Similarly, Klumpe et al. (2023) investigated single-cell time series prediction, albeit also using synthetic data with no specific focus on metabolic pathways. To the best of our knowledge, our work is the first to comprehensively study the dynamically varying flux and balance of metabolic pathways derived from real-world single-cell gene expression time-series data.

Neural ODE. The neural ODE family of models has shown strong capabilities in modeling dynamic systems, particularly when the underlying dynamics are known to follow an ODE (Rubanova et al., 2019). While latent neural ODEs have been applied to interpolation and extrapolation tasks, they are not suitable for modeling random processes. In contrast, neural processes (NP) (Garnelo et al.,

![](images/5dba2009bfeb9f2e0e42177cea3527892cf81d7d23c22a9bdcfbaa1f88f7f9e4.jpg)  
Figure 1: Overall pipeline of our framework for predicting time-varying distributions, such as gene expressions, flux, and balance, with (green) and without (orange) gene knockouts.

2018) can be used for modeling time-varying distributions, but they have no consideration for the underlying dynamics. These observations motivate us to explore models like neural ODE processes (NODEP) (Norcliffe et al., 2021), where the dynamics are defined over the parametric space of these distributions. Other models, such as those proposed in Kidger et al. (2021), assume a noisy evolution of dynamics, which does not align with our prediction problems of time-varying distributions in metabolic systems. Our work adapts standard neural ODE processes (Norcliffe et al., 2021) to better suit our specific settings, showing improvements across various tasks and metabolic pathways.

# 2 PROBLEM FORMULATION

Classical methods like DFBA estimate time-varying metabolic flux and balance by solving an optimization problem to maximize the biomass at each timestep (see Appendix B.1 for more details). Our work proposes to directly train a model on scFEA-estimated flux-balance values until a certain timestep and then predict the distributions of gene expression, flux, and balance in future timesteps, expecting that the trained model will learn the underlying dynamics. Figure 1 illustrates the overview of our pipeline. Due to page limits, we defer more details on scFEA proposed by Alghamdi et al. (2021) to Appendix B.2. Below, we provide detailed descriptions of our problem setup. The key notations and their descriptions are provided in Appendix A.

Predicting Gene Expression, Flux and Balance. Suppose we have a gene count matrix of dimension  $K \times N$ , where  $N$  is the total number of cells and  $K$  is the total number of genes, with gene counts measured at each regular timestep  $t$  and total  $V$  timesteps. Let  $\mathbb{B}_t$  be the index set representing the cells whose gene counts  $\in \mathbb{R}^K$  are observed at time  $t$ . Then, we have  $\sum_{t} |\mathbb{B}_t| = N$ , indicating that all  $N$  cells get their expressions counted over various timesteps.

For a metabolic pathway, we only extract the relevant  $d$  genes from the total set of  $K$  genes. Let  $g_{i,t} \in \mathbb{R}^d$  be the gene-expression array for cell  $i \in \mathbb{B}_t$  at time  $t$ , and  $\mathbf{G}_t \in \mathbb{R}^{d \times |\mathbb{B}_t|}$  be the corresponding matrix. For a certain metabolic pathway with  $u$  modules and  $v$  metabolites and each batch  $\mathbb{B}_t$  of cells at time  $t$ , we estimate the flux  $m^f$  and balance  $m^b$  using the scFEA framework detailed in Appendix B.2. Specifically, we define:

-  $S_{t}^{f}: \{g_{i,t}\}_{i \in \mathbb{B}_{t}} \to \{m_{i,t}^{f}\}_{i \in \mathbb{B}_{t}}$  as the mapping that estimates the flux  $m_{i,t}^{f} \in \mathbb{R}^{u}$  for each cell  $i$  based on its gene expression. Let  $\mathbf{M}_{t}^{f} \in \mathbb{R}^{u \times |\mathbb{B}_{t}|}$  be the matrix of the flux samples.  
-  $S_{t}^{b} : \{g_{i,t}\}_{i \in \mathbb{B}_{t}} \to \{m_{i,t}^{b}\}_{i \in \mathbb{B}_{t}}$  as the analogous mapping for estimating the metabolic balance  $m_{i,t}^{b} \in \mathbb{R}^{v}$ . Let  $\mathbf{M}_{t}^{b} \in \mathbb{R}^{v \times |\mathbb{B}_{t}|}$  be the matrix of the balance samples.

We note that scFEA was originally developed for static-FBA, but since the static-DFBA formulation (Equation 4) can be interpreted as solving the static-FBA for different timesteps, we use scFEA to estimate flux-balance values for different timesteps.

Gene-knockout. Gene knockout is a way to understand how a gene influences the metabolic network, for example, in understanding how essential genes in pathogens affect metabolic pathways to design drugs to inhibit those pathways (Larsson et al., 2020); it's also widely used in synthetic

biology Dalvie et al. (2021). In the gene-knockout simulations in FBA models, the constraints of the reaction fluxes affected by essential genes are usually modified (Maranas & Zomorrodi, 2016). In contrast, we train a model on certain gene-knockout configurations and then predict the distribution on unseen configurations and timesteps. To simulate gene-knockout conditions, we randomly sample  $S$  subsets of  $k$  most expressed genes, set the gene-expression of genes from those subsets to zero (See Algorithm 1 for more details), and estimate the flux-balance values again based on the scFEA techniques. For  $s \in \{1,2,\dots,S\}$ , we analogously define  $\{\tilde{m}_{i,t}^{f}\}_{s,i \in \mathbb{B}_{t}}$  and  $\{\tilde{m}_{i,t}^{b}\}_{s,i \in \mathbb{B}_{t}}$  as gene-knockout flux and balance estimates, respectively, where we use  $\tilde{\mathbf{M}}_{s,t}^{f} \in \mathbb{R}^{u \times |\mathbb{B}_{s,t}|}$  and  $\tilde{\mathbf{M}}_{s,t}^{b} \in \mathbb{R}^{v \times |\mathbb{B}_{s,t}|}$  to denote the corresponding matrix of samples.

Essentially, our framework assumes that metabolic flux and balance from scRNA-seq data can be estimated using scFEA techniques, that knocking out a subset of genes does not change the expression levels of the rest of the genes, and that gene essentiality and gene expression levels are correlated.

Learning Objective. For each timestep  $t \in \{t_1, t_2, \ldots, t_V\}$ , we collect samples of gene expression  $\{g_{i,t}\}_{i \in \mathbb{B}_t}$ , flux  $\{m_{i,t}^f\}_{i \in \mathbb{B}_t}$  and balance  $\{m_{i,t}^b\}_{i \in \mathbb{B}_t}$  and their gene-knockout samples  $\{\{\tilde{m}_{i,t}^f\}_{j,i \in \mathbb{B}_t}, \{\tilde{m}_{i,t}^b\}_{j,i \in \mathbb{B}_t}\}$  with cells  $\mathbb{B}_t$  using previous steps. We assume these samples are drawn from some underlying distributions corresponding to gene expression  $G(\theta_g(t))$ , flux  $M^f(\theta_f(t))$ , balance  $M^b(\theta_b(t))$  and their gene knockout versions  $\{\tilde{M}^f(\theta_f(t)), \tilde{M}^b(\theta_b(t))\}$ , respectively. The goal is to learn a model  $F: t \to Y(\theta_t)$  that can predict the underlying dynamics of time-varying distributions, which depend on some latent distribution  $L$ . In our setup,  $F$  is considered as an encoder-decoder neural network, with a different network for each distribution in  $\{G, M^f, M^b, \tilde{M}^f, \tilde{M}^b\}$ .

Let  $C < T < V$  be the length of context, target, and total available time points, respectively. Given a distribution  $Y$ , let  $y_{i} \sim Y(\theta_{t})$  for any  $i \in \{1, \ldots, V\}$ . When we say  $y_{i} \sim Y(\theta_{t})$ , it means a random sample from the set  $\{y_{i,t}\}_{i \in \mathbb{B}_{t}}$ . During training, our model's encoder takes as input the context data, which includes samples from context points  $\mathcal{C} = \{(t_{1}, y_{1}), \ldots, (t_{C}, y_{C})\}$ . The decoder then predicts samples from the target points  $\mathcal{T} = \{(t_{1}, y_{1}), \ldots, (t_{T}, y_{T})\}$ . During inference, the model is used to predict every timestep available, including hitherto unseen timesteps  $\mathcal{V} = \{(t_{1}, y_{1}), \ldots, (t_{V}, y_{V})\}$ . In the following discussions, we denote  $\mathbb{I}_{\mathcal{C}} = \{1, \ldots, C\}$  and  $\mathbb{I}_{\mathcal{T}} = \{1, \ldots, T\}$  for simplicity.

# 3 METHODOLOGY

# 3.1 ISSUES WITH STANDARD NEURAL ODE PROCESS

The standard neural ODE process (NODEP) model (Norcliffe et al., 2021) employs an encoder-decoder model architecture, where the context points  $\{(t_i,y_i)\}_{i\in \mathbb{I}_c}$  are used to calculate the latent distributions  $L_{0}(\theta_{l_{0}})$  and  $D(\theta_d)$ , and the latent  $l_0\sim L_0$  evolves over target timesteps  $\{t_i\}_{i\in \mathbb{I}_T}$  according to an ODE that is modeled by a neural network  $\mathbf{f}_w$  as follows:

$$
l \left(t _ {i}\right) = l _ {0} + \int_ {t _ {0}} ^ {t _ {i}} \mathbf {f} _ {w} (l (t), d, t) d t. \tag {1}
$$

The time-evolving latent distributions are then fed into a decoder to obtain the target distributions:  $\{N_{i}(y_{i}|\mu_{w_{1}}(l(t_{i})),\sigma_{w_{2}}(l(t_{i})))\}_{i\in \mathbb{I}_{\mathcal{T}}}$ . Although NODEP has been shown effective in modeling ODE dynamics for scientific discovery, there are a few limitations with NODEP if applied to our settings:

1. The latent and decoding distributions are treated as Normal. This is not the best choice of distributions to model gene-expression data, which is usually discrete and Poisson-like (Thattai & Van Oudenaarden, 2001) and confirmed by Figures 9a and 9b in Appendix E.  
2. The encoded representation  $r_i = f_e(\{t_i^\mathcal{C}, y_i^\mathcal{C}\})$  is calculated using context points without any particular structure in NODEP. These  $r_i$ 's are then order-invariantly aggregated to give  $r$ , and finally  $D \sim q_D(d|\mathcal{C}) = \mathcal{N}(d|\mu_D(r), \mathrm{diag}(\sigma_D(r)))$ , similarly for  $L_0$ . The order between the context points and their sequential dependence on each other is not efficiently utilized. Enforcing this sequential dependence can be highly useful for guiding the ODE decoder because otherwise, it might lead to unintended attention to certain context points.

This sequential dependence of context points is even more important for irregularly sampled data, where an order-invariant encoder might lead to different representations for different timesteps sam

![](images/c225ee6df1ece6ff6bc8bf7d5ee8cb903f4e1eeac16fce46254ce16a01b29829.jpg)  
Figure 2: Illustration of the overall pipeline of the proposed SNODEP architecture.

pled, even though the underlying condition is the same. This further motivates us to employ a GRU-ODE encoder to capture the underlying dynamics and thus not be sensitive to irregularity.

# 3.2 STRUCTURED NEURAL ODE PROCESS (SNODEP)

Encoder with Regularly Sampled Data. To address the above issues, we propose a modified architecture where the encoder leverages Long Short-Term Memory (LSTM) (Hochreiter, 1997). The LSTM encoder is designed to capture dependencies between context points across time, allowing for a more informed and contextually-aware calculation of latent distributions  $L_0(\theta_{l_0})$  and  $D(\theta_d)$ . We run the LSTM backward since we want the initial value of the latent variable  $l_0$ . Formally, the encoder takes the context sequence  $\{(t_i,y_i)\}_{i\in \mathbb{I}_c}$  and computes hidden representations  $\{h_i\}_{i\in \mathbb{I}_c}$ :

$$
h _ {i} ^ {\mathrm {b w d}} = \operatorname {L S T M} _ {\mathrm {b w d}} \left(y _ {i}, h _ {i + 1} ^ {\mathrm {b w d}}\right), \text {f o r} i \in \mathbb {I} _ {\mathcal {C}}.
$$

Encoder with Irregularly Sampled Data. Recurrent networks assume inputs to be regularly spaced and have no consideration for the actual time the input was sampled, not applicable to irregularly sampled data (Rubanova et al., 2019). Thus, our hidden state varies according to a GRU-ODE:

$$
\hat {h} _ {i - 1} ^ {\mathrm {b w d}} = h _ {i} ^ {\mathrm {b w d}} + \int_ {t _ {i}} ^ {t _ {i - 1}} \mathbf {g} _ {\phi} (h ^ {\mathrm {b w d}} (t))   d t, h _ {i - 1} ^ {\mathrm {b w d}} = \mathrm {G R U} (y _ {i}, \hat {h} _ {i - 1} ^ {\mathrm {b w d}}), \text {f o r} i \in \mathbb {I} _ {\mathcal {C}},
$$

where  $\mathbf{g}_{\phi}$  is the network supposed to capture the time-dependent underlying dynamics of the hidden state, and GRU stands for the Gated Recurrent Unit (Cho, 2014), a gating mechanism typically employed in recurrent neural networks. For irregular data, our encoder uses the final hidden state  $h_0^{\mathrm{bwd}}$  to calculate the parameters of the initial latent  $l_{0}$  and control  $d$ , which then evolves to give us the time-varying probability distributions. But in Rubanova et al. (2019), the  $h_0^{\mathrm{bwd}}$  is used to get the initial latent,  $l_{0}$  which then evolves directly, giving us quantities of interest and there's no time-varying distribution involved. For both regular and irregular scenarios, the final hidden state from the backward pass gives us the representation  $r = [h_0^{\mathrm{bwd}}]$ , which is then used to parameterize the latent distributions  $L_{0}$  and  $D$ , via a feed-forward layer (FFW in Figure 2).

Latent distributions. The latent distributions,  $L_0(\theta_{l_0})$  and  $D(\theta_d)$ , are chosen based on the dataset. For gene-expression data, we model the latent distribution as a LogNormal distribution, to resemble the Poisson-like nature of the data:

$$
l _ {0} \sim \mathrm {L o g N o r m a l} (\mu_ {L _ {0}} (r), \sigma_ {L _ {0}} (r)), \quad d \sim \mathrm {L o g N o r m a l} (\mu_ {D} (r), \sigma_ {D} (r)),
$$

whereas for metabolic-flux and balance data, we use a Gaussian distribution:

$$
l _ {0} \sim \mathcal {N} \left(\mu_ {L _ {0}} (r), \operatorname {d i a g} \left(\sigma_ {L _ {0}} (r)\right)\right), \quad d \sim \mathcal {N} \left(\mu_ {D} (r), \operatorname {d i a g} \left(\sigma_ {D} (r)\right)\right).
$$

where  $\mu_{L_0},\sigma_{L_0},\mu_D$  and  $\sigma_{D}$  are learned functions. Using LogNormal ensures that we can resemble the Poisson-like nature of gene-expression data while still being able to use the re-parametrization trick (Kingma & Welling, 2013).

Table 1: Illustration of considered pathways with the number of genes, metabolites, and modules.  

<table><tr><td>Pathway</td><td>Num Genes</td><td>Num Metabolites</td><td>Num Modules</td></tr><tr><td>M171</td><td>623</td><td>70</td><td>168</td></tr><tr><td>MHC-i</td><td>281</td><td>6</td><td>9</td></tr><tr><td>Iron Ions</td><td>136</td><td>8</td><td>15</td></tr><tr><td>Glucose-TCACycle</td><td>84</td><td>11</td><td>15</td></tr></table>

Decoder. The decoder relies on evolving the latent variable  $l(t)$  over time based on a neural ODE. For a given latent state at time  $t_0$ , the evolution is governed by:

$$
l (t _ {i}) = l _ {0} + \int_ {t _ {0}} ^ {t _ {i}} \mathbf {f} _ {\theta} (l (t), d, t) d t,
$$

where  $\mathbf{f}_{\theta}$  represents the dynamics defined by the Neural ODE, and  $d$  is used for tuning the trajectory. At each target time  $\{t_i\}_{i\in \mathbb{I}_T}$ , the latent state  $l(t_{i})$  is used to determine the target distributions. For gene expressions, we model the predicted distributions as a Poisson distribution:

$$
y _ {i} \sim \operatorname {P o i s s o n} \left(\lambda_ {y} (l (t))\right) \quad \text {f o r} i \in \mathbb {I} _ {\mathcal {T}}.
$$

Whereas for metabolic flux and balances, we model the predicted distributions as a Gaussian:

$$
y _ {i} \sim \mathcal {N} \left(\mu_ {y} (l (t)), \sigma_ {y} (l (t))\right) \quad \text {f o r} i \in \mathbb {I} _ {\mathcal {T}},
$$

where  $\lambda_y, \mu_y$  and  $\sigma_y$  are again learned functions. The decoding distributions are meant to capture the nature of the corresponding data. The output distribution is motivated by the nature of distribution that we observe in the datasets as seen in Figure 9. During inference, we use the learned  $\mathbf{f}_{\theta}$  to give latent values over unseen timesteps, from  $\mathcal{V}$ , as well.

# 3.3 OPTIMIZATION OBJECTIVE

Since the generative process is highly nonlinear, the true posterior is intractable. Thus, the model is trained using the amortized variational inference method using the evidence lower bound (ELBO):

$$
\mathbb {E} _ {q \left(l _ {0}, d \mid \mathcal {T}\right)} \left[ \sum_ {i \in \mathbb {I} _ {\mathcal {T}}} \log Y \left(y _ {i} \mid l _ {0}, d, t _ {i}\right) + \log \left(\frac {L _ {0} \left(l _ {0} \mid \mathcal {C}\right)}{L _ {0} \left(l _ {0} \mid \mathcal {T}\right)}\right) + \log \left(\frac {D (d \mid \mathcal {C})}{D (d \mid \mathcal {T})}\right) \right], \tag {2}
$$

where the expectation is taken over joint latent distribution  $q(l_0,d) = L_0(\theta_{l_0})\times D(\theta_d)$ .

# 4 EXPERIMENTS

# 4.1 EXPERIMENTAL SETTINGS

Datasets. We use the gene-expression time-series dataset from Ori et al. (2021), which investigates the differentiation of human pluripotent stem cells into lung and hepatocyte progenitors using single-cell RNA sequencing to map the transcriptional changes during this process. The gene-count matrix has dimensions  $10667 \times 26936$ , with 10667 cells and 26936 genes. The gene expression is counted regularly across 16 days in batches with  $\mathbb{B}_t$  being the index set of cells being counted on day  $t$  and  $|\mathbb{B}_0| + |\mathbb{B}_1| + \ldots + |\mathbb{B}_{15}| = 10667$ . For each cell batch  $\mathbb{B}_t$  and a given metabolic pathway, we only consider genes responsible for encoding the metabolites from the pathway. Table 1 summarizes the four metabolic pathways from Alghamdi et al. (2021) we considered in this study. Alghamdi et al. (2021) considered the metabolic reactions from the KEGG database (Kanehisa & Goto, 2000), import and export reactions, and reorganized them into modules based on the topological structure. This reorganization is, in essence, the simplification of the system of reactions by coercing connected reactions into a module. Thus, when we say flux or balance, we mean it with regard to a module.

Methods. We compare performances on model architectures, including neural process (NP) (Garnelo et al., 2018), neural ODE process (NODEP) (Norcliffe et al., 2021), and our structured neural

![](images/25ebe97986223d195d6d37836667cb1ed5ae77c7d4875c17522bae4ba2c349a1.jpg)  
Figure 3: Comparison of test-MSE in log-scale between NP, NODEP, and SNODEP across different metabolic pathways on ground-truth gene-expression time-series data.

![](images/80ad41ef5f1431a7712b8e89a7ab9e7b813786dad4f97a88cc2c981839edb00d.jpg)

![](images/9a6d5387846b0ef45fcc5bcb2d00887aec0867720f20eb3db65145bf8ed99e4a.jpg)

![](images/c27e1e22dff2b627578d4f4f7cd7ab51bd1308bef60967419ce324596e75820e.jpg)

ODE process (SNODEP). We treat NP architecture as a baseline model to get insights on modeling our problems as a differentiable random process without considering the underlying dynamics. We also compare performances between NODEP and SNODEP, which have a neural-ODE decoder, with the latter exploiting sequential relationships between the context points via its encoder.

Hyperparameters. We vary the context length on the largest metabolic pathway, M171, to specify the hyperparameter setup for context length and train-test splits (see Appendix D). We observe that setting the context length as 8 had a small test-MSE, corresponding to a  $80/20$  split for train and test timesteps available. Thus for the experiments below, we set our context as  $\mathbb{I}_{\mathcal{C}} = \{0,1,\dots ,8\}$  and our target as  $\mathbb{I}_{\mathcal{T}} = \{0,1,\dots ,12\}$ , while at inference, we predict for all the timesteps  $\mathbb{I}_{\mathcal{V}} = \{0,1,\dots ,15\}$ . Our training input is a sample  $y\sim \Pi_{t = 0}^{|\mathcal{T}|}Y(\theta_y(t))$  with context being  $y[0:|\mathcal{C}|]$  and target being  $y[0:|\mathcal{T}|]$ . And during testing, we sample  $y\sim \Pi_{t = 0}^{|\mathcal{V}|}Y(\theta_y(t))$ . For gene-knockout experiments, we set the train-test split of gene-knockout subsets as  $80 / 20$ .

Evaluation Metric. We use the MSE loss to measure model performance in predicting unseen timesteps of time-varying distributions. Let  $s \sim Y_s$  and  $s_* \sim Y_{s_*}$ , where  $Y_{s}$  is the learned distribution, and  $Y_{s_*}$  is the ground-truth distribution, where  $s, s_* \in \mathbb{R}$ . Let  $\mu_r$  and  $\mathrm{Var}(r)$  be the mean and variance for any random variable  $r$ . Then the Mean Squared Error (MSE) is given by:

$$
\mathbb {E} \left[ (\mu_ {s} - s _ {*}) ^ {2} \right] = \mathbb {E} \left[ \mu_ {s} ^ {2} + s _ {*} ^ {2} - 2 \mu_ {s} s _ {*} \right] = \mathrm {V a r} (s _ {*}) + (\mu_ {s} - \mu_ {s _ {*}}) ^ {2}.
$$

Assuming independence between dimensions,  $\mathrm{MSE} = \sum_{i=1}^{d} \mathrm{Var}(s_*, i) + (\mu_{s,i} - \mu_{s,*i})^2$  for  $s \in \mathbb{R}^d$ . For gene-expression data, assume  $Y_s = \mathrm{Poisson}(\lambda)$  and  $Y_{s_*} = \mathrm{Poisson}(\lambda_*)$ . We thus get  $\mathrm{MSE} = \sum_{i=1}^{d} [\lambda_{*,i} + (\lambda_i - \lambda_{*,i})^2]$ . Note that gene-expression data is usually very sparse (Figures 9a and 9b), and hence  $\lambda_g$  is usually very low. So in this case, minimizing MSE essentially boils down to getting as close to the Poisson approximation as possible. For metabolic flux and balance data, suppose  $Y_s = \mathcal{N}(\mu, \sigma^2)$  and  $Y_{s_*} = \mathcal{N}(\mu_* \sigma_*^2)$ . Then, we have  $\mathrm{MSE} = \sum_{i=1}^{d} [\sigma_{*,i}^2 + (\mu_i - \mu_{*,i})^2]$ . Since we observe the estimated flux and balance are of low variances (Figures 9c-9f), minimizing MSE essentially boils down to bringing the model mean  $\mu$  closer to ground-truth mean  $\mu_*$ .

# 4.2 GENE-EXPRESSION

Ideally, we would like to collect the ground-truth metabolic flux and balance at an individual cell or tissue level. However, this is difficult because there is very little data on them. Gene-expression counts can be considered as a rough approximation for the concentration of proteins, metabolites, and enzymes they encode since they are highly correlated. Specifically, mRNA molecules are transcribed at a certain rate from the template DNA strand, which are then translated into proteins at some rate. Thus, we explore the timestep prediction task on log-normalized and scaled gene-expression time-series data. Here,  $Y = G(\theta_g(t))$ , which is defined in Section 2.

From Figure 3, we can clearly see that SNODEP achieves much lower MSE across different pathways, showing the efficacy of our proposed SNODEP. Both setting the sampling distribution as Poisson and using the contextual information for the latent variables, in conjunction, help in obtaining better performance. Even though we are working with ground-truth gene expressions, this result should encourage further study on ground-truth flux datasets.

![](images/4bfa6f5e14a36212d3bc70b66a4a3b398d2613ff6420e6d7dd2cda16ebd61edc.jpg)  
(a) Predicting Metabolic-flux without Gene-knockout

![](images/89ae1ef560c61d05ad2c378abd6967d1d72a012a397b0e6b033b0ba34500bb05.jpg)

![](images/d6ca7fe422549818681ceeffd2b20fee0c6488ca3dae6dd9a05acd533f941b05.jpg)

![](images/02188143ed4ed09e5545116ba96fa5542e23fc49d4d30de64df61f41ce7f22ad.jpg)

![](images/983aa17bb9f2d73752cada1a0d54857d2e1b5e8f3c93da339e6bddafda393c8c.jpg)  
(b) Predicting Metabolic-flux with Gene-knockout

![](images/86b85bf097efae8a9df0739be2d2008fda6508d33b9e74f2eb3fa15e18ed5c1d.jpg)  
Figure 4: Comparison of test-MSE in log-scale between NP, NODEP, and SNODEP across different metabolic pathways on the scFEA-estimated metabolic-flux data with and without gene-knockout.

![](images/01f34634ecc54312617a287233f5522d336a06c87ac174a2881f366b236a38f2.jpg)

![](images/d176884314883349d267159d82c567959ee1658288cb04fb3e96cabee4c0afed.jpg)

# 4.3 METABOLIC-FLUX

Applying techniques from single-cell flux estimation analysis on the gene-expression data, we obtain samples of metabolic fluxes for the metabolic pathways. Specifically for each time  $t$ , given a metabolic pathway with  $d$  genes and  $u$  modules with gene-expression matrix  $\mathbf{G}_t \in \mathbb{R}^{d \times |\mathbb{B}_t|}$ , we get  $S_t^f(\mathbf{G}_t) = \mathbf{M}_t^f$ , where  $\mathbf{M}_t^f \in \mathbb{R}^{u \times |\mathbb{B}_t|}$  is the flux values for cell batch  $\mathbb{B}_t$ . From Figure 4a, we can observe that SNODEP performs generally better than the other two methods across different metabolic pathways, though the difference is not visually significant in some of them. We hypothesize that this is due to the nature of distributions as seen for some modules in Figures 9c and 9d, they have low variances, and if the mean of the distributions varies in an uncomplicated manner like linear or Markovian, incorporating the context in the latent is expected not to help much.

Gene-knockout. Gene-knockout experiments are meant to simulate the effect of disturbances in the pathway, such as the effect of any drug or environmental stress. Algorithm 1 in Appendix C describes the algorithmic form for our creation of knockout data. We model this by assuming that the gene expression level is correlated with how sensitive the metabolic pathway is with respect to the enzymes/proteins encoded by the gene. We consider  $k$  most-expressed genes in the dataset and sample random subsets of these  $k$  genes with the maximum cardinality of  $k // 2$ . We call these random subsets as knockout sets where the gene expression for the genes contained is set to zero. We again calculate flux samples using scFEA (Appendix B.2) corresponding to each of knockout set, with train and test containing data corresponding to different knockout sets. In our experiments, we set  $k = 20$  and the number of subsets  $S = 5$  for all pathways. Figure 4b shows that our methodology is robust to gene knockout predictions, and overall, SNODEP performs better than NP and NODEP. This validates that we can use our model to predict behaviors of unseen gene knockout configurations experiments and unseen timesteps.

# 4.4 METABOLIC-BALANCE

Once we get the flux values for all the modules, we can immediately obtain the change in concentration of a particular metabolite, known as the balance in flux balance analysis, by multiplying the flux with the stoichiometric matrix. We thus perform analogous experiments as in Section 4.3, where for each time  $t$  for a metabolic pathway with  $d$  genes and  $v$  metabolites with gene-expression matrix  $\mathbf{G}_t \in \mathbb{R}^{d \times |\mathbb{B}_t|}$  we get  $S_t^b(\mathbf{G}_t) = \mathbf{M}_t^b$ , with  $\mathbf{M}_t^b \in \mathbb{R}^{v \times |\mathbb{B}_t|}$  as defined in Section 2. Figure 5a

![](images/c38b4f4b491f60d73cd11b77c3bf3a55a0e21daa0d5d24aa585ea20a8bc34442.jpg)  
(a) Predicting Metabolic-balance without Gene-knockout

![](images/3c1a951f928cd2d9ae2b2ba70e1418d952be0743bf83697d38bf09c90e092138.jpg)

![](images/b68ed065635b33ae2f0adc736e21ee39c46d9070249f744417f2b541085d094f.jpg)

![](images/7713247d333baf41cd36808045b9af84abc2b23ff372ce66d9840525fb59f2a4.jpg)

![](images/153051466ae0ab415906364c185b24e349b7f3f62169ebefc921e8c05069532d.jpg)  
(b) Predicting Metabolic-balance with Gene-knockout

![](images/f163a03f9780e45a41539b88c189475867bcf4f6deb2ee35d457fb890a03fc6a.jpg)  
Figure 5: Comparison of test-MSE in log-scale between NODEP and SNODEP across metabolic pathways on scFEA-estimated metabolic-balance data with and without gene-knockout.

![](images/afbe2f7fb1b0a8ebee4bed571250055c722855a1edcb9f89f4a0997c6776fbf4.jpg)

![](images/3f832ee4be3915d7ad91b33ead001748711b6fcc1f777bbff002c49d0f21a248.jpg)

shows SNODEP generally outperforms NODEP, especially for the Iron Ions pathway. We believe that the performance is similar in pathways like MHC-i and M171 due to the simplistic nature of distributions (Figures 9e and 9f), akin to what we have mentioned in Section 4.3. Similar to the previous section, we follow the steps mentioned in Algorithm 1 to get the metabolic balance samples corresponding to gene knockout, and the test MSE is shown in Figure 5b. We can observe that the overall performance of SNODEP is better than that of NP and NODEP for all pathways.

# 4.5 IRREGULARLY SAMPLED TIMESTEPS

Data collection in experiments involving temporal profiling of gene expression is often performed irregularly (Rade et al., 2023; Nouri et al., 2023). Therefore, we also performed experiments where the points are irregularly sampled. To tackle the irregularity, we use GRU-ODE (Rubanova et al., 2019) to calculate latent distributions 3. Our context  $\mathbb{I}_{\mathcal{C}}$  and target  $\mathbb{I}_{\mathcal{T}}$  are similarly chosen to earlier sections, and we randomly set data from a fraction of timesteps to zero for each batch. During test time, we predict the remaining unseen timesteps. Figure 6 depicts heatmap visualizations of the difference between MSE of NODEP and SNODEP with GRU-ODE encoder. Entries in the heatmap with a positive value indicate that our SNODEP outperforms NODEP, and the higher the value is, the better the performance is. The negative values, where NODEP has a smaller test-MSE, are very low. We clearly see that SNODEP outperforms NODEP most of the time, especially towards lower frequencies, confirming the value of our model on irregularly sampled data.

# 5 CONCLUSION AND FUTURE WORK

In this work, we have shown how to get the time-varying metabolic flux of a system using genomics data rather than metabolomics data, which is much harder to procure. Through our framework, we intend to use the learned dynamics to generate quantities from future time steps and unseen gene-knockout configurations without any particular domain expertise. Nevertheless, we want to point out that our results with respect to flux and balance and their corresponding gene-knockout results are on data estimated via scFEA. Ideally, we would've preferred a gene-expression time-series that was sampled keeping metabolic pathways in mind, meaning time-series for normal conditions and several metabolic stresses, along with ground-truth metabolic flux and balance measurement. Such an experiment should also have the alternative DFBA formulation available so that we can benchmark

![](images/b36fe284231ce9b4f20e1743a894a7c1b88b56af72f5d33d6964779313621304.jpg)  
Figure 6: Heatmap of test-MSE difference  $(\times 10^{-2})$  between NODEP and SNODEP with GRU-ODE encoder across metabolic pathways for flux, balance, and their knockout versions. Frequency refers to the fraction of the timesteps present. In Appendix F, we provide the corresponding tables.

our method with it. However, we could not find such an open-sourced dataset, so we provided our evaluations on scFEA estimated values instead of an ideal real-world dataset. Apart from such an evaluation, several future directions could be taken, like making the scFEA methods differentiable, enabling a single end-to-end differentiable pipeline, incorporating hypergraph structure into them, modifying the loss and distribution appropriately for the sparsity of gene-expression data, and exploring non-parametric probability estimations for the decoder, to name a few. We believe our work can also be helpful for integrating genomic and metabolomic data by using our pre-trained framework to fine-tune metabolomic data, for example. In conclusion, we believe our work can serve as a starting point for several interesting directions in making metabolic analysis more scalable.

# REFERENCES

Mohammad Aghaee, Stephane Krau, Melih Tamer, and Hector Budman. Graph neural network representation of state space models of metabolic pathways. International Symposium on Advanced Control of Chemical Processes, 2024.  
Norah Alghamdi, Wennan Chang, Pengtao Dang, Xiaoyu Lu, Changlin Wan, Silpa Gampala, Zhi Huang, Jiashi Wang, Qin Ma, Yong Zang, et al. A graph neural network model to estimate cellwise metabolic flux using single-cell rna-seq data. Genome research, 31(10):1867-1884, 2021.  
Heba Askr, Enas Elgeldawi, Heba Aboul Ella, Yaseen AMM Elshaier, Mamdouh M Gomaa, and Aboul Ella Hassanien. Deep learning in drug discovery: an integrative review and future challenges. Artificial Intelligence Review, 56(7):5975-6037, 2023.  
Geng Chen, Baitang Ning, and Tieliu Shi. Single-cell rna-seq technologies and related computational data analysis. Frontiers in genetics, 10:317, 2019.  
Kyunghyun Cho. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
Gabriele Corso, Hannes Stärk, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. Diffdock: Diffusion steps, twists, and turns for molecular docking. arXiv preprint arXiv:2210.01776, 2022.  
Zak Costello and Hector Garcia Martin. A machine learning approach to predict metabolic pathway dynamics from time-series multiomics data. NPJ systems biology and applications, 4(1):1-14, 2018.  
Neil C Dalvie, Timothy Lorigeree, Andrew M Biedermann, Kerry R Love, and J Christopher Love. Simplified gene knockout by crispr-cas9-induced homologous recombination. ACS Synthetic Biology, 11(1):497-501, 2021.  
George B Dantzig. Linear programming. Operations research, 50(1):42-47, 2002.  
Jonas Degrave, Federico Felici, Jonas Buchli, Michael Neunert, Brendan Tracey, Francesco Carpanese, Timo Ewalds, Roland Hafner, Abbas Abdelmaleki, Diego de Las Casas, et al. Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602(7897):414-419, 2022.

Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Eslami, and Yee Whye Teh. Neural processes. arXiv preprint arXiv:1807.01622, 2018.  
S Hochreiter. Long short-term memory. Neural Computation MIT-Press, 1997.  
Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In International conference on machine learning, pp. 8867-8887. PMLR, 2022.  
Minoru Kanehisa and Susumu Goto. Kegg: kyoto encyclopedia of genes and genomes. Nucleic acids research, 28(1):27-30, 2000.  
Patrick Kidger, James Foster, Xuechen Li, and Terry J Lyons. Neural sdes as infinite-dimensional gans. In International conference on machine learning, pp. 5453-5463. PMLR, 2021.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Heidi E Klumpe, Jean-Baptiste Lugagne, Ahmad S Khalil, and Mary J Dunlop. Deep neural networks for predicting single-cell responses and probability landscapes. ACS Synthetic Biology, 12 (8):2367-2381, 2023.  
Gioele La Manno, Ruslan Soldatov, Amit Zeisel, Emelie Braun, Hannah Hochgerner, Viktor Petukhov, Katja Lidschreiber, Maria E Kastriti, Peter Lönnerberg, Alessandro Furlan, et al. RNA velocity of single cells. Nature, 560(7719):494-498, 2018.  
Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Graphcast: Learning skillful medium-range global weather forecasting. arXiv preprint arXiv:2212.12794, 2022.  
Ida Larsson, Mathias Uhlen, Cheng Zhang, and Adil Mardinoglu. Genome-scale metabolic modeling of glioblastoma reveals promising targets for drug development. Frontiers in genetics, 11: 381, 2020.  
Shitong Luo, Yufeng Su, Xingang Peng, Sheng Wang, Jian Peng, and Jianzhu Ma. Antigen-specific antibody design and optimization with diffusion-based generative models for protein structures. Advances in Neural Information Processing Systems, 35:9754-9767, 2022.  
Radhakrishnan Mahadevan, Jeremy S Edwards, and Francis J Doyle. Dynamic flux balance analysis of diauxic growth in escherichia coli. Biophysical journal, 83(3):1331-1340, 2002.  
Costas D Maranas and Ali R Zomorrodi. Optimization methods in metabolic networks. John Wiley & Sons, 2016.  
Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, and Aditya Grover. Climax: A foundation model for weather and climate. arXiv preprint arXiv:2301.10343, 2023.  
Alexander Norcliffe, Cristian Bodnar, Ben Day, Jacob Moss, and Pietro Lio. Neural ode processes. arXiv preprint arXiv:2103.12413, 2021.  
Nima Nouri, Raquel Cao, Eleonora Bunsow, Djamel Nehar-Belaid, Radu Marches, Zhaohui Xu, Bennett Smith, Santtu Heinonen, Sara Mertz, Amy Leber, Gaby Smits, Fiona van der Klis, Asuncion Mejias, Jacques Banchereau, Virginia Pascual, and Octavio Ramilo. Young infants display heterogeneous serological responses and extensive but reversible transcriptional changes following initial immunizations. Nature Communications, 14, 12 2023. doi: 10.1038/s41467-023-43758-2.  
Chaido Ori, Meshal Ansari, Ilias Angelidis, Fabian J. Theis, Herbert B. Schiller, and Micha Drukker. Single cell trajectory analysis of human pluripotent stem cells differentiating towards lung and hepatocyte progenitors. bioRxiv, 2021. doi: 10.1101/2021.02.23.432413.  
Jeffrey D Orth, Ines Thiele, and Bernhard Ø Palsson. What is flux balance analysis? Nature biotechnology, 28(3):245-248, 2010.

Michael Rade, Sebastian Böhlen, Vanessa Neuhaus, Dennis Löffler, Conny Blumert, Ulrike Köhl, Susann Dehmel, Katherina Sewald, and Kristin Reiche. A time-resolved meta-analysis of consensus gene expression profiles during human t-cell activation. *bioRxiv*, 2023. doi: 10.1101/2023.05.03.538418.  
Vytautas Raškevičius, Valeryia Mikalayeva, Ieva Antanaviciūte, Ieva Cesleviciené, Vytenis Arvydas Skeberdis, Visvaldas Kairys, and Sergio Bordel. Genome scale metabolic models as tools for drug design and personalized medicine. *PloS one*, 13(1):e0190636, 2018.  
Yulia Rubanova, Ricky TQ Chen, and David K Duvenaud. Latent ordinary differential equations for irregularly-sampled time series. Advances in neural information processing systems, 32, 2019.  
Ankur Sahu, Mary-Ann Blätke, Jedrzej Jakub Szymański, and Nadine Töpfer. Advances in flux balance analysis by integrating machine learning and mechanism-based models. Computational and structural biotechnology journal, 19:4626-4640, 2021.  
MH Jr Saier, CV Tran, and RD Barabote. Tcdb: the transporter classification database for membrane transport protein analyses and information. *Nucleic Acids Res*, 34(Database issue):D181–D186, Jan 2006. doi: 10.1093/nar/gkj001.  
Partho Sen and Matej Orešić. Integrating omics data in genome-scale metabolic modeling: A methodological perspective for precision medicine. Metabolites, 13(7):855, 2023.  
Mukund Thattai and Alexander Van Oudenaarden. Intrinsic noise in gene regulatory networks. Proceedings of the National Academy of Sciences, 98(15):8614-8619, 2001.  
Ingrid von Glehn, James S Spencer, and David Pfau. A self-attention ansatz for ab-initio quantum chemistry. arXiv preprint arXiv:2211.13672, 2022.
