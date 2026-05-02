# NASI: LABEL- AND DATA-AGNOSTIC NEURAL ARCHITECTURE SEARCH AT INITIALIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recent years have witnessed a surging interest in Neural Architecture Search (NAS). Various algorithms have been proposed to improve the search efficiency and effectiveness of NAS, i.e., to reduce the search cost and improve the generalization performance of the selected architectures, respectively. However, the search efficiency of these algorithms is severely limited by the need for model training during the search process. To overcome this limitation, we propose a novel NAS algorithm called NAS at Initialization (NASI) that exploits the capability of a Neural Tangent Kernel in being able to characterize the performance of candidate architectures at initialization, hence allowing model training to be completely avoided to boost the search efficiency. Besides the improved search efficiency, NASI also achieves competitive search effectiveness on various datasets like CIFAR-10/100 and ImageNet. Further, NASI is shown to be label- and data-agnostic under mild conditions, which guarantees the transferability of architectures selected by our NASI over different datasets.

# 1 INTRODUCTION

The past decade has witnessed the wide success of deep neural networks (DNNs) in computer vision and natural language processing. These DNNs, e.g., VGG (Simonyan & Zisserman, 2015), ResNet (He et al., 2016), and MobileNet (Howard et al., 2017), are typically handcrafted by human experts with considerable trials and errors. The human efforts devoting to the design of these DNNs are, however, not affordable nor scalable due to an increasing demand of customizing DNNs for different tasks. To reduce such human efforts, Neural Architecture Search (NAS) (Zoph & Le, 2017) has recently been introduced to automate the design of DNNs. As summarized in (Elsken et al., 2019), NAS conventionally consists of a search space, a search algorithm, and a performance evaluation. Specifically, the search algorithm aims to select the best-performing neural architecture from the search space based on its evaluated performance via performance evaluation. In the literature, various search algorithms (Luo et al., 2018; Zoph et al., 2018; Real et al., 2019) have been proposed to search for architectures with comparable or even better performance than the handcrafted ones.

However, these NAS algorithms are inefficient due to the requirement of model training for numerous candidate architectures during the search process. To improve the search inefficiency, one-shot NAS algorithms (Dong & Yang, 2019; Pham et al., 2018; Liu et al., 2019; Xie et al., 2019) have trained a single one-shot architecture and then evaluated the performance of candidate architectures with model parameters inherited from this fine-tuned one-shot architecture. So, these algorithms can considerably reduce the cost of model training, but still require the training of the one-shot architecture. This naturally leads to the question whether NAS is realizable at initialization such that model training can be completely avoided during the search process? To the best of our knowledge, only a few efforts to date have been devoted to developing NAS algorithms without model training empirically (Mellor et al., 2020; Park et al., 2020; Abdelfattah et al., 2021; Chen et al., 2021).

This paper presents a novel NAS algorithm called NAS at Initialization (NASI) that can completely avoid model training to boost search efficiency. To achieve this, NAS exploits the capability of a Neural Tangent Kernel (NTK) (Jacot et al., 2018; Lee et al., 2019a) in being able to formally characterize the performance of infinite-wide DNNs at initialization, hence allowing the performance of candidate architectures to be estimated and realizing NAS at initialization. Specifically, given the estimated performance of candidate architectures by NTK, NAS can be reformulated into an

optimization problem without model training (Sec. 3.1). However, NTK is prohibitively costly to evaluate. Fortunately, we can approximate it with a similar form to gradient flow (Wang et al., 2020) (Sec. 3.2). This results in a reformulated NAS problem that can be solved efficiently by a gradient-based algorithm via additional relaxation with Gumbel-Softmax (Jang et al., 2017; Maddison et al., 2017) (Sec. 3.3). Interestingly, NASI is shown to be label- and data-agnostic under mild conditions, which thus implies the transferability of architectures selected by NASI over different datasets (Sec. 4).

We will firstly empirically demonstrate the improved search efficiency and the competitive search effectiveness achieved by NASI in NAS-Bench-1Shot1 (Zela et al., 2020b) (Sec. 5.1). Compared with other NAS algorithms, NASI incurs the smallest search cost while preserving the competitive performance of its selected architectures. Meanwhile, the architectures selected by NASI from the DARTS (Liu et al., 2019) search space over CIFAR-10 consistently enjoy the competitive or even outperformed performance when evaluated on different benchmark datasets, e.g., CIFAR-10/100 and ImageNet (Sec. 5.2), indicating the guaranteed transferability of architectures selected by our NASI. In Sec. 5.3, NASI is further demonstrated to be able to select well-performing architectures on CIFAR-10 even with randomly generated labels or data, which strongly supports the label- and data-agnostic search and also the guaranteed transferability achieved by our NASI.

# 2 RELATED WORKS AND BACKGROUND

# 2.1 NEURAL ARCHITECTURE SEARCH

A growing body of NAS algorithms have been proposed in the literature (Zoph & Le, 2017; Liu et al., 2018; Luo et al., 2018; Zoph et al., 2018; Real et al., 2019) to automate the design of neural architectures. However, scaling existing NAS algorithms to large datasets is notoriously hard. Recently, attention has thus been shifted to improving the search efficiency of NAS without sacrificing the generalization performance of its selected architectures. In particular, a one-shot architecture is introduced by Pham et al. (2018) to share model parameters among candidate architectures, thereby reducing the cost of model training substantially. Recent works (Chen et al., 2019; Dong & Yang, 2019; Liu et al., 2019; Xie et al., 2019; Chen & Hsieh, 2020; Chu et al., 2020) along this line have further formulated NAS as a continuous and differentiable optimization problem to yield efficient gradient-based solutions. These one-shot NAS algorithms have achieved considerable improvement in search efficiency. However, the model training of the one-shot architecture is still needed.

More recently, a number of algorithms have been proposed to estimate the performance of candidate architectures without model training. For example, Mellor et al. (2020) have explored the correlation between the divergence of linear maps induced by data points at initialization and the performance of candidate architectures heuristically. Meanwhile, Park et al. (2020) have approximated the performance of candidate architectures by the performance of their corresponding Neural Network Gaussian Process (NNGP) with only initialized model parameters, which is yet computationally costly. Abdelfattah et al. (2021) have investigated several training-free proxies to rank candidate architectures in the search space, while Chen et al. (2021) intuitively adopt theoretical aspects in deep networks (e.g., NTK (Jacot et al., 2018) and linear regions of deep networks (Raghu et al., 2017)) to select architectures with a good trade-off between its trainability and expressivity. Our NASI significantly advances this line of work in (a) providing theoretically grounded performance estimation by NTK (compared with (Mellor et al., 2020; Abdelfattah et al., 2021; Chen et al., 2021)), (b) guaranteeing the transferability of its selected architectures with its provable label- and data-agnostic search under mild conditions (compared with (Mellor et al., 2020; Park et al., 2020; Abdelfattah et al., 2021; Chen et al., 2021))) and (c) achieving SOTA performance in a large search space over various benchmark datasets (compared with (Mellor et al., 2020; Park et al., 2020; Abdelfattah et al., 2021)).

# 2.2 NEURAL TANGENT KERNEL (NTK)

Let a dataset  $(\mathcal{X},\mathcal{Y})$  denote a pair comprising a set  $\mathcal{X}$  of  $m$ $n_0$ -dimensional vectors of input features and a vector  $\mathcal{V}\in \mathbb{R}^{mn\times 1}$  concatenating the  $m$ $n$ -dimensional vectors of corresponding output values, respectively. Let a DNN be parameterized by  $\pmb{\theta}_t\in \mathbb{R}^p$  at time  $t$  and output a vector

$\pmb{f}(\mathcal{X}; \pmb{\theta}_t) \in \mathbb{R}^{mn \times 1}$  (abbreviated to  $\pmb{f}_t$ ) of the predicted values of  $\mathcal{V}$ . Jacot et al. (2018) have revealed that the training dynamics of DNNs with gradient descent can be characterized by an NTK. Formally, define the NTK  $\Theta_t(\mathcal{X}, \mathcal{X}) \in \mathbb{R}^{mn \times mn}$  (abbreviated to  $\Theta_t$ ) as

$$
\boldsymbol {\Theta} _ {t} (\mathcal {X}, \mathcal {X}) \triangleq \nabla_ {\boldsymbol {\theta} _ {t}} \boldsymbol {f} (\mathcal {X}; \boldsymbol {\theta} _ {t}) \nabla_ {\boldsymbol {\theta} _ {t}} \boldsymbol {f} (\mathcal {X}; \boldsymbol {\theta} _ {t}) ^ {\top}. \tag {1}
$$

Given a loss function  $\mathcal{L}_t$  at time  $t$  and a learning rate  $\eta$ , the training dynamics of the DNN can then be characterized as

$$
\nabla_ {t} \boldsymbol {f} _ {t} = - \eta \Theta_ {t} (\mathcal {X}, \mathcal {X}) \nabla_ {\boldsymbol {f} _ {t}} \mathcal {L} _ {t}, \quad \nabla_ {t} \mathcal {L} _ {t} = - \eta \nabla_ {\boldsymbol {f} _ {t}} \mathcal {L} _ {t} ^ {\top} \Theta_ {t} (\mathcal {X}, \mathcal {X}) \nabla_ {\boldsymbol {f} _ {t}} \mathcal {L} _ {t}. \tag {2}
$$

Interestingly, as proven in (Jacot et al., 2018), the NTK stays asymptotically constant during the course of training as the width of DNNs goes to infinity. NTK at initialization (i.e.,  $\Theta_0$ ) can thus characterize the training dynamics and also the performance of infinite-width DNNs.

Lee et al. (2019a) have further revealed that, for DNNs with over-parameterization, the aforementioned training dynamics can be governed by their first-order Taylor expansion (or linearization) at initialization. In particular, define

$$
\boldsymbol {f} ^ {\text {l i n}} \left(\boldsymbol {x}; \boldsymbol {\theta} _ {t}\right) \triangleq \boldsymbol {f} \left(\boldsymbol {x}; \boldsymbol {\theta} _ {0}\right) + \nabla_ {\boldsymbol {\theta} _ {0}} \boldsymbol {f} \left(\boldsymbol {x}; \boldsymbol {\theta} _ {0}\right) ^ {\top} \left(\boldsymbol {\theta} _ {t} - \boldsymbol {\theta} _ {0}\right) \tag {3}
$$

for all  $\pmb{x} \in \mathcal{X}$ . Then,  $\pmb{f}(\pmb{x};\pmb{\theta}_t)$  and  $\pmb{f}^{\mathrm{lin}}(\pmb{x};\pmb{\theta}_t)$  share similar training dynamics over time, as described formally in Appendix A.2. Besides, following the definition of NTK in (1), this linearization  $\pmb{f}^{\mathrm{lin}}$  achieves a constant NTK over time.

Given the mean squared error (MSE) loss defined as  $\mathcal{L}_t\triangleq m^{-1}\| \mathcal{Y} - \pmb {f}(\mathcal{X};\pmb {\theta}_t)\| _2^2$  and the constant  $\mathrm{NTK}\Theta_t = \Theta_0$ , the loss dynamics in (2) above can be analyzed in a closed form while applying gradient descent with learning rate  $\eta$  (Arora et al., 2019):

$$
\mathcal {L} _ {t} = m ^ {- 1} \sum_ {i = 1} ^ {m n} (1 - \eta \lambda_ {i}) ^ {2 t} \left(\boldsymbol {u} _ {i} ^ {\top} \mathcal {Y}\right) ^ {2}, \tag {4}
$$

where  $\Theta_0 = \sum_{i=1}^{mn} \lambda_i(\Theta_0) \pmb{u}_i \pmb{u}_i^\top$ , and  $\lambda_i(\Theta_0)$  and  $\pmb{u}_i$  denote the  $i$ -th largest eigenvalue and the corresponding eigenvector of  $\Theta_0$ , respectively.

# 3 NEURAL ARCHITECTURE SEARCH AT INITIALIZATION

# 3.1 REFORMULATING NAS VIA NTK

Given a loss function  $\mathcal{L}$  and model parameters  $\theta(\mathcal{A})$  of architecture  $\mathcal{A}$ , we denote the training and validation loss as  $\mathcal{L}_{\mathrm{train}}$  and  $\mathcal{L}_{\mathrm{val}}$ , respectively. NAS is conventionally formulated as a bi-level optimization problem (Liu et al., 2019):

$$
\min  _ {\mathcal {A}} \mathcal {L} _ {\text {v a l}} \left(\boldsymbol {\theta} ^ {*} (\mathcal {A}); \mathcal {A}\right)
$$

$$
\text {s . t .} \theta^ {*} (\mathcal {A}) \triangleq \arg \min  _ {\theta (\mathcal {A})} \mathcal {L} _ {\text {t r a i n}} (\theta (\mathcal {A}); \mathcal {A}). \tag {5}
$$

Notably, model training is required to evaluate the validation performance of each candidate architecture in (5). The search efficiency of NAS algorithms (Real et al., 2019; Zoph et al., 2018) based on (5) is thus severely limited by the cost of model training for each candidate architecture. Though recent works (Pham et al., 2018) have considerably reduced this training cost by introducing a one-shot architecture for model parameter sharing, such a one-shot architecture requires training and hence incurs the training cost.

To completely avoid this training cost, we exploit the capability of NTK for characterizing the performance of DNNs at initialization. Specifically, Sec. 2.2 has revealed that the training dynamics of an over-parameterized DNN can be governed by its linearization at initialization. With the MSE loss, the training dynamics of such linearization are further determined by its constant NTK. Therefore, the training dynamics and hence the performance of a DNN can be characterized by the constant NTK of its linearization. However, this constant NTK is computationally costly to evaluate. To this end, we instead characterize the training dynamics (i.e., MSE) of DNNs in Proposition 1 using the trace norm of NTK at initialization, which can be efficiently approximated. For simplicity, we use this MSE loss in our analysis. Other widely adopted loss functions (e.g., cross entropy with softmax) can also be applied, as supported in our experiments. Note that throughout this paper, the parameterization and initialization of DNNs follow that of Jacot et al. (2018). For a  $L$ -layer DNN, we denote the output dimension of its hidden layers and the last layer as  $n_1 = \dots = n_{L-1} = k$  and  $n_L = n$ , respectively.

Proposition 1. Suppose that  $\| \pmb{x}\| _2\leq 1$  for all  $\pmb {x}\in \mathcal{X}$  and  $\mathcal{V}\in [0,1]^{mn}$  for a given dataset  $(\mathcal{X},\mathcal{Y})$  of size  $|\mathcal{X}| = m$ , a given  $L$ -layer neural architecture  $\mathcal{A}$  outputs  $\pmb {f}_t\in [0,1]^m n$  as predicted labels of  $\mathcal{V}$  with the corresponding MSE loss  $\mathcal{L}_t$ ,  $\lambda_{\mathrm{min}}(\Theta_0) > 0$  for the given NTK  $\Theta_0$  w.r.t.  $\pmb{f}_{t}$  at initialization, and gradient descent (or gradient flow) is applied with learning rate  $\eta < \lambda_{\mathrm{max}}^{-1}(\Theta_0)$ . Then, for any  $t\geq 0$ , there exists a constant  $c_{0} > 0$  such that as  $k\to \infty$

$$
\mathcal {L} _ {t} \leq m n ^ {2} \left(1 - \eta \bar {\lambda} \left(\Theta_ {0}\right)\right) ^ {q} + \epsilon \tag {6}
$$

with probability arbitrarily close to 1 where  $q$  is set to 2t if  $t < 0.5$ , and 1 otherwise,  $\overline{\lambda}(\Theta_0) \triangleq (mn)^{-1} \sum_{i=1}^{mn} \lambda_i(\Theta_0)$ , and  $\epsilon \triangleq 2c_0 \sqrt{n/(mk)} \left(1 + c_0 \sqrt{1/k}\right)$ .

Its proof is in Appendix A.3. Proposition 1 implies that NAS can be realizable at initialization. Specifically, given a fixed and sufficiently large training budget  $t$ , in order to select the best-performing architecture, we can simply minimize the upper bound of  $\mathcal{L}_t$  in (6) over all the candidate architectures in the search space. Here,  $\mathcal{L}_t$  can be applied to approximated  $\mathcal{L}_{\mathrm{val}}$  since both strong theoretical (Mohri et al., 2018) and empirical (Hardt et al., 2016) justifications in the literature have shown that training and validation loss are generally highly related. Hence, (5) can be reformulated as

$$
\min  _ {\mathcal {A}} m n ^ {2} \left(1 - \eta \bar {\lambda} \left(\Theta_ {0} (\mathcal {A})\right)\right) + \epsilon \quad \text {s . t .} \bar {\lambda} \left(\Theta_ {0} (\mathcal {A})\right) <   \eta^ {- 1}. \tag {7}
$$

Note that the constraint in (7) is derived from the condition  $\eta < \lambda_{\max}^{-1}(\Theta_0(\mathcal{A}))$  in Proposition 1, and  $\eta$  and  $\epsilon$  are typically constants $^2$  during the search process. Following the definition of trace norm, (7) can be further reduced into

$$
\max  _ {\mathcal {A}} \| \Theta_ {0} (\mathcal {A}) \| _ {\mathrm {t r}} \quad \text {s . t .} \| \Theta_ {0} (\mathcal {A}) \| _ {\mathrm {t r}} <   m n \eta^ {- 1}. \tag {8}
$$

Notably,  $\Theta_0(\mathcal{A})$  only relies on the initialization of  $\mathcal{A}$ . So, no model training is required in optimizing (8), which achieves our objective of realizing NAS at initialization.

Furthermore, (8) suggests an interesting interpretation of NAS: NAS intends to select architectures with a good trade-off between their model complexity and the optimization behavior in their model training. Particularly, architectures containing more model parameters will usually achieve a larger  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  according to the definition in (1), which hence provides an alternative to measuring the complexity of architectures. So, maximizing  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  leads to architectures with large complexity and therefore strong representation power. On the other hand, the complexity of the selected architectures is limited by the constraint in (8) to ensure a well-behaved optimization with a large learning rate  $\eta$  in their model training. By combining these two effects, the optimization of (8) naturally trades off between the complexity of the selected architectures and the optimization behavior in their model training for the best performance. Appendix C.1 will validate such trade-off. Interestingly, Chen et al. (2021) have revealed a similar insight of NAS to us.

# 3.2 APPROXIMATING THE TRACE NORM OF NTK

The optimization of our reformulated NAS in (8) requires the evaluation of  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  for each architecture  $\mathcal{A}$  in the search space, which can be obtained by

$$
\left\| \boldsymbol {\Theta} _ {0} (\mathcal {A}) \right\| _ {\mathrm {t r}} = \sum_ {\boldsymbol {x} \in \mathcal {X}} \left\| \nabla_ {\boldsymbol {\theta} _ {0} (\mathcal {A})} \boldsymbol {f} \left(\boldsymbol {x}, \boldsymbol {\theta} _ {0} (\mathcal {A})\right) \right\| _ {\mathrm {F}} ^ {2}, \tag {9}
$$

where  $\| \cdot \|_{\mathrm{F}}$  denotes the Frobenius norm. However, the Frobenius norm of the Jacobian matrix in (9) is costly to evaluate. So, we propose to approximate this term. Specifically, given a  $\gamma$ -Lipschitz continuous loss function  $\mathcal{L}_{\pmb{x}}$  (i.e.,  $\| \nabla_{\pmb{f}}\mathcal{L}_{\pmb{x}}\|_{2}\leq \gamma$  for all  $\pmb {x}\in \mathcal{X}$ ),

$$
\left. \gamma^ {- 1} \left\| \nabla_ {\boldsymbol {\theta} _ {0} (\mathcal {A})} \mathcal {L} _ {\boldsymbol {x}} \right\| _ {2} = \gamma^ {- 1} \left\| \nabla_ {\boldsymbol {f}} \mathcal {L} _ {\boldsymbol {x}} ^ {\top} \nabla_ {\boldsymbol {\theta} _ {0} (\mathcal {A})} \boldsymbol {f} (\boldsymbol {x}, \boldsymbol {\theta} _ {0} (\mathcal {A})) \right\| _ {2} \leq \left\| \nabla_ {\boldsymbol {\theta} _ {0} (\mathcal {A})} \boldsymbol {f} (\boldsymbol {x}, \boldsymbol {\theta} _ {0} (\mathcal {A})) \right\| _ {\mathrm {F}} . \right. \tag {10}
$$

The Frobenius norm  $\left\| \nabla_{\pmb{\theta}_0(\mathcal{A})}\pmb {f}(\pmb {x},\pmb {\theta}_0(\mathcal{A}))\right\|_{\mathrm{F}}$  can therefore be approximated efficiently by its lower bound given in (10) through automatic differentiation (Baydin et al., 2017).

Meanwhile, the evaluation of  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  in (9) requires iterating over the entire dataset of size  $m$ , which incurs  $\mathcal{O}(m)$  time. Fortunately, this incurred time can be reduced by parallelization over

multi-batches. Let the set  $\mathcal{X}_j$  denote the input feature vectors of the  $j$ -th randomly sampled mini-batches of size  $|\mathcal{X}_j| = b$ . By combining (9) and (10),

$$
\left\| \boldsymbol {\Theta} _ {0} (\mathcal {A}) \right\| _ {\mathrm {t r}} \geq \gamma^ {- 1} \sum_ {\boldsymbol {x} \in \mathcal {X}} \left\| \nabla_ {\boldsymbol {\theta} _ {0} (\mathcal {A})} \mathcal {L} _ {\boldsymbol {x}} \right\| _ {2} ^ {2} \geq b \gamma^ {- 1} \sum_ {j = 1} ^ {m / b} \left\| b ^ {- 1} \sum_ {\boldsymbol {x} \in \mathcal {X} _ {j}} \nabla_ {\boldsymbol {\theta} _ {0} (\mathcal {A})} \mathcal {L} _ {\boldsymbol {x}} \right\| _ {2} ^ {2}, \tag {11}
$$

where the last inequality follows from Jensen's inequality. Note that (11) provides an approximation of  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  incurring  $\mathcal{O}(m / b)$  time because the gradients within a mini-batch can be evaluated in parallel. Moreover, we further approximate the summation over  $m / b$  mini-batches in (11) by one single uniformly randomly sampled mini-batch  $\mathcal{X}_j$ . Formally, under the definition of  $\| \widetilde{\Theta}_0(\mathcal{A})\|_{\mathrm{tr}} \triangleq \| b^{-1}\sum_{\boldsymbol{x} \in \mathcal{X}_j}\nabla_{\theta_0(\mathcal{A})}\mathcal{L}_{\boldsymbol{x}}\| _2^2$ , our final approximation of  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  becomes

$$
\left\| \Theta_ {0} (\mathcal {A}) \right\| _ {\mathrm {t r}} \approx m \gamma^ {- 1} \left\| \widetilde {\Theta} _ {0} (\mathcal {A}) \right\| _ {\mathrm {t r}}. \tag {12}
$$

This final approximation incurs only  $\mathcal{O}(1)$  time and can effectively characterize the performance of neural architectures, as demonstrated in our experiments. Interestingly, a similar form called gradient flow (Wang et al., 2020) has also been applied in network pruning at initialization.

# 3.3 OPTIMIZATION AND SEARCH ALGORITHM

The approximation of  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  in Sec. 3.2 engenders an efficient optimization of our reformulated NAS in (8): Firstly, we apply a penalty method to transform (8) into an unconstrained optimization problem. Given a penalty coefficient  $\mu$  and an exterior penalty function  $F(x)\triangleq \max (0,x)$  with a pre-defined constant  $\nu \triangleq \gamma n\eta^{-1}$ , and a randomly sampled mini-batch  $\mathcal{X}_j$ , by replacing  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  with the approximation in (12), our reformulated NAS problem (8) can be transformed into

$$
\max  _ {\mathcal {A}} \left[ \| \widetilde {\Theta} _ {0} (\mathcal {A}) \| _ {\mathrm {t r}} - \mu F \left(\| \widetilde {\Theta} _ {0} (\mathcal {A}) \| _ {\mathrm {t r}} - \nu\right) \right]. \tag {13}
$$

Interestingly, (13) implies that the complexity of the final selected architectures is limited by not only the constraint  $\nu$  (discussed in Sec. 3.1) but also the penalty coefficient  $\mu$ : For a fixed constant  $\nu$ , a larger  $\mu$  imposes a stricter limitation on the complexity of architectures (i.e.,  $\|\widetilde{\Theta}_0(\mathcal{A})\|_{\mathrm{tr}} < \nu$ ) in the optimization of (13).

The optimization of (13) in the discrete search space, however, is intractable. So, we apply some optimization tricks to simplify it: Following that of Pham et al. (2018); Liu et al. (2019); Xie et al. (2019), we represent the search space as a one-shot architecture such that the candidate architectures are subgraphs of this one-shot architecture. Next, instead of optimizing (13), we introduce a distribution  $p_{\alpha}(\mathcal{A})$  (parameterized by  $\alpha$ ) over the candidate architectures in this search space like that in (Zoph & Le, 2017; Pham et al., 2018; Xie et al., 2019), and optimize the expected performance of architectures sampled from  $p_{\alpha}(\mathcal{A})$ :

$$
\max  _ {\boldsymbol {\alpha}} \mathbb {E} _ {\mathcal {A} \sim p _ {\boldsymbol {\alpha}} (\mathcal {A})} [ R (\mathcal {A}) ] \quad \text {s . t .} R (\mathcal {A}) \triangleq \| \widetilde {\Theta} _ {0} (\mathcal {A}) \| _ {\mathrm {t r}} - \mu F \left(\| \widetilde {\Theta} _ {0} (\mathcal {A}) \| _ {\mathrm {t r}} - \nu\right). \tag {14}
$$

Then, we apply Gumbel-Softmax (Jang et al., 2017; Maddison et al., 2017) to relax the optimization of (14) to be continuous and differentiable using the reparameterization trick. Specifically, for a given  $\alpha$ , to sample an architecture  $\mathcal{A}$ , we simply have to sample  $\pmb{g}$  from  $p(\pmb{g}) = \mathrm{Gumbel}(\mathbf{0},\mathbf{1})$  and then determine  $\mathcal{A}$  using  $\alpha$  and  $\pmb{g}$  (more details in Appendix B.3). Consequently, (14) can be transformed into

$$
\left. \max  _ {\boldsymbol {\alpha}} \mathbb {E} _ {\boldsymbol {g} \sim p (\boldsymbol {g})} \left[ R \left(\mathcal {A} \left(\boldsymbol {\alpha}, \boldsymbol {g}\right)\right) \right]. \right. \tag {15}
$$

After that, we approximate (15) based on its first-order Taylor expansion at initialization such that it can be optimized efficiently through a gradient-based algorithm. In particular, given the first-order approximation within the  $\xi$ -neighborhood of initialization  $\alpha_0$  (i.e.,  $\| \Delta \|_2 \leq \xi$ ):

$$
\mathbb {E} _ {\boldsymbol {g} \sim p (\boldsymbol {g})} \left[ R \left(\mathcal {A} \left(\boldsymbol {\alpha} _ {0} + \Delta , \boldsymbol {g}\right)\right) \right] \approx \mathbb {E} _ {\boldsymbol {g} \sim p (\boldsymbol {g})} \left[ R \left(\mathcal {A} \left(\boldsymbol {\alpha} _ {0}, \boldsymbol {g}\right)\right) + \nabla_ {\boldsymbol {\alpha} _ {0}} R \left(\mathcal {A} \left(\boldsymbol {\alpha} _ {0}, \boldsymbol {g}\right)\right) ^ {\top} \Delta \right], \tag {16}
$$

the maximum of (16) is achieved when

$$
\Delta^ {*} = \underset {\| \Delta \| _ {2} \leq \xi} {\arg \max } \mathbb {E} _ {\boldsymbol {g} \sim p (\boldsymbol {g})} \left[ \nabla_ {\boldsymbol {\alpha} _ {0}} R (\mathcal {A} (\boldsymbol {\alpha} _ {0}, \boldsymbol {g})) ^ {\top} \Delta \right] = \xi \mathbb {E} _ {\boldsymbol {g} \sim p (\boldsymbol {g})} \left[ \frac {\nabla_ {\boldsymbol {\alpha} _ {0}} R (\mathcal {A} (\boldsymbol {\alpha} _ {0} , \boldsymbol {g}))}{\left\| \mathbb {E} _ {\boldsymbol {g} \sim p (\boldsymbol {g})} [ \nabla_ {\boldsymbol {\alpha} _ {0}} R (\mathcal {A} (\boldsymbol {\alpha} _ {0} , \boldsymbol {g})) ] \right\| _ {2}} \right]. \tag {17}
$$

# Algorithm 1 NAS at Initialization (NASI)

1: Input: dataset  $\mathcal{D} \triangleq (\mathcal{X}, \mathcal{Y})$ , batch size  $b$ , steps  $T$ , penalty coefficient  $\mu$ , constraint constant  $\nu$ , initialized model parameters  $\theta_0$  for one-shot architecture and distribution  $p_{\alpha_0}(\mathcal{A})$  with initialization  $\alpha_0 = 0$ , set  $\xi = 1$  
2: for step  $t = 1,\dots ,T$  do  
3: Sample data  $\mathcal{D}_t\sim \mathcal{D}$  of size  $b$  
4: Sample  $\pmb{g}_t\sim p(\pmb {g}) = \mathrm{Gumbel}(\mathbf{0},\mathbf{1})$  and determine sampled architecture  $\mathcal{A}_t$  based on  $\alpha_0,\pmb {g}_t$  
5: Evaluate gradient  $G_{t} = \nabla_{\alpha_{0}}R(\mathcal{A}_{t})$  with data  $\mathcal{D}_t$  
6: end for  
7: Estimate  $\Delta^{*}$  with (18) and get  $\alpha^{*} = \alpha_{0} + \Delta^{*}$  
8: Select architecture  $\mathcal{A}^* = \arg \max_{\mathcal{A}} p_{\alpha^*}(\mathcal{A})$

![](images/5880192b39303f9c5b5ac01ed116e7384fdff4228258df23acc5ff058fe57f6b.jpg)  
(a) Label-agnostic

![](images/8256f4c2f0d6b5090988b360bf62102d7693847ca2c81feea9c0d99bf2ea6d6c.jpg)  
Figure 1: Comparison of the approximated  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  following (12) in the three search spaces of NAS-Bench-1Shot1 (Zela et al., 2020b) on CIFAR-10 (a) between random vs. true labels, and (b) between random vs. true data. Each pair  $(x,y)$  denotes the approximation of one candidate architecture in the search space with true vs. random labels (or data), respectively. The trends of these approximations are further illustrated by the lines in orange. In addition, Pearson correlation coefficient  $\rho$  of the approximations with random vs. true labels (or data) is given in the corner.

![](images/3a449e157c1220f3863e7ff5ece602eddc44442c93c55b3f98d6f3551ef8d0c6.jpg)

![](images/5f2e8de1a4dd403b5922fcfe76b697359c19e5911a501f7c82b404dbf0e2fdbc.jpg)  
(b) Data-agnostic

![](images/dddaa662be79bc9dc4da752f47136083a113a35efe594dce33cced55abe50cf8.jpg)

![](images/d0b8f4faee6d675139d87520e9a6c1e21fd6dad04d417c7b8639cbf9fad43bc3.jpg)

The closed-form solution in (17) follows from the definition of dual norm and requires only a one-step optimization, i.e., without the iterative update of  $\Delta$ . Similar one-step optimizations have also been adopted by other works (Goodfellow et al., 2015; Wang et al., 2020).

Unfortunately, the expectation in (17) makes the evaluation of  $\Delta^{*}$  intractable. Monte Carlo sampling is thus applied to estimate  $\Delta^{*}$  efficiently: Given  $T$  sequentially sampled  $\pmb{g}$  (i.e.,  $\pmb{g}_1,\dots ,\pmb{g}_T$ ) and let  $G_{i}\triangleq \nabla_{\alpha_{0}}R(\mathcal{A}(\pmb{\alpha}_{0},\pmb{g}_{i}))$ ,  $\Delta^{*}$  can be approximated as

$$
\Delta^ {*} \approx \frac {\xi}{T} \sum_ {t = 1} ^ {T} \frac {G _ {t}}{\operatorname* {m a x} \left(\left\| G _ {1} \right\| _ {2} , \dots , \left\| G _ {t} \right\| _ {2}\right)}. \tag {18}
$$

Note that the expectation  $\left\| \mathbb{E}_{g \sim p(g)}[\nabla_{\alpha_0} R(\mathcal{A}(\alpha_0, g))]\right\|_2$  in (17) has been approximated by  $\max(\|G_1\|_2, \ldots, \|G_t\|_2)$  in (18) for the sample of  $g$  at time  $t$ , which is somehow inspired by AMSGrad (Reddi et al., 2018). Interestingly, this approximation is non-decreasing in  $t$  and therefore achieves a similar effect of learning rate decay, which may lead to a better-behaved optimization of  $\Delta^*$ . With the optimal  $\Delta^*$  and  $\alpha^* = \alpha_0 + \Delta^*$ , the final architecture can then be selected as  $\mathcal{A}^* \triangleq \arg \max_A p_{\alpha^*}(\mathcal{A})$ , which completes our NAS at Initialization (NASI) algorithm detailed in Algorithm 1. Interestingly, this simple and efficient solution in (18) can already allow us to select architectures with competitive performances, as shown in our experiments (Sec. 5).

# 4 LABLE- AND DATA-AGNOSTIC SEARCH OF NASI

Besides the improved search efficiency by completely avoiding model training during the search, NASI can even guarantee the transferability of its selected architectures with its provable label- and data-agnostic search under mild conditions shown in Sec. 4.1 and Sec. 4.2, respectively. Particularly, with such provable label- and data-agnostic search, the final selected architectures by NASI on a proxy dataset are also likely to be selected and hence guaranteed to perform well on the target datasets under aforementioned mild conditions. So, the transferability of architectures selected via such label- and data-agnostic search can be guaranteed, as validated in Sec. 5 empirically.

# 4.1 LABEL-AGNOSTIC SEARCH

Our reformulated NAS problem (8) explicitly reveals that it can be optimized without the need of the labels from a dataset. Though the approximation of  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  in (12) seemingly depends on the labels, (12) can, however, be derived using random labels. This is because the Lipschitz continuity assumption on the loss function required by (10), which is necessary for the derivation of (12), remains satisfied when random labels are used. So, the approximation in (12) (and hence our optimization objective (13) that is based on this approximation) is label-agnostic, which justifies the label-agnostic nature of NASI. Interestingly, NAS algorithms with a similar label-agnostic search have already been developed in (Liu et al., 2020), which further implies the reasonableness of such label-agnostic search.

The label-agnostic approximation of  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  is demonstrated in Fig. 1a using the three search spaces of NAS-Bench-1Shot1 with randomly selected labels. According to Fig. 1a, the large Pearson correlation coefficient (i.e.,  $\rho \approx 1$ ) implies a strong correlation between the approximations with random vs. true labels, which consequently validates the label-agnostic approximation of  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$ . Overall, these empirical observations have verified that the approximation of  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  and hence NASI based on the optimization over this approximation are label-agnostic, which will be further validated empirically in Sec. 5.3.

# 4.2 DATA-AGNOSTIC SEARCH

Besides being label-agnostic, NASI is also guaranteed to be data-agnostic. To justify this, we prove in Proposition 2 (following from the notations in Sec. 3.1) below that  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  is data-agnostic under mild conditions.

Proposition 2. Suppose that  $\pmb{x} \in \mathbb{R}^{n_0}$  and  $\| \pmb{x} \|_2 \leq 1$  for all  $\pmb{x} \in \mathcal{X}$  given a dataset  $(\mathcal{X}, \mathcal{Y})$  of size  $|\mathcal{X}| = m$ , a given  $L$ -layer neural architecture  $\mathcal{A}$  is randomly initialized, and the  $\gamma$ -Lipschitz continuous nonlinearity  $\sigma$  satisfies  $|\sigma(x)| \leq |x|$ . Then, for any two data distributions  $P(\pmb{x})$  and  $Q(\pmb{x})$ , denote  $Z \triangleq \int \| P(\pmb{x}) - Q(\pmb{x}) \| \, \mathrm{d}\pmb{x}$ , as  $n_1, \ldots, n_{L-1} \to \infty$  sequentially,

$$
(m n) ^ {- 1} \left| \| \Theta_ {0} (\mathcal {A}; P) \| _ {\mathrm {t r}} - \| \Theta_ {0} (\mathcal {A}; Q) \| _ {\mathrm {t r}} \right| \leq n _ {0} ^ {- 1} Z D (\gamma)
$$

with probability arbitrarily close to 1.  $D(\gamma)$  is set to  $L$  if  $\gamma = 1$ , and  $(1 - \gamma^{2L}) / (1 - \gamma^2)$  otherwise.

Its proof is in Appendix A.4. Proposition 2 reveals that for any neural architecture  $\mathcal{A}$ ,  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  is data-agnostic if either one of the following conditions is satisfied: (a) Different datasets achieve a small  $Z$  or (b) the input dimension  $n_0$  is large. Interestingly, these two conditions required by the data-agnostic  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  can be well-satisfied in practice. Firstly, we always have  $Z < 2$  according to the property of probability distributions. Moreover, many real-world datasets are of high dimensions such as  $\sim 10^3$  for CIFAR-10 (Krizhevsky et al., 2009) and  $\sim 10^5$  for COCO (Lin et al., 2014). Since  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  under such mild conditions is data-agnostic, NASI using  $\|\Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  as the optimization objective in (8) is also data-agnostic.

While  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  is costly to evaluate, we demonstrate in Fig. 1b that the approximated  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  in (12) is also data-agnostic using random data following the standard Gaussian distribution. Similar to the results using true vs. random labels in Fig. 1a, the approximated  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  using random vs. true data are also highly correlated with a large Pearson correlation coefficient (i.e.,  $\rho > 0.9$ ). Interestingly, the correlation here is slightly smaller than the label-agnostic approximations in Fig. 1a, which implies that the approximated  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  is more agnostic to the labels than data. Based on these results, the approximated  $\| \Theta_0(\mathcal{A})\|_{\mathrm{tr}}$  is guaranteed to be data-agnostic. So, NASI based on the optimization over such a data-agnostic approximation is also data-agnostic, which will be further validated empirically in Sec. 5.3.

# 5 EXPERIMENTS

# 5.1 SEARCH IN NAS-BENCH-1SHOT1

We firstly validate the search efficiency and effectiveness of NASI in the three search spaces of NAS-Bench-1Shot1 (Zela et al., 2020b) on CIFAR-10. As the three search spaces are relatively

![](images/5e30c60757002c966252b1d8fc6cfd0f0031727532f7b6966063cb437fd984d8.jpg)  
Figure 2: Comparison of search efficiency (search budget in  $x$ -axis) and effectiveness (test error evaluated on CIFAR-10 in  $y$ -axis) between NASI and other NAS algorithms in the three search spaces of NAS-Bench-1Shot1. The test error for each algorithm is reported with the mean and standard error after ten independent searches.

![](images/5691ec0639af3e24e9dc01542c0c3254689dbce81dafbd2e7aa829f63ae41ea9.jpg)  
Random PC-DARTS ENAS DARTS GDAS NASI

![](images/b2e7feb3c48ce982af4a2f26b81a51eeff3b549da8d70e9d8717854c158d2c34.jpg)

Table 1: Performance comparison among state-of-the-art (SOTA) image classifiers on CIFAR-10/100. The performance of the final architectures selected by NASI is reported with the mean and standard deviation of five independent evaluations. The search costs are evaluated on a single Nvidia 1080Ti.  

<table><tr><td rowspan="2">Architecture</td><td colspan="2">Test Error (%)</td><td colspan="2">Params (M)</td><td rowspan="2">Search Cost (GPU Hours)</td><td rowspan="2">Search Method</td></tr><tr><td>C10</td><td>C100</td><td>C10</td><td>C100</td></tr><tr><td>DenseNet-BC (Huang et al., 2017)</td><td>3.46*</td><td>17.18*</td><td>25.6</td><td>25.6</td><td>-</td><td>manual</td></tr><tr><td>NASNet-A (Zoph et al., 2018)</td><td>2.65</td><td>-</td><td>3.3</td><td>-</td><td>48000</td><td>RL</td></tr><tr><td>AmoebaNet-A (Real et al., 2019)</td><td>3.34±0.06</td><td>18.93†</td><td>3.2</td><td>3.1</td><td>75600</td><td>evolution</td></tr><tr><td>PNAS (Liu et al., 2018)</td><td>3.41±0.09</td><td>19.53*</td><td>3.2</td><td>3.2</td><td>5400</td><td>SMBO</td></tr><tr><td>ENAS (Pham et al., 2018)</td><td>2.89</td><td>19.43*</td><td>4.6</td><td>4.6</td><td>12</td><td>RL</td></tr><tr><td>NAONet (Luo et al., 2018)</td><td>3.53</td><td>-</td><td>3.1</td><td>-</td><td>9.6</td><td>NAO</td></tr><tr><td>DARTS (2nd) (Liu et al., 2019)</td><td>2.76±0.09</td><td>17.54†</td><td>3.3</td><td>3.4</td><td>24</td><td>gradient</td></tr><tr><td>GDAS (Dong &amp; Yang, 2019)</td><td>2.93</td><td>18.38</td><td>3.4</td><td>3.4</td><td>7.2</td><td>gradient</td></tr><tr><td>NASP (Yao et al., 2020)</td><td>2.83±0.09</td><td>-</td><td>3.3</td><td>-</td><td>2.4</td><td>gradient</td></tr><tr><td>P-DARTS (Chen et al., 2019)</td><td>2.50</td><td>-</td><td>3.4</td><td>-</td><td>7.2</td><td>gradient</td></tr><tr><td>DARTS-(avg) (Chu et al., 2020)</td><td>2.59±0.08</td><td>17.51±0.25</td><td>3.5</td><td>3.3</td><td>9.6</td><td>gradient</td></tr><tr><td>SDARTS-ADV (Chen &amp; Hsieh, 2020)</td><td>2.61±0.02</td><td>-</td><td>3.3</td><td>-</td><td>31.2</td><td>gradient</td></tr><tr><td>R-DARTS (L2) (Zela et al., 2020a)</td><td>2.95±0.21</td><td>18.01±0.26</td><td>-</td><td>-</td><td>38.4</td><td>gradient</td></tr><tr><td>TE-NAS# (Chen et al., 2021)</td><td>2.83±0.06</td><td>17.42±0.56</td><td>4.3</td><td>4.4</td><td>0.5</td><td>training-free</td></tr><tr><td>NASI-FIX</td><td>2.79±0.07</td><td>16.12±0.38</td><td>4.4</td><td>4.6</td><td>0.24</td><td>training-free</td></tr><tr><td>NASI-ADA</td><td>2.90±0.13</td><td>16.84±0.40</td><td>4.2</td><td>4.4</td><td>0.24</td><td>training-free</td></tr></table>

† Reported by Dong & Yang (2019) with their experimental settings.  
* Obtained by training corresponding architectures without cutout (Devries & Taylor, 2017) augmentation.  
Evaluated using our experimental settings in Appendix B.4.

small, a lower penalty coefficient  $\mu$  and a larger constraint  $\nu$  (i.e.,  $\mu = 1$  and  $\nu = 1000$ ) are adopted to encourage the selection of high-complexity architectures in the optimization of (13). Here,  $\nu$  is determined adaptively as shown in Appendix B.1.

Figure 2 shows the results comparing the efficiency and effectiveness between NASI with a one-epoch search budget and other NAS algorithms with a maximum search budget of 20 epochs to allow sufficient model training during their search process. Figure 2 reveals that among all these three search spaces, NASI consistently selects architectures of better generalization performance than other NAS algorithms with a search budget of only one epoch. Interestingly, the selected architectures by the one-epoch NASI achieve performances that are comparable to the best-performing NAS algorithms with  $19 \times$  more search budget. Above all, NASI guarantees its benefits of improving the search efficiency of NAS algorithms considerably without sacrificing the generalization performance of its selected architectures.

# 5.2 SEARCH IN THE DARTS SEARCH SPACE

We then compare NASI with other NAS algorithms in a more complex search space than NAS-Bench1Shot1, i.e., the DARTS (Liu et al., 2019) search space (detailed in Appendix B.2). Here, NASI

Table 2: Performance comparison of architectures selected with random or true labels/data by NASI on CIFAR-10. The standard method denotes the search with the true labels and data of CIFAR-10 and each test error is reported with the mean and standard deviation of five independent searches.  

<table><tr><td rowspan="2">Method</td><td colspan="3">NAS-Bench-1Shot1</td><td rowspan="2">DARTS</td></tr><tr><td>S1</td><td>S2</td><td>S3</td></tr><tr><td>Standard</td><td>7.3±1.1</td><td>7.2±0.4</td><td>7.2±0.6</td><td>2.95±0.13</td></tr><tr><td>Random Label</td><td>6.8±0.3</td><td>7.0±0.4</td><td>7.5±1.4</td><td>2.90±0.12</td></tr><tr><td>Random Data</td><td>6.6±0.2</td><td>7.5±0.7</td><td>7.3±0.9</td><td>2.97±0.10</td></tr></table>

selects the architecture with a search budget of  $T = 100$ , batch size of  $b = 64$  and  $\mu = 2$ . Besides, two different methods are applied to determine the constraint  $\nu$  during the search process: the adaptive determination with an initial value of 500 and the fixed determination with a value of 100. The final selected architectures with adaptive and fixed  $\nu$  are, respectively, called NASI-ADA and NASI-FIX (visualized in Appendix C.4), which are then evaluated on CIFAR-10/100 (Krizhevsky et al., 2009) and ImageNet (Deng et al., 2009) following Appendix B.4.

Table 1 summarizes the generalization performance of the final architectures selected by various NAS algorithms on CIFAR-10/100. Compared with popular training-based NAS algorithms, NASI achieves a substantial improvement in search efficiency and maintains a competitive generalization performance. Even when compared with the training-free NAS algorithm (i.e., TE-NAS), NASI is also able to select competitive or even outperformed architectures with a smaller search cost. Besides, NASI-FIX achieves the smallest test error on CIFAR-100, which demonstrates the transferability of the architectures selected by NASI over different datasets. Consistent results on ImageNet can be found in Appendix C.5.

# 5.3 LABEL- AND DATA-AGNOSTIC SEARCH

To further validate the label- and data-agnostic search achieved by our NASI as discussed in Sec. 4, we compare the generalization performance of the final architectures selected by NASI using random labels and data on CIFAR-10. The random labels are randomly selected from all possible categories while the random data is i.i.d. sampled from the standard Gaussian distribution. Both NAS-Bench1Shot1 and the DARTS search space are applied in this performance comparison where the same search and training settings in Sec. 5.1 and Sec. 5.2 are adopted.

Table 2 summarizes the performance comparison. Interestingly, among all the four search spaces, comparable generalization performances are obtained on CIFAR-10 for both the architectures selected with random labels (or data) and the ones selected with true labels and data. These results hence confirm the label- and data-agnostic search achieved by NASI, which therefore also further validates the transferability of the architectures selected by NASI over different datasets.

# 6 CONCLUSION

This paper describes a novel NAS algorithm called NASI that exploits the capability of NTK for estimating the performance of candidate architectures at initialization. Consequently, NASI can completely avoid model training during the search process to achieve higher search efficiency than existing NAS algorithms. NASI can also achieve competitive generalization performance across different search spaces and benchmark datasets. Interestingly, NASI is guaranteed to be label- and data-agnostic under mild conditions, which implies the transferability of the final architectures selected by NASI over different datasets. With all these advantages, NASI can thus be adopted to select well-performing architectures for unsupervised tasks and larger-scale datasets efficiently, which to date remains challenging to other training-based NAS algorithms. Furthermore, NASI can also be integrated into other training-based one-shot NAS algorithms to improve their search efficiency while preserving the search effectiveness of these training-based algorithms.

# REPRODUCIBILITY STATEMENT

In order to guarantee the reproducibility of the theoretical analysis in our paper, we have provided complete proof of our propositions and also the justification of certain assumptions in Appendix A. Moreover, we have conducted adequate ablation studies to further investigate the impacts of these assumptions and also the approximations used in our method on the final search results in Appendix C.6. To guarantee the reproducibility of the empirical results in our paper, we have provided our codes in the supplementary materials and detailed experimental settings in Appendix B.

# REFERENCES

Mohamed S. Abdelfattah, Abhinav Mehrotra, Lukasz Dudziak, and Nicholas D. Lane. Zero-cost proxies for lightweight NAS. In Proc. ICLR, 2021.  
Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In Proc. ICML, pp. 322-332, 2019.  
Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: A survey. Journal of Machine Learning Research, 18:153:1-153:43, 2017.  
Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In Proc. ICLR, 2019.  
Wuyang Chen, Xinyu Gong, and Zhangyang Wang. Neural architecture search onImagenet in fourgpu hours: A theoretically inspired perspective. In Proc. ICLR, 2021.  
Xiangning Chen and Cho-Jui Hsieh. Stabilizing differentiable architecture search via perturbation-based regularization. In Proc. ICML, pp. 1554-1565, 2020.  
Xin Chen, Lingxi Xie, Jun Wu, and Qi Tian. Progressive differentiable architecture search: Bridging the depth gap between search and evaluation. In Proc. ICCV, pp. 1294-1303, 2019.  
Xiangxiang Chu, Xiaoxing Wang, Bo Zhang, Shun Lu, Xiaolin Wei, and Junchi Yan. DARTS: Robustly stepping out of performance collapse without indicators. arXiv:2009.01027, 2020.  
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proc. CVPR, 2009.  
Terrance Devries and Graham W. Taylor. Improved regularization of convolutional neural networks with cutout. arXiv:1708.04552, 2017.  
Xuanyi Dong and Yi Yang. Searching for a robust neural architecture in four GPU hours. In Proc. CVPR, pp. 1761-1770, 2019.  
Xuanyi Dong and Yi Yang. Nas-bench-201: Extending the scope of reproducible neural architecture search. In Proc. ICLR, 2020.  
Simon S. Du, Xiyu Zhai, Barnabás Póczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In Proc. ICLR, 2019.  
Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Neural architecture search: A survey. Journal of Machine Learning Research, 20:55:1-55:21, 2019.  
Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In Proc. ICLR, 2015.  
Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In Proc. ICML, pp. 1225-1234, 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In Proc. ICCV, pp. 1026-1034, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. CVPR, pp. 770-778, 2016.  
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861, 2017.  
Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In Proc. CVPR, pp. 2261-2269, 2017.  
Arthur Jacot, Clément Hongler, and Franck Gabriel. Neural Tangent Kernel: Convergence and generalization in neural networks. In Proc. NeurIPS, pp. 8580-8589, 2018.  
Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-Softmax. In Proc. ICLR, 2017.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.  
Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In Proc. NeurIPS, pp. 8570-8581, 2019a.  
Namhoon Lee, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Snip: Single-shot network pruning based on connection sensitivity. In Proc. ICLR, 2019b.  
Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In Proc. ECCV, pp. 740-755, 2014.  
Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan L. Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. In Proc. ECCV, pp. 19–35, 2018.  
Chenxi Liu, Piotr Dólár, Kaiming He, Ross B. Girshick, Alan L. Yuille, and Saining Xie. Are labels necessary for neural architecture search? In Proc. ECCV, pp. 798-813, 2020.  
Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In Proc. ICLR, 2019.  
Renqian Luo, Fei Tian, Tao Qin, Enhong Chen, and Tie-Yan Liu. Neural architecture optimization. In Proc. NeurIPS, pp. 7827-7838, 2018.  
Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. ShuffleNet V2: Practical guidelines for efficient CNN architecture design. In Proc. ECCV, pp. 122-138, 2018.  
Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. In Proc. ICML, pp. 3, 2013.  
Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete distribution: A continuous relaxation of discrete random variables. In Proc. ICLR, 2017.  
Joseph Mellor, Jack Turner, Amos J. Storkey, and Elliot J. Crowley. Neural architecture search without training. arXiv:2006.04647, 2020.  
Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning. MIT press, 2018.  
Daniel S. Park, Jaehoon Lee, Daiyi Peng, Yuan Cao, and Jascha Sohl-Dickstein. Towards NNGP-guided neural architecture search. arXiv:2011.06006, 2020.  
Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, and Jeff Dean. Efficient neural architecture search via parameter sharing. In Proc. ICML, pp. 4092-4101, 2018.

Maithra Raghu, Ben Poole, Jon M. Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the expressive power of deep neural networks. In Proc. ICML, pp. 2847-2854, 2017.  
Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Regularized evolution for image classifier architecture search. In Proc. AAAI, pp. 4780-4789, 2019.  
Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of Adam and beyond. In Proc. ICLR, 2018.  
Yao Shu, Wei Wang, and Shaofeng Cai. Understanding architectures learnt by cell-based neural architecture search. In Proc. ICLR, 2020.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Proc. ICLR, 2015.  
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proc. CVPR, pp. 1-9, 2015.  
Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V. Le. MnasNet: Platform-aware neural architecture search for mobile. In Proc. CVPR, pp. 2820-2828, 2019.  
Hidenori Tanaka, Daniel Kunin, Daniel L. Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. In Proc. NeurIPS, 2020.  
Chaoqi Wang, Guodong Zhang, and Roger B. Grosse. Picking winning tickets before training by preserving gradient flow. In Proc. ICLR, 2020.  
Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. SNAS: Stochastic neural architecture search. In Proc. ICLR, 2019.  
Jingjing Xu, Liang Zhao, Junyang Lin, Rundong Gao, Xu Sun, and Hongxia Yang. KNAS: green neural architecture search. In Proc. ICML, pp. 11613-11625, 2021.  
Quanming Yao, Ju Xu, Wei-Wei Tu, and Zhanxing Zhu. Efficient neural architecture search via proximal iterations. In Proc. AAAI, pp. 6664-6671, 2020.  
Arber Zela, Thomas Elsken, Tonmoy Saikia, Yassine Marrakchi, Thomas Brox, and Frank Hutter. Understanding and robustifying differentiable architecture search. In Proc. ICLR, 2020a.  
Arber Zela, Julien Siems, and Frank Hutter. NAS-Bench-1Shot1: Benchmarking and dissecting one-shot neural architecture search. In Proc. ICLR, 2020b.  
Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In Proc. ICLR, 2017.  
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. In Proc. CVPR, pp. 8697-8710, 2018.
