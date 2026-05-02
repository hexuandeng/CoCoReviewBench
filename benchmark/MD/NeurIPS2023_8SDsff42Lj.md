# Continual Learning with Global Prototypes: Beyond the Scope of Task Supervision

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Continual learning aims to sequentially learn from different tasks without catastrophic forgetting. With no assumptions of task dependence, the knowledge learned from observed tasks may not align with that required for future tasks. This may result in models' disruptive updates for learning future tasks, causing abrupt changes to previously learned knowledge (e.g. representation drift [7]) which induces catastrophic forgetting. To reduce such disruptive updates, we connect knowledge for observed and unknown tasks by learning task data representations properly related to a set of global prototypes, which have general-purpose connections and are shared across all tasks. We derive global prototypes and the corresponding objective for NLP tasks. For those tasks, the correlated global prototypes can be obtained from a model pre-trained by masked language modeling. And the data representations that have proper relationships to global prototypes can be learned by specific adaptations of the pre-trained model. We investigate existing adaptation models and propose a neighbor attention model which combines different advantages of existing models for our objective. Experiments show that models learning data representations well related to global prototypes can induce significantly less catastrophic forgetting, without memorizing information from past tasks.

# 1 Introduction

In the continual learning paradigm, models progressively learn a sequence of tasks. This paradigm supports real-world applications which face continuous streams of data and tasks [35, 20]. In practice, models may be under storage constraints to use a fixed structure and under privacy considerations that restrict revisiting of previous tasks' data. These introduce the challenge of catastrophic forgetting, where models lose knowledge of previously learned tasks after learning new tasks.

Most prior works address catastrophic forgetting using models that integrate the knowledge of the past and present tasks, i.e. the observed tasks. For example, regularization-based models constrain the deviation of current parameters from the previous ones [27, 56, 2, 29]; replay-based models memorize samples from past tasks and rehearse when learning present tasks [35, 9, 46, 26]. However, since there are no assumptions on task dependence in continual learning, models learned from a set of observed tasks may not contain knowledge needed for unknown future tasks [28, 16]. To learn such a future task, these models may have disruptive changes on previously learned knowledge (e.g. representation drift [7]), which still induces catastrophic forgetting. One way to reduce such disruptive updates is to make models consider potential knowledge connections to future tasks.

Our key idea is to build connections between observed and unknown tasks by connecting task-specific data representations to a general-purpose representation base that is shared across all tasks. In many domains, task-specific information about classes can be represented by specific combinations of general units. For example, consider the data instance 'A boy in a red hooded top is smiling. The

![](images/44516393f7f355277da9a861b3fdb10ac6b6f52fdb2bd3b5903897b0bb05dad8.jpg)

![](images/c75f9ca328e67649bb103a566d3b7e4a80a190d9d0608dcb708165092b4e9bf4.jpg)  
(b) Continual Learning with Knowledge of Global Prototypes  
Figure 1: Representations learned with or without global prototypes. The shaded regions cover data representations for each class. In (a), with knowledge only learned for observed supervised tasks, models may have disruptive updates that cause data representation drift when learning a new task. In (b), with reference to correlated global prototypes (dots) in each task learning, representations for different tasks (shaded regions) can properly connect to each other which reduces representation drift.

boy is upset.' from 'contradiction' class in an entailment classification task. The set  $\{\text{smiling, upset}\}$  conveys the task-specific information of 'contradiction' using the general (i.e. not task-specific) semantics of the token units 'smiling' and 'upset'. Based on this, we construct a general-purpose representation base consisting a set of unit representations, which we call global prototypes. These global prototypes are pre-learned to reflect semantic connections between them. Then we learn data representations with appropriate task-specific connections to global prototypes. This allows knowledge learned from observed tasks to connect to that of future tasks via the interconnection of global prototypes, which is beyond the scope of task supervision from observed tasks. Our idea mimics mechanism in the brain, a biological continual learning system [56] which rewrites existing neurons instead of creating new neurons to learn new tasks [17]. Here, global prototypes mimic the neurons, and learning different connections between data representations and global prototypes mimic the rewiring process. A figure of the idea is shown in Figure 1.

We address two main challenges in realizing this idea: (1). constructing the representation base with correlated global prototypes; (2). learning data representations with task-specific connections to global prototypes. We investigate the above challenges for NLP tasks. For text, the non-contextual token representations are a natural choice for global prototypes, as any text information can be represented by sets of tokens from a fixed vocabulary. For the first challenge, we obtain the global prototypes from a pre-trained language model which learns semantic connections between tokens through self-supervised learning [11]. For the second challenge, we learn data representations by lightly adapting a pre-trained model to obtain task-specific connections to the global prototypes (Section 3). We investigate existing adaptation models with learnable projections (Adapters [21]), learnable embeddings (Prompt Tuning [30]), and propose a neighbor attention module combining properties of these two (Section 4). Results show that catastrophic forgetting can be significantly mitigated with models that can learn representations well connected to global prototypes. In addition, our neighbor attention model combines the advantages of existing adaptation models, and achieves superior performance in both vanilla and replay settings.

In conclusion, our contributions in this paper are:

1. We propose to learn task-specific information over a general-purpose base with global prototypes to address general task connections in continual learning. Specifically, we derive the construction of the base and the corresponding objective for NLP tasks.  
2. We investigate existing adaptation models and propose a new neighbor attention model to learn data representations that have proper relationships to global prototypes.  
3. We conduct experiments on different adaptation models and continual learning frameworks. Results show our model can significantly reduce forgetting without replay.

# 2 Related Work

Continual Learning Continual learning aims to sequentially learn new tasks while not forgetting previously learned tasks. Models for continual learning can be divided into three main categories: (1). regularization-based models which constrain the deviation of new parameters from the older ones [27, 56, 2, 29]; (2) replay-based models which reduce catastrophic forgetting by rehearsing on real or pseudo samples from previous tasks [35, 9] or generative models [46, 26]; (3). architecture-based models which learn evolving architectures for sequential tasks, with their capacities for each task carefully assigned [44, 53]. Most works above focus on knowledge based on observed tasks.

Some recent works show that knowledge only from observed task supervision is insufficient for continual learning. Knoblauch et al. [28] claim that optimal continual learning requires perfect memory and is NP-hard; Guo et al. [16] suggest preservingholistic information which may not benefit current task but help future tasks. With biased knowledge, models can have disruptive updating when learning a new task, causing problems like representation drift [7, 36, 24]. In this paper, we propose to consider knowledge beyond observed task supervision through a general-purpose base with pre-learned global prototypes. Unlike previous works [39, 4] which use a pre-defined classifier to help class separation [38], our global prototypes are pre-learned with general semantic connections and thus can build connections between tasks. Some works use self-supervised learning to learn more general representations for continual learning [36, 14, 15]. However, those representations do not necessarily connect to specific global prototypes, which is different from our objective.

Continual learning for NLP is an emerging area [3]. Liu et al. [32] introduce a sentence encoder with matrix conceptors; MBPA++ [10] uses an episodic memory with replay and local adaptation to mitigate catastrophic forgetting; LAMOL [49] learns to generate training samples for replay based on pre-trained knowledge; IDBR [23] disentangles hidden spaces to distinguish task-agnostic and task-specific information. Most of them require a memory of past task information, or converting data to a question-answering format along with text-to-text models [6, 42]. Our model does not have such restriction. There are also works [25] focusing on knowledge transfer in continual learning.

Adaptation Models In this work, we use adaptation models to learn representations connected to global prototypes. Prior works using pre-trained model with light adaptation for target tasks were originally aimed at parameter efficient tuning. Different methods include adding limited trainable parameters on the frozen transformer layer [21, 40, 18, 22]; or selectively updating existing parameters during training [41, 55]. Recent prompt tuning works [31, 30, 34] learn target tasks by trainable prompt embeddings for generalization purposes as well.

Most closely related work are adaptation models used for continual learning [51, 13, 43]. However, most use the models' parameter efficiency to construct progressive memory. Whether utilizing the pre-trained knowledge can help continual learning, why and how they help remain unexplored. Our approach is based on a fixed model without progressive memory of parameters. We use the adaptation model for our desiderata, which also provides a metric to interpret whether the model can benefit continual learning. We believe our work can inspire further utilization of adaptation models for CL.

# 3 Learning over Global Prototypes

We consider the following continual learning setting: the model learns from a sequence of tasks, where each task consists of data  $\mathcal{D}_{\tau} = \{(\mathbf{x}_{\tau}^{(i)},y_{\tau}^{(i)})_{i = 1}^{n_{\tau}}\}$ .  $\mathbf{x}_{\tau}$  is the input data and  $y_{\tau}$  is the class label. A task identifier  $\tau$  is provided at the training time. We consider two scenarios: task-incremental and class-incremental learning, where models are task-aware or task-agnostic at the inference time [37]. Without replay, we use the same training objective for both task-incremental and class-incremental learning while evaluating them in different ways.

Notation  $C_{\tau}$  represents a set of all classes for each task  $\tau$ ,  $C = [C_1, \ldots, C_\tau, \ldots]$  represents all classes for all tasks. For NLP tasks,  $V$  represents the set of tokens with global prototypes in the representation base.  $\mathbf{w}^i$  is the  $i$ -th column of a matrix  $\mathbf{w}$ . Our main model consists of two components: an encoder  $f_{\theta}$  to generate representation  $f_{\theta}(\mathbf{x}_{\tau})$  for each data instance  $\mathbf{x}_{\tau}$ ; and a classifier with matrix  $\mathbf{w}_{\gamma} \in \mathbb{R}^{d \times |C|}$  for class prediction, where  $d$  represents the dimension of data representations. At the inference time, the class label is predicted by  $\arg \max_{i \in C_{\text{candidate}}} f_{\theta}(\mathbf{x}_{\tau}) \cdot \mathbf{w}_{\gamma}^i$ . For task-incremental inference we have  $C_{\text{candidate}} = C_{\tau}$ , while for class-incremental inference we have  $C_{\text{candidate}} = C_{1:\tau}$ .

# 3.1 The Learning Objective

Classification Loss For a task  $\tau$ , the typical classification objective is to minimize the cross-entropy loss  $\mathcal{L}_c(\mathbf{x}_{\tau};\theta ,\gamma)$  over the training data for the task, as shown below:

$$
\mathcal {L} _ {c} \left(\mathbf {x} _ {\tau}; \theta , \gamma\right) = - \log \frac {\exp \left(\mathbf {w} _ {\gamma} ^ {y _ {\tau}} \cdot f _ {\theta} \left(\mathbf {x} _ {\tau}\right)\right)}{\sum_ {c \in C _ {\tau}} \exp \left(\mathbf {w} _ {\gamma} ^ {c} \cdot f _ {\theta} \left(\mathbf {x} _ {\tau}\right)\right)}. \tag {1}
$$

After learning task  $\tau$ , models have knowledge about data  $\mathbf{x}_{1:\tau}$  and class vectors  $\mathbf{w}_{\gamma}^{c \in C_{1:\tau}}$  from observed tasks  $1: \tau$ . However, the knowledge may not align with that required for the unknown future task  $(\tau + 1)$ . Specifically, after adjusting  $\theta$  in task  $(\tau + 1)$ , the alignment between  $\mathbf{w}_{\gamma}^{y\tau}$  learned from task  $\tau$  and  $f_{\theta}(\mathbf{x}_{\tau})$  with adjusted  $\theta$  may shift and degrade. In other words, to learn a future task, models may have disruptive updates which make abrupt changes to previously learned knowledge (e.g. representation drift [7]), and induce forgetting.

Prototype Loss To mitigate models' disruptive updates, we consider potential connections between observed and unknown tasks. The connection is built by learning task-specific data representations connected to a general-purpose representation base, which is shared across all tasks. The base consists of global token prototypes (denoted proto[v] for token  $v$ ) which reflect semantic connections between them. In particular, we want the data representation  $f_{\theta}(\mathbf{x}_{\tau})$  to be connected to the task-relevant global prototypes. Given a reference probability distribution  $p(v|\mathbf{x}_{\tau},\mathbf{y}_{\tau})$  which indicates the strength of connection between data representation and proto[v], we push the data representations towards the prototypes in proportion to their reference probability. Formally, we define the prototype loss as:

$$
\mathcal {L} _ {v} \left(\mathbf {x} _ {\tau}; \theta\right) = - \sum_ {v \in V} p \left(v \mid \mathbf {x} _ {\tau}, y _ {\tau}\right) \log \frac {\exp \left(\operatorname {p r o t o} [ v ] \cdot f _ {\theta} \left(\mathbf {x} _ {\tau}\right)\right)}{\sum_ {v ^ {\prime} \in V} \exp \left(\operatorname {p r o t o} \left[ v ^ {\prime} \right] \cdot f _ {\theta} \left(\mathbf {x} _ {\tau}\right)\right)}. \tag {2}
$$

In Eq.(2), the softmax is calculated over all global prototypes, i.e.  $\mathrm{proto}[v]$  for any  $v\in V$ , regardless of task difference. Such calculation is task-agnostic, while the referenced probability  $p(v|\mathbf{x}_{\tau},\mathbf{y}_{\tau})$  gives task-specific guidance for representation learning. By doing this, Eq. (2) learns representations with task-specific connections to global prototypes. Since global prototypes are pre-learned to reflect semantic connections, representations learned by Eq. (2) can connect across tasks via connections of global prototypes. This can reduce abrupt representation change caused by disruptive updating.

The reference probability  $p(v|\mathbf{x}_{\tau},\mathbf{y}_{\tau})$  gives task-specific guidance for representation learning, where tokens with task-specific information of  $\mathbf{x}_{\tau}$  should have high probabilities. Considering both task-specific and holistic information of the data [16, 36], we set  $p(v|\mathbf{x}_{\tau},y_{\tau}) = 1 / r_{\tau}$  when  $v$  is one of data's  $r_{\tau}$  rationale tokens, i.e. tokens in the data that are essential for class prediction [8], otherwise  $p(v|\mathbf{x}_{\tau},y_{\tau}) = 0$ . Using multiple rationale tokens as task-specific guidance brings extra benefits to the expressiveness of data representations and global prototypes. First, different data representations from the same class have different guidance. Second, a small number of global prototypes can convey rich information when connecting representations to different sets of global prototypes.

Learning Objective Based on the above analysis, our learning objective is to learn data representations that can correctly predict class labels (Eq. (1)); and properly connect to global prototypes (Eq. (2)). The optimal parameters  $\theta^{*}$ ,  $\gamma^{*}$  for task  $\tau$  should satisfy the desiderata below:

- Task performance.  $\mathcal{L}_c(\mathbf{x}_{\tau};\theta^*,\gamma^*) \leq \mathcal{L}_c(\mathbf{x}_{\tau};\theta,\gamma)$  for any  $\theta \neq \theta^*$ ,  $\gamma \neq \gamma^*$  (3)  
- Global alignment.  $\mathcal{L}_v(\mathbf{x}_{\tau};\theta^*)\leq a_{\tau}$  (4)

where  $a_{\tau} > 0$  is a threshold value of the prototype loss. Task performance desiderata (Eq. (3)) can be satisfied by optimization on classification loss in Eq. (1). In the rest of this section, we discuss two questions that are necessary for our desiderata: (1). How to get the semantically connected global prototype proto[v] for Eq. (2)? (2). How to get feasible models for the second desiderata in Eq. (4)?

# 3.2 Pre-trained Models for Prototypes and Data Representations

To get correlated global prototypes and learn data representations with reference to them, we utilize a model pre-trained by masked language modeling (MLM). The MLM objective is to predict masked

![](images/d0e4cd77244a815390416f75322f65a748e8647f67ed2500c2d4f83cc21b4273.jpg)  
Figure 2: Layers of the transformer and different adaptation models. Shaded blocks are learnable.

![](images/3f54d71e75e6758b7647404c471e238034ededccd64431765fd088a31a223979.jpg)

![](images/4fe5d283938495e683c3f95297887a2db684c4ce221f5ce1f42dc91556c0b863.jpg)

![](images/1fcb2abc21c3737470f3fb780821a6816dd85e86332dd6ce1a7f7a875fc7b7e0.jpg)

token  $v_{m}$  from a masked input  $\tilde{\mathbf{x}}$ , with the following loss:

$$
\mathcal {L} _ {m} (\tilde {\mathbf {x}}; \delta , \phi) = - \sum_ {v \in V} p (v | \tilde {\mathbf {x}}) \log \frac {\exp \left(\mathbf {w} _ {\delta} ^ {v} \cdot f _ {\phi} (\tilde {\mathbf {x}})\right)}{\sum_ {v ^ {\prime} \in V} \exp \left(\mathbf {w} _ {\delta} ^ {v ^ {\prime}} \cdot f _ {\phi} (\tilde {\mathbf {x}})\right)}, \tag {5}
$$

where  $f_{\phi}$  denotes the encoder for MLM,  $\mathbf{w}_{\delta}$  consists of the token vector  $\mathbf{w}_{\delta}^{v}$  for each token  $v$ . The probability  $p(v|\tilde{\mathbf{x}}) = 1$  if  $v$  is the masked token  $v_{m}$ , and 0 otherwise.

Pre-Trained Model for Global Prototypes The MLM objective learns token vectors  $\mathbf{w}_{\delta}$  that reflect semantic connections between tokens, which suits our requirement for global prototypes. Therefore, we can get the global prototype proto[v] as the  $v$ -th token vector (proto[v] =  $\mathbf{w}_{\delta}^{v}$ ) from a model pre-trained by MLM. Extending to cases when pre-trained models are unavailable, we can first train a model by self-supervised learning which learns global prototypes. Global prototypes are fixed once learned. We leave improving them during continual task learning for future study.

Adapting Pre-Trained Models for Feasibility To get feasible models for the desiderata in Eq.(4), we have two options: (a). learning with the prototype loss in Eq.(2); (b). designing a model which can satisfy the desiderata without direct supervision of probabilities  $p(v|\mathbf{x}_{\tau},y_{\tau})$ . Option (a) needs rationale tokens to get  $p(v|\mathbf{x}_{\tau},y_{\tau})$ , which requires expensive human annotations. In this work, we investigate models for option (b). Specifically, we investigate whether adapting a pre-trained model where we get global prototypes can satisfy our desiderata. Comparing Eq.(5) and Eq.(2), when having proto[ v = \mathbf{w}_{\delta}^{v} ], models for Eq.(5) learn representations that have task-agnostic connections to global prototypes, which is a variant of Eq.(2). When lightly adapting a pre-trained encoder  $f_{\phi}$  to task encoder  $f_{\theta}$ , data representations are learned with reference to those task-agnostic connections. Therefore, the adapted representations may have better connections to global prototypes.

In general, our learning includes two stages: first training a model by self-supervised learning for global prototypes (can be skipped if starting from a pre-trained language model); then lightly adapting this model for target tasks while satisfying the desiderata in Eq. (4). We investigate different adaptation models and whether they satisfy our desiderata in the following sections.

# 4 Adaptation Models for Global Alignment

We investigate the potential of different adaptation models for our desiderata of global alignment in Eq.(4). In this section, we first introduce existing adaptation models (Section 4.1) and propose a new neighbor attention model for the desiderata (Section 4.2). A comparison of models is shown in Fig. 2.

# 4.1 Existing Adaptation Models

For a transformer model, representations are calculated by the self-attention mechanism. Given input representations  $\mathbf{H} = [\mathbf{h}_1,\dots,\mathbf{h}_n]$ , each output representation  $\mathbf{o}_i$  after self-attention is:

$$
\mathbf {o} _ {i} = f \left(\operatorname {M H A} \left(\mathbf {Q} _ {\phi} (\mathbf {h} _ {i}), \mathbf {K} _ {\phi} (\mathbf {H}), \mathbf {V} _ {\phi} (\mathbf {H})\right)\right), \tag {6}
$$

where MHA is the multi-head attention function (Appendix A),  $f$  is the feed-forward function,  $\mathbf{Q}_{\phi}, \mathbf{K}_{\phi}, \mathbf{V}_{\phi}$  are linear functions for query, key and value. Adaptation models utilize pre-trained parameters for self-attention, while adding extra components to adapt the model for target tasks. According to He et al. [18], different adaptations can be viewed as combining different modification vectors  $\Delta_{\theta} \mathbf{o}_i$  to pre-trained representation  $\mathbf{o}_i$ . We investigate two types of modifications below.

Learnable Projections Models like Adapters [21] insert adaptation modules between transformer layers. The module applies linear projections to the self-attention output  $\mathbf{o}_i$ , with the non-linear

activation between them. With a residual connection [19], the adapted output  $\mathbf{o}_i^{(\mathrm{new})}$  is:

$$
\mathbf {o} _ {i} ^ {\text {(n e w)}} \leftarrow \mathbf {o} _ {i} + \Delta_ {\theta} \mathbf {o} _ {i}, \quad \Delta_ {\theta} \mathbf {o} _ {i} := \mathbf {W} _ {\theta} \mathbf {o} _ {i}. \tag {7}
$$

$\mathbf{W}_{\theta} \in \mathbb{R}^{d \times d}$  represents the linear projections. (We omit the non-linear activation for simplicity).

Learnable Embeddings Models like Prompt Tuning [30] add learnable embeddings in the input. Then self-attention is performed based on the input with prompts. The adapted output is [18]:

$$
\mathbf {o} _ {i} ^ {\text {(n e w)}} \leftarrow (1 - \lambda (\mathbf {h} _ {i})) \mathbf {o} _ {i} + \lambda (\mathbf {h} _ {i}) \Delta_ {\theta} \mathbf {o} _ {i}, \quad \Delta_ {\theta} \mathbf {o} _ {i} := \mathrm {M H A} \left(\mathbf {Q} _ {\phi} (\mathbf {h} _ {i}), \mathbf {K} _ {\phi} (\mathbf {P} _ {\theta}), \mathbf {V} _ {\phi} (\mathbf {P} _ {\theta})\right). \tag {8}
$$

$\mathbf{P}_{\theta}$  are learnable prompt embeddings in  $\mathbb{R}^{p\times d}$ ,  $p$  is the number of prompts.  $\lambda (\mathbf{h}_i)$  is a gate value computed from self-attention which decides the ratio of pre-trained and modified representations.

Choices for Global Alignment Both of the adaptations show effectiveness in single-task performance for our desiderata Eq. (3) [21, 31]. For global alignment in Eq. (4), Prompt Tuning has a gate  $\lambda (\mathbf{h}_i)$  to mix pre-trained and modified representations. With a small gate value, this may generate representations close to pre-trained representations, and thus better connect to global prototypes. However, the gate  $\lambda (\mathbf{h}_i)$  in Eq. (8) is decided by self attention over inputs and prompts, thus can lean to modified representations  $\Delta_{\theta}\mathbf{o}_i$ . Also, the learned prompts  $\mathbf{P}_{\theta}$  may convey information far away from the original data. These may degrade the models' capacity for global alignment. Because of this, we propose a model that has a controlled gate value and relies on neighbors of tokens instead of searching from random prompts for task adaptation. In addition, the training for prompt embeddings is not as easy as that for linear projections [30, 22], which may cause efficiency issues when adapting multiple tasks. We also introduce learnable projections in our model for fast adaptations.

# 4.2 Transformer with Neighbor Attentions

We design a neighbor attention module added to the pre-trained model for task adaptations. The module has three properties: (1). utilizing learnable linear projections to learn modified representations; (2). acquiring neighbor representations for extra information; (3). using a controlled gate to mix pre-trained and modified representations. The adapted output of the neighbor attention module is:

$$
\mathbf {o} _ {i} ^ {\text {(n e w)}} \leftarrow (1 - \lambda) \mathbf {o} _ {i} + \lambda \Delta_ {\theta} \mathbf {o} _ {i}, \Delta_ {\theta} \mathbf {o} _ {i} := \mathrm {M H A} \left(\mathbf {Q} _ {\phi} (\mathbf {h} _ {i}), \mathbf {K} _ {\theta} (\mathbf {M} _ {i} \| \mathbf {h} _ {i}), \mathbf {V} _ {\theta} (\mathbf {M} _ {i} \| \mathbf {h} _ {i})\right). \tag {9}
$$

where  $\lambda$  is the ratio of modified representations in the mix-up,  $||$  denotes the concatenation operation.  $\mathbf{K}_{\theta}$ ,  $\mathbf{V}_{\theta}$  are learnable linear functions for key and value.  $\mathbf{M}_i = [\mathbf{m}_{i1},\dots,\mathbf{m}_{ik}]$  are  $k$  neighbor representations of the input representation  $\mathbf{h}_i$ .

Comparing Eq. (9) to Eq. (8), neighbor attention has learnable linear functions for key and value. Moreover, we manually control the gate by setting  $\lambda = 0.1$  to push the module to focus more on the pre-trained representations. This is for our desiderata to have representations close to pre-trained ones which are trained over global prototypes. Finally, we introduce neighbor representations  $\mathbf{M}_i$  for information out of the inputs, which can improve the model's expressivity. Details are shown below.

**Neighbor Representations** Before the first neighbor attention layer, we find the initial neighbor representations  $\mathbf{M}_i$  for a hidden representation  $\mathbf{h}_i$ . Neighbors of  $\mathbf{h}_i$  can be obtained by comparing the dot product between  $\mathbf{h}_i$  and token embeddings from the pre-trained embedding layer, then selecting  $k$  tokens which have top- $K$  scores as neighbors.  $K$  decides the range of the neighborhood.

Then we transform neighbor embeddings to the space of  $\mathbf{h}_i$ . We disentangle  $\mathbf{h}_i$ 's  $j$ -th neighbor representation  $\mathbf{m}_{ij}$  into two parts: one related to the hidden representation  $\mathbf{h}_i$ ; and the other related to neighbor information out of  $\mathbf{h}_i$ . The latter can be obtained by deviating neighbor embedding  $\mathbf{e}_{ij}$  from  $\mathbf{h}_i$ 's token embedding  $\mathbf{e}_i$ . Then the transformed neighbor representation is:  $\mathbf{m}_{ij} = \alpha (\mathbf{e}_{ij} - \mathbf{e}_i) + \beta \mathbf{h}_i$  where  $0 < \alpha, \beta < 1$  are scalars. In this paper, we set  $\alpha = \beta = 0.2$ .

After that, the neighbor representation  $\mathbf{M}_i$  is updated at each neighbor attention layer. For the  $j$ -th neighbor representation  $\mathbf{m}_{ij}$ , the updated representation  $\mathbf{m}_{ij}^{(\mathrm{new})}$  for the next layer is:

$$
\mathbf {m} _ {i j} ^ {\left(\text {n e w}\right)} \leftarrow \mathbf {m} _ {i j} + \Delta_ {\theta} \mathbf {m} _ {i j}, \Delta_ {\theta} \mathbf {m} _ {i j} := f \big (\operatorname {M H A} \left(\mathbf {Q} _ {\phi} \left(\mathbf {m} _ {i j}\right), \mathbf {K} _ {\theta} \left(\mathbf {M} _ {i} | | \mathbf {h} _ {i}\right), \mathbf {V} _ {\theta} \left(\mathbf {M} _ {i} | | \mathbf {h} _ {i}\right)\right) \big).
$$

Adding neighbor attention on more layers will increase the model capacity, but also cause more risk of over-smoothing [45], i.e., neighbor tokens all have the same representations. In practice, we add

neighbor attention to less than half of the transformer layers, and leave the last layer untouched for guidance. In continual learning, the optimal layer selections for different tasks may vary.

# 5 Experimental Settings

Single Task Evaluation for Desiderata We first evaluate the models' capacities for our desiderata Eq. (3) and Eq. (4) on single tasks. We test classification accuracies for desiderata of task performance on tasks from the GLUE benchmark [50] and SNLI data [5]. For the desiderata of global alignment, we predict top-20 tokens from the learned representation by the pre-trained decoder (global prototypes), and compute the ratio of rationle tokens in the top-20 predictions (i.e. Recall@20). We evaluate this on e-SNLI dataset [8], where data's rationale tokens [5] are highlighted by human annotators.

Continual Learning (CL) Evaluation We evaluate four sequences of tasks: (1) Yahoo 1: a split of Yahoo dataset for news question-answer categorization [57] with 5 disjoint tasks containing 2 classes each; (2) Yahoo 2: a Yahoo sequence with the same split as (1) but with more data; (3) DB: a split of DBPedia data for Wikipedia article classification [57] with 7 disjoint tasks containing 2 classes each; (4) News Series: a sequence of tasks on news-related data, including AG_news (news classification, 4 classes), MRPC (paraphrase detection, 2 classes) [12], RTE (text entailment, 2 classes) [52] and SST (sentiment analysis, 2 classes) [47]. For the above sequences except (2), we randomly sample 1245 samples per class, which is the least number of class samples in our datasets. For (2), we sample 10000 samples per class. We measure the average accuracy and forgetting (Appendix C) with standard deviations. For each sequence, we test five random orders of tasks.

We evaluate for both task-incremental and class-incremental learning. Task identifiers are available at inference time for task-incremental learning but not for class-incremental learning [37]. For class-incremental learning, the original cross-entropy loss over all seen classes will cause significant forgetting [54, 1]. Since our work does not focus on the problem of cross-entropy, we apply the asymmetric strategy (ACE) [7]: the current task's classification loss is calculated over in-task classes, while the replay loss is calculated over all seen classes in the memory (if applicable).

Models and CL Frameworks We compare different adaptation models on BERT-base. Data representation is from a [MASK] token added to the beginning of input to match the pre-training format. Models for comparison are: (1) NeiAttn: our standard neighbor attention model. (2) NeiReg: our neighbor attention model with extra regularization for holistic information (Appendix B). (3) Fine-tuning (FT): a model in which all parameters are learnable. (4) Prompt Tuning (ProT) [30]: the model adding learnable embeddings only to data inputs. (5) Prefix Tuning v2 (PT2) [33]: an adaptation model adding learnable embeddings to inputs of all attention layers. (6) Adapter [21]: an adaptation model with learnable linear projections injected in each layer. (7) BitFit [55]: an adaptation model tuning only bias terms in the pre-trained model. More settings are in the appendix.

We consider different frameworks (methods) for continual learning: (1) Vanilla: the vanilla online learning framework; (2) MBPA: an episodic memory framework retrieving stored samples to locally adapt the model at inference time [48]. (3) ER: an episodic memory framework storing all seen examples and performs sparse  $(1\%)$  experience replay; (4) A-GEM: an episodic memory framework constraining on gradients to prevent degrading performance of previous tasks [9]; (5) Probing: a framework which learns the encoder with Vanilla setting while tunes the classifier for each task using all task data. This is used to evaluate the discrimination of data representations; (6). MTL: a multi-task framework that jointly trains on all tasks (not continual learning). For class-incremental cases, we have the above replay-based methods combined with the ACE strategy. The baseline performance for each continual learning framework is that on FT model.

# 6 Experimental Results

Models for Desiderata in Eq.(3) and Eq.(4) Figure 3 shows models' capacities for our desiderata. We compare the classification accuracy for desiderata in Eq.(3) and Recall@20 of rationale tokens for desiderata in Eq.(4). The higher scores on both metrics, the better model capacities for our desiderata.

Overall, NeiAttn and PT2 consistently achieve a superior balance between classification and recall scores on different NLI tasks. However, Adapter and FT achieve high classification scores but do not generate representations well related to global prototypes (low recall scores). This supports our intuition that mixing pre-trained and modified representations with a gate can result representations better connected to global prototypes. With explicit regularization on holistic information, NeiReg

![](images/498143d5a39747577b59839892f502d618d1f86cab92302c8d47204cc1fc04ad.jpg)  
Figure 3: Results for single-task learning. Dashed lines split figure regions based on scores of NeiAttn. Results with higher accuracy and recall (upper right corner) are better. We test on three random seeds.  
Table 1: Results for task-incremental learning. We report average accuracy (Acc) and forgetting (Forget) with their standard deviations (std) on five random seeds. Bold scores are the best scores and underline scores are the second best. Models in blue have prototype loss larger than the threshold. Models in red satisfy the desiderata Eq. (4). Models with (*) are baselines for each CL framework.

<table><tr><td rowspan="2">CL Framework</td><td rowspan="2">Model</td><td colspan="2">Yahoo 1</td><td colspan="2">Yahoo 2</td><td colspan="2">DB</td><td colspan="2">News Series</td></tr><tr><td>Acc std</td><td>Forget std</td><td>Acc std</td><td>Forget std</td><td>Acc std</td><td>Forget std</td><td>Acc std</td><td>Forget std</td></tr><tr><td rowspan="5">Vanilla</td><td>Pretrained</td><td>82.953.64</td><td>7.343.64</td><td>83.704.16</td><td>7.714.15</td><td>95.382.34</td><td>4.082.37</td><td>66.664.47</td><td>5.353.06</td></tr><tr><td>FT (*)</td><td>73.075.32</td><td>18.675.41</td><td>79.824.29</td><td>13.274.25</td><td>73.155.36</td><td>24.905.17</td><td>59.988.94</td><td>21.137.44</td></tr><tr><td>Adapter</td><td>79.851.83</td><td>11.861.83</td><td>71.902.45</td><td>20.922.47</td><td>98.701.10</td><td>1.191.10</td><td>65.434.73</td><td>15.534.29</td></tr><tr><td>PT2</td><td>88.620.80</td><td>3.040.79</td><td>90.640.76</td><td>2.380.71</td><td>99.830.04</td><td>0.070.04</td><td>75.030.97</td><td>6.130.98</td></tr><tr><td>NeiAttn</td><td>88.961.14</td><td>2.801.12</td><td>89.840.70</td><td>3.240.69</td><td>97.343.41</td><td>2.543.41</td><td>71.952.20</td><td>9.892.29</td></tr><tr><td rowspan="4">MBPA</td><td>FT (*)</td><td>72.404.42</td><td>19.344.49</td><td>78.713.29</td><td>14.383.26</td><td>73.015.45</td><td>25.045.27</td><td>60.608.30</td><td>20.526.67</td></tr><tr><td>Adapter</td><td>78.502.12</td><td>13.132.09</td><td>73.662.95</td><td>19.152.95</td><td>99.091.10</td><td>0.801.10</td><td>65.284.74</td><td>15.674.11</td></tr><tr><td>PT2</td><td>90.690.78</td><td>0.970.75</td><td>91.700.51</td><td>1.330.58</td><td>99.900.06</td><td>-0.010.06</td><td>76.160.81</td><td>4.991.46</td></tr><tr><td>NeiAttn</td><td>90.691.36</td><td>1.671.35</td><td>91.180.90</td><td>1.900.88</td><td>97.533.28</td><td>2.353.28</td><td>73.282.53</td><td>8.562.11</td></tr><tr><td rowspan="4">ER</td><td>FT (*)</td><td>70.776.72</td><td>20.926.72</td><td>90.310.72</td><td>2.670.67</td><td>91.058.74</td><td>8.758.69</td><td>70.445.87</td><td>10.934.85</td></tr><tr><td>Adapter</td><td>77.443.39</td><td>14.133.42</td><td>75.793.44</td><td>17.083.39</td><td>98.921.54</td><td>0.971.54</td><td>68.111.96</td><td>13.162.10</td></tr><tr><td>PT2</td><td>88.910.42</td><td>2.760.35</td><td>91.020.50</td><td>2.190.78</td><td>99.840.04</td><td>0.030.03</td><td>69.603.06</td><td>11.582.96</td></tr><tr><td>NeiAttn</td><td>84.023.10</td><td>7.873.12</td><td>91.540.22</td><td>1.520.24</td><td>99.680.18</td><td>0.200.18</td><td>75.050.94</td><td>7.310.48</td></tr><tr><td rowspan="4">A-GEM</td><td>FT (*)</td><td>87.561.32</td><td>4.111.40</td><td>89.980.71</td><td>3.170.68</td><td>84.4510.16</td><td>15.3410.12</td><td>75.066.17</td><td>5.484.01</td></tr><tr><td>Adapter</td><td>80.862.36</td><td>10.652.26</td><td>77.473.20</td><td>15.373.24</td><td>99.520.23</td><td>0.380.24</td><td>73.801.16</td><td>6.721.61</td></tr><tr><td>PT2</td><td>90.400.21</td><td>1.390.16</td><td>90.840.19</td><td>2.220.21</td><td>99.880.01</td><td>0.010.01</td><td>73.310.73</td><td>4.291.02</td></tr><tr><td>NeiAttn</td><td>90.470.26</td><td>1.380.21</td><td>91.350.43</td><td>1.810.47</td><td>98.223.48</td><td>1.663.49</td><td>77.071.56</td><td>4.430.85</td></tr><tr><td rowspan="4">Probing (classifier non-CL)</td><td>FT (*)</td><td>90.180.41</td><td>1.560.49</td><td>92.160.14</td><td>0.930.14</td><td>97.733.58</td><td>0.310.04</td><td>77.172.09</td><td>3.941.98</td></tr><tr><td>Adapter</td><td>91.110.25</td><td>0.510.25</td><td>88.987.25</td><td>3.847.28</td><td>99.870.01</td><td>0.020.01</td><td>78.470.76</td><td>2.491.70</td></tr><tr><td>PT2</td><td>91.490.12</td><td>0.170.09</td><td>92.810.11</td><td>0.210.11</td><td>99.890.01</td><td>0.010.01</td><td>77.620.32</td><td>3.531.06</td></tr><tr><td>NeiAttn</td><td>91.470.16</td><td>0.290.16</td><td>92.720.11</td><td>0.370.11</td><td>99.870.01</td><td>0.010.02</td><td>78.830.51</td><td>3.011.02</td></tr><tr><td>MTL (non-CL)</td><td>FT (*)</td><td>91.690.26</td><td>—</td><td>92.670.71</td><td>—</td><td>99.610.41</td><td>—</td><td>79.671.99</td><td>—</td></tr></table>

performs best in in-task (SNLI  $\rightarrow$  E-SNLI) rationale recalls, while losing its superiority in cross-task (GLUE  $\rightarrow$  E-SNLI) rationale recalls. This may suggest the explicit regularization may not generalize well across tasks. With prompts only in the input, ProT has insufficient capacity for task performance.

For desiderata Eq.(4), NeiAttn and PT2 perform much better than Adapter and FT. We set  $a_{\tau}$  to make NeiAttn and PT2 satisfy Eq.(4) while Adapter and FT fail to, then we evaluate them for CL scenarios.

Task-Incremental Learning We test models' capacities for task-incremental learning under different CL frameworks. Results are shown in Table 1. Models are split into two categories according to our desiderata (Eq.(4)) experiment above: (NeiAttn, PT2) which satisfy it and (FT, Adapters) in opposite.

In the vanilla setting, both PT2 and NeiAttn significantly outperform other models with minor forgetting. Adapter on most CL frameworks performs worse than PT2 and NeiAttn, marginally better than FT. This supports our claim that models learning representations better connected to global prototypes perform better in continual learning. Combined with ER and A-GEM, NeiAttn can improve more than PT2 in most cases. FT has significant improvement with replay but can also suffer from overfitting to the replay buffer (ER for Yahoo 1). We also evaluate on a probing framework with only the classifier retrained over task data to evaluate whether the forgetting will cause representations to lose separation. PT2 and NeiAttn also preserve the most separation of representations in this case.

In general, (NeiAttn, PT2) consistently outperform (FT, Adapter) under different CL frameworks. This supports that our desiderata Eq. (4) helps improve models' continual learning ability. NeiAttn performs better with replay. The capacity of models also depends on different data distributions in

the sequence. On News Series, when with replay, FT can even outperform PT2. This may happen because News Series includes data from similar distributions related to the news. And models should have the capacity to deal with knowledge transfer besides catastrophic forgetting.

Results for Class-Incremental Learning Figure 4 shows models' performance on class-incremental learning. PT2 and NeiAttn perform well in the vanilla case, where the training is the same as that for taskincremental learning. This indicates that they can address connections between classes from different tasks even without supervision. On the

![](images/c7cd3b742d997ff7ad64430fee9b52a2fee9da1903368e80404e37677ea9357b.jpg)  
Figure 4: Results on class incremental learning. Dashed lines show scores of a pre-trained model in the vanilla setting.

![](images/13cc7b3f15dac80c8ca0a399053f57fd9b414ce98e21b783531856b12335fb8e.jpg)

other side, Adapter and FT perform much worse in this case. Then we evaluate three frameworks with replay: one is the full ER-ACE [7] with experience replay at each step; one is the ER-ACE (sparse) with sparse experience replay; the other is the ACE strategy with only previous task's data stored in the replay (AGEM-ACE). We observe that performance on class-incremental learning heavily relies on the quality of replay. In most cases, FT, Adapter and NeiAttn can benefit more from the replay. We hypothesize that it is related to the fast adaptation ability related to linear projections.

Influence of Parameter-Efficiency With limited parameters, adaptation models have less risk of deviating fast from previously learned knowledge compared to FT, and thus may perform better in CL. However, different models' improvements come not just from having fewer trainable

Table 2: The ratio of models' learnable parameters compared to FT.

<table><tr><td>Models</td><td>FT</td><td>Bitfit</td><td>Adapter</td><td>ProT</td><td>PT2</td><td>NeiAttn</td></tr><tr><td>Parameters (%)</td><td>1</td><td>0.5</td><td>2.3</td><td>0.5</td><td>0.8</td><td>4.9</td></tr></table>

parameters. Table 2 shows the comparison of parameters in each model. NeiAttn has better performance in most cases compared to Adapter and Pre-trained models, which have fewer or no trainable parameters in the encoder. Even with more parameters, NeiAttn performs on par with PT2 with Vanilla and outperform PT2 with replay. NeiAttn also requires much less time to train (5 vs 20 epochs). These suggest the adaptation model structure will highly influence its performance on CL.

Visualization of Representations In Figure 5, we visualize NeiAttn and FT's data representations for class incremental DB under Vanilla and ER-ACE frameworks. Even trained with in-task classes, Vanilla NeiAttn can well disperse data representations. Learning a model includes learning the encoder (representations) and classifier (class vectors). The learned class vectors may not well align with representations even with replay (left bottom). We hypothesize this may result from different training paces for the encoder and classifier. For FT, the encoder quickly learns representations close to single class centroids, which may degrade the function of the classifier. However, with con

![](images/2cb41f25eb1d7a5fa0c483b77096ac101816d10606c08afc9508c3131c932e61.jpg)  
Figure 5: T-SNE plot of FT, NeiAttn representations. Triangles are class vectors.

nections to multiple different global prototypes, NeiAttn representations may not quickly move to one centroid. Therefore, it can better balance the training of the encoder and classifier (right bottom).

# 7 Conclusion

In this paper, we investigate models which consider potential connections between observed and unknown tasks to reduce disruptive updating in CL. Specifically, we learn task-specific data representations appropriately connected to a general-purpose representation base with global prototypes. For NLP tasks, the global prototypes can be obtained from a pre-trained language model. And the representation connected to global prototypes can be obtained by lightly adapting the pre-trained model. We investigate existing adaptation models and propose a neighbor attention model which combines advantages of existing models. Experimental results show that models learning representations appropriately connected to global prototypes have significantly less catastrophic forgetting in CL, even without using experience replay. Specifically, when neighbor attention is used, we suffer from less catastrophic forgetting than FT and Adapter, and surpass PT2 when experience replay is applied. We consider the main limitations of our work as: (1) requiring extra memory to compute neighbor attentions; (2) the optimal number of neighbor attention layers may vary for different tasks.

# References

[1] Ahn, H., Kwak, J., Lim, S., Bang, H., Kim, H., and Moon, T. Ss-il: Separated softmax for incremental learning. In Proceedings of the IEEE/CVF International conference on computer vision, pp. 844-853, 2021.  
[2] Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., and Tuytelaars, T. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 139-154, 2018.  
[3] Biesialska, M., Biesialska, K., and Costa-jussa, M. R. Continual lifelong learning in natural language processing: A survey. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 6523-6541, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.574. URL https://www.aclweb.org/anthology/2020.coling-main.574.  
[4] Biondi, N., Pernici, F., Bruni, M., Mugnai, D., and Bimbo, A. D. Cl2r: Compatible lifelong learning representations. ACM Transactions on Multimedia Computing, Communications and Applications, 18(2s):1-22, 2023.  
[5] Bowman, S. R., Angeli, G., Potts, C., and Manning, C. D. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 632-642, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. URL https://www.aclweb.org/anthology/D15-1075.  
[6] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
[7] Caccia, L., Aljundi, R., Asadi, N., Tuytelaars, T., Pineau, J., and Belilovsky, E. New insights on reducing abrupt representation change in online continual learning. arXiv preprint arXiv:2203.03798, 2022.  
[8] Camburu, O.-M., Rocktäschel, T., Lukasiewicz, T., and Blunsom, P. e-snli: Natural language inference with natural language explanations. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.  
[9] Chaudhry, A., Ranzato, M., Rohrbach, M., and Elhoseiny, M. Efficient lifelong learning with a-GEM. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Hkf2_sC5FX.  
[10] d'Autume, C. d. M., Ruder, S., Kong, L., and Yogatama, D. Episodic memory in lifelong language learning. arXiv preprint arXiv:1906.01076, 2019.  
[11] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
[12] Dolan, B. and Brockett, C. Automatically constructing a corpus of sentential paraphrases. In Third International Workshop on Paraphrasing (IWP2005), 2005.  
[13] Ermis, B., Zappella, G., Wistuba, M., Rawal, A., and Archambeau, C. Memory efficient continual learning with transformers. Advances in Neural Information Processing Systems, 35: 10629-10642, 2022.  
[14] Fini, E., Da Costa, V. G. T., Alameda-Pineda, X., Ricci, E., Alahari, K., and Mairal, J. Self-supervised models are continual learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9621-9630, 2022.  
[15] Gomez-Villa, A., Twardowski, B., Yu, L., Bagdanov, A. D., and van de Weijer, J. Continually learning self-supervised representations with projected functional regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3867-3877, 2022.

[16] Guo, Y., Liu, B., and Zhao, D. Online continual learning through mutual information maximization. In International Conference on Machine Learning, pp. 8109-8126. PMLR, 2022.  
[17] Gurbuz, M. B. and Dovrolis, C. Nispa: Neuro-inspired stability-plasticity adaptation for continual learning in sparse networks. arXiv preprint arXiv:2206.09117, 2022.  
[18] He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., and Neubig, G. Towards a unified view of parameter-efficient transfer learning. arXiv preprint arXiv:2110.04366, 2021.  
[19] He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
[20] Hou, S., Pan, X., Loy, C. C., Wang, Z., and Lin, D. Learning a unified classifier incrementally via rebalancing. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pp. 831-839, 2019.  
[21] Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and Gelly, S. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pp. 2790-2799. PMLR, 2019.  
[22] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.  
[23] Huang, Y., Zhang, Y., Chen, J., Wang, X., and Yang, D. Continual learning for text classification with information disentanglement based regularization. arXiv preprint arXiv:2104.05489, 2021.  
[24] Javed, K. and White, M. Meta-learning representations for continual learning. Advances in Neural Information Processing Systems, 32, 2019.  
[25] Ke, Z., Liu, B., Ma, N., Xu, H., and Shu, L. Achieving forgetting prevention and knowledge transfer in continual learning. Advances in Neural Information Processing Systems, 34:22443-22456, 2021.  
[26] Kemker, R. and Kanan, C. Fearnet: Brain-inspired model for incremental learning. arXiv preprint arXiv:1711.10563, 2017.  
[27] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521-3526, 2017.  
[28] Knoblauch, J., Husain, H., and Diethe, T. Optimal continual learning has perfect memory and is NP-hard. In III, H. D. and Singh, A. (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 5327-5337. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/knoblauch20a.html.  
[29] Lee, S.-W., Kim, J.-H., Jun, J., Ha, J.-W., and Zhang, B.-T. Overcoming catastrophic forgetting by incremental moment matching. arXiv preprint arXiv:1703.08475, 2017.  
[30] Lester, B., Al-Rfou, R., and Constant, N. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.  
[31] Li, X. L. and Liang, P. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.  
[32] Liu, T., Ungar, L., and Sedoc, J. Continual learning for sentence representations using conceptors. arXiv preprint arXiv:1904.09187, 2019.  
[33] Liu, X., Ji, K., Fu, Y., Du, Z., Yang, Z., and Tang, J. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv preprint arXiv:2110.07602, 2021.  
[34] Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., and Tang, J. Gpt understands, too. arXiv preprint arXiv:2103.10385, 2021.

[35] Lopez-Paz, D. and Ranzato, M. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017.  
[36] Madaan, D., Yoon, J., Li, Y., Liu, Y., and Hwang, S. J. Representational continuity for unsupervised continual learning. In International Conference on Learning Representations, 2021.  
[37] Masana, M., Liu, X., Twardowski, B., Menta, M., Bagdanov, A. D., and van de Weijer, J. Class-incremental learning: survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.  
[38] Pernici, F., Bruni, M., Baecchi, C., and Del Bimbo, A. Fix your features: Stationary and maximally discriminative embeddings using regular polytope (fixed classifier) networks. arXiv preprint arXiv:1902.10441, 2019.  
[39] Pernici, F., Bruni, M., Baecchi, C., Turchini, F., and Del Bimbo, A. Class-incremental learning with pre-allocated fixed classifiers. In 2020 25th International Conference on Pattern Recognition (ICPR), pp. 6259–6266. IEEE, 2021.  
[40] Pfeiffer, J., Kamath, A., Rückle, A., Cho, K., and Gurevych, I. AdapterFusion: Non-destructive task composition for transfer learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 487-503, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.39.  
[41] Radiya-Dixit, E. and Wang, X. How fine can fine-tuning be? learning efficient language models. In Chiappa, S. and Calandra, R. (eds.), Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pp. 2435-2443. PMLR, 26-28 Aug 2020.  
[42] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.  
[43] Razdaibiedina, A., Mao, Y., Hou, R., Khabsa, M., Lewis, M., and Almahairi, A. Progressive prompts: Continual learning for language models. arXiv preprint arXiv:2301.12314, 2023.  
[44] Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., and Hadsell, R. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.  
[45] Shi, H., GAO, J., Xu, H., Liang, X., Li, Z., Kong, L., Lee, S. M. S., and Kwok, J. Revisiting oversmoothing in BERT from the perspective of graph. In International Conference on Learning Representations, 2022.  
[46] Shin, H., Lee, J. K., Kim, J., and Kim, J. Continual learning with deep generative replay. arXiv preprint arXiv:1705.08690, 2017.  
[47] Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631-1642, 2013.  
[48] Sprechmann, P., Jayakumar, S. M., Rae, J. W., Pritzel, A., Badia, A. P., Uria, B., Vinyals, O., Hassabis, D., Pascanu, R., and Blundell, C. Memory-based parameter adaptation. arXiv preprint arXiv:1802.10542, 2018.  
[49] Sun, F.-K., Ho, C.-H., and Lee, H.-Y. Lamol: Language modeling for lifelong language learning. arXiv preprint arXiv:1909.03329, 2019.  
[50] Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJ4km2R5t7.

[51] Wang, Z., Zhang, Z., Lee, C.-Y., Zhang, H., Sun, R., Ren, X., Su, G., Perot, V., Dy, J., and Pfister, T. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 139-149, 2022.  
[52] Williams, A., Nangia, N., and Bowman, S. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1112–1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1101. URL https://www.aclweb.org/anthology/N18-1101.  
[53] Yoon, J., Yang, E., Lee, J., and Hwang, S. J. Lifelong learning with dynamically expandable networks. arXiv preprint arXiv:1708.01547, 2017.  
[54] Yu, L., Twardowski, B., Liu, X., Herranz, L., Wang, K., Cheng, Y., Jui, S., and Weijer, J. v. d. Semantic drift compensation for class-incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6982-6991, 2020.  
[55] Zaken, E. B., Ravfogel, S., and Goldberg, Y. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. arXiv preprint arXiv:2106.10199, 2021.  
[56] Zenke, F., Poole, B., and Ganguli, S. Continual learning through synaptic intelligence. In International Conference on Machine Learning, pp. 3987-3995. PMLR, 2017.  
[57] Zhang, X., Zhao, J., and LeCun, Y. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015.