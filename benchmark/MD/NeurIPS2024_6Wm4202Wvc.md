# Label Privacy in Split Learning for Large Models with Parameter-Efficient Training

Anonymous Author(s)

Affiliation

Address

email

# Abstract

As deep learning models become larger and more expensive, many practitioners turn to fine-tuning APIs. These web services allow fine-tuning a model between two parties: the client that provides the data, and the server that hosts the model. While convenient, these APIs raise a new concern: the data of the client is at risk of privacy breach during the training procedure. This challenge presents an important practical case of vertical federated learning, where the two parties perform parameter-efficient fine-tuning (PEFT) of a large model. In this study, we systematically search for a way to fine-tune models over an API while keeping the labels private. We analyze the privacy of LoRA, a popular approach for parameter-efficient fine-tuning when training over an API. Using this analysis, we propose P $^3$ EFT, a multi-party split learning algorithm that takes advantage of existing PEFT properties to maintain privacy at a lower performance overhead. To validate our algorithm, we fine-tune DeBERTa-v2-XXLarge, Flan-T5 Large and LLaMA-2 7B using LoRA adapters on a range of NLP tasks. We find that P $^3$ EFT is competitive with existing privacy-preserving methods in multi-party and two-party setups while having higher accuracy.

# 1 Introduction

One of the main reasons behind deep learning success is its ability to transfer knowledge between tasks [34]. When training a model for any particular problem, it is common to reuse previously trained models from other, related problems. In the past, this was typically done by downloading pre-trained model weights from public hubs, then fine-tuning the said models on the downstream task. However, as models grow larger and more compute-intensive, fine-tuning them locally becomes an increasingly difficult task. Furthermore, many recent models are not released, but instead made available as proprietary services.

When a model cannot be fine-tuned locally, many practitioners opt instead for the so-called fine-tuning APIs [27, 16, 6, 26]. These APIs are web services that host one or several pre-trained models and allow clients to perform limited fine-tuning. More specifically, APIs usually allow their clients to run parameter-efficient fine-tuning (PEFT), such as LoRA [15] or Prefix-tuning [21]. These techniques allow adapting a model to a dataset while training a relatively small number of additional weights, which is particularly important for large language or image generation models that have billions of parameters.

Although the fine-tuning APIs can be convenient, they also introduce new risk in terms of data privacy. When a client uses such API to train on sensitive data, they need to ensure that their data will stay private [7]. This is particularly important when dealing with patient's medical records, personal user data or trade secrets [24, 19]. The two main threats to data privacy are that the API provider obtains the private data and that a third party intercepts data in transit. Therefore, data privacy is

not guaranteed even if the API provider is trusted. Several recent works propose LLM fine-tuning protocols that establish a certain level of privacy for multi-party fine-tuning [42, 7, 22]. Unfortunately, these algorithms work for a narrow class of fine-tuning algorithms or assume that a client can run LLM training locally using an obfuscated version of the model, provided by a remote server [42]. As a result, these algorithms are impractical for our use case of fine-tuning over an API. The few algorithms that are suitable for API fine-tuning guarantee the privacy of input tokens [22], meaning that the attacker can infer private training labels.

In this work, we seek to alleviate this problem by designing a two-party fine-tuning protocol that performs standard parameter-efficient fine-tuning with privacy guarantees. We formulate our protocol as a special case of split learning (or vertical federated learning), where one side (server) holds the pre-trained model and the other (client) has private training data. More specifically, we focus on the privacy of client's training labels. While input privacy is often crucial, there are scenarios where input data is publicly available, such as social media user pages. In these cases, labels could include ad clicks (known to the social network) or financial information (known to a bank that matches social profiles to its internal records). This example further justifies the use of LLMs, as social media pages often contain substantial amounts of text, and LLMs excel at processing long-context data.

Instead of developing a specific privacy-preserving architecture, we seek algorithms that can work with popular existing models and PEFT algorithms. Furthermore, our approach relies on the properties of parameter-efficient fine-tuning. Notably, since the adapters are compact, both parties can maintain multiple sets of adapters and swap between them with relative ease. This allows us to design a PEFT-specific algorithm that can solve its task more effectively than general split learning strategies [18].

We summarize our main contributions as follows:

- We analyze Low-Rank Adaptation, a common parameter-efficient fine-tuning algorithm, from the perspective of label privacy in the split learning setup. We observe that, despite fine-tuning less than  $0.1\%$  of model parameters, PEFT algorithms leak client's training labels against simple attacks that work for modern pretrained transformers.  
- Based on our analysis, we formulate a framework for privacy-preserving parameter-efficient fine-tuning (P $^3$ EFT). This framework leverages the properties of PEFT to obfuscate the gradients and parameters communicated during fine-tuning with little impact on the fine-tuned model quality.  
- To verify the practical viability of  $\mathrm{P}^{3}\mathrm{EFT}$ , we conduct experiments on popular real-world PEFT workloads<sup>1</sup>. Specifically, we fine-tune DeBERTa-v2-XXL [13], Flan-T5-Large [4] and LLaMA-2 7B [35] on a set of standard language understanding problems. We find that, compared to prior split learning algorithms,  $\mathrm{P}^{3}\mathrm{EFT}$  can maintain label privacy throughout training with a significantly smaller accuracy drop.

# 2 Background

# 2.1 Federated learning and split learning

Privacy preservation in machine learning has been a subject of active study within several frameworks. An important branch of privacy-preserving learning methods is federated learning, or FL [24], which can be broadly described as an approach allowing several parties to train a model jointly without sharing their private data. In particular, vertical federated learning [12, 43] targets the scenario where different features (including the label) of each training instance are kept by different parties.

One of the most popular approaches to vertical FL for neural networks is split learning [10, 37], where each party stores its part of the overall model. To train the model in such an approach, it is only necessary to transfer the intermediate activations and the gradients between layers, while the data itself is stored at the premises of the participant hosting each layer. In this work, we focus on the two-party formulation of split learning, where one side stores the features for each example and another one stores the labels.

Recent works have investigated the setting of two-party split learning from the label leakage perspective [38, 28]: because the label party needs to pass the gradients of the loss function to the non-label party, it is possible for the latter party to deduce the labels by inspecting the gradients or activations or by hijacking the training procedure. Li et al. [18] provide a set of attack methods that allow recovering private labels and propose a defense mechanism that injects noise into the gradients; however, they test the approach on pretraining smaller models, and we study finetuning large models on private downstream data.

# 2.2 Parameter-efficient finetuning

The majority of large neural networks today are not trained with a specific task in mind: instead, they are pretrained on a general objective and then adapted for the downstream problem. Importantly, the growth in the size of foundation models has led to the increased popularity of parameter-efficient finetuning (PEFT) methods that adapt the model to a given task by training a small number of task-specific parameters. There are several prominent approaches to parameter-efficient finetuning, ranging from trainable prompts [21, 11], to residual adapters [14, 29]. We focus on Low-Rank Adaptation (or LoRA, 15), one of the most popular PEFT methods that adds extra parameters to each weight matrix in the form of a low-rank factorization (see Appendix C for a more detailed description). Such formulation allows LoRA adapters to be merged into the original weights after finetuning; this ability, combined with the simplicity of the method, has made LoRA a broadly popular approach in multiple domains. Still, the approach we propose can be applied to any PEFT method.

Several recent lines of work explore the problem of fine-tuning LLMs with privacy guarantees [44, 31]. Zhao et al. [46] analyze the viability of prompt tuning for federated learning, and Zhang et al. [45], Liu et al. [23] study PEFT algorithms in the setting of horizontal federated learning, that is, where multiple users train a shared model on their local private data. Another, more relevant research direction considers private fine-tuning in a vertical federated learning scenario, where participants hold different model layers [22, 40]. Most of these studies leverage the idea of differential privacy to prove an upper bound on how much information is leaked [8]. Unfortunately, these upper bounds are typically loose and do not match practical observations for real models. Furthermore, the majority of these studies only guarantees privacy of specific parts of the training procedure: for instance, Li et al. [22] only protects the input features, and not labels or model parameters. Finally, Xiao et al. [42] presents an alternative algorithm that protects client data by running the entire fine-tuning on client side by emulating the server-side model layers. While this approach is more holistic, it assumes that clients can run fine-tuning locally, which makes it impractical for many real-world users of LLM fine-tuning APIs. The primary distinction between our work and these studies is that we investigate parameter-efficient adaptation in the setting of split learning: we aim to finetune a model without disclosing the labels of examples to the model provider.

# 3 Privacy-preserving parameter-efficient fine-tuning

In this section, we analyze the privacy of parameter-efficient fine-tuning and propose a protocol for two-party parameter-efficient fine-tuning with the desired privacy guarantees. We begin by analyzing the privacy of API fine-tuning with popular PEFT algorithms in Sections 3.1 and 3.2. Then, in Section 3.3, we formulate a protocol for privately computing gradients over fine-tuning APIs. Finally, we formulate the full  $\mathrm{P}^3\mathrm{EFT}$  protocol in Section 3.4.

# 3.1 Setup

To analyze the privacy of API fine-tuning, we first need to formulate a common framework for this type of APIs and develop private learning protocols. This step is important, because existing fine-tuning APIs greatly vary in what they offer to the client: from closed APIs that require users to submit their full training data [27] to more flexible APIs where clients can run individual training steps [20, 2, 30]. Similarly to most existing works on split learning, we focus on the latter type of APIs that allows clients to run individual forward and backward passes over a remote model. Thus, a client can use these APIs to obtain the training gradients for their PEFT adapters, then update adapters locally with any optimization method. In our work, we adopt this archetype of fine-tuning API as it offers sufficient flexibility to develop privacy-preserving algorithms.

We formulate fine-tuning over an API for two or more parties: a client, and one or several servers. The client owns a training dataset with inputs  $X$  and labels  $Y$ . In turn, each server has the same pre-trained model  $h(x_{i},\theta)\in \mathcal{R}^{d}$ . Note that the parameters  $\theta$  denote not the pre-trained model

![](images/b304a1d98718cf77a60f3810687164653ec0379c6a62fc2c367e9c3df9f04b4b.jpg)

![](images/e45615c867475a9df66a4fbb944848ebed98a49c0d59fcbef3e2330c58ac3153.jpg)

![](images/543cec656a6c4fa730c9bc07c74d6bfb3fd76fca13529dd891766ad54ffb9b1f.jpg)

![](images/973f03c9e640c18df31d581bfcb1275283ad847f1ae3863503278e7def6abdd7.jpg)

![](images/d179c718e39267ab949d15829475e1f220997266a14aa7dcb0c092409fcf7e4c.jpg)  
Figure 1: A visualization of top-2 principal components of gradients (top) and activations (bottom) from different fine-tuning steps (left to right). Color indicates the training labels (binary).

![](images/09c6a6f93fe0015bb04a4e1395b646406a383efef98d537150211f28f1221a1a.jpg)

![](images/1ea9c0b8f873ff80a267043123034b2ea1f7d9b1614aeaff0e66411c31c848fe.jpg)

![](images/67a99662768a72137826624f30cd6691b5231f3c4f010bc0bcada98115bd199d.jpg)

weights, but the trainable adapter weights for a certain PEFT algorithm. A model can encode an input  $x_{i} \in X$  and produce a  $d$ -dimensional vector of activations that depend on the learned adapter weights  $\theta$ .

To allow fine-tuning, a server offers two API methods:

1.  $\mathbf{forward}(x,\theta)\to h(x,\theta)$  that computes model activations on input  $x$  using adapter weights  $\theta$ ;  
2. backprop(x,θ,gh)→gθ that receives gradients of an arbitrary loss function w.r.t. model activations gh =  $\frac{\partial L(h(x,\theta))}{\partial h(x,\theta)}$  and returns the gradients w.r.t. adapter parameters,  $g_{\theta} = \frac{\partial L(h(x,\theta))}{\partial\theta}$ .

We further assume that both forward(·) and backprop(·) APIs are stateless and deterministic, i.e. calling the same API method multiple times (or on multiple servers) with the same inputs produces identical results. Thus, if the model uses dropout or any other form of non-determinism, we assume that clients provide the random seed as a part of  $x$ .

To fine-tune a model with this API, a client can initialize adapters locally, alongside with a small task-specific head $^2$ , then train both adapters and the head. For each training batch  $(x,y) \in D$ , a client calls forward  $(x,\theta)$  to compute feature representations, then predicts with local "head" and computes task-specific loss function  $L$ . After that, a client performs backward pass: first, it computes gradients w.r.t. local head inputs  $g_h = \frac{\partial L}{\partial h}$ , then passes those gradients to a remote server via backprop  $(x,\theta,g_h)$  API call to compute gradients w.r.t.  $\frac{\partial L}{\partial\theta}$ . Finally, a client updates both  $\theta$  and local "head" parameters using the optimizer of choice.

Before building more advanced algorithms, let us analyze the privacy of client's labels under standard fine-tuning. We consider an "honest, but curious" attacker model. This means that the server will faithfully run the forward and backprop computations as requested by the client without changing the results. Furthermore, we assume that servers are independent and do not communicate client's data between each other. However, a server can recover client's labels by performing arbitrary computations using any information it receives from the client. When training in this way, a client does not directly communicate training labels to the server. However, it communicates inputs, adapter parameters, and gradients. Furthermore, the server communicates input representations that can be intercepted by a third party.

# 3.2 Label Leakage of Standard Split Learning

In Figure 1, we train a DeBERTa-v2-XXL model on the SST-2 [32] sentiment classification dataset. The top row depicts the gradients  $g_{h}$  communicated by the client when calling backprop(·) at different training stages. In the bottom row, we similarly track activations  $h(x,\theta)$  that server may compute based on the specified  $x,\theta$ . We defer further additional figures and details to Section 4.1.

As we can see, both gradients and activations are arranged in such a way that simple k-means clustering would reveal which objects have the same label. The training activations (bottom row) do

![](images/4ea9779b6f75186a17a91f53dbd2b66f6d44e6d29e85082d4c2dd48da55bb0d9.jpg)  
Figure 2: An intuitive illustration of the proposed fine-tuning protocol.

not reveal labels right away (at least not against this attack). However, they gradually "leak" private label information during training. Informally, it appears that the training gradients gradually pull apart the feature representations for each label, until eventually they turn into separate clusters. From an information-theoretic perspective, knowing just one vector of gradients or trained activations allows the attacker to learn all but one bit $^3$  of information about client's private labels.

To summarize, leaving any one data source unprotected (gradients, activations or parameters) would already compromise label privacy. However, we found that gradients and activations require different means of protection.

# 3.3 Privacy-preserving backpropagation

In this section, we formulate an algorithm for "anonymizing" the gradients communicated over a single training step with arbitrary PEFT type. Several prior works approach this by modifying the training objective or model architecture. However, when dealing with a real-world PEFT workload with optimized hyperparameters, changing the model or loss function often results in reduced model accuracy<sup>4</sup>. Thus, we seek an algorithm that preserves both model and training objective.

We design our algorithm based on an observation that backpropagation is conditionally linear in output gradients, even when the model itself is nonlinear. Formally, if we take a model  $h(\cdot ,\cdot)$ , a fixed set of trainable parameters  $\theta$  and input samples  $x$ , the backprop function computes backprop(x,θ,∂L/∂h(x,θ)) = ∂L/∂θ. For convenience, we shorten it to backprop(x,θ,gh) = gh, where  $g_{h} = \frac{\partial L}{\partial h(x,\theta)}$  represents the gradients of some objective function with respect to model activations (outputs), and  $g_{\theta} = \frac{\partial L}{\partial\theta}$  are gradients of the same objective function w.r.t. trainable parameters. In this notation, backprop is linear in terms of  $g_{h}$  for any fixed  $x,\theta$ .

This becomes self-evident if we view backprop as multiplying  $g_{h}$  by the Jacobian of model outputs w.r.t. trainable parameters,  $\frac{\partial h(x,\theta)}{\partial\theta}$ . If  $x,\theta$  are constant, the Jacobian is also constant, and backprop is a linear operator:

$$
\operatorname {b a c k p r o p} (x, \theta , \frac {\partial L}{\partial h (x , \theta)}) = \frac {\partial L}{\partial \theta} = \frac {\partial L}{\partial h (x , \theta)} \times \frac {\partial h (x , \theta)}{\partial \theta}. \tag {1}
$$

This observation allows us to design a private backpropagation protocol. To illustrate this protocol, let us first consider a distributed API with two identical independent servers that offer backprop API. Then, for arbitrary vector  $z$ , we can rewrite  $\text{backprop}(x, \theta, g_h)$  as  $\text{backprop}(x, \theta, g_h + z) + \text{backprop}(x, \theta, g_h - z)$ .

During API fine-tuning, we obtain  $\mathrm{backprop}(x,\theta ,g_h + z)$  using an API call to server 1, whereas the second term  $\mathrm{backprop}(x,\theta ,g_h - z)$  translates to an API call to server 2. Note that neither of two servers has access to the true gradient  $g_{h}$ : they only receive the sum  $[g_h + z]$ . If we sample a large noise vector  $z\left(\operatorname {Var}(z)\gg \| g_h\| _2^2\right)$ , this sum also becomes dominated by noise. However, when both API calls finish, a client can sum the results to recover the true gradient of the loss with respect to parameters.

If both requests are processed by the same server, it can obviously recover  $g_{h}$  by adding up gradients from both calls, which leads us to the final step. Instead of generating a single noise vector, a client

needs to generate (privately) a set of  $m > 1$  random vectors  $\hat{g}_h^1, \dots, \hat{g}_h^m$  and scalars  $\alpha_1, \dots, \alpha_m$  such that

$$
g _ {h} = \sum_ {i = 1} ^ {m} \alpha_ {i} \cdot \hat {g} _ {h} ^ {i}. \tag {2}
$$

Then, for each  $\hat{g}_h^i$ , client computes backprop  $(x,\theta ,\hat{g}_h^i)$  as  $m$  parallel API calls. Once this is done, client recovers

$$
g _ {\theta} = \sum_ {i = 1} ^ {m} \alpha_ {i} \cdot \operatorname {b a c k p r o p} \left(x, \theta , \hat {g} _ {h} ^ {i}\right). \tag {3}
$$

Note that the client does not reveal  $\alpha_{1},\ldots ,\alpha_{m}$  to anyone.

The resulting procedure is formulated in Algorithm 1. This algorithm is conceptually similar to the secure aggregation protocol for conventional (horizontal) federated learning [1]. This protocol allows clients to average their local vector with peers while keeping each individual vector provably private. Similarly to our scheme, clients perturb the vector in such a way that the average of perturbed vectors remains the same. Unlike Bonawitz et al. [1], our protocol privately backpropagates through a server-hosted model by leveraging the conditional linearity of the backpropagation operator.

Algorithm 1 private_backprop — Privacy-Preserving Backpropagation (from the client's perspective)  
1: Input:  $x$  inputs,  $\theta$  adapter weights,  $g_{h}$  gradients w.r.t. activations,  $m > 1$  - number of passes  
2:  $\hat{g}_h^1,\dots ,\hat{g}_h^m,\alpha_1,\dots ,\alpha_m = \mathrm{obfucate}(g_h,m)$   
3: for  $j = 1,\ldots ,m$  do  
4:  $\hat{g}_{\theta}^{j} = \mathrm{backprop}(x,\theta ,\hat{g}_{h}^{j})$   
5: end for  
6:  $g_{\theta} = \sum_{j = 1}^{m}\alpha_{j}\cdot \hat{g}_{\theta}^{j}$   
7: Return:  $g_{\theta}$

The private backpropagation algorithm can allow client to safely compute gradients once, but, in practice, client usually needs to run many consecutive steps. This creates an additional vector of attack: if the same server receives two sets of parameters  $\theta_t, \theta_{t+1}$ , they could potentially recover  $g_\theta$  by inverting the optimizer.

In the simplest case, if the server somehow knows that the client computes  $\theta_{t + 1} = \theta_t - \eta \cdot g_\theta$ , then they can compute  $g_{\theta} = (\theta_t - \theta_{t + 1}) / \eta$ . While  $g_{\theta}$  does not necessarily leak private labels, a server could, in some cases, use  $g_{\theta}$  to recover  $g_{h}$ , either fully (e.g. if Jacobian is invertible), or partially.

The client has two ways to prevent this attack. The first one is to ensure that no single server runs backprop on two consecutive steps. This is easy to do in decentralized systems where there are many potential servers. However, even when there is a single server, they could be required to set up multiple trusted execution environments [25]. A more risky alternative is to ensure that the gradients cannot be reversed from consecutive parameters: randomize initial optimizer statistics or add noise to parameters. This solution is easier, but it can slow down training in some cases.

To summarize, we formulated a procedure that allows a client to compute gradients privately for any given model and PEFT type. Furthermore, since Equation 3 recovers true gradients, this obfuscation method does not affect the training dynamics. However, as we have shown in Section 3.1, gradients are not the only source of privacy leakage.

# 3.4 Full fine-tuning

The other major attack vector are training activations. As the model fits to training data, it's intermediate activations  $h(x,\theta)$  allow attackers to recover labels, e.g. by clustering (see Figure 1). To combat this issue, we take advantage of the fact that PEFT has few trainable parameters. Instead of learning just one set of trainable parameters, a client creates  $n$  independent adapter sets  $\theta_{1},\ldots ,\theta_{n}$ . Note that this does not require  $n$  unique servers: a single server can run multiple sets of adapters. Furthermore, a client can alternate between using different servers for the same adapters. During forward pass, the outputs of different adapters are mixed together using randomized mixing weights  $W\in \mathcal{R}^{n,d}$ :

$$
h ^ {\prime} \left(x, \theta_ {1}, \dots , \theta_ {n}\right) = \sum_ {i = 1} ^ {n} W _ {i} \odot h \left(x, \theta_ {i}\right) \tag {4}
$$

Overall, we design this model in such a way the combined model  $h^\prime$  can predict the labels, but the adapters  $h(x,\theta_i)$  do not allow predicting these labels without knowing the mixing weights W. The mixing weights are generated such that initial activations  $h^{\prime}(x,\ldots)$  are equal to mean  $h(x,\cdot)$  for all  $x$ . To achieve this, we generate W as follows: first, we generate  $n\cdot (n - 1) / 2$  d-dimensional random vectors  $\xi_{i,j}\in \mathcal{R}^{d}\forall i\in [1,n],j\in [i + 1,n]$ . Then, we add them up in the following way:

$$
W = \left( \begin{array}{c} \frac {1}{n} e + \xi_ {1, 2} + \xi_ {1, 3} + \dots + \xi_ {1, n} \\ - \xi_ {1, 2} + \frac {1}{n} e + \xi_ {2, 3} + \dots + \xi_ {2, n} \\ \dots \\ - \xi_ {1, n} - \xi_ {2, n} - \xi_ {3, n} - \dots + \frac {1}{n} e \end{array} \right) \tag {5}
$$

Here,  $e$  stands for a vector of all ones. The purpose of these mixing weights is to ensure that the gradients w.r.t. individual  $h(x,\theta_i)$  are obfuscated, but the averaged model behaves the same as regular PEFT adapter. To illustrate this, consider  $n = 2$  identical LoRA adapters  $\theta_{1},\theta_{2}$ . During the first training step  $h(x,\theta_1) = h(x,\theta_2)$ . Therefore,

$$
h ^ {\prime} \left(x, \theta_ {1}, \dots , \theta_ {n}\right) = \left(1 / 2 e + \xi_ {1, 2}\right) \odot h \left(x, \theta_ {1}\right) + \left(1 / 2 e - \xi_ {1, 2}\right) \odot h \left(x, \theta_ {2}\right) = h \left(x, \theta_ {1}\right) \tag {6}
$$

However, the two adapters will learn different functions as they receive different gradients. From the first update on,  $h'$  will be equal to an average of adapter predictions.

Finally, to ensure that individual adapters  $h(x,\theta)$  do not accidentally "learn to leak" labels, we maintain this over the course of training with a privacy regularizer inspired by [9]. This ensures that it is impossible to predict labels from individual adapters  $h(x,\theta_i)$ . Intuitively, on each training step, client fits  $n$  linear "heads" that learn to predict labels  $y$  from  $h(x,\theta_i)$ , then performs an adversarial update of  $\theta_i$  to prevent the "head" from predicting  $y$ . Formally, each of  $n$  "heads" minimize the same objective function as the full model. For instance, if the full model solves multi-class classification, each head is trained to minimize cross-entropy:

$$
\eta_ {i} ^ {*} = \arg \min  _ {\eta_ {i}} \sum_ {x, y \in D} - y \cdot \log \frac {e ^ {\langle \eta_ {i j} , h (x , \theta_ {i}) \rangle}}{\sum_ {k} e ^ {\langle \eta_ {i k} , h (x , \theta_ {i}) \rangle}}, \tag {7}
$$

where  $y$  is one-hot encoding of the correct class.

The whole adversarial update takes place locally on client's side, using the same  $h(x,\theta)$  it uses for the main training objective. The resulting procedure appears complicated but it typically takes negligible time compared to running the large pre-trained model  $h(x,\theta)$ . Furthermore, since adversarial "heads" are linear, minimizing the objective above is done with standard logistic regression solver.

To summarize, our approach combines the two proposed ideas: we use the private backpropagation algorithm from Section 3.3 to protect the gradients, then trains a mixture of adapters in such a way that obfuscates learned activators leaking labels. The resulting procedure is described in Algorithm 2. In the next section, we will evaluate the efficacy of  $\mathrm{P}^3\mathrm{EFT}$  on popular NLP benchmarks.

# 4 Experiments

The main goal of our study is to find a practical method of private fine-tuning that would scale to large models. Because our approach leverages parameter-efficient fine-tuning techniques, we evaluate  $\mathrm{P^3EFT}$  with fine-tuning Transformer models on popular NLP benchmarks that these techniques were designed for.

To that end, we chose three pre-trained models: DeBERTa-XXLarge [13], Flan-T5-Large [4] and LLaMA-2 7B [35]. We train these models on several datasets from the GLUE benchmark [39]: SST-2 [32], MNLI [41] and QNLI.

# 4.1 Privacy of gradients and activations

For this experiment, we train DeBERTa-XXLarge on SST-2 dataset using LoRA adapters with hyperparameters from [15]. First, we train the model locally and track model activations  $h$  and gradients w.r.t. those activations. We apply principal component analysis to them and plot the first

![](images/395add61508e61cae93c1e76a1eeff9bc5bd03e4daadd9048c70bf98581be3be.jpg)  
Step: 0

![](images/d2b1667b304a33cc999bea6e31bb57e2a972bfd4c7d92ce662d33edcb73a7b77.jpg)  
Step: 1000

![](images/a231a1cc72a0527e071f6884595b2ebd9df6db5baa86bd29da64f0189b98e99f.jpg)  
Step: 4000

![](images/c5c79ff850222c7324b6a812cc5d4190f9b87dacae0cdcdfaaa0fa2e25ca0e45.jpg)  
Step: 16000

![](images/d7b0ca95da2a40c9a259664ab29b7590612bf6cbee7007017ff6652656ce3028.jpg)  
Step: 0  
Figure 3: Gradients of cross-entropy w.r.t. LoRA parameters for DeBERTa-v2-XXLarge. The top row corresponds to normal backpropagation and the bottom row uses privacy-preserving backprop.

![](images/c042f186e3b4a0ca13f098b3510b1ad32ebc89fa38e886c3edb2ec9c5e18a881.jpg)  
Step: 1000

![](images/93fcd8271194f09bbcd30b95fd12d7681422b677bf2ced8e3da1d3456d598856.jpg)  
Step: 4000

![](images/2c59351ad417b9da0355bbaeff129d5586c01377b2530449b71613dc5a25d101.jpg)  
Step: 16000

2 dimensions in Figure 1. Similarly, we visualize gradients of individual per-sample loss functions w.r.t. LoRA parameters  $\theta$  in Figure 3 (top row). The results suggest that a hypothetical attacker could easily recover private labels by performing K-Means clustering over any data source: activations, gradients with respect to activations, or individual gradients with respect to parameters.

Next, we run the same experiment using privacy-preserving backpropagation as defined in Section 3.3. We use  $n = 2$  with the noise variance set to 1000. As expected, we observed the same learning curve as with normal training. However, instead of sending gradients w.r.t. activations to the server, a client uses specially crafted random noise vectors that are not informative. In Figure 3 (bottom) we plot the same kind of individual gradients as in the top row, except that we visualize the gradients computed by the first of the two servers. Finally, we train XGBoost [3] with default hyperparameters to predict labels given the noisy gradients (pre-PCA): the resulting classifier is able to fit the training data perfectly, but has at most  $50.4\%$  accuracy on a balanced test set.

# 4.2 Main fine-tuning experiments

Next, we evaluated the entire P3EFT algorithm. To control tasks and model type, we examined DeBERTa and Flan-T5 across all four datasets mentioned above, in addition to evaluating LLaMA on SST2 and QNLI datasets. For each setup, we compare against three baselines:

- Without LoRAs. In this baseline, the client gathers  $h$  activations at the beginning (with no adapters), then proceeds to train local "head" layers using these activations. This method cannot leak information about training labels except for what is stored in X.  
- Regular fine-tuning (Regular FT) refers to training a single LoRA adapter normally. This baseline represents an upper bound on model accuracy, but lacks privacy.  
- Distance Correlation (DC). Our re-implementation of the distance correlation defense formulated in [33] for Transformer models.

For each algorithm, we evaluated a task-specific metric (accuracy or F1), as well as the privacy leakage value for the 3 following measures:

- Spectral attack AUC — a measure of vulnerability to an attack proposed in [33], measured as classifier ROC AUC: lower value corresponds to better privacy.  
- Norm attack AUC — vulnerability to a variant of attack proposed in [18], measured as classifier ROC AUC (lower is better). Despite the initial proposal of this approach for attacking gradients, we observed that it is also well-suited for attacking activations.  
- K-means accuracy — vulnerability to clusterization attack, measured in the percentage of correctly clustered activations, lower is better.

For all setups, we report the worst (least private) value among these metrics throughout the entire training period as a measure of privacy leakage, because it is the worst possible scenario that matters from the client's perspective. For DC and  $\mathrm{P^3EFT}$ , we specify the values for the best configuration in terms of the utility-privacy trade-off. See details in Appendix A. We also report adjusted standard deviations for the two privacy aware algorithms:  $\mathrm{P^3EFT}$  and DC. To do so, we run the full training procedure from scratch with 3 random seeds.

Table 1: Accuracy and privacy metrics. DeBERTa XXLarge.  

<table><tr><td colspan="2">Dataset</td><td>Without LoRAs</td><td>Regular FT</td><td>DC</td><td>P3EFT</td></tr><tr><td rowspan="2">SST2</td><td>acc</td><td>82.9</td><td>96.9</td><td>96.6±0.4</td><td>96.5±0.2</td></tr><tr><td>leak</td><td>53.9</td><td>99.1</td><td>93.3±6.8</td><td>62.6±2.6</td></tr><tr><td rowspan="2">QNLI</td><td>acc</td><td>72.6</td><td>96.0</td><td>95.8±0.3</td><td>95.6±0.5</td></tr><tr><td>leak</td><td>51.5</td><td>99.1</td><td>85.0±11.6</td><td>74.6±11.1</td></tr><tr><td rowspan="2">MNLI</td><td>acc</td><td>49.2</td><td>91.9</td><td>—</td><td>86.9±0.5</td></tr><tr><td>leak</td><td>34.2</td><td>91.5</td><td>—</td><td>37.4±0.7</td></tr></table>

Table 2: Accuracy and privacy metrics. Flan-T5-Large.  

<table><tr><td colspan="2">Dataset</td><td>Without LoRAs</td><td>Regular FT</td><td>DC</td><td>P3EFT</td></tr><tr><td rowspan="2">SST2</td><td>acc</td><td>92.8</td><td>96.1</td><td>95.0±0.1</td><td>96.1±0.1</td></tr><tr><td>leak</td><td>55.8</td><td>98.3</td><td>68.1±5.0</td><td>74.1±3.0</td></tr><tr><td rowspan="2">QNLI</td><td>acc</td><td>83.2</td><td>95.3</td><td>95.2±0.1</td><td>94.7±0.0</td></tr><tr><td>leak</td><td>58.7</td><td>98.9</td><td>67.0±1.2</td><td>63.0±0.8</td></tr><tr><td rowspan="2">MNLI</td><td>acc</td><td>73.9</td><td>90.5</td><td>89.8±0.1</td><td>90.1±0.1</td></tr><tr><td>leak</td><td>34.6</td><td>85.9</td><td>45.6±0.8</td><td>40.0±1.1</td></tr></table>

The results for DeBERTa are presented in Table 1. To improve reproducibility, we reuse the hyperparameters from original paper, with the exception of the LoRA dropout value. We disable dropout because it interferes with the mixing weights (5). In preliminary experiments, we observed that with dropout enabled, both our algorithm and DC begin to perform significantly worse.

We use  $n = 2$  adapter sets for  $\mathrm{P}^3\mathrm{EFT}$  for all datasets and adhered to the same approach for the other models as well. Overall,  $\mathrm{P}^3\mathrm{FT}$  achieves nearly the same accuracy as traditional (non-private) fine-tuning, outperforming the DC-based algorithm in terms of accuracy given the same privacy level. On the MNLI dataset, we could not find the hyperparameters for DC that ensure stable training while maintaining privacy. Meanwhile,  $\mathrm{P}^3\mathrm{EFT}$  maintains consistent performance on this task with a slight drop in quality.

Table 2 a reports evaluation for the Flan-T5 base model[4]. For this model, we adapt the exact same hyperparameters as in the previous evaluation with DeBERTa-XXLarge. Compared to DeBERTa, these results are more closely matched. Both both our algorithm and DC consistently solve all three tasks, but  $\mathbf{P}^3\mathrm{EFT}$  slightly outperforms DC in terms of privacy.

Table 3: Accuracy and privacy metrics for LLaMA-2 7B.  

<table><tr><td colspan="2">Dataset</td><td>Without LoRAs</td><td>Regular FT</td><td>DC</td><td>P3EFT</td></tr><tr><td rowspan="2">SST2</td><td>acc</td><td>94.6</td><td>97.4</td><td>97.1±0.1</td><td>95.8±0.1</td></tr><tr><td>leak</td><td>59.1</td><td>99.3</td><td>83.6±10.6</td><td>68.9±2.6</td></tr><tr><td rowspan="2">QNLI</td><td>acc</td><td>77.0</td><td>95.0</td><td>95.2±0.1</td><td>94.7±0.2</td></tr><tr><td>leak</td><td>53.3</td><td>85.5</td><td>66.6±4.1</td><td>62.9±0.8</td></tr></table>

To evaluate how our algorithm scales to larger models, we also fine-tune Llama-2 7B [35] on SST2 [32] and QNLI [39] datasets. For these evaluations, we use LoRA hyperparameters that Hu et al. [15] used when fine-tuning GPT-3, with several changes inspired by Dettmers et al. [5]. Namely, we use the NF4 weight format, apply LoRA to both attention and MLP layers with rank 16. We fine-tune both tasks with maximum context length of 512 and weight decay 0.01. Table 3 summarizes our results: for QNLI,  $\mathrm{P}^3\mathrm{EFT}$  achieves somewhat better privacy-accuracy trade-off. On SST2,  $\mathrm{P}^3\mathrm{EFT}$  shows similarly favorable trade-offs while DC struggles to preserve privacy.

# 5 Conclusion and Discussion

In this work, we analyze privacy-preserving fine-tuning of large neural networks in the context of parameter-efficient fine-tuning and the two-party split learning setting. We show that while standard fine-tuning suffers from label leakage even in the parameter-efficient case, it is possible to leverage the efficiency of PEFT to alter the procedure without any significant performance drawbacks. We test the resulting method, named  $\mathrm{P}^3\mathrm{EFT}$ , on a range of pretrained language models and multiple datasets, showing that it is competitive with a strong baseline in terms of label privacy while having higher task performance.

In future work, it is natural to explore how this approach can be extended to establish holistic privacy in both labels and inputs. This problem can be approached from two directions: either adapt the ideas of  $\mathrm{P}^3\mathrm{EFT}$  for input privacy, or combine it with an existing work like [22]. Another important direction for future research is exploring the privacy of the long-term client-provider interaction. In a typical real-world use case of API fine-tuning, a client performs multiple training runs on overlapping data and hyperparameters. This could open additional attacks vectors that combine information from multiple training runs.

# References

[1] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacy-preserving machine learning. In proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pages 1175–1191, 2017.  
[2] Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel. Petals: Collaborative inference and fine-tuning of large models. arXiv preprint arXiv:2209.01188, 2022. URL https://arxiv.org/abs/2209.01188.  
[3] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pages 785-794, New York, NY, USA, 2016. ACM. ISBN 978-1-4503-4232-2. doi: 10.1145/2939672.2939785. URL http://doi.acm.org/10.1145/2939672.2939785.  
[4] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022. URL https://arxiv.org/abs/2210.11416.  
[5] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.  
[6] Dreambooth API. Dreambooth API - Easily finetune Stable Diffusion and generate customised AI images — dreamboothapi.ai. https://dreamboothapi.ai/, 2023. [Accessed 28-09-2023].  
[7] Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. Flocks of stochastic parrots: Differentially private prompt learning for large language models. arXiv preprint arXiv:2305.15594, 2023.  
[8] Cynthia Dwork. Differential privacy. In International colloquium on automata, languages, and programming, pages 1-12. Springer, 2006.  
[9] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 1180-1189, Lille, France, 07-09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/ganin15.html.  
[10] Otkrist Gupta and Ramesh Raskar. Distributed learning of deep neural network over multiple agents. Journal of Network and Computer Applications, 116:1-8, 2018. ISSN 1084-8045. doi: https://doi.org/10.1016/j.jnca.2018.05.003. URL https://www.sciencedirect.com/science/article/pii/S1084804518301590.  
[11] Karen Hambardzumyan, Hrant Khachatrian, and Jonathan May. WARP: Word-level Adversarial ReProgramming. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4921-4933, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.381. URL https://aclanthology.org/2021.acl-long.381.  
[12] Stephen Hardy, Wilko Henecka, Hamish Ivey-Law, Richard Nock, Giorgio Patrini, Guillaume Smith, and Brian Thorne. Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption, 2017.  
[13] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=XPZIaotutsD.

[14] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790-2799. PMLR, 09-15 Jun 2019. URL https://proceedings.mlrpress/v97/houlsby19a.html.  
[15] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.  
[16] Hugging Face. AutoTrain — huggingface.co. https://huggingface.co/autotrain, 2023. [Accessed 28-09-2023].  
[17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
[18] Oscar Li, Jiankai Sun, Xin Yang, Weihao Gao, Hongyi Zhang, Junyuan Xie, Virginia Smith, and Chong Wang. Label leakage and protection in two-party split learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=c0tBRgsf2f0.  
[19] Qinbin Li, Zeyi Wen, Zhaomin Wu, Sixu Hu, Naibo Wang, Yuan Li, Xu Liu, and Bingsheng He. A survey on federated learning systems: Vision, hype and reality for data privacy and protection. IEEE Transactions on Knowledge and Data Engineering, 2021.  
[20] Shen Li, Pritam Damania, Luca Wehrstedt, and Rohan Varma. PyTorch RPC: Distributed Deep Learning Built on Tensor-Optimized Remote Procedure Calls. In Proceedings of Machine Learning and Systems 5 (MLSys), 2023.  
[21] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582-4597, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.353. URL https://aclanthology.org/2021.acl-long.353.  
[22] Yansong Li, Zhixing Tan, and Yang Liu. Privacy-preserving prompt tuning for large language model services. ArXiv, abs/2305.06212, 2023. URL https://api-semanticscholar.org/CorpusID:258588141.  
[23] Xiao-Yang Liu, Rongyi Zhu, Daochen Zha, Jiechao Gao, Shan Zhong, and Meikang Qiu. Differentially private low-rank adaptation of large language model using federated learning. arXiv preprint arXiv:2312.17493, 2023.  
[24] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Aarti Singh and Jerry Zhu, editors, Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 1273-1282. PMLR, 20-22 Apr 2017. URL https://proceedings.mlr.press/v54/mcmahan17a.html.  
[25] Nvidia. Nvidia confidential computing. https://www.nvidia.com/en-us/data-center/solutions/confidential-computing, 2023. [Accessed 28-09-2023].  
[26] OctoAI. Fine-tuning Stable Diffusion — docs.octoai.cloud. https://docs.octoai.cloud/docs/fine-tuning-stable-diffusion, 2023. [Accessed 28-09-2023].  
[27] OpenAI. OpenAI Platform — platform.openai.com. https://platform.openai.com/docs/guides/fine-tuning, 2023. [Accessed 28-09-2023].

[28] Dario Pasquini, Giuseppe Ateniese, and Massimo Bernaschi. Unleashing the tiger: Inference attacks on split learning. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security, CCS '21, page 2113-2129, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450384544. doi: 10.1145/3460120.3485259. URL https://doi.org/10.1145/3460120.3485259.  
[29] Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. Adapterfusion: Non-destructive task composition for transfer learning, 2021.  
[30] Yuma Rao, Jacob Steeves, Ala Shaabana, Daniel Attevelt, and Matthew McAteer. Bittensor: A peer-to-peer intelligence market, 2021.  
[31] Weiyan Shi, Ryan Shea, Si Chen, Chiyuan Zhang, Ruoxi Jia, and Zhou Yu. Just fine-tune twice: Selective differential privacy for large language models. arXiv preprint arXiv:2204.07667, 2022.  
[32] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631-1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/D13-1170.  
[33] Jiankai Sun, Xin Yang, Yuanshun Yao, and Chong Wang. Label leakage and protection from forward embedding in vertical federated learning. arXiv preprint arXiv:2203.01451, 2022.  
[34] Chuanqi Tan, Fuchun Sun, Tao Kong, Wenchang Zhang, Chao Yang, and Chunfang Liu. A survey on deep transfer learning. In Věra Kürková, Yannis Manolopoulos, Barbara Hammer, Lazaros Iliadis, and Ilias Maglogiannis, editors, Artificial Neural Networks and Machine Learning - ICANN 2018, pages 270-279, Cham, 2018. Springer International Publishing. ISBN 978-3-030-01424-7.  
[35] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.  
[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.  
[37] Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and Ramesh Raskar. Split learning for health: Distributed deep learning without sharing raw patient data, 2018.  
[38] Praneeth Vepakomma, Otkrist Gupta, Abhimanyu Dubey, and Ramesh Raskar. Reducing leakage in distributed deep learning for sensitive health data. 05 2019.  
[39] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.  
[40] Yiming Wang, Yu Lin, Xiaodong Zeng, and Guannan Zhang. Privatelora for efficient privacy preserving lIm. arXiv preprint arXiv:2311.14030, 2023.  
[41] Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage challenge corpus for sentence understanding through inference. arXiv preprint arXiv:1704.05426, 2017.  
[42] Guangxuan Xiao, Ji Lin, and Song Han. Offsite-tuning: Transfer learning without full model. arXiv preprint arXiv:2302.04870, 2023.  
[43] Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. Federated machine learning: Concept and applications. ACM Trans. Intell. Syst. Technol., 10(2), jan 2019. ISSN 2157-6904. doi: 10.1145/3298981. URL https://doi.org/10.1145/3298981.

[44] Da Yu, Saurabh Naik, Arturs Backurs, Sivakanth Gopi, Huseyin A Inan, Gautam Kamath, Janardhan Kulkarni, Yin Tat Lee, Andre Manoel, Lukas Wutschitz, Sergey Yekhanin, and Huishuai Zhang. Differentially private fine-tuning of language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=Q42f0dfjEC0.  
[45] Zhuo Zhang, Yuanhang Yang, Yong Dai, Qifan Wang, Yue Yu, Lizhen Qu, and Zenglin Xu. FedPETuning: When federated learning meets the parameter-efficient tuning methods of pre-trained language models. In Findings of the Association for Computational Linguistics: ACL 2023, pages 9963-9977, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023-findings-acl.632. URL https://aclanthology.org/2023-findings-acl.632.  
[46] Haodong Zhao, Wei Du, Fangqi Li, Peixuan Li, and Gongshen Liu. Fedprompt: Communication-efficient and privacy preserving prompt tuning in federated learning, 2023.
