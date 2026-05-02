# CAUSE: POST-HOC NATURAL LANGUAGE EXPLANATION OF MULTIMODAL CLASSIFIERS THROUGH CAUSAL ABSTRACTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

The increasing integration of AI models in critical areas, such as healthcare, finance, and security has raised concerns about their "black-box" nature, limiting trust and accountability. To ensure robust and trustworthy AI, interpretability is essential. In this paper, we propose CAuSE (Causal Abstraction under Simulated Explanation), a novel framework for post-hoc explanation of multimodal classifiers. Unlike existing interpretability methods, such as Amnesic Probing and Integrated Gradients, CAuSE generates causally faithful natural language explanations of fine-tuned multimodal classifiers' decisions. CAuSE integrates Interchange Intervention Training (IIT) within a Language Model (LM) based module to simulate the causal reasoning behind a classifier's outputs. We introduce a novel metric Counterfactual F1 score to measure causal faithfulness and demonstrate that CAuSE achieves state-of-the-art performance on this metric. We also provide a rigorous theoretical underpinning for causal abstraction between two neural networks and implement this within our CAuSE framework. This ensures that CAuSE's natural language explanations are not only simulations of the classifier's behavior but also reflect its underlying causal processes. Our method is task-agnostic and achieves state-of-the-art results on benchmark multimodal classification datasets, such as e-SNLI-VE and Facebook Hateful Memes, offering a scalable, faithful solution for interpretability in multimodal classifiers.

# 1 INTRODUCTION

With the rise of Visual Language Models (VLMs), AI systems have evolved to handle multiple data types like images, text, and audio. Multimodal classifiers, central to this advancement, are crucial in applications such as healthcare, where they combine medical images and patient data to improve diagnostic accuracy for diseases like COVID-19 and Alzheimer's (Baltrusaitis et al., 2017). Similarly, in autonomous driving, they enhance decision-making by integrating visual, LiDAR, and radar inputs (Xiao et al., 2022). These classifiers boost performance by leveraging diverse modalities, making them vital in real-world scenarios.

However, as multimodal classifiers grow in complexity, the need for interpretability becomes paramount. Current interpretability methods, such as Integrated Gradients(Sundararajan et al., 2017a), are designed to highlight explicit input features but fall short of capturing the implicit causal relationships that often drive the decisions of these models. While some techniques, like CausaLM(Feder et al., 2022) and Amnesic Probing(Elazar et al., 2021), aim to incorporate causal mechanisms for interpretability, they struggle with scalability. Other methods, such as Semantify(Bandyopadhyay et al., 2024), manage implicit concepts efficiently but are restricted to specific use cases and fail to generate comprehensive natural language explanations.

To address these limitations, large Visual Language Models (VLMs) have been utilized to generate natural language explanations for decisions made by visual-text multimodal classifiers. However, these models often inject their own biases and opinions, leading to explanations that are inconsistent or detached from the actual workings of the classifier(Agarwal et al., 2024). Recent studies(Madsen et al., 2024) have highlighted these faithfulness issues, revealing inconsistencies when models are further probed.

In this paper, we introduce CAuSE (Causal Abstraction under Simulated Explanation), a novel framework designed to generate faithful natural language explanations for the decisions of a pretrained classifier, offering post-hoc interpretability. CAuSE combines Interchange Intervention Training(Geiger et al., 2021a) with Language Model (LM)-based modules, ensuring that the generated explanations are both causally accurate and reflective of the classifier's internal decision-making process. We introduce a new metric, the Counterfactual F1 score, to assess the causal faithfulness of explanations. CAuSE sets a new benchmark on this metric, achieving state-of-the-art performance. Through case studies, we showcase successful generations from our framework and conduct error analysis to identify common mistakes and their underlying causes.

Our framework is task-agnostic and demonstrates state-of-the-art performance on benchmark datasets, such as e-SNLI-VE(Do et al., 2021) and Facebook Hateful Memes(Kiela et al., 2021), providing robust, faithful explanations across diverse multimodal tasks. The codes are available at https://anonymous.4open.science/r/CAuSE-5BD0.

# 2 ARCHITECTURE

Our framework, CAuSE, generates faithful natural language explanations for decisions made by a pre-trained multimodal classifier (called the post-hoc classifier). As detailed in Section 3.2, CAuSE acts as a causal abstraction of the post-hoc classifier, ensuring its explanations are rooted in the actual decision-making process. This is supported by the high Counterfactual F1 scores CAuSE achieves compared to the other ablated components, as shown in Table 2. This section introduces the post-hoc classifier and provides a detailed description of the CAuSE framework, with a working diagram of both presented in Figure 1.

# 2.1 POST-HOC CLASSIFIER

The post-hoc classifier is assumed to be composed of a multimodal encoder  $E$  and a feed-forward neural network (FFN)  $\mathcal{C}_1$ .

Multimodal Encoder. The multimodal encoder  $E$  accepts as inputs the text  $(t \in \mathbb{R}^{m \times 1})$  and image representation  $(v \in \mathbb{R}^{m \times 1})$ . The image and text representation are fused via either i) early-fusion or ii) late-fusion modules. The final multimodal representation is denoted as  $c \in \mathbb{R}^{m \times 1}$ , where  $c = E(t, v)$ .

This module serves as a plug-and-play replacement for any multimodal encoder, whether based on early-fusion or late-fusion. In our implementation for this paper, we use a late-fusion-based module, which consists of CLIP(Radford et al., 2021) and MFB(Yu et al., 2017), as commonly adopted in the literature(Bandyopadhyay et al., 2024).

Classifier  $\mathcal{C}_1$ . The classifier gets the multimodal representation  $c$  and via a chain of feed-forward neural nets, it gets transformed into a vector  $z \in \mathbb{R}^{L \times 1}$ , where  $L$  is the number of classes in the output label. A softmax function is used which converts logit  $z$  into a probability distribution  $y_1 = \text{softmax}(z)$ . Supposing the one-hot ground truth probability distribution is  $\hat{y}_1$ , the cross-entropy loss which is used to optimize the post-hoc classifier is

$$
L _ {P H} = - \left[ \hat {y} _ {1} \log \left(y _ {1}\right) \right] \tag {1}
$$

# 2.2 CAUSE

The CAuSE is composed of i) A language model (LM) called  $\phi_1$  which reconstructs the input text. ii) Another LM  $\phi_2$  which generates the explanation.  $\phi_2$  is coupled with another classifier  $(\mathcal{C}_2)$  which is trained to predict the outputs of the original classifier  $\mathcal{C}_1$ . It is important to note that  $\phi_1$  and  $\phi_2$  share the same weights and are both implemented using a single GPT-2 small model with 350 million parameters, reducing memory consumption.

Training the LMs. The LMs are trained using vanilla causal language modelling (CLM) loss. Specifically, the multimodal representation  $c$  is broken into two components  $c_{0}$  and  $c_{1}$  by passing them through two separate FFNs ( $F_{0}$  and  $F_{1}$ ) which bring their dimension to match with LM embedding dimension  $\mathbb{R}^{768 \times 1}$ , such that  $c_{0} = F_{0}(c)$ , and  $c_{1} = F_{1}(c)$ .

![](images/fbfe2b08d73032774152eaa7fb03f0103dc5948c9712caa95bc4b739b23d61e2.jpg)  
Figure 1: Diagram of our proposed framework CAuSE and the post-hoc classifier.

Given  $c_{0}$ ,  $\phi_{1}$  reconstructs next word  $(x_{i})$  for the  $i$ -th step via the following loss over a total of  $T'$  time-steps:

$$
\mathcal {L} _ {\phi_ {1}} = - \sum_ {i = 1} ^ {T ^ {\prime}} \log P _ {\phi_ {1}} \left(x _ {i} \mid x _ {i - 1}\right) \quad \text {w h e r e} \quad x _ {0} = c _ {0} \tag {2}
$$

Similar equation is used to train  $\phi_{2}$

$$
\mathcal {L} _ {\phi_ {2}} = - \sum_ {i = 1} ^ {T} \log P _ {\phi_ {2}} \left(x _ {i} \mid x _ {i - 1}\right) \quad \text {w h e r e} \quad x _ {0} = c _ {1} \tag {3}
$$

Aggregator  $A$ . The logits  $x_{i}$  retrieved from  $\phi_2$  has the dimension  $\mathbb{R}^{1\times T\times V}$ , where  $V$  is the vocabulary size. These logits are first summed up along the time axis, which yields an intermediate vector  $x$  having dimension of  $\mathbb{R}^{1\times V}$ . This is then passed through another FFN which converts into a dimension same as  $c$ , which is  $\mathbb{R}^{m\times 1}$ .

Classifier  $\mathcal{C}_2$ . The aggregated output having the same dimension as  $c$  is passed through a classifier  $\mathcal{C}_2$  architecturally identical to  $\mathcal{C}_1$ .  $\mathcal{C}_2$  is then trained to predict labels from  $\mathcal{C}_1^1$ .  $y_1$  is the output distribution from  $\mathcal{C}_1$ . Similarly, the probability distribution of  $\mathcal{C}_2$  is  $y_2 = \text{softmax}(\mathcal{C}_2(x))$ , where  $x = (A \circ \phi_2 \circ F_1)(c)$ . We minimize the Cross-Entropy loss between outputs of  $\mathcal{C}_2$  and  $\mathcal{C}_1$  as:

$$
\mathcal {L} _ {C} = - \left[ y _ {1} \log \left(y _ {2}\right) \right] \tag {4}
$$

# 3 TRAINING METHODOLOGY

Training CAuSE involves two steps other than using  $\mathcal{L}_C$  to align  $\mathcal{C}_2$  to  $\mathcal{C}_1$ . They are i) Linguistic Infusion, ii) Causal Intervention.

# 3.1 LINGUISTIC INFUSION (LI)

We denote the input to the classifier  $\mathcal{C}_1$  as  $c$ , which is a multimodal encoding from the encoder. This captures the overall encoded representation of the multimodal source input. Through LI, we want to enrich  $c$  with input source  $(t,v)$  such that the latter could possess enough source information. LI is performed because: We only use a projected version of  $c$  as the input token representation  $c_2$  to  $\phi_2$ . This essentially serves as a bottleneck and most of the source information is lost when input is given to the LLM.

Assuming  $M = (t,v)$ , in LI, the enrichment of  $c$  through source can be defined as the following constrained maximization problem following Plug and Play Language Model (PPLM)(Dathathri et al., 2020).

$$
\hat {c} = \underset {c} {\arg \max } P (c | M) \quad \text {s u c h t h a t} \quad \mathcal {C} _ {1} (\hat {c}) = \mathcal {C} _ {1} (c) \tag {5}
$$

Applying Bayes' theorem,  $P(c|M) \propto P(c)P(M|c)$ . Subsequently, the optimization Equation 5 can be written as:  $\hat{c} = \arg \max_c P(M|c)$ .

To estimate  $P(M|c)$ , we use an autoencoder which tries to predict  $M$  from  $c$ . Formally, we try to estimate  $P(d|c)$  by training an autoencoder which is trained to minimize a loss denoted by  $L_{AE} = |d - M|$ . This ensures  $d$  becomes as close to  $M$  as possible. Specifically, to find  $\hat{c}$ , we train the autoencoder first and then perform gradient descent of  $c$  along the loss. We use  $\hat{c} \gets c - \gamma \nabla_c L_{AE}$  as the iterative update formula to get  $\hat{c}$  from  $c$ .

# 3.2 CAUSAL INTERVENTION

Causal Abstraction. In Geiger et al. (2021c), the authors introduced the concept of causal abstraction for neural models. They define a neural network,  $N_{2}$ , as a causal abstraction of a higher-level causal model,  $N_{1}$ , if the neural representations of  $N_{2}$  exhibit the same causal properties as the corresponding high-level variables in  $N_{1}$ . This alignment is achieved through the Interchange Intervention Training (IIT) objective.

A natural extension of this idea is to consider  $N_{1}$  as a structurally identical neural network to  $N_{2}$  and apply IIT between them, keeping  $N_{1}$  frozen. This process ensures that  $N_{2}$  becomes a causal abstraction of  $N_{1}$ . In our framework, we replace  $N_{1}$  with  $\mathcal{C}_1$  and  $N_{2}$  with  $\mathcal{C}_2$ . Through IIT, we aim to ensure that the structurally identical classifier  $\mathcal{C}_2$  becomes a causal abstraction of  $\mathcal{C}_1$ .

Benefits of Causal Abstraction. The type of causal abstraction learned through IIT is referred to as constructive abstraction in the causality literature. This concept ensures a systematic correspondence between interventions on the neurons in  $N_{1}$  and those in  $N_{2}$ . Unlike a traditional teacher-student loss, which merely teaches the student to mimic the teacher's output, causal abstraction ensures that the student model internally mirrors the teacher's decision-making process. Through IIT, we guarantee that interventions on  $N_{1}$  have corresponding effects on  $N_{2}$ , meaning that  $N_{2}$  operates in the same causal manner as  $N_{1}$ .

We theoretically demonstrate that applying IIT can have significant implications if specific conditions are met. Notably, when the weights of  $\mathcal{C}_1$  and  $\mathcal{C}_2$  remain the same throughout the IIT process:

- The LLM machinery (i.e.,  $A, \phi_2$  along with  $F_1$ , combined as  $F(z) = (A \circ \phi_2 \circ F_1)(z)$ ) perfectly simulates the encoder, such that for any input  $x$ ,  $F(E(x)) = E(x)$ . Hence, the output from the LLM machinery matches that of the encoder [proven in Theorem 1].  
- Building on this result, under a specific set of assumptions, we further show that the LLM machinery, together with  $\mathcal{C}_2$  (referred to as the "explanator"), forms a causal abstraction of the encoder and  $\mathcal{C}_1$  (the "post-hoc classifier") [proven in Theorem 2].

Teacher-student objective. Figure 2 illustrates the training process for  $\mathcal{C}_2$ . A sample input, consisting of both an image and a text from the dataset, is passed through the encoder. The encoder produces an output  $c$ , represented as a 3-dimensional vector, which is then fed into  $\mathcal{C}_1$ . Assuming the weights in the first layer are all set to one, the activation of the  $i_1$ -th neuron (as shown in the diagram) would be calculated as  $1 \times 0.1 + 1 \times 0.2 + 1 \times 0.3 = 0.6$ . The final activation is then computed as  $y_1 = 3 \times 0.6 + 2 \times 0.6 = 3$ .

![](images/71799e90f4c7d8f4f88b77c617886bdb524d93fcf54d3175700a9bcbb9aa43ba.jpg)  
Figure 2: Causal Abstraction is enabled by IIT objective. Along with the teacher-student training objective, IIT poses as indispensable for  $\mathcal{C}_2$  to be a causal abstraction of  $\mathcal{C}_1$ .

Simultaneously, the output  $c$  is passed through the LLM machinery, which generates an activation that is forwarded to  $\mathcal{C}_2$ , producing an activation denoted as  $y_2$ . To ensure  $\mathcal{C}_2$  mirrors the behavior of  $\mathcal{C}_1$ , we calculate the final loss using the KL divergence between their outputs:

$$
\mathcal {L} _ {T S} = K L \left(P _ {y _ {1}} \mid P _ {y _ {2}}\right) \tag {6}
$$

where  $P_{y_1} = [\sigma(y_1), 1 - \sigma(y_1)]$  and  $P_{y_2} = [\sigma(y_2), 1 - \sigma(y_2)]$ . This approach can be generalized to handle multiple outputs by applying the softmax function.

IIT objective. The Interchange Intervention (II) process is depicted in Figure 2. A neuron is randomly selected from  $\mathcal{C}_1$  (denoted as  $i_1$ ), and the II is applied. For a given source input, let  $c = [0.5, 0.6, 0.7]$  (shown on the right-hand side). The II process ensures that the value of neuron  $i_1$  is replaced with its original value, 0.6, which was obtained when the base input was processed. The final value after this intervention, referred to as the "intervened output," is represented as  $y_1^{INT}$  for  $\mathcal{C}_1$ .

The same operation is carried out for  $\mathcal{C}_2$ , and the resulting "intervened output" is denoted as  $y_2^{INT}$ . Following the methodology of Geiger et al. (2021c), to ensure that  $\mathcal{C}_2$  becomes a causal abstraction of  $\mathcal{C}_1$ , we minimize the IIT loss between the two outputs:

$$
\mathcal {L} _ {I I T} = K L \left(P _ {y _ {1} ^ {I N T}} \mid P _ {y _ {2} ^ {I N T}}\right) \tag {7}
$$

CAuSE Loss Function. The final loss used to train CAuSE (i.e.  $\mathcal{L}_{CAuSE}$ ) is defined as a sum of all individual loss terms.

$$
\mathcal {L} _ {C A u S E} = \mathcal {L} _ {\phi_ {1}} + \mathcal {L} _ {\phi_ {2}} + \mathcal {L} _ {I I T} + \mathcal {L} _ {T S} + \mathcal {L} _ {C} + \| W _ {\mathcal {C} _ {1}} - W _ {\mathcal {C} _ {2}} \| _ {F} \tag {8}
$$

where  $\| W_{\mathcal{C}_1} - W_{\mathcal{C}_2}\| _F$  denotes the Frobenius norm between the weights of  $\mathcal{C}_1$  and  $\mathcal{C}_2$  respectively. This term ensures that weights of  $\mathcal{C}_1$  and  $\mathcal{C}_2$  remain the same during training.

Counterfactual F1 score. We hypothesize that if the explainer becomes a causal abstraction of the post-hoc classifier, it should still mimic the classifier under counterfactual input. To evaluate this, we introduce the counterfactual F1 (c-F1) score. Our empirical analysis shows that using only teacher-student training results in poor performance on counterfactual input, as reflected by a low c-F1 score. However, when combined with IIT, the explainer achieves a robust c-F1 score. Algorithm 1 details c-F1 calculation, and Table 2 compares methods based on their c-F1 scores.

# 3.2.1 CALCULATING COUNTERFACTUAL F1 SCORE

Suppose  $x \in \mathcal{T}$  is a data-point from test set. As posed in Feder et al. (2022), the corresponding counterfactual input  $x'$  for the post-hoc classifier would satisfy the following:

$$
x ^ {\prime} = \underset {x ^ {\prime} \in \mathcal {T}} {\arg \min } d \left(x, x ^ {\prime}\right) \quad \text {s u c h t h a t} \quad \mathcal {C} _ {1} (x) \neq \mathcal {C} _ {1} \left(x ^ {\prime}\right) \tag {9}
$$

$d$  is any kind of distance metric (e.g. manhattan, euclidean etc) between these data points.  $\mathcal{C}_1(z)$  denotes the output class from  $\mathcal{C}_1$  for any input  $z$ .

Subsequently, any counterfactual for  $x$  can be expressed as:  $x' = x + \mu$ , where  $\mu = x' - x$  is the perturbation between normal and counterfactual input. Note that  $E(x')$  could not be a good counterfactual input for the LLM machinery, as  $x' \in \mathcal{T}$  and high simulation performance between  $\mathcal{C}_2$  and  $\mathcal{C}_1$  means  $\mathcal{C}_2$  could easily find label of  $x'$ . Therefore, we resort to the following three constraints while designing a counterfactual input  $z'$  for the LLM Machinery: i)  $z'$  should be a counterfactual for  $\mathcal{C}_1$ , as our task is to measure how many counterfactuals for  $\mathcal{C}_1$  are also counterfactual for  $\mathcal{C}_2$ . ii)  $z'$  should not be representation of any data-point from  $\mathcal{T}$ , iii) It should be a transformation of the original data-point  $x$  and its perturbation  $\mu$ .

We assume  $z'$  has the following generic form (satisfying ii. and iii.),  $z' = z + T(\mu)$ , where  $z = E(x)$  is an input to the LLM machinery. So,  $z' = E(x) + T(\mu)$ . Note that to ensure  $T(\mu)$  is an invertible function of  $\mu$  (satisfying iii.), we use an autoencoder which maps  $\mu$  to  $T(\mu)$  and then back to  $\mu$  again. Finally, to satisfy the first constraint, we ensure the following holds true:

$$
\mathcal {C} _ {1} (E (x) + T (\mu)) = \mathcal {C} _ {1} (E (x + \mu)) \tag {10}
$$

Note that this can be enforced by standard KL divergence loss between  $\mathcal{C}_1$  and  $\mathcal{C}_2$ .

Algorithm 1: Counterfactual F1 Score for  $\mathcal{C}_1$  and  $\mathcal{C}_2$  
Input: Data-point  $\pmb {x}\in \mathcal{T}$    
Function CounterFactual  $(\pmb {x})$  ..   
\(\begin{array}{rl} & {\pmb{x}^{\prime}\gets \arg \min_{\pmb{x}^{\prime}\in \mathcal{T}}d(\pmb {x},\pmb{x}^{\prime})\mathrm{s.t.}\mathcal{C}_{1}(\pmb {x})\neq \mathcal{C}_{1}(\pmb{x}^{\prime});}\\ & {\mu \gets \pmb{x}^{\prime} - \pmb {x};\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \mathrm{//~Compute~the~perturbation}}\\ & {\pmb {z}\gets E(\pmb {x});\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \mathrm{//~Encode~the~original~input}}\\ & {T(\mu)\gets f(\mu)\mathrm{where~}g(f(\mu)) = \mu ;\qquad \qquad \qquad \qquad \qquad \qquad \mathrm{//~Transform~the~perturbation}}\\ & {\pmb {z}^{\prime}\gets \pmb {z} + T(\mu);}\\ & {\mathrm{return}\pmb {z}^{\prime},\pmb{x}^{\prime}}\\ & {\mathrm{Procedure~Calculate~Counterfactual~F1~score}}\\ & {\mathrm{ZList}\gets [];}\\ & {\mathrm{XList}\gets [];}\\ & {\mathrm{while~}\mathcal{T}\neq \phi\mathrm{do}}\\ & {\mathrm{Sample~}\pmb {x}\in\mathcal{T};\qquad\mathrm{//~Draw~a~new~data~point}}\\ & {\pmb {z}^{\prime},\pmb{x}^{\prime}\gets\mathrm{CounterFactual}(\pmb {x});}\\ & {\mathrm{Ensure:}\mathcal{C}_{1}(\pmb {z}^{\prime})=\mathcal{C}_{1}(E(\pmb{x}^{\prime}));\qquad\mathrm{//~constraint~i.}}\\ & {\mathrm{ZList}\gets\mathrm{ZList}\cup\{\mathcal{C}_{2}(\pmb {z}^{\prime})\};\qquad\mathrm{/ /~Append~}\mathcal{C}_{2}(\pmb {z}^{\prime})\mathrm{~to~the~list}}\\ & {\mathrm{XList}\gets\mathrm{XList}\cup\{\mathcal{C}_{1}(\pmb{x}^{\prime})\};\qquad\mathrm{/ /~Append~}\mathcal{C}_{1}(\pmb{x}^{\prime})\mathrm{~to~the~list}}\\ & {\mathcal{T}\gets\mathcal{T}-\{\pmb{x}\};}\\ & {\mathrm{return}\boldsymbol{F}_{1}-\mathrm{score(XList,ZList)}}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {} \\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ & {}\\ &
\end{array}\)

Table 1: Ablation studies.  $\mathcal{L}_{MSE}$  refers to an MSE loss between  $c$  and  $x$ , such that  $F(E(x)) = E(x)$ . B-1, B-2, B-3, B- refers to Bleu scores with various  $n$  gram precisions.  

<table><tr><td></td><td></td><td>F1</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>BertScore</td></tr><tr><td rowspan="5">Hateful Meme</td><td>\( \mathcal{L}_{\phi_2} \)</td><td>97.29</td><td>0.65</td><td>0.53</td><td>0.47</td><td>0.39</td><td>0.971</td></tr><tr><td>\( \mathcal{L}_{\phi_1} + \mathcal{L}_{\phi_2} \)</td><td>98.44</td><td>0.65</td><td>0.53</td><td>0.47</td><td>0.39</td><td>0.971</td></tr><tr><td>\( \mathcal{L}_{\phi_1} + \mathcal{L}_{\phi_2} + \mathcal{L}_{MSE} \)</td><td>98.55</td><td>0.64</td><td>0.53</td><td>0.46</td><td>0.39</td><td>0.971</td></tr><tr><td>\( \mathcal{L}_{\phi_1} + \mathcal{L}_{\phi_2} + \mathcal{L}_C \)</td><td>98.33</td><td>0.64</td><td>0.53</td><td>0.46</td><td>0.38</td><td>0.971</td></tr><tr><td>\( \mathcal{L}_{CAuSE} \)</td><td>98.09</td><td>0.64</td><td>0.51</td><td>0.44</td><td>0.36</td><td>0.969</td></tr><tr><td rowspan="5">e-SNLI-VE</td><td>\( \mathcal{L}_{\phi_2} \)</td><td>94.66</td><td>0.39</td><td>0.27</td><td>0.19</td><td>0.15</td><td>0.905</td></tr><tr><td>\( \mathcal{L}_{\phi_1} + \mathcal{L}_{\phi_2} \)</td><td>94.08</td><td>0.39</td><td>0.27</td><td>0.19</td><td>0.15</td><td>0.905</td></tr><tr><td>\( \mathcal{L}_{\phi_1} + \mathcal{L}_{\phi_2} + \mathcal{L}_{MSE} \)</td><td>94.39</td><td>0.39</td><td>0.27</td><td>0.20</td><td>0.15</td><td>0.905</td></tr><tr><td>\( \mathcal{L}_{\phi_1} + \mathcal{L}_{\phi_2} + \mathcal{L}_C \)</td><td>94.94</td><td>0.38</td><td>0.27</td><td>0.20</td><td>0.15</td><td>0.905</td></tr><tr><td>\( \mathcal{L}_{CAuSE} \)</td><td>91.96</td><td>0.39</td><td>0.27</td><td>0.20</td><td>0.15</td><td>0.904</td></tr></table>

Table 2: In addition to the Counterfactual F1 score, we also report the number of comprehensible generations (#gen), as many outputs from CAuSE tend to be gibberish when counterfactual input is provided. To provide a more holistic evaluation of CAuSE's performance on counterfactual inputs, we compute the harmonic mean (HM) of the F1 score and #gen, capturing both accuracy and the quality of generated explanations.  

<table><tr><td></td><td colspan="3">Hateful Meme</td><td colspan="3">e-SNLI-VE</td></tr><tr><td></td><td>F1</td><td># gen.</td><td>HM</td><td>F1</td><td># gen.</td><td>HM</td></tr><tr><td>Lφ1 + Lφ2</td><td>55.02</td><td>17</td><td>32.98</td><td>93.81</td><td>167</td><td>28.35</td></tr><tr><td>Lφ1 + Lφ2 + LMSE</td><td>33.33</td><td>2</td><td>3.976</td><td>85.94</td><td>322</td><td>46.84</td></tr><tr><td>Lφ1 + Lφ2 + LC</td><td>53.78</td><td>91</td><td>15.56</td><td>73.48</td><td>850</td><td>78.82</td></tr><tr><td>LCAUSE</td><td>75.81</td><td>755</td><td>75.61</td><td>85.24</td><td>986</td><td>91.43</td></tr></table>

# 4 RESULTS AND ANALYSIS

# 4.1 AUTOMATIC EVALUATION

The proposed system is evaluated across two verticals: i) Mimicking capability of the explainator when compared to post-hoc classifier, and ii) performance under counterfactual input. The automatic evaluation metric used to evaluate CAuSE performance can be grouped into two categories, i) Faithfulness: This is measured by the obtained F1 score measured between the predicted class by the LLM machinery (or  $\mathcal{C}_2$ ) and the predicted class by the post-hoc classifier  $\mathcal{C}_1$ . The predicted class obtained from the LLM machinery is extracted either from the prediction of  $\phi_2$  or from  $\mathcal{C}_2$  classifier head. ii) Plausibility: This is measured as the BLEU scorePapineni et al. (2002) and BERTScoreZhang et al. (2020) between the generated explanation and the ground truth explanation from the test set.

Baselines. To the best of our knowledge, ours is the first approach that generates faithful natural language explanations directly from a classifier's hidden state. Nonetheless, we compare our method with several Visual Language Model (VLM) baselines as there are no existing techniques for this task in the literature. Specifically, we use zero-shot and few-shot ( $k = 2$  or 3) prompting with i) PaLiGemma(Beyer et al., 2024), ii) LLaVA(Liu et al., 2023), to simulate the predicted class from a given classifier  $(\mathcal{C}_1)$ , based on previous input-output examples<sup>2</sup>. Since it is challenging to simulate a model's behaviour without access to its hidden activations, few-shot prompting often performs similarly or even worse than zero-shot prompting. The faithfulness of the explanations, as measured by the F1 score, is inconsistent and random (below  $50\%$  for the Hateful Memes dataset and below  $33\%$  for e-SNLI-VE), as shown in Table 3 The fine-tuned models (shown through FT suffix) perform the best, where the F1 score reaches close to  $\sim 70\%$ .

Table 3: Various VLM-based baselines. FT as a suffix denotes finetuned model. Note that LLaVA has 7B and PaLiGemma has 3.5B parameters respectively.  

<table><tr><td>Dataset</td><td>Baselines</td><td>F1</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>BertScore</td></tr><tr><td rowspan="4">Hateful Meme</td><td>LLaVA-0-shot</td><td>58.44</td><td>0.09</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.889</td></tr><tr><td>LLaVA-2-shot</td><td>46.55</td><td>0.12</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.864</td></tr><tr><td>PaLiGemma -FT</td><td>72.33</td><td>0.41</td><td>0.27</td><td>0.15</td><td>0.09</td><td>0.891</td></tr><tr><td>LLaVA-FT</td><td>72.38</td><td>0.40</td><td>0.27</td><td>0.17</td><td>0.13</td><td>0.894</td></tr><tr><td rowspan="4">e-SNLI-VE</td><td>LLaVA-0-shot</td><td>33.12</td><td>0.22</td><td>0.07</td><td>0.03</td><td>0.02</td><td>0.876</td></tr><tr><td>LLaVA-3-shot</td><td>35.77</td><td>0.22</td><td>0.07</td><td>0.03</td><td>0.01</td><td>0.869</td></tr><tr><td>PaLiGemma -FT</td><td>64.90</td><td>0.19</td><td>0.04</td><td>0.01</td><td>0.01</td><td>0.866</td></tr><tr><td>LLaVA-FT</td><td>64.29</td><td>0.22</td><td>0.08</td><td>0.03</td><td>0.02</td><td>0.859</td></tr></table>

Table 4: Case studies: A few example where our model succeeds. Pred: Explanation generated from the model, GT: Ground truth explanation.  

<table><tr><td>Image Path</td><td>Pred</td><td>GT</td><td>y1</td><td>y2</td></tr><tr><td>489134459.jpg</td><td>A woman is a female. 
Just because she is sitting on a curb, it means she is outside..</td><td>A boy and a girl are two kids. 
The front of a house is located outside..</td><td>E</td><td>E</td></tr><tr><td>5631556013.jpg</td><td>A man is performing on the street in front of a group of people..</td><td>man jumping from someone</td><td>E</td><td>E</td></tr><tr><td>12507.png</td><td>it promotes negative stereotypes about people who are Muslim and suggests that all Muslims are violent or dangerous</td><td>it promotes harmful stereotypes about Muslims, suggesting that they are violent and intolerant.</td><td>O</td><td>O</td></tr><tr><td>91462.png</td><td>it promotes racism, specifically by implying that white people are superior to other people.</td><td>it promotes harmful stereotypes about black women.</td><td>O</td><td>O</td></tr></table>

# 4.1.1 ABLATION STUDIES

What is the use of various loss function other than  $\mathcal{L}_{\phi_1}$  and  $\mathcal{L}_{\phi_2}$  loss? As seen from Table 1, it can be posed as a valid question. Indeed, when using our proposed method which uses  $\mathcal{L}_{IIT}$  and other losses seem to achieve slightly lower F1 score (indicating slightly lower faithfulness) and slightly lower BLEU score / BERTScore (indicating slightly lower plausibility). Note that this difference is very small and it is compensated by very high counterfactual F1 score as shown in Table 2 obtained by our method.

Why is IIT required? As can be seen from Table 1 and Table 2, good mimicking performance under normal condition does not always entail good performance when posed with counterfactual input. IIT ensures causal abstraction between  $\mathcal{C}_2$  and  $\mathcal{C}_1$  theoretically and this is also being verified empirically by the high counterfactual F1 score obtained by our method which uses IIT.

Is  $\mathcal{L}_{\phi_1}$  necessary?  $\mathcal{L}_{\phi_1}$  which is used to train the LLM  $(\phi_{1})$  which reconstructs the content is required, because that shows better mimicking performance (at least in Hateful meme dataset) coupled with  $\mathcal{L}_{\phi_2}$  than using  $\mathcal{L}_{\phi_2}$  alone. This can be attributed to the joint training objective which ensures that  $c$  possesses enough input information to aid in explanation generation by the second LLM  $\phi_{2}$ .

![](images/329d9d6dfdcfbeaef31d451bb63534ebc5ddd5b8e6cf45aa457b8c7497e868f4.jpg)  
two kids are outside

![](images/c0c0333d29e80081c305a248f73f39a92a8c02059a5710f570f1b2ce02ab686e.jpg)  
There is a man jumping from someone

![](images/eacfe0bd2ec0e0a8ab9c0c888217736590e26176539d748e262a313297caaba4.jpg)  
Figure 3: Examples corresponding to Table 4

![](images/682cc2dbc2b23c7ef6d562b27cdb2cbc72b36c1fe4c332a0bb1f2d0ebd72e284.jpg)

Table 5: Error Analysis: These cases demonstrate four kinds of error cases that is prevalent among our proposed framework CAuSE.  

<table><tr><td>Image Path</td><td>Gen</td><td>GT</td><td>y</td><td>ŷ</td></tr><tr><td>7046014201.jpg</td><td>Construction work necessitates working outdoor.</td><td>A juggler is juggling clubs at an outdoor plaza.</td><td>E</td><td>E</td></tr><tr><td>2731298834.jpg</td><td>A dog that is jumping into the water will be wet.</td><td>swimming is perform in a water.</td><td>E</td><td>E</td></tr><tr><td>151215569.jpg</td><td>A man is pulling on the street so he is outdoors.</td><td>A young blond girl describes a child and a man describes and adult.</td><td>E</td><td>E</td></tr><tr><td>59260.png</td><td>it suggests that white people are superior to other people, which is not accurate</td><td>it promotes anti-Semitism and hatred towards Jewish people.</td><td>O</td><td>O</td></tr></table>

![](images/dec313479bb7e8495c50c065d884467a84ed7f34ddd9a01e9eceeb9e2cf039b5.jpg)  
A juggler is performing outdoors.

![](images/2fa60f70493872d32ee705cd62154683c5a406480db417951904dc8e67b81a64.jpg)  
There is water

![](images/d6173116f6bcd14b9e898567e0fd125eff89a66c1b1153df99d8d79166c85430.jpg)  
A child is riding an adult.

![](images/616fddd420ca51c1d9a084bc484d4bcc64c0432df4f3b9eefe599e090dcc2726.jpg)  
Figure 4: Memes pertaining to Error Analysis shown in Table 5

# 4.2 QUALITATIVE STUDIES

# 4.2.1 CASE STUDIES

In Table 4, we present four successful examples from the e-SNLI-VE and Facebook Hateful Memes datasets (two from each). The first two examples are from e-SNLI-VE, while the latter two are from the Hateful Memes dataset. In the e-SNLI-VE examples, CAuSE produces semantically accurate explanations while correctly predicting the class as "Entailment." A noticeable pattern emerges from these successful cases: CAuSE tends to perform well when the class-level information can be explicitly inferred from the combination of the image and text. Specifically, for e-SNLI-VE, when CAuSE generates accurate explanations, the hypothesis often functions like a caption for the image premise, which aids in classification.

For the Hateful Memes examples, CAuSE also generates correct explanations. In these cases, the image and the embedded text are semantically aligned rather than contradictory (i.e., where the image-text mismatch is used to evoke negative sentiment). In such instances, CAuSE effectively provides explanations and correctly predicts the appropriate output class.

# 4.2.2 ERROR ANALYSIS

We selected four examples from the e-SNLI-VE and Hateful Memes datasets to highlight common types of errors made by CAuSE (in Table 5). These errors can be categorized into three main types:

Lack of representation capability: In the first example, the hypothesis reads, "A juggler is performing outdoors," and the premise is entailed, as confirmed by the ground truth explanation: "A juggler is juggling clubs at an outdoor plaza." However, CAuSE incorrectly generates the explanation: "Construction work necessitates working outdoors," confusing the act of juggling with construction work. This error likely stems from insufficient information in the initial representation,  $c$ , used by CAuSE.

Lack of object-level representation: The post-hoc classifier relies on unimodal representations from the CLIP architecture, which lacks fine-grained object-level details, compared to models like Faster R-CNNRen et al. (2016). In the second example, instead of recognizing a "dog," CAuSE should have identified "a woman and children" for a more accurate representation.

The third example illustrates both issues: lack of object-level representation and general representation capability. These limitations prevent CAuSE from correctly describing the relationship between "a young blonde girl," "an adult," and "a man pulling outdoors."

Implicit semantic category: In the fourth example, although CAuSE correctly predicts the output class as offensive, it does so for the wrong reasons. Even a human might struggle to recognize the

implicit anti-Semitism in this meme, as neither the image nor the text explicitly convey the historical context of the Holocaust, where six million Jews were killed. Without this prior knowledge, CAuSE cannot fully comprehend the offence.

# 5 RELATED WORK

Interpretability. Interpretability is crucial for building trust in AI systems within human society. Techniques like LIME, SHAP and RISE (Ribeiro et al., 2016; Lundberg & Lee, 2017; Petsiuk et al., 2018) explain classifier predictions by providing feature-level explanations for local interpretability. Although model-agnostic, these methods lack global interpretability, which is addressed by GALE van der Linden et al. (2019), where local explanations are aggregated into a global model understanding. Approaches like SmoothGrad Smilkov et al. (2017) and Integrated Gradients (Sundararajan et al., 2017b) utilize input gradients for model explanation, while CAM Zhou et al. (2015) highlights critical pixels for decision making in visual classification. Counterfactual generations (Chang et al., 2019; Mothilal et al., 2020; Goyal et al., 2019) also offer insights into the inner working of the model by revealing decision boundaries. However, most of these methods often overlook implicit features behind model decisions and lack natural language explanations. To address these limitations, we propose a novel framework for classifier explanations which generates both faithful and plausible natural language outputs.

Causal Interpretability. Causal interpretability refers to the ability to explain a model's decisions by identifying the cause-effect relationships between input features and the model's output. Feder et al. (2022) demonstrated how incorporating causal reasoning in NLP tasks can improve model predictions and enhance interpretability by going beyond simple correlations between input features and outputs. Further works by Geiger et al. (2021b); Vig et al. (2020); Meng et al. (2023) have focused on causal abstraction and causal mediation analysis, helping to create causally faithful models and identify both direct and indirect causal factors behind certain model behaviors. In addition to generating counterfactuals, testing models on counterfactual inputs is another critical aspect of understanding model behavior. Since creating exact counterfactuals is challenging, Abraham et al. (2022); Calderon et al. (2022), recent research has focused on approximations Geiger et al. (2021b) or counterfactual representations Feder et al. (2021); Elazar et al. (2021); Ravfogel et al. (2021). Our proposed counterfactual metric is inspired by these counterfactual representations. Moreover, most of the existing works focuses on single modality (e.g., text or vision) Feder et al. (2021); Goyal et al. (2020). In contrast, the natural language causal explanation provided by our framework is model-agnostic, task-agnostic, and capable of handling multimodal inputs.

# 6 CONCLUSION AND FUTURE WORK

In this paper, we presented CAuSE (Causal Abstraction under Simulated Explanation), a novel framework for generating causally faithful natural language explanations for multimodal classifiers. By integrating Interchange Intervention Training (IIT) with a Language Model (LM) based module, CAuSE addresses the limitations of existing interpretability methods, ensuring explanations are directly tied to the classifier's causal reasoning. Our new Counterfactual F1 score highlights CAuSE's state-of-the-art performance on datasets like e-SNLI-VE and Facebook Hateful Memes.

While CAuSE demonstrates robust task-agnostic performance, future work will focus on enhancing fine-grained object-level representations and extending the framework to temporal data, such as video and audio. Additionally, we aim to explore how self-supervised learning and deeper integration of implicit cultural knowledge can further improve the framework's scalability and contextual understanding in real-world applications.

# ETHICS STATEMENT

The datasets used in this study are publicly available. The explanations for hateful memes were generated from publicly accessible meme data, and we adhered to copyright regulations to prevent any infringement. Furthermore, our research received approval from the Institutional Review Board (IRB). Since the hateful meme dataset includes content that may be offensive, we recommend that readers approach it with discretion.

# REPRODUCIBILITY STATEMENT

To ensure reproducibility, we consistently use a random seed of 42 across all experiments. The code is available at https://anonymous.4open.science/r/CAuSE-5BD0, and model outputs will be shared upon paper acceptance. These outputs can be cross-verified with the results generated from the provided code. Our method is theoretically sound, supported by the proof of the proposed theorem and proposition outlined in Appendix A, with all underlying assumptions clearly stated and justified.

# REFERENCES

Eldar David Abraham, Karel D'Oosterlinck, Amir Feder, Yair Ori Gat, Atticus Geiger, Christopher Potts, Roi Reichart, and Zhengxuan Wu. Gebab: Estimating the causal effects of real-world concepts on nlp model behavior, 2022.  
Chirag Agarwal, Sree Harsha Tanneru, and Himabindu Lakkaraju. Faithfulness vs. plausibility: On the (un)reliability of explanations from large language models, 2024.  
Tadas Baltrusaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: A survey and taxonomy, 2017. URL https://arxiv.org/abs/1705.09406.  
Dibyanayan Bandyopadhyay, Asmit Ganguly, Baban Gain, and Asif Ekbal. Semantify: Unveiling memes with robust interpretability beyond input attribution. In Kate Larson (ed.), Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pp. 6189-6197. International Joint Conferences on Artificial Intelligence Organization, 8 2024. doi: 10.24963/ijcai.2024/684. URL https://doi.org/10.24963/ijcai.2024/684. Main Track.  
Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmesen, and Xiaohua Zhai. Paligemma: A versatile 3b vlm for transfer, 2024. URL https://arxiv.org/abs/2407.07726.  
Nitay Calderon, Eyal Ben-David, Amir Feder, and Roi Reichart. DoCoGen: Domain counterfactual generation for low resource domain adaptation. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7727-7746, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.533. URL https://aclanthology.org/2022.acl-long.533.  
Chun-Hao Chang, Elliot Creager, Anna Goldenberg, and David Duvenaud. Explaining image classifiers by counterfactual generation, 2019. URL https://arxiv.org/abs/1807.08024.  
Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to controlled text generation, 2020. URL https://arxiv.org/abs/1912.02164.  
Virginie Do, Oana-Maria Camburu, Zeynep Akata, and Thomas Lukasiewicz. e-snli-ve: Corrected visual-textual entailment with natural language explanations, 2021. URL https://arxiv.org/abs/2004.03744.  
Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals, 2021.  
Amir Feder, Nadav Oved, Uri Shalit, and Roi Reichart. CausaLM: Causal model explanation through counterfactual language models. Computational Linguistics, 47(2):333-386, June 2021. doi: 10.1162/coli_a_00404. URL https://aclanthology.org/2021.cl-2.13.

Amir Feder, Katherine A. Keith, Emaad Manzoor, Reid Pryzant, Dhanya Sridhar, Zach Wood-Doughty, Jacob Eisenstein, Justin Grimmer, Roi Reichart, Margaret E. Roberts, Brandon M. Stewart, Victor Veitch, and Diyi Yang. Causal inference in natural language processing: Estimation, prediction, interpretation and beyond. Transactions of the Association for Computational Linguistics, 10:1138-1158, 2022. doi: 10.1162/tacl_a_00511. URL https://aclanthology.org/2022.tacl-1.66.  
Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. CoRR, abs/2106.02997, 2021a. URL https://arxiv.org/abs/2106.02997.  
Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks, 2021b.  
Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks, 2021c. URL https://arxiv.org/abs/2106.02997.  
Yash Goyal, Ziyan Wu, Jan Ernst, Dhruv Batra, Devi Parikh, and Stefan Lee. Counterfactual visual explanations, 2019. URL https://arxiv.org/abs/1904.07451.  
Yash Goyal, Amir Feder, Uri Shalit, and Been Kim. Explaining classifiers with causal concept effect (cace), 2020.  
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuzhhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/abs/2106.09685.  
Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringsha, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes, 2021.  
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. URL https://arxiv.org/abs/2304.08485.  
Scott Lundberg and Su-In Lee. A unified approach to interpreting model predictions, 2017. URL https://arxiv.org/abs/1705.07874.  
Andreas Madsen, Sarath Chandar, and Siva Reddy. Are self-explanations from large language models faithful?, 2024.  
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt, 2023.  
Ramaravind K. Mothilal, Amit Sharma, and Chenhao Tan. Explaining machine learning classifiers through diverse counterfactual explanations. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, FAT* '20. ACM, January 2020. doi: 10.1145/3351095.3372850. URL http://dx.doi.org/10.1145/3351095.3372850.  
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin (eds.), Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311-318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040.  
Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of black-box models, 2018. URL https://arxiv.org/abs/1806.07421.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.  
Shauli Ravfogel, Grusha Prasad, Tal Linzen, and Yoav Goldberg. Counterfactual interventions reveal the causal effect of relative clause representations on agreement prediction. In Arianna Bisazza and Omri Abend (eds.), Proceedings of the 25th Conference on Computational Natural Language Learning, pp. 194-209, Online, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.conll-1.15. URL https://aclanthology.org/2021.conll-1.15.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks, 2016.  
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier, 2016. URL https://arxiv.org/abs/1602.04938.  
Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. Smoothgrad: removing noise by adding noise, 2017. URL https://arxiv.org/abs/1706.03825.  
Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks, 2017a.  
Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks, 2017b. URL https://arxiv.org/abs/1703.01365.  
Ilse van der Linden, Hindu Haned, and Evangelos Kanoulas. Global aggregations of local explanations for black box models, 2019. URL https://arxiv.org/abs/1907.03039.  
Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Simas Sakenis, Jason Huang, Yaron Singer, and Stuart Shieber. Causal mediation analysis for interpreting neural nlp: The case of gender bias, 2020.  
Yi Xiao, Felipe Codevilla, Akhil Gurram, Onay Urfalioglu, and Antonio M. Lopez. Multimodal end-to-end autonomous driving. IEEE Transactions on Intelligent Transportation Systems, 23 (1):537-547, January 2022. ISSN 1558-0016. doi: 10.1109/tits.2020.3013234. URL http://dx.doi.org/10.1109/TITS.2020.3013234.  
Zhou Yu, Jun Yu, Jianping Fan, and Dacheng Tao. Multi-modal factorized bilinear pooling with co-attention learning for visual question answering, 2017. URL https://arxiv.org/abs/1708.01471.  
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert, 2020. URL https://arxiv.org/abs/1904.09675.  
Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization, 2015. URL https://arxiv.org/abs/1512.04150.
