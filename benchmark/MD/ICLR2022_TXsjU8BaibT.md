# TRIGGER HUNTING WITH A TOPOLOGICAL PRIOR FOR TROJAN DETECTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Despite their success and popularity, deep neural networks (DNNs) are vulnerable when facing backdoor attacks. This impedes their wider adoption, especially in mission critical applications. This paper tackles the problem of Trojan detection, namely, identifying Trojaned models – models trained with poisoned data. One popular approach is reverse engineering, i.e., recovering the triggers on a clean image by manipulating the model's prediction. One major challenge of reverse engineering approach is the enormous search space of triggers. To this end, we propose innovative priors such as diversity and topological simplicity to not only increase the chances of finding the appropriate triggers but also improve the quality of the found triggers. Moreover, by encouraging a diverse set of trigger candidates, our method can perform effectively in cases with unknown target labels. We demonstrate that these priors can significantly improve the quality of the recovered triggers, resulting in substantially improved Trojan detection accuracy as validated on both synthetic and publicly available TrojanAI benchmarks.

# 1 INTRODUCTION

Deep learning has achieved superior performance in various computer vision tasks, such as image classification (Krizhevsky et al., 2012), image segmentation (Long et al., 2015), object detection (Girshick et al., 2014), etc. However, the vulnerability of DNNs against backdoor attacks raises serious concerns. In this paper, we address the problem of Trojan attacks, where during training, an attacker injects polluted samples. While resembling normal samples, these polluted samples contain a specific type of perturbation (called triggers). These polluted samples are assigned with target labels, which are usually different from the expected class labels. Training with this polluted dataset results in a Trojaned model. At the inference stage, a Trojaned model behaves normally given clean samples. But when the trigger is present, it makes unexpected, yet consistently incorrect predictions.

One major constraint for Trojan detection is the limited access to the polluted training data. In practice, the end-users, who need to detect the Trojaned models, often only have access to the weights and architectures of the trained DNNs. State-of-the-art (SOTA) Trojan detection methods generally adopt a reverse engineering approach (Guo et al., 2019; Wang et al., 2019; Huster & Ekwedike, 2021; Wang et al., 2020; Chen et al., 2019c; Liu et al., 2019). They start with a few clean samples, using either gradient descent or careful stimuli crafting, to find a potential trigger that alters model prediction. Characteristics of the recovered triggers along with the associated network activations are used as features to determine whether a model is Trojaned or not.

![](images/144bb5fad25f5e981d0aa3d07ada0a89a5524b3a105c347f9a4d9b62980e3976.jpg)  
(a)

![](images/e1892d41617411d27905c78104427666a283936e4b1f0a7eb2e6949e84858558.jpg)  
(b)

![](images/eab264e302bd6b1540b2a0f74e25e3ab00fe31514d4daf14fa5a9d3141f6897c.jpg)  
Figure 1: Illustration of recovered triggers: (a) clean image, (b) poisoned image, (c) image with a trigger recovered without topological prior, (d)-(f) images with candidate triggers recovered with the proposed method. Topological prior contributes to improved compactness. We run the trigger reconstruction for multiple rounds with a diversity prior to ensure a diverse set of trigger candidates.  
(c)

![](images/fd82b03eebdc775955264d0d95bbce573bda2d2aec1d6d23df5a3c6fb24d0b8f.jpg)  
(d)

![](images/38b19095f690069087692932df8dddc98643bc8b51d3ff16b5a3b7ec1252d2dd.jpg)  
(e)

![](images/9cdf73d4c8adf06557fa05f72c0379651cd6337750a7903f1283990b91d80797.jpg)  
(f)

Trojan triggers can be of arbitrary patterns (e.g., shape, color, texture) at arbitrary locations of an input image (e.g., Fig. 1). As a result, one major challenge of the reverse engineering-based approach is the enormous search space for potential triggers. Meanwhile, just like the trigger is unknown, the target label (i.e., the class label to which a triggered model predicts) is also unknown in practice. Gradient descent may flip a model's prediction to the closest alternative label, which may not necessarily be the target label. This makes it even more challenging to recover the true trigger. Note that many existing methods (Guo et al., 2019; Wang et al., 2019; 2020) require a target label. These methods achieve target label independence by enumerating through all possible labels, which can be computationally prohibitive especially when the label space is huge.

We propose a novel target-label-agnostic reverse engineering method. First, to improve the quality of the recovered triggers, we need a prior that can localize the triggers, but in a flexible manner. We, therefore, propose to enforce a topological prior to the optimization process of reverse engineering, i.e., the recovered trigger should have fewer connected components. This prior is implemented through a topological loss based on the theory of persistent homology (Edelsbrunner & Harer, 2010). It allows the recovered trigger to have arbitrary shape and size. Meanwhile, it ensures the trigger is not scattered and is reasonably localized. See Fig. 1 for an example - comparing (d)-(f) vs. (c).

As a second contribution, we propose to reverse engineer multiple diverse trigger candidates. Instead of running gradient decent once, we run it for multiple rounds, each time producing one trigger candidate, e.g., Fig. 1 (d)-(f). Furthermore, we propose a trigger diversity loss to ensure the trigger candidates to be sufficiently different from each other (see Fig. 2). Generating multiple diverse trigger candidates can increase the chance of finding the true trigger. It also mitigates the risk of unknown target labels. In the example of Fig. 2, the first trigger candidate flips the model prediction to a label different from the target, while only the third candidate hits the true target label.

Generating multiple trigger candidates, however, adds difficulties in filtering out already-subtle cues for Trojan detection. We also note reverse engineer

ing approaches often suffer from false positive triggers such as adversarial perturbations or direct modification of the crucial objects of the image<sup>1</sup>. In practice, we systematically extract a rich set of features to describe the characteristics of the reconstructed trigger candidates based on geometry, color, and topology, as well as network activations. A Trojan-detection network is then trained to detect Trojaned models based on these features. Our main contributions are summarized as follows:

- We propose a topological prior to regularize the optimization process of reverse engineering. The prior ensures the locality of the recovered triggers, while being sufficiently flexible regarding the appearance. It significantly improves the quality of the reconstructed triggers.  
- We propose a diversity loss to generate multiple diverse trigger candidates. This increases the chance of recovering the true trigger, especially for cases with unknown target labels.  
- Combining the topological prior and diversity loss, we propose a novel Trojan detection framework. On both synthetic and public TrojanAI benchmarks, our method demonstrates substantial improvement in both trigger recovery and Trojan detection.

# 2 RELATED WORK

Trojan detection. Many Trojan detection methods have been proposed recently. Some focus on detecting poisoned inputs via anomaly detection (Chou et al., 2020; Gao et al., 2019; Liu et al., 2017; Ma & Liu, 2019). For example, SentiNet (Chou et al., 2020) tries to identify adversarial inputs, and

uses the behaviors of these adversarial inputs to detect Trojaned models. Others focus on analyzing the behaviors of the trained models (Chen et al., 2019a; Guo et al., 2019; Shen et al., 2021; Sun et al., 2020). Specifically, (Chen et al., 2019a) propose the Activation Clustering (AC) methodology to analyze the activations of neural networks to determine if a model has been poisoned or not.

While early works require all training data to detect Trojans (Chen et al., 2019a; Gao et al., 2019; Tran et al., 2018), recent approaches have been focusing on a more realistic setting – when one has limited access to the training data. A particular promising direction is reverse engineering approaches, which recover Trojan triggers with only a few clean samples. Neural cleansse (NC) (Wang et al., 2019) develops a Trojan detection method by identifying if there is a trigger that would produce misclassified results when added to an input. However, as pointed out by (Guo et al., 2019), NC becomes futile when triggers vary in terms of size, shape, and location.

Since NC, different approaches have been proposed, extending the reverse engineering idea. Using a conditional generative model, DeepInspect (Chen et al., 2019c) learns the probability distribution of potential triggers from a model of interest. (Kolouri et al., 2020) propose to learn universal patterns that change predictions of the model (called Universal Litmus Patterns (ULPs)). The method is efficient as it only involves forward passes through a CNN and avoids backpropagation. ABS (Liu et al., 2019) analyzes inner neuron behaviors by measuring how extra stimulation can change the network's prediction. Wang et al. (2020) propose a data-limited TrojanNet detector (TND) by comparing the impact of per-sample attack and universal attack. Guo et al. (2019) cast Trojan detection as a non-convex optimization problem and it is solved through optimizing an objective function. Huster & Ekwedike (2021) solve the problem by observing that, compared with clean models, adversarial perturbations transfer from image to image more readily in poisoned models.

Existing methods are generally demanding on training data access, neural network architectures, types of triggers, target class, etc. This limits their deployment to real-world applications. As for reverse engineering approaches, it remains challenging, if not entirely infeasible, to recover the true triggers. We propose a novel reverse engineering approach that can recover the triggers with high quality using the novel diversity and topological prior. Our method shares the common benefit of reverse engineering methods; it only needs a few clean input images per model. Meanwhile, our approach is agnostic of model architectures, trigger types, and target labels.

Topological data analysis and persistent homology. Topological data analysis (TDA) is a field in which one analyzes datasets using topological tools such as persistent homology (Edelsbrunner & Harer, 2010; Edelsbrunner et al., 2000). The theory has been applied to different applications (Kulp et al., 2015; Kwitt et al., 2015; Wong et al., 2016; Chazal et al., 2013; Ni et al., 2017; Adams et al., 2017; Bubenik, 2015; Varshney & Ramamurthy, 2015; Hu et al., 2019; 2021).

As the advent of deep learning, some works have tried to incorporate the topological information into deep neural networks, and the differentiable property of persistent homology (Edelsbrunner et al., 2000) make it a possible choice. The main idea is that the persistence diagram/barcodes can capture all the topological changes, and it is differentiable to the original data. Hu et al. (2019) first propose a topological loss to learn to segment images with correct topology, by matching persistence diagrams in a supervised manner. Similarly, Clough et al. (2020) propose to use the persistence barcodes to enforce a given topological prior of the target object. These methods achieve better quality results especially in structural accuracy. In other contexts, persistent-homology-based losses have been applied to different learning problems (Hofer et al., 2019; 2020; Carrière et al., 2020; Chen et al., 2019b).

The aforementioned methods use topological priors in supervised learning tasks (namely, segmentation). Instead, in this work, we propose to leverage the topological prior in an unsupervised setting; we use a topological prior for the reverse engineering pipeline to reduce the search space of triggers and enforce the recovered triggers to have fewer connected components.

# 3 METHOD

Our reverse engineering framework is illustrated in Fig. 3. Given a trained DNN model, either clean or Trojaned, and a few clean images, we use gradient descent to reconstruct triggers that can flip the model's prediction. To increase the quality of reconstructed triggers, we introduce novel diversity loss and topological prior. They help recover multiple diverse triggers of high quality.

The common hypothesis of reverse engineering approaches is that the reconstructed triggers will appear different for Trojaned and clean models. To fully exploit the discriminative power of the reconstructed triggers for Trojan detection, we extract features based on trigger characteristics and associated network activations. These features are used to train a classifier, called the Trojan-detection network, to classify a given model as Trojaned or clean.

We note the discriminative power of the extracted trigger features, and thus the Trojan-detection network, are highly dependent on the quality of the reconstructed triggers. Empirical results will show the proposed diversity loss and topological prior are crucial in reconstructing high quality triggers, and ensures a high quality Trojan-detection network. We will show that our method can learn to detect Trojaned models even when trained with a small amount of labeled DNN models.

For the rest of this section, we mainly focus on the reverse engineering module. We also add details of the Trigger feature extraction to Sec. 3.3 and details of the Trojan-detection network to Sec. A.1.

![](images/db5d4e88e5c39bf8a2bbe11fa0d915485523730e48d3e482f9986529d64b7755.jpg)  
Figure 3: Our Trojan detection framework.

# 3.1 REVERSE ENGINEERING OF MULTIPLE DIVERSE TRIGGER CANDIDATES

We first introduce the reverse engineering approach. Given a trained DNN model, let  $f(\cdot)$  be the mapping from an input clean image  $\mathbf{x} \in \mathbb{R}^{M \times N}$  to the output  $\mathbf{y} \in \mathbb{R}^K$  with  $K$  classes, where  $M$  and  $N$  denote the height and width of the image, respectively. Denote by  $f_k(\cdot)$  the  $k$ -th output of  $f$ . The predicted label  $c^*$  is given by  $c^* = \arg \max_k f_k(\mathbf{x})$ ,  $1 \leq k \leq K$ . We introduce parameters  $\theta$  and  $\mathbf{m}$  to convert  $\mathbf{x}$  into an altered sample

$$
\phi (\mathbf {x}, \mathbf {m}, \boldsymbol {\theta}) = (\mathbf {1} - \mathbf {m}) \odot \mathbf {x} + \mathbf {m} \odot \boldsymbol {\theta}, \tag {1}
$$

![](images/c93638c8fc89525587ed9b8e3a563f93ca19fd021a99bd880022c7a651466c29.jpg)  
Figure 4:  $\mathbf{m}$  and  $\pmb{\theta}$  convert an input image  $\mathbf{x}$  into an altered one  $\phi (\mathbf{x},\mathbf{m},\pmb {\theta})$  The  $\odot$  is omitted here for simplification.

where the binary mask  $\mathbf{m} \in \{0,1\}^{M \times N}$  and the pattern  $\pmb{\theta} \in \mathbb{R}^{M \times N}$  determine the trigger. 1 denotes an all-one matrix. The symbol “ $\odot$ ” denotes Hadamard product. See

Fig. 4 for an illustration. We intend to find a triggered image  $\hat{\mathbf{x}} = \phi (\mathbf{x},\hat{\mathbf{m}},\hat{\boldsymbol{\theta}})$  so that the model prediction  $\hat{c} = \arg \max_{k}f_{k}(\hat{\mathbf{x}})$  is different from the prediction on the original image  $c^*$ .

We find the triggered image,  $\hat{\mathbf{x}}$ , by minimizing a loss over the space of  $\mathbf{m}$  and  $\theta$ :

$$
L (\mathbf {m}, \boldsymbol {\theta}; \mathbf {x}, f, c ^ {*}) = L _ {f l i p} (\dots) + \lambda_ {1} L _ {d i v} (\dots) + \lambda_ {2} L _ {t o p o} (\dots) + R (\mathbf {m}), \tag {2}
$$

where  $L_{flip}$ ,  $L_{div}$ , and  $L_{topo}$  denote the label-flipping loss, diversity loss, and topological loss, respectively. We temporarily dropped their arguments for convenience.  $\lambda_1, \lambda_2$  are the weights to balance the loss terms.  $R(\mathbf{m})$  is a regularization term penalizing the size and range of the mask (more details will be provided in the Sec. A.1 of Appendix).

To facilitate optimization, we relax the constraint on the mask  $\mathbf{m}$  and allow it to be a continuous-valued function, ranging between 0 and 1 and defined over the image domain,  $\mathbf{m} \in [0,1]^{M \times N}$ . Next, we introduce the three loss terms one-by-one.

Label-flipping loss  $L_{flip}$ : The label-flipping loss  $L_{flip}$  penalizes the prediction of the model regarding the ground truth label, formally:

$$
L _ {f l i p} (\mathbf {m}, \boldsymbol {\theta}; \mathbf {x}, f, c ^ {*}) = f _ {c ^ {*}} \left(\phi (\mathbf {x}, \mathbf {m}, \boldsymbol {\theta})\right). \tag {3}
$$

Minimizing  $L_{flip}$  means minimizing the probability that the altered image  $\phi (\mathbf{x},\mathbf{m},\boldsymbol {\theta})$  is predicted as  $c^*$ . In other words, we are pushing the input image out of its initial decision region.

Note that we do not specify which label we would like to flip the prediction to. This makes the optimization easier. Existing approaches often run optimization to flip the label to a target label and enumerate through all possible target labels (Wang et al., 2019; 2020). This can be rather expensive in computation, especially with large label space.

The downside of not specifying a target label during optimization is we will potentially miss the correct target label, i.e., the label which the Trojaned model predicts on a triggered image. To this end, we propose to reconstruct multiple candidate triggers with diversity constraints. This will increase the chance of hitting the correct target label. See Fig. 2 for an illustration.

Diversity loss  $L_{div}$ : With the label-flipping loss  $L_{flip}$ , we flip the label to a different one from the original clean label and recover the corresponding triggers. The new label, however, may not be the same as the true target label. Also considering the huge trigger search space, it is difficult to recover the triggers with only one attempt. Instead, we propose to search for multiple trigger candidates to increase the chance of capturing the true trigger.

We run our algorithm for  $N_T$  rounds, each time reconstructing a different trigger candidate. To avoid finding similar trigger candidates, we introduce the diversity loss  $L_{div}$  to encourage different trigger patterns and locations. Let  $\mathbf{m}_j$  and  $\theta_j$  denote the trigger mask and pattern found in the  $j$ -th round. At the  $i$ -th round, we compare the current candidates with triggers from all previous founds in terms of  $L_2$  norm. Formally:

$$
L _ {d i v} (\mathbf {m}, \boldsymbol {\theta}) = - \sum_ {j = 1} ^ {i - 1} \left\| \mathbf {m} \odot \boldsymbol {\theta} - \mathbf {m} _ {j} \odot \boldsymbol {\theta} _ {j} \right\| _ {2}. \tag {4}
$$

Minimizing  $L_{div}$  ensures the eventual trigger  $\mathbf{m}_i\odot \pmb{\theta}_i$  to be different from triggers from previous rounds. Fig. 1(d)-(f) demonstrates the multiple candidates recovered with sufficient diversity.

# 3.2 TOPOLOGICAL PRIOR

Quality control of the trigger reconstruction remains a major challenge in reverse engineering methods, due to the huge search space of triggers. Even with the regularizer  $R(\mathbf{m})$ , the recovered triggers can still be scattered and unrealistic. See Fig. 1(c) for an illustration. We propose a topological prior to improve the locality of the reconstructed trigger. We introduce a topological loss enforcing that the recovered trigger mask  $\mathbf{m}$  to have as few number of connected components as possible. The loss is based on the theory of persistent homology (Edelsbrunner et al., 2000; Edelsbrunner & Harer, 2010), which models the topological structures of a continuous signal in a robust manner.

Persistent homology. We introduce persistent homology in the context of 2D images. A more comprehensive treatment of the topic can be found in (Edelsbrunner & Harer, 2010; Dey & Wang, 2021). Recall we relaxed the mask function  $\mathbf{m}$  to a continuous-valued function defined over the image domain (denoted by  $\Omega$ ). Given any threshold  $\alpha$ , we can threshold the image domain with regard to  $\mathbf{m}$  and obtain the superlevel set,  $\Omega^{\alpha} := \{p \in \Omega | \mathbf{m}(p) \geq \alpha\}$ . A superlevel set can have different topological structures, e.g., connected components and holes. If we continuously decrease the value  $\alpha$ , we have a continuously growing superlevel set  $\Omega^{\alpha}$ . This sequence of superlevel set is called a filtration. The topology of  $\Omega^{\alpha}$  continuously changes through the filtration. New connected components are born and later die (are merged with others). New holes are born and later die (are sealed up). For each topological structure, the threshold at which it is born is called its birth time. The threshold at which it dies is called its death time. The difference between birth and death time is called the persistence of the topological structure.

We record the lifespan of all topological structures over the filtration and encode them via a 2D point set called persistence diagram, denoted by  $\mathrm{Dgm}(\mathbf{m})$ . Each topological structure is represented by a 2D point within the diagram,  $p \in \mathrm{Dgm}(\mathbf{m})$ , called a persistent dot. We use the birth and death times of the topological structure to define the coordinates of the corresponding persistent dot. For each dot  $p \in \mathrm{Dgm}(\mathbf{m})$ , we abuse the notation and call the birth/death time of its corresponding topological

structure as  $\mathrm{birth}(p)$  and  $\mathrm{death}(p)$ . Then we have  $p = (1 - \mathrm{birth}(p), 1 - \mathrm{death}(p))$ . See Fig. 5 for an example function  $\mathbf{m}$  (viewed as a terrain function) and its corresponding diagrams. There are five dots in the diagram, corresponding to five peaks in the landscape view.

To compute persistence diagram, we use the classic algorithm (Edelsbrunner & Harer, 2010; Edelsbrunner et al., 2000) with an efficient implementation (Chen & Kerber, 2011; Wagner et al., 2012). The image is first discretized into a cubical complex consisting of vertices (pixels), edges and squares. A boundary matrix is then created to encode the adjacency relationship between these elements. The algorithm essentially carries out a matrix reduction algorithm over the boundary matrix, and the reduced matrix reads out the persistence diagram.

![](images/0fbdcd81f1e673aeedb5595b5da74472e2b8b8897e8176c78cad4ffbe153b563.jpg)  
Figure 5: From the left to right: (Left) a sample landscape for a continuous function. The values at the peaks  $\alpha_0 < \alpha_1 < \alpha_2 < \alpha_3 < \alpha_4 < \alpha_5$ . As we decrease the threshold, the topological structures of the superlevel set change, (Middle-Left) and (Middle) correspond to topological structures captured by different thresholds, (Middle-Right) highlighted region in (Left), (Right) the changes are captured by the persistence diagram (right figure). We focus on the 0-dimensional topological structures (connected components). Each persistent dot in the persistence diagram denotes a specific connected component. The topological loss is introduced to reduce the connected components, which means pushing most of the persistent dots to the diagonal (along the green lines).

![](images/022f461d0717953df963b36980107749c18ee3c18d851731bb55dd3e7f4065e1.jpg)

![](images/d164d530b6ba3d9e938092f2e1d4510a00720cf2f53d16f109cd64d896e38403.jpg)

![](images/9af63f98d7b3027f1bd86cb2424866f83c2e7a778841b6b8c88a3524b7b59991.jpg)

![](images/62127c056d18c586d18bdb8a386f53e2cecad5dbc24c1df50a42a730270957ce.jpg)

Topological loss  $L_{topo}$ : We formulate our topological loss based on persistent homology described above. Minimizing our loss reduces the number of connected components of triggers. We will focus on zero-dimensional topological structure, i.e., connected components. Intuitively speaking, each dot in the diagram corresponds to a connected component. The ones far away from the diagonal line are considered salient as its birth and death times are far apart. And the ones close to the diagonal line are considered trivial. In Figure 5, there is one salient dot far away from the diagonal line. It corresponds to the highest peak. The other four dots are closer to the diagonal like and correspond to the smaller peaks. The topological loss will reduce the number of connected components by penalizing the distance of all dots from the diagonal line, except for the most salient one. Formally, the loss  $L_{topo}$  is defined as:

$$
L _ {t o p o} (\mathbf {m}) = \sum_ {p \in \operatorname {D g m} (m) \backslash \{p ^ {*} \}} \left[ \operatorname {b i r t h} (p) - \operatorname {d e a t h} (p) \right] ^ {2}, \tag {5}
$$

where  $p^*$  denotes the persistent dot that is farthest away from the diagonal (with the highest persistence). Minimizing this loss will keep  $p^*$  intact, while pushing all other dots to the diagonal line, thus making their corresponding components either disappear or merged with the main component.

Differentiability and the gradient: The loss function (Eq. (5)) is differentiable almost everywhere in the space of functions. To see this, we revisit the filtration, i.e., the growing superlevel set as we continuously decrease the threshold  $\alpha$ . The topological structures change at specific locations of the image domain. A component is born at the corresponding local maximum. It dies merging with another component at the saddle point between the two peaks. In fact, these locations correspond to critical points of the function. And the function values at these critical points correspond to the birth and death times of these topological structures. For a persistent dot,  $p$ , we call the critical point corresponding to its birth,  $c_{b}(p)$ , and the critical point corresponding to its death,  $c_{d}(p)$ . Then we have  $\mathrm{birth}(p) = \mathbf{m}(c_b(p))$  and  $\mathrm{death}(p) = \mathbf{m}(c_d(p))$ . The loss function (Eq. 5) can be rewritten as a polynomial function of the function  $\mathbf{m}$  at different critical points.

$$
L _ {t o p o} (\mathbf {m}) = \sum_ {p \in \operatorname {D g m} (m) \backslash \{p ^ {*} \}} [ \mathbf {m} \left(c _ {b} (p)\right) - \mathbf {m} \left(c _ {d} (p)\right) ] ^ {2}, \tag {6}
$$

The gradient can be computed naturally.  $L_{topo}$  is a piecewise differentiable loss function over the space of all possible functions  $\mathbf{m}$ . In a gradient decent step, for all dots except for  $p^*$ , we push up the function at the death critical point  $c_d(p)$  (the saddle), and push down the function value at the birth

critical point  $c_{b}(p)$  (the local maximum). This is illustrated by the arrows in Fig. 5(Middle-Right). This will kill the non-salient components and push them towards the diagonal.

# 3.3 TRIGGER FEATURE EXTRACTION AND TROJAN DETECTION NETWORK

Next we summarize the features we extract from recovered triggers. The recovered Trojan triggers can be characterized via their capability in flipping model predictions (i.e., the label-flipping loss). Moreover, they are different from adversarial noise as they tend to be more regularly shaped and are also distinct from actual objects which can be recognized by a trained model. We introduce appearance-based features to differentiate triggers from adversarial noise and actual objects, .

Specifically, for label flipping capability, we directly use the label-flipping loss  $L_{flip}$  and diversity loss  $L_{div}$  as features. For appearance features, we use trigger size and topological statistics as their features: 1) The number of foreground pixels divided by total number of pixels in mask m; 2) To capture the size of the triggers in the horizontal and vertical directions, we fit a Gaussian distribution to the mask m and record mean and std in both directions; 3) The trigger we find may have multiple connected components. The final formulated topological descriptor includes the topological loss  $L_{topo}$ , the number of connected components, mean and std in terms of the size of each component.

After the features are extracted, we build a neural network for Trojan detection, which takes the bag of features of the generated triggers as inputs, and outputs a scalar score of whether the model is Trojaned or not. More details are provided in Sec. A.1 of Appendix.

# 4 EXPERIMENTS

We evaluate our method on both synthetic datasets and publicly available TrojanAI benchmarks. We provide quantitative and qualitative results, followed by ablation studies, to demonstrate the efficacy of the proposed method. All clean/Trojaned models are DNNs trained for image classification.

Synthetic datasets (Trojaned-MNIST and Trojaned-CIFAR10): We adopt the codes provided by NIST $^2$  to generate 200 DNNs (50% of them are Trojaned) trained to classify MNIST and CIFAR10 data, respectively. The Trojaned models are trained with images poisoned by square triggers. The poison rate is set as 0.2.

TrojAI benchmarks (TrojAI-Round1, Round2, Round3 and Round4): These datasets are provided by US IARPA/NIST<sup>3</sup>, who recently organized a Trojan AI competition. Polygon triggers are generated randomly with variations in shape, size, and color. Filter-based triggers are generated by randomly choosing from five distinct filters. Trojan detection is more challenging on these Trojan AI datasets as compared to Triggered-MNIST due to the use of deeper DNNs and larger variations in appearances of foreground/background objects, trigger patterns etc. Round1, Round2, Round3 and Round4 have 1000, 1104, 1008 and 1008 models, respectively. Descriptions of the difference among these rounds are provided in Sec. A.2 of Appendix.

Baselines: We choose recently published methods including NC (Neural Cleanse) (Wang et al., 2019), ABS (Liu et al., 2019), TABOR (Guo et al., 2019), ULP (Kolouri et al., 2020), and DLTND (Wang et al., 2020) as SOTA baselines.

Implementation details: We set  $\lambda_{1} = 1$ ,  $\lambda_{2} = 10$  and  $N_{T} = 3$  for all our experiments (i.e., we generate 3 trigger candidates for each input image and each model). The parameters of Trojan detection network are learned using a set of clean and Trojaned

Table 1: Comparison on Trojaned-MNIST/CIFAR10.  

<table><tr><td>Method</td><td>Metric</td><td>Trojaned-MNIST</td><td>Trojaned-CIFAR10</td></tr><tr><td>NC</td><td>AUC</td><td>0.57 ± 0.07</td><td>0.75 ± 0.07</td></tr><tr><td>ABS</td><td>AUC</td><td>0.63 ± 0.04</td><td>0.67 ± 0.06</td></tr><tr><td>TABOR</td><td>AUC</td><td>0.65 ± 0.07</td><td>0.71 ± 0.05</td></tr><tr><td>ULP</td><td>AUC</td><td>0.59 ± 0.03</td><td>0.55 ± 0.03</td></tr><tr><td>DLTND</td><td>AUC</td><td>0.62 ± 0.05</td><td>0.52 ± 0.08</td></tr><tr><td>Ours</td><td>AUC</td><td>0.88 ± 0.04</td><td>0.91 ± 0.05</td></tr><tr><td>NC</td><td>ACC</td><td>0.60 ± 0.04</td><td>0.73 ± 0.06</td></tr><tr><td>ABS</td><td>ACC</td><td>0.65 ± 0.02</td><td>0.69 ± 0.04</td></tr><tr><td>TABOR</td><td>ACC</td><td>0.62 ± 0.04</td><td>0.69 ± 0.08</td></tr><tr><td>ULP</td><td>ACC</td><td>0.57 ± 0.02</td><td>0.59 ± 0.06</td></tr><tr><td>DLTND</td><td>ACC</td><td>0.64 ± 0.07</td><td>0.55 ± 0.07</td></tr><tr><td>Ours</td><td>ACC</td><td>0.89 ± 0.02</td><td>0.92 ± 0.04</td></tr></table>

models with ground truth labeling. We train the detection network by optimizing cross entropy loss

![](images/d75a42e37b3d232ced854f3343bce1c82361d965b6fabd0603b2c4df1d0056c1.jpg)  
Figure 6: Examples of recovered triggers overlaid on clean images. From left to right: (a) clean image, (b) triggers recovered by (Wang et al., 2019), (c) triggers recovered by (Liu et al., 2019), (d) triggers recovered by (Guo et al., 2019), (e) triggers recovered by our method without topological prior, and (f) triggers recovered by our method with topological prior.

using the Adam optimizer (Kingma & Ba, 2014). The hidden state size, number of layers of  $MLP_{\alpha}$ ,  $MLP_{\beta}$ , as well as optimizer learning rate, weight decay and number of epochs are optimized using Bayesian hyperparameter search for 500 rounds on 8-fold cross-validation.

Evaluation metrics: We follow the settings in (Sikka et al., 2020). We report the mean and standard deviation of two metrics: area under the ROC curve (AUC) and accuracy (ACC). Specifically, we use  $80\%$  of the models for training,  $10\%$  for validation, and the rest  $10\%$  for testing.

Results: Tables 1 and 2 show the quantitative results on the Trojaned-MNIST/CIFAR10 and TrojanAI datasets, respectively. The reported performances of baselines are reproduced using source codes provided by the authors or quoted from related papers. The best performing numbers are highlighted in bold. From Tab. 1 and 2, we observe that our method performs substantially better than the baselines. It is also worth noting that, compared with these baselines, our proposed method extracts fix-sized features for each model, independent of the number of classes, architectures, trigger types, etc. By using the extracted features, we are able to train a separate Trojan detection network, which is salable and model-agnostic.

Table 2: Performance comparison on the TrojAI dataset.  

<table><tr><td>Method</td><td>Metric</td><td>TrojAI-Round1</td><td>TrojAI-Round2</td><td>TrojAI-Round3</td><td>TrojAI-Round4</td></tr><tr><td>NC</td><td>AUC</td><td>0.50 ± 0.03</td><td>0.63 ± 0.04</td><td>0.61 ± 0.06</td><td>0.58 ± 0.05</td></tr><tr><td>ABS</td><td>AUC</td><td>0.68 ± 0.05</td><td>0.61 ± 0.06</td><td>0.57 ± 0.04</td><td>0.53 ± 0.06</td></tr><tr><td>TABOR</td><td>AUC</td><td>0.71 ± 0.04</td><td>0.66 ± 0.07</td><td>0.50 ± 0.07</td><td>0.52 ± 0.04</td></tr><tr><td>ULP</td><td>AUC</td><td>0.55 ± 0.06</td><td>0.48 ± 0.02</td><td>0.53 ± 0.06</td><td>0.54 ± 0.02</td></tr><tr><td>DLTND</td><td>AUC</td><td>0.61 ± 0.07</td><td>0.58 ± 0.04</td><td>0.62 ± 0.07</td><td>0.56 ± 0.05</td></tr><tr><td>Ours</td><td>AUC</td><td>0.90 ± 0.02</td><td>0.87 ± 0.05</td><td>0.89 ± 0.04</td><td>0.92 ± 0.06</td></tr><tr><td>NC</td><td>ACC</td><td>0.53 ± 0.04</td><td>0.49 ± 0.02</td><td>0.59 ± 0.07</td><td>0.60 ± 0.04</td></tr><tr><td>ABS</td><td>ACC</td><td>0.70 ± 0.04</td><td>0.59 ± 0.05</td><td>0.56 ± 0.03</td><td>0.51 ± 0.05</td></tr><tr><td>TABOR</td><td>ACC</td><td>0.70 ± 0.03</td><td>0.68 ± 0.08</td><td>0.51 ± 0.05</td><td>0.55 ± 0.06</td></tr><tr><td>ULP</td><td>ACC</td><td>0.58 ± 0.07</td><td>0.51 ± 0.03</td><td>0.56 ± 0.04</td><td>0.57 ± 0.04</td></tr><tr><td>DLTND</td><td>ACC</td><td>0.59 ± 0.04</td><td>0.61 ± 0.05</td><td>0.65± 0.04</td><td>0.59 ± 0.06</td></tr><tr><td>Ours</td><td>ACC</td><td>0.91 ± 0.03</td><td>0.89 ± 0.04</td><td>0.90 ± 0.03</td><td>0.91 ± 0.04</td></tr></table>

Fig. 6 shows a few examples of recovered triggers. We observe that, compared with the baselines, the triggers found by our method are more compact and of better quality. This is mainly due to the introduction of topological constraints. The improved quality of recovered triggers directly results in improved performance of Trojan detection.

Ablation study of loss weights: For the loss weights  $\lambda_{1}$  and  $\lambda_{2}$ , we empirically choose the weights which make reverse engineering converge the fastest. This is a reasonable choice as in practice, time is one major concern for reverse engineering pipelines.

Despite the seemingly ad hoc choice, we have observed that our performances are quite robust to all these loss weights. As topological loss is a major contribution of this paper, we conduct an ablation study in terms of its weight  $(\lambda_{2})$  on TrojAI-Round4 dataset. The results are reported in Fig. 7. We observe that the proposed method is quite robust to  $\lambda_{2}$ , and when  $\lambda = 10$ , it achieves slightly better performance (AUC:  $0.92 \pm 0.06$ ) than other choices.

Ablation study of number of training model samples: The trigger features and Trojan detection network are important in achieving SOTA performance. To further demonstrate the efficacy of the proposed diversity and topological loss terms, we conduct another ablation study to investigate the case with less training model samples, and thus a weaker Trojan detection network.

The ablation study in terms of number of training samples on TrojanAI-Round4 data is illustrated in Tab. 3. We observe that the proposed topological loss and diversity loss will boost the performance with or without a fully trained

![](images/de1047627eb6f0a89c888dbcb704e1289b6a4d2f8e84813463a5f1c7b94ddfef.jpg)  
Figure 7: Ablation study results for  $\lambda_{2}$ .

Trojan-detection network. These two losses improve the quality of the recovered trigger, in spite of how the trigger information is used. Thus even with a limited number of training samples (e.g., 25), the proposed method could still achieve significantly better performance than the baseline methods.

Ablation study for loss terms: We investigate the individual contribution of different loss terms used to search for the latent triggers. Tab. 4 lists the corresponding performance on the TrojAI-Round4 dataset. We observe a decrease in AUC (from 0.92 to 0.89) if the topological loss is removed. This drop is expected as the topological loss helps to find more compact triggers. Also, the performance drops signifi

Table 3: Ablation study for # of training samples.  

<table><tr><td># of samples</td><td>Ours</td><td>w/o topo</td><td>w/o diversity</td></tr><tr><td>25</td><td>0.77 ± 0.04</td><td>0.73 ± 0.03</td><td>0.68 ± 0.04</td></tr><tr><td>50</td><td>0.81 ± 0.03</td><td>0.76 ± 0.05</td><td>0.73 ± 0.02</td></tr><tr><td>100</td><td>0.84 ± 0.05</td><td>0.78 ± 0.06</td><td>0.76 ± 0.03</td></tr><tr><td>200</td><td>0.86 ± 0.04</td><td>0.82 ± 0.04</td><td>0.79 ± 0.05</td></tr><tr><td>400</td><td>0.90 ± 0.05</td><td>0.85 ± 0.03</td><td>0.82 ± 0.04</td></tr><tr><td>800</td><td>0.92 ± 0.06</td><td>0.89 ± 0.04</td><td>0.85 ± 0.02</td></tr></table>

cantly (from 0.92 to 0.85 in AUC) if the diversity loss is removed. We also report the performance by setting  $N_{T} = 2$ ; when  $N_{T} = 2$ , the performance increases from 0.85 to 0.89 in AUC. The reason is that with diversity loss, we are able to generate multiple diverse trigger candidates, which increases the probability of recovering the true trigger when the target class is unknown. Our ablation study justifies the use of both diversity and topological losses.

In practice, we found that topological loss can improve the convergence of trigger search. Without topological loss, it takes  $\approx 50$  iterations to find a reasonable trigger (Fig. 6(e)). In contrast, with the topological loss, it takes only  $\approx 30$  iterations to converge to a better recovered trigger (Fig. 6(f)). The rationale is that, as the topological loss imposes strong

<table><tr><td colspan="2">Table 4: Ablation results of loss terms.</td></tr><tr><td>Method</td><td>TrojAI-Round4</td></tr><tr><td>w/o topological loss</td><td>0.89 ± 0.04</td></tr><tr><td>w/o diversity loss (NT=1)</td><td>0.85 ± 0.02</td></tr><tr><td>NT=2</td><td>0.89 ± 0.05</td></tr><tr><td>with all loss terms (NT=3)</td><td>0.92 ± 0.06</td></tr></table>

constraints on the number of connected components, it largely reduces the search space of triggers, consequently, making the convergence of trigger search much faster. This is worth further investigation.

# 5 CONCLUSION

In this paper, we propose a diversity loss and a topological prior to improve the quality of the trigger reverse engineering for Trojan detection. These loss terms help finding high quality triggers efficiently. They also avoid the dependant of the method to the target label. On both synthetic datasets and publicly available TrojanAI benchmarks, our approach recovers high quality triggers and achieves SOTA Trojan detection performance.

Ethics Statement: As we have developed better Trojan detection algorithm and introduce the method in details, the attackers may inversely create Trojaned models that are more difficult to detect based on the limitations of current method. Attack and defense will always coexist, which pushes researchers to keep developing more efficient algorithms.  
Reproducibility Statement: The implementation details are mentioned in Sec. 4. The details of the data are provided in Sec. A.2 of Appendix. The details of Trojan detection classifier are described in Sec. A.1 of Appendix. The used computation resources are specified in Sec. A.4 of Appendix.

# REFERENCES

Henry Adams, Tegan Emerson, Michael Kirby, Rachel Neville, Chris Peterson, Patrick Shipman, Sofya Chepushtanova, Eric Hanson, Francis Motta, and Lori Ziegelmeier. Persistence images: A stable vector representation of persistent homology. The Journal of Machine Learning Research, 18(1):218-252, 2017.  
Peter Bubenik. Statistical topological data analysis using persistence landscapes. The Journal of Machine Learning Research, 16(1):77-102, 2015.  
Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, and Yuhei Umeda. Perslay: A neural network layer for persistence diagrams and new graph topological signatures. In International Conference on Artificial Intelligence and Statistics, pp. 2786-2796. PMLR, 2020.  
Frédéric Chazal, Leonidas J Guibas, Steve Y Oudot, and Primoz Skraba. Persistence-based clustering in riemannian manifolds. Journal of the ACM (JACM), 60(6):41, 2013.  
Bryant Chen, Wilka Carvalho, Nathalie Baracaldo, Heiko Ludwig, Benjamin Edwards, Taesung Lee, Ian Molloy, and Biplav Srivastava. Detecting backdoor attacks on deep neural networks by activation clustering. In SafeAI@ AAAI, 2019a.  
Chao Chen and Michael Kerber. Persistent homology computation with a twist. In Proceedings 27th European Workshop on Computational Geometry, volume 11, pp. 197-200, 2011.  
Chao Chen, Xiuyan Ni, Qinxun Bai, and Yusu Wang. A topological regularizer for classifiers via persistent homology. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2573-2582. PMLR, 2019b.  
Huili Chen, Cheng Fu, Jishen Zhao, and Farinaz Koushanfar. Deepinspect: A black-box trojan detection and mitigation framework for deep neural networks. In *IJCAI*, pp. 4658-4664, 2019c.  
Edward Chou, Florian Tramér, and Giancarlo Pellegrino. Sentinel: Detecting localized universal attacks against deep learning systems. In 2020 IEEE Security and Privacy Workshops (SPW), pp. 48-54. IEEE, 2020.  
James Clough, Nicholas Byrne, Ilkay Oksuz, Veronika A Zimmer, Julia A Schnabel, and Andrew King. A topological loss function for deep-learning based image segmentation using persistent homology. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.  
Tamal K. Dey and Yusu Wang. Computational Topology for Data Analysis. Cambridge University Press, 2021.  
Herbert Edelsbrunner and John Harer. Computational topology: an introduction. American Mathematical Soc., 2010.  
Herbert Edelsbrunner, David Letscher, and Afra Zomorodian. Topological persistence and simplification. In Proceedings 41st Annual Symposium on Foundations of Computer Science, pp. 454-463. IEEE, 2000.  
Yansong Gao, Change Xu, Derui Wang, Shiping Chen, Damith C Ranasinghe, and Surya Nepal. Strip: A defence against trojan attacks on deep neural networks. In Proceedings of the 35th Annual Computer Security Applications Conference, pp. 113-125, 2019.

Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 580-587, 2014.  
Wenbo Guo, Lun Wang, Xinyu Xing, Min Du, and Dawn Song. Tabor: A highly accurate approach to inspecting and restoring trojan backdoors in ai systems. 20th IEEE International Conference on Data Mining, 2019.  
Christoph Hofer, Roland Kwitt, Marc Niethammer, and Mandar Dixit. Connectivity-optimized representation learning via persistent homology. In International Conference on Machine Learning, pp. 2751-2760. PMLR, 2019.  
Christoph Hofer, Florian Graf, Bastian Rieck, Marc Niethammer, and Roland Kwitt. Graph filtration learning. In International Conference on Machine Learning, pp. 4314-4323. PMLR, 2020.  
Xiaoling Hu, Fuxin Li, Dimitris Samaras, and Chao Chen. Topology-preserving deep image segmentation. Advances in neural information processing systems, 32, 2019.  
Xiaoling Hu, Yusu Wang, Li Fuxin, Dimitris Samaras, and Chao Chen. Topology-aware segmentation using discrete Morse theory. International Conference on Learning Representations, 2021.  
Todd Huster and Emmanuel Ekwedike. Top: Backdoor detection in neural networks via transferability of perturbation. arXiv preprint arXiv:2103.10274, 2021.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Soheil Kolouri, Aniruddha Saha, Hamed Pirsiavash, and Heiko Hoffmann. Universal litmus patterns: Revealing backdoor attacks in cnns. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 301-310, 2020.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25:1097-1105, 2012.  
Scott Kulp, Chao Chen, Dimitris Metaxas, and Leon Axel. Ventricular blood flow analysis using topological methods. In 2015 IEEE 12th International Symposium on Biomedical Imaging (ISBI), pp. 663-666. IEEE, 2015.  
Roland Kwitt, Stefan Huber, Marc Niethammer, Weili Lin, and Ulrich Bauer. Statistical topological data analysis-a kernel perspective. In Advances in neural information processing systems, pp. 3070-3078, 2015.  
Yingqi Liu, Wen-Chuan Lee, Guanhong Tao, Shiqing Ma, Yousra Aafer, and Xiangyu Zhang. Abs: Scanning neural networks for back-doors by artificial brain stimulation. In Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, pp. 1265-1282, 2019.  
Yuntao Liu, Yang Xie, and Ankur Srivastava. Neural trojans. In 2017 IEEE International Conference on Computer Design (ICCD), pp. 45-48. IEEE, 2017.  
Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3431-3440, 2015.  
Shiqing Ma and Yingqi Liu. Nic: Detecting adversarial samples with neural network invariant checking. In Proceedings of the 26th Network and Distributed System Security Symposium (NDSS 2019), 2019.  
Xiuyan Ni, Novi Quadrianto, Yusu Wang, and Chao Chen. Composing tree graphical models with persistent homology features for clustering mixed-type data. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 2622-2631. JMLR.org, 2017.  
Guangyu Shen, Yingqi Liu, Guanhong Tao, Shengwei An, Qiuling Xu, Siyuan Cheng, Shiqing Ma, and Xiangyu Zhang. Backdoor scanning for deep neural networks through k-arm optimization. arXiv preprint arXiv:2102.05123, 2021.

Karan Sikka, Indranil Sur, Susmit Jha, Anirban Roy, and Ajay Divakaran. Detecting trojaned dnns using counterfactual attributions. arXiv preprint arXiv:2012.02275, 2020.  
Mingjie Sun, Siddhant Agarwal, and J Zico Kolter. Poisoned classifiers are not only backdoored, they are fundamentally broken. arXiv preprint arXiv:2010.09080, 2020.  
Brandon Tran, Jerry Li, and Aleksander Madry. Spectral signatures in backdoor attacks. Advances in neural information processing systems, 2018.  
Kush R Varshney and Karthikeyan Natesan Ramamurthy. Persistent topology of decision boundaries. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3931-3935. IEEE, 2015.  
Hubert Wagner, Chao Chen, and Erald Vuçini. Efficient computation of persistent homology for cubical data. In Topological methods in data analysis and visualization II, pp. 91-106. Springer, 2012.  
Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y Zhao. Neural cleansse: Identifying and mitigating backdoor attacks in neural networks. In 2019 IEEE Symposium on Security and Privacy (SP), pp. 707-723. IEEE, 2019.  
Ren Wang, Gaoyuan Zhang, Sijia Liu, Pin-Yu Chen, Jinjun Xiong, and Meng Wang. Practical detection of trojan neural networks: Data-limited and data-free cases. European Conference on Computer Vision (ECCV), 2020.  
Eleanor Wong, Sourabh Palande, Bei Wang, Brandon Zielinski, Jeffrey Anderson, and P Thomas Fletcher. Kernel partial least squares regression for relating functional brain network topology to clinical measures of behavior. In 2016 IEEE 13th International Symposium on Biomedical Imaging (ISBI), pp. 1303-1306. IEEE, 2016.  
Eric Wong, Leslie Rice, and J Zico Kolter. Fast is better than free: Revisiting adversarial training. In International Conference on Learning Representations, 2019.
