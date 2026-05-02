# OBJECT PURSUIT: BUILDING A SPACE OF OBJECTS VIA DISCRIMINATIVE WEIGHT GENERATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a framework to continuously learn object-centric representations for visual learning and understanding. Existing object-centric representations either rely on supervisions that individualize objects in the scene, or perform unsupervised disentanglement that can hardly deal with complex scenes in the real world. To mitigate the annotation burden and relax the constraints on the statistical complexity of the data, our method leverages interactions to effectively sample diverse variations of an object and the corresponding training signals while learning the object-centric representations. Throughout learning, objects are streamed one by one in random order with unknown identities, and are associated with latent codes that can synthesize discriminative weights for each object through a convolutional hypernetwork. Moreover, re-identification of learned objects and forgetting prevention are employed to make the learning process efficient and robust. We perform an extensive study of the key features of the proposed framework and analyze the characteristics of the learned representations. Furthermore, we demonstrate the capability of the proposed framework in learning representations that can improve label efficiency in downstream tasks. Our code and trained models will be made publicly available.

# 1 INTRODUCTION

What are human infants and toddlers learning while they are manipulating a discovered object? And, how do such continual interaction and learning experiences, i.e., objects are discovered and learned one by one, help develop the capability to understand the scenes that consist of individual objects? Inspired by these questions, we aim for training frameworks that enable an autonomous agent to continuously learn object-centric representations through self-supervised discovery and manipulation of objects, so that the agent can later use the learned representations for visual scene understanding.

A majority of object-centric representation learning methods focus on encoding images or video clips into disentangled latent codes, each of which explains an entity in the scene, and together they should reconstruct the input. However, without explicit supervision and more sophisticated inductive biases beyond parsimony, the disentanglement usually has difficulties aligning with objects, especially for complex scenes. We leverage the fact that an autonomous agent can actively explore the scene, and propose that the data collected by manipulating a discovered object can serve as an important source for building inductive biases for object-level disentanglement.

In our proposed framework, whenever an object is discovered by the agent, a dataset containing images and instance masks of this object can easily be sampled via interaction compared to annotating all the objects. Theoretically speaking, any function of the images induced by the discovered object could be a representation of the object. For example, let  $\phi$  be an encoder implemented by a neural network, and let  $\mathbf{x}$  be the image of an object, we can say that  $\phi(\mathbf{x})$  is a representation of the object. Similarly, the encoder itself can also be a representation of this object since  $\phi = \arg \min_{\phi} \mathcal{L}(\phi, \mathbf{x})$ , i.e.,  $\phi$  is the output of an optimization procedure that takes the object's images as input.

We employ network weights as the object-centric representations. Specifically, the proposed method learns an object-centric representation from the data collected by manipulating a single object, through learning a latent code that can be translated into a neural network. The neural network is produced by a discriminative weight generation hypernetwork and is able to distinguish the represented object from anything else. In order to learn representations for objects that stream in one

by one, the proposed framework is augmented with an object re-identification procedure to avoid learning seen objects. Moreover, we hypothesize that object representations are embedded in a low-dimensional manifold, so the proposed framework first checks whether a new object can be represented by learned objects; if not, the new object will be learned as a base object serving the purpose of representing future objects, thus the name object pursuit. Furthermore, the proposed framework deals with the catastrophic forgetting of learned object representations by enforcing the hypernetwork to maintain the mapping between the learned representations and their corresponding network weights.

In summary, our work makes the following contributions: 1) we propose a novel framework named object pursuit that can continuously learn object-centric representations using training data collected from interactions with individual objects, 2) we perform an extensive study to understand the pursuit dynamics and characterize its typical behaviors regarding the key design features, and 3) we analyze the learned object space, in terms of its succinctness and effectiveness in representing objects, and empirically demonstrate its potential for label efficient visual learning.

# 2 RELATED WORK

Object-centric representation learning falls in the field of disentangled representation learning (Higgins et al., 2016; Kim & Mnih, 2018; Press et al., 2019; Chen et al., 2018b; Karras et al., 2019; Li et al., 2020; Locatello et al., 2020a; Zhou et al., 2021). However, object-centric representations require that the disentangled latents correspond to objects in the scene. For example, (Eslami et al., 2016; Kosiorek et al., 2018) model image formation as a structured generative process so that each component may represent an object in the generated image. One can also apply inverse graphics (Yao et al., 2018; Wu et al., 2017) or spatial mixture models (Greff et al., 2017; 2019; Engelcke et al., 2020b) to decompose images into interpretable latents. Monet (Burgess et al., 2019) jointly predicts segmentation and representation with a recurrent variational auto-encoder. Capsule autoencoders (Kosiorek et al., 2019) are proposed to decompose images into parts and poses that can be arranged into objects. To deal with complex images or scenes, (Yang et al., 2020; Bear et al., 2020) employ motion to encourage decomposition into objects. Besides motion, (Klindt et al., 2021) shows that the transition statistics can be informative about objects in natural videos. Similarly, (Kabra et al., 2021) infers object latents and frame latents from videos. Slot-attention (Locatello et al., 2020b; Jiang et al., 2020) employs the attention mechanism that aggregates features with similar appearance, while Giraffe (Niemeyer & Geiger, 2021) factorizes the scene using neural feature fields. Even though better performance is achieved with more sophisticated network designs, scenes with complex geometry and appearance still lag. As shown in (Engelcke et al., 2020a), the reconstruction bottleneck has critical effects on the disentanglement quality. Instead of relying on reconstruction as a learning signal, our work calls for interactions that stimulate and collect training data from complex environments.

Rehearsal-based continual learning. In general, continual learning methods can be divided into three streams: rehearsal-based, regularization-based, and expansion-based. The rehearsal-based method manages buffers to replay past samples, in order to prevent from forgetting knowledge of the preceding tasks. The regularization-based methods learn to regularize the changes in parameters of the models. The expansion-based methods aim to expand model architectures in a dynamic manner. Among these three types, rehearsal-based methods are widely-used due to their simplicity and effectiveness (Luders et al., 2016; Kemker & Kanan, 2017; Rebuffi et al., 2017; Cha et al., 2021; von Oswald et al., 2019; Riemer et al., 2018; Lopez-Paz & Ranzato, 2017; Buzzega et al., 2020; Aljundi et al., 2019; Chaudhry et al., 2020; Parisi et al., 2018; Lopez-Paz & Ranzato, 2017). Samples from previous tasks can either be the data or corresponding network activations on the data. For example, (Shin et al., 2017) proposes a dual-model architecture where training data from learned tasks can be sampled from a generative model and (Draelos et al., 2017; Kamra et al., 2017) propose sampling in the output space of an encoder for training tasks relying on an auto-encoder architecture. ICaRL Rebuffi et al. (2017) allows adding new classes progressively based on the training samples with a small number of classes, while (Pellegrini et al., 2020; Li & Hoiem, 2017) store activations volumes at some intermediate layer to alleviate the computation and storage requirement. Co $^2$ L (Cha et al., 2021) proposes continual learning within the contrastive representation learning framework, and (Balaji et al., 2020) studies continual learning in large scale where tasks in the input sequence are not limited to classification. Similar to the forgetting prevention component in our framework, von

![](images/6b3622995edf2865f92190cc0f1befe72956294901cd52d35ea857de8a568a22.jpg)  
Figure 1: Object space as discriminative weights. Objects live in a low-dimensional manifold of a high-dimensional latent space. A latent code representing a specific object is translated into segmentation weights that can distinguish the object from anything else at different viewing conditions. The hypernetwork consists of blocks built of convolutional and upsampling layers.

Oswald et al. (2019) applies a task-conditioned hypernetwork to rehearse the task-specific weight realizations. Please refer to (Parisi et al., 2019; Delange et al., 2021) for a more comprehensive review on this subject.

Hypernetwork. The goal of hypernetworks is to generate the weights of a target network, which is responsible for the main task (Ha et al., 2016; Krueger et al., 2017; Chung et al., 2016; Bertinetto et al., 2016; Lorraine & Duvenaud, 2018; Sitzmann et al., 2020; Nirkin et al., 2021). For example, (Krueger et al., 2017) proposes Bayesian hypernetworks to learn the variational inference in neural networks and (Bertinetto et al., 2016) proposes to learn the network parameters in one shot. HyperSeg (Nirkin et al., 2021) presents real-time semantic segmentation by employing a U-Net within a U-Net architecture, and (Finn et al., 2019) applies hypernetwork to adapt to new tasks for continual lifelong learning. Moreover, (Tay et al., 2020) proposes a new transformer architecture that leverages task-conditioned hypernetworks for controlling its feed-forward layers, whereas (Ma et al., 2021) proposes hyper-convolution, which implicitly represents the convolution kernel as a function of kernel coordinates. Hypernetworks have shown great potential in different meta-learning settings (Rusu et al., 2018; Munkhdalai & Yu, 2017; Wang et al., 2019), mainly due to that hypernetworks are effective in compressing the primary networks' weights as proved in (Galanti & Wolf, 2020).

# 3 METHOD

We consider an agent that can explore the environment and manipulate objects which are discovered in an unknown order. Suppose there are  $N$  objects in the scene, each of which randomly appears in an image  $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ , whose ground-truth instance segmentation mask is  $\mathbf{y} \in \mathbb{R}^{H \times W \times N}$ . One can train a deep neural network that maps an image  $\mathbf{x}$  to its mask  $\mathbf{y}$  with a dataset  $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}$  that consists of such paired training samples. However, sampling from the joint distribution  $\mathrm{p}(\mathbf{x}, \mathbf{y})$  can be extremely time-consuming, e.g., someone may have to manually draw the instance masks for every object in an image.

On the other hand, sampling from the marginals can be much more accessible through interactions. Let  $\mathcal{D}^k$  be the dataset collected by observing an image  $\mathbf{x}_i$  and the corresponding binary mask of the  $k$ -th object  $\mathbf{y}_i^k \in \mathbb{R}^{H \times W}$ , i.e.,  $\mathcal{D}^k = \{(\mathbf{x}_i, \mathbf{y}_i^k)\} \sim \mathrm{p}(\mathbf{x}, \mathbf{y}^k)$ , which is the marginal distribution obtained by integrating out other objects' masks in  $\mathbf{y}$ . The goal of the proposed object pursuit framework is to learn object-centric representations from the data collected by continuously sampling the marginals. Next, we detail the representations used for objects (as illustrated in Fig. 1), and how we can learn them without catastrophic forgetting.

# 3.1 REPRESENTING OBJECTS VIA DISCRIMINATIVE WEIGHT GENERATION

In order to represent an object, one can compute any functions of the data produced with this object. For example, the encoding of an image containing a specific object that can be used to reconstruct the input image. Here we take a conjugate perspective instead of asking the representation to store information of an object that is good for reconstruction. We propose that the object-centric representation of an object shall generate the mechanisms for performing certain downstream tasks on this object, e.g., distinguishing this object from the others.

Let  $\phi$  be a segmentation network with learnable weights  $\theta$  that maps an image to a binary mask, i.e.,  $\phi : \Theta \times \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{H \times W}$ . Moreover, let  $\psi : \zeta \to \Theta$  be the mapping from the latent space  $\zeta$  to the weights of the segmentation backbone  $\phi$ . We define the object-centric representation of an object  $o$  as a latent  $z_o \in \zeta$ , such that:

$$
\mathbb {E} _ {(\mathrm {x}, \mathrm {y} ^ {o}) \sim \mathrm {p} (x, \mathrm {y} ^ {o})} \Delta (\phi (\psi (\mathrm {z} _ {o}), \mathrm {x}), \mathrm {y} ^ {o}) \geq \tau , \tag {1}
$$

where the expectation is computed according to  $\mathrm{p(x,y^o)}$ , i.e., the marginal distribution of object  $o$ , and  $\Delta$  is a similarity measure between the prediction from  $\phi$  and the sampled mask  $\mathbf{y}^o$ . In other words,  $\mathbf{z}_o$  is a representation of object  $o$ , if the network weights generated from  $\mathbf{z}_o$  are capable of predicting high-quality instance masks regarding the object under the corresponding marginal distribution. The threshold  $\tau$  is a scalar parameter that will be studied in the experiments. Now we detail the proposed object pursuit framework, which unifies object re-identification, succinctness of the representation space, and forgetting prevention, for continuously learning object representations.

# 3.2 OBJECT PURSUIT

Given the definition of object-centric representations in Eq. 1, our goal is to construct a low-dimensional manifold to embed objects in the input space  $\zeta$  of the weight generation hypernetwork  $\psi$ . We conjecture that the low-dimensional manifold can be spanned by a set of base object representations. More explicitly, we instantiate two lists  $\mathbf{z}$  and  $\pmb{\mu}$ , which store the representations of the base objects and the embeddings of the learned objects, respectively. We denote  $\mathbf{z}^{t - 1} = \{z_i\}_{i = 1}^m$  and  $\pmb{\mu}^{t - 1} = \{\mu_i\}_{i = 1}^n (n\geq m)$  as the constructed lists after encountering a  $(t - 1)$ -th object. Note that  $\mu_{i}$  has the same dimension as the number of base object representations. Similarly, we denote  $\psi^{t - 1}$  as the corresponding hypernetwork parameters.

As discussed, when the  $t$ -th object  $o_t$  is discovered, a dataset  $\mathcal{D}^t = \{\left(x_j,y_j^t\right)\}$  can be easily sampled from the marginal distribution  $\mathrm{p}(\mathrm{x},\mathrm{y}^t)$  through interactions. However, such object might already be seen previously. Thus, it is necessary to apply re-identification to avoid repetitively learning the same object. According to the definition in Eq. 1, object  $o_t$  will be claimed as a seen or learned object if the following condition is true:

$$
\max  _ {i \leq | \boldsymbol {\mu} ^ {t - 1} |} \mathbb {E} _ {\left(\mathrm {x} _ {j}, \mathrm {y} _ {j} ^ {t}\right) \in \mathcal {D} ^ {t}} \Delta \left(\phi \left(\psi^ {t - 1} \left(\mathrm {z} _ {i}\right), \mathrm {x} _ {j}\right), \mathrm {y} _ {j} ^ {t}\right) \geq \tau . \tag {2}
$$

In this case, object  $o_t$  will be assigned the identity  $i^*$  that achieves the maximum value. Otherwise, if Eq. 2 is not valid,  $o_t$  is considered as an object that has not been learned.

Learning base object representations. An object  $o_t$  that can not be identified with the list of learned objects  $\pmb{\mu}^{t-1}$  can potentially serve as a base object whose representation should be added to the list of base representations  $\mathbf{z}$ . To ensure that object  $o_t$  qualifies as a base object, we propose the following test which checks whether  $o_t$  can be embedded in the current manifold spanned by  $\mathbf{z}^{t-1}$ :

$$
\mu^ {*} = \underset {\mu \in \mathbb {R} ^ {| \mathbf {z} ^ {t - 1} |}} {\arg \max } \mathbb {E} _ {(\mathrm {x} _ {j}, \mathrm {y} _ {j} ^ {t}) \in \mathcal {D} ^ {t}} \Delta (\phi (\psi^ {t - 1} (\mu^ {T} \mathbf {z} ^ {t - 1}), \mathrm {x} _ {j}), \mathrm {y} _ {j} ^ {t}) + \alpha \| \mu \| _ {1}, \tag {3}
$$

where  $\mu^{*}$  is the optimal embedding for object  $o_t$  regarding  $\mathbf{z}^{t-1}$  under the  $\ell_1$  regularizer to encourage sparsity. If the first term of Eq. 3 passes the threshold  $\tau$  with the representation  $\mu^{*T}\mathbf{z}^{t-1}$ , then we consider  $o_t$  as an object that should not be added to the list of bases since it can already be represented by the existing base objects.

Next, if  $o_t$  does not fall on the manifold spanned by  $\mathbf{z}^{t-1}$ , a joint learning of the representation of  $o_t$  and the hypernetwork  $\psi$  shall be performed so that a new base object representation can be added to the list. However, since updating the hypernetwork could result in catastrophic forgetting of the

![](images/f04c60c6672cf53e430baae2159ade08ab2d8f39663df5700f3b4e5af9213e26.jpg)  
Figure 2: Data collected in iThor. Target objects are highlighted by their instance masks.

previously learned object representations, it is also necessary to constrain the learning process, and the training loss is:

$$
\begin{array}{l} z ^ {*}, \psi^ {*} = \underset {z, \psi} {\arg \max } \mathbb {E} _ {(x _ {j}, y _ {j} ^ {t}) \in \mathcal {D} ^ {t}} \Delta (\phi (\psi (z), x _ {j}), y _ {j} ^ {t}) + \alpha \| z \| _ {1} \\ + \beta \sum_ {i \leq | \boldsymbol {\mu} ^ {t - 1} |} \| \psi \left(\mu_ {i} ^ {T} \mathbf {z} ^ {t - 1}\right) - \psi^ {t - 1} \left(\mu_ {i} ^ {T} \mathbf {z} ^ {t - 1}\right) \| _ {1}, \tag {4} \\ \end{array}
$$

where the first two terms help to find a good representation for object  $o_t$  under the sparsity constraint, and the third term enforces that the updated weight generation hypernetwork maintains the previously learned object representations. The value of the negative scalar coefficients  $\alpha, \beta$  will be detailed in the experiments.

Backward redundancy removal. The last but not the least component of the proposed object pursuit framework is to have a backward redundancy check. Since the weight generation hypernetwork is updated to  $\psi^t = \psi^*$  with Eq. 4, there may now exist an embedding  $\mu^{*}$  (computed using Eq. 3) that re-certifies object  $o_t$  as an object falls on the manifold spanned by  $\mathbf{z}^{t-1}$  under  $\psi^t$ . If this is true, we set  $\mathbf{z}^t = \mathbf{z}^{t-1}$ , otherwise,  $\mathbf{z}^*$  is added to the list of base object representations since object  $o_t$  is now confirmed as a base object. In some rare cases, object  $o_t$  might be hard to learn, e.g.,  $\mathbf{z}^*$  may not satisfy the criterion described in Eq. 1 under the current hypernetwork  $\psi^t$ . In this case, we simply toss away this object so that it can be better learned in the future as the pursuit process evolves. The proposed object pursuit framework is also summarized in Algorithm 1.

# 4 EXPERIMENTS

We target the learning scenario where a scene consists of multiple objects, each of them can be discovered and manipulated through interactions. The objects are learned one by one in a continuous manner but with unknown orders. There are two main aspects of the whole pipeline, i.e., data collection by sampling the marginals of individual objects and construction of the object-centric representations with Object Pursuit. We focus on continuous object-centric representation learning, and thus orient our study on the behavior and characteristics of the proposed object pursuit algorithm. We also perform experiments on one-shot and few-shot learning, and show the potential of the learned object-centric representations in effectively reducing supervisions for object detection. Next, we brief our data collection process.

# 4.1 SETUP

Data collection. To learn diverse objects from variant positions and viewing angles, we collect synthetic data within the iThor environment ((Kolve et al., 2017)), which provides a set of interactive objects and scenes, as well as accurate modeling of the physics. We collect data of 138 different objects to generate their images and masks. The 138 objects are divided into 52 pretraining objects, 60 train objects for the pursuit process, and 25 test unseen objects. To focus on the representation learning part, we abstract the interaction policy, and the data collection procedure of a single object can be summarized as follows: 1) Randomly set the positions of all the objects in the scene. 2) Calculate all available camera positions and viewing angles from which the target object (to be learned) is visible so that the sampling is effective. The camera position, yaw angle, and pitch angle change within the range of 0.4 (grid size),  $4^{\circ}$  and  $30^{\circ}$  respectively. 3) For each camera position and viewing angle, we collect a  $572 \times 572$  RGB image and a binary mask of the target object. 4) Repeat (1-3) for all objects in the stream. Please check Fig. 6 for the sampled data.

Table 1: Re-identification: recall and precision on seen objects.  

<table><tr><td>τ</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td></tr><tr><td>recall</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>precision</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr></table>

Table 2: Re-identification: rate of unseen objects been identified along the course of the pursuit process.  

<table><tr><td rowspan="2">τ</td><td colspan="7">No. of trained objects</td></tr><tr><td>8</td><td>16</td><td>24</td><td>32</td><td>40</td><td>48</td><td>56</td></tr><tr><td>0.5</td><td>0.40</td><td>0.52</td><td>0.56</td><td>0.60</td><td>0.64</td><td>0.64</td><td>0.72</td></tr><tr><td>0.6</td><td>0.08</td><td>0.20</td><td>0.28</td><td>0.44</td><td>0.56</td><td>0.60</td><td>0.60</td></tr><tr><td>0.7</td><td>0.16</td><td>0.28</td><td>0.32</td><td>0.40</td><td>0.40</td><td>0.48</td><td>0.44</td></tr><tr><td>0.8</td><td>0.00</td><td>0.08</td><td>0.16</td><td>0.24</td><td>0.28</td><td>0.28</td><td>0.28</td></tr></table>

Network implementation In our experiment, we use Deeplab v3+ (Chen et al., 2018a) as the segmentation network  $\phi$ , which consists of 3 parts: a backbone to encode features at different levels, an aspp module, and a decoder to predict the segmentation probability per pixel. We use resnet18 as the backbone (encoder), whose weights are fixed both in the pretraining and the pursuit process. The weights of the aspp module and the decoder are generated by the convolutional hypernetwork  $\psi$ . For each convolution layer in the aspp module and the decoder,  $\psi$  takes object representation  $\mathbf{z}$  as input, and predict weights of the convolution kernel using an upsampling convolution block. The input representation  $\mathbf{z}$  first expanded to a 1024-dim vector by a linear mapping and resized to a  $1 \times 1 \times 32 \times 32$  tensor. After going through several upsampling blocks, each of which consists of an upsampling followed by a convolution and a leaky Relu, the  $1 \times 1 \times 32 \times 32$  tensor turns into the output kernel weight. For other network weights like 'running_mean' and 'running_var' in batch normalization, the hypernetwork linearly maps representation  $\mathbf{z}$  to generate them.

Training details. For the similarity measure  $\Delta$ , we use the dice score proposed in (Milletari et al.). In addition to  $\Delta$ , we find that it will be beneficial to add an extra binary cross-entropy term when learning base object representations using Eq. 4. Note, to deal with imbalanced foreground and background sizes, we also put a weighting on the entropy terms that correspond to the object so the learning can be more efficient. The sparsity constraint  $\alpha$  is set to 0.2, 0.1 for Eq. 3 and Eq. 4 respectively, and  $\beta = 0.04$  for all our experiments. To improve the convergence, we also warm up the hypernetwork using the pretraining objects. During pretraining, each mini-batch contains training data from one object, and we randomly choose which object to use in the next batch. In backpropagation, we update the hypernetwork  $\psi$  and representation  $z$  for each object. When the pretraining is done, we perform a redundancy check to get rid of the objects that can be represented by others. For simplicity, this check is performed in sequential order, and we are left with a set of base object representations to carry out the following studies.

# 4.2 ON THE REPRESENTATION QUALITY MEASURE

The learning dynamics and the output of Algorithm. 1, i.e., the lists of base object representations  $\mathbf{z}$  and the learned objects  $\mu$ , together with the weight generation hypernetwork  $\psi$ , are primarily affected by the representation quality measure  $\tau$  introduced in Eq. 1. For example,  $\tau$  controls whether an object will be claimed as seen, and it also determines whether or not an object falls on the manifold spanned by the current base object representations. We study each of them in the following.

# 4.2.1 RE-IDENTIFICATION

As described in Eq. 2, when an object is discovered, it will be first checked against the learned objects and re-identified if the maximum expected similarity passes  $\tau$ . To examine how the quality measure  $\tau$  influences the re-identification process, we run multiple object pursuit processes with different  $\tau$ 's. All runs are performed with the same training object order so that the only variant is the value of  $\tau$ . For evaluation, we preserve a separate set of 25 objects (unseen test objects) that never appear during training. Note, among these unseen test objects, there are also objects that are similar to the training ones. And we use 27 objects (seen test objects) from the warp-up joint training described above to check the re-identification accuracy.

First, we check how  $\tau$  affects the re-identification for seen objects. As reported in Tab. 1, if an object is learned and added to the object list  $\mu$ , it will be claimed as seen by Eq. 2, and the re-identification accuracy is always one. This is true for  $\tau$  varying between 0.5 and 0.8, which demonstrates the robustness of the re-identification process against  $\tau$  for objects learned.

Second, we check the behavior of the re-identification component for unseen objects under different  $\tau$ 's along the pursuit process. In Tab. 2, we can observe that as more and more objects are learned during the pursuit, the unseen objects that are claimed as seen from the re-identification process also increase. This observation is consistent across different  $\tau$ 's. Furthermore, the rate of unseen objects identified as seen converges at the end of the pursuit process, but at different levels for different  $\tau$ 's, i.e., the converged rate is lower for larger  $\tau$ . It may seem incorrect if an unseen object is claimed as seen by the re-identification component. However, if we examine the unseen objects (see Fig. 3), we can see that it is quite natural for these unseen objects to be labeled as seen, because they are similar to one or multiple objects in the object list  $\mu$ . This is indeed a desired characteristic since representing

or learning an object that is similar to existing ones may not be informative. Moreover, one can adjust  $\tau$  to tune the similarity level. For example, if one insists on learning an object similar to previously seen objects, increasing the value of  $\tau$  should work as evidenced by the converged rates for  $\tau$ 's in Tab. 2.

In a nutshell, the representation quality measure  $\tau$  has little effect on the re-identification recall and accuracy for learned objects. Yet, it controls the granularity of the learned representations by modulating the rate of unseen objects that would be identified as learned ones.

# 4.2.2 SUCCINCTNESS AND EXPRESSIVENESS

We want to study how the representation quality measure  $\tau$  affects the overall learning dynamics in terms of the succinctness and expressiveness of the learned base object representations. By checking Eq. 2, Eq 3 and also the previous experiment, we conjecture that if  $\tau$  is small, objects similar to the learned ones will be more easily identified as seen and certified as on the manifold. If so, the number of objects that will be used for learning the base representations may also be small, thus increasing the succinctness of the final representations. Conversely, when  $\tau$  increases, we would expect that more objects will contribute to the base representations, thus increasing the expressiveness. We like to check if the observations align with our conjecture and how such behav-

Table 3: Pursuit dynamics by varying  $\tau$ . Please see the enclosing description for the meaning of the metrics and corresponding analysis.  

<table><tr><td>τ</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td></tr><tr><td>|z|/N</td><td>0.34</td><td>0.42</td><td>0.42</td><td>0.40</td></tr><tr><td>|μ|/N</td><td>0.46</td><td>0.58</td><td>0.46</td><td>0.40</td></tr><tr><td>Re</td><td>0.19</td><td>0.21</td><td>0.00</td><td>0.00</td></tr><tr><td>Rf</td><td>0.08</td><td>0.18</td><td>0.42</td><td>0.54</td></tr><tr><td>Aμ</td><td>0.75</td><td>0.77</td><td>0.83</td><td>0.86</td></tr></table>

ior affects the quality of the learned bases. To facilitate the analysis, we propose to check the following quantities: 1)  $|\mathbf{z}| / N$ , which is the portion of objects that contribute to base representations; 2)  $|\pmb{\mu}| / N$ , which is the portion of learnable objects that are added to the object list  $\pmb{\mu}$ ; 3)  $\mathcal{R}_e$ , rate of objects that are confirmed unseen but can be expressed by the base object representations; 4)  $\mathcal{R}_f$ , rate of objects to be learned as base representations, which are later considered as redundant or unqualified; 5)  $\mathcal{A}_{\mu}$ , segmentation accuracy on learned objects.

We report the above metrics across different  $\tau$ 's in Tab. 3. As expected, a larger  $\tau$  generally encourages more objects to be learned as base representations. For example, the number of base objects learned is much larger when  $\tau$  increases from 0.5 to 0.6 (first row). This is also evidenced by the third row of Tab. 3, which shows that the probability of an unseen object to be expressed by base rep

![](images/c41803649ce403bc010c28c3d0e0e2b90317730c100dbfef3f78a6842e7d7d51.jpg)

![](images/516d821f868136b7ca50fb7adec5b4a1ac8c48000d8f838d5b117dc2831dddd9.jpg)  
unseen objects

![](images/25a1e9075c75bc36b62e97e09ca2f1ca98dc624ac5cdc915e5d01becf1e40802.jpg)

![](images/2822f06e690bbc7c7b6091f34ee51673062a61b6e0f70404cbd44a5c1b4eb230.jpg)

![](images/9db33f583eaa209ac4f47bdaa9b316c2f38f7423ef44b53b8ebc79487e15133c.jpg)

![](images/5d5e2100961b07688d1c37c6ba7db8b3ca0e55a69c81b9755f936d0b3a21e499.jpg)

![](images/82c5d90dd5593dd3c63fe484d1d72a1b764915b7a2f0463575e362d15b9b9c2e.jpg)  
candidate seen objects

![](images/aad141c6616dc93366f47bcd6975582e54da6505330230f35cf0c4f6b09006fd.jpg)

![](images/61111b6e4e19631e161541bfd335a3c5fcb4faf0d3d7bd3233b03a41be114c7d.jpg)

![](images/4bdd2b6d14f7ea1c5245e67261af7c9f041070030f2c6954a70028b11fc02875.jpg)

![](images/4dd08fe9cd7faf6cd2373180fbd81286ca869da9abce64cf853840535f123196.jpg)

![](images/636e4a5e4d950055be373c56937fe8612d429f2a2a7a236e739cb0bc49b2c0f4.jpg)

![](images/b6f9eee14f35b2acb1c97ce7ee6d4fd5fc053efe3aed5a76a679b10beed78016.jpg)

![](images/ad81a156346c63e736898388f366044e88d97ff4be7a8a9451cf3634821f9740.jpg)

![](images/3b783de4f6e2b286a681dc076d339d77521aae4c4f5e1b1e1fe249afc03d0686.jpg)

![](images/b386f6c0b2b855270a50cf57ec250b1516c87f491c8d78ffa295bf87681fc2ba.jpg)  
Figure 3: Unseen objects re-identified as learned. 1st row: unseen objects, 2nd to 4th row: similar objects from the learned object list. Bounding boxes highlight the objects with embedded text indicating the instance identity.

![](images/968b4fa9148ff2a494b31a890a3f965d60715828043b3beb0b97cf8e14394a60.jpg)

![](images/715f7c93ca0232b585be3acf55e1e18c16764732644949fd81fd57d3491c6b46.jpg)

![](images/84ce3c0a666ff94a7a70adc23051bb627b5440d55025e8519b6f6545b530f690.jpg)

![](images/008d8a2ea87ab7e709693ed862e5abe32b43d6fc359bda1bb8616f6a268c0e57.jpg)

resentations will decrease as  $\tau$  increases, creating more attempts to learn objects as bases. However, the number of learnable objects, i.e., a base object or an object that falls on the manifold spanned by the bases, attains the maximum at a medium value  $\tau = 0.6$  (second row). The underlying reason is two-fold: First, a very small  $\tau$  means that many objects will be identified as seen and thus discarded to save computation; Second, a very large  $\tau$  can make the qualification of an object representation extremely difficult such that it will be put aside for future learning. The latter is also supported by the metric  $\mathcal{R}_f$  shown in the fourth row, i.e., the probability that an object will be considered redundant or unqualified after learning as a base object will increase as  $\tau$  becomes large. Lastly, when checking the quality of the base representations in expressing a common set of learned objects, we can see that the segmentation accuracy correlates with  $\tau$  in a positive manner (fifth row).

In general,  $\tau$  directly impacts the quality of the base representations for learned objects, but its effect on the number of base representations produced by the pursuit procedure is not monotone. Within a moderate range, we can increase  $\tau$  to encourage learning more base representations, however, we may not want  $\tau$  to be too large that only a few objects are qualified as base representations.

# 4.2.3 LABEL EFFICIENCY

Besides the training dynamics, we evaluate the usefulness of the learned object base representations in terms of how it facilitates learning the representation of a new object with only a few annotations. For comparison, we also perform learning of the object representations over the entire representation space. Training is similar to Eq. 3. The quality of the learned object representations is measured by their segmentation accuracy on test data.

Table 4: N-shot learning the representation of a new object. Training is performed by searching the optimal representation either on the manifold spanned by the base objects, or over the entire representation space. Segmentation accuracy on the test set is reported for bases and hypernetworks learned at different  $\tau^*$ s.  

<table><tr><td rowspan="2">n</td><td colspan="4">over base object representations</td><td colspan="4">full representation space</td></tr><tr><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td></tr><tr><td>1</td><td>0.377</td><td>0.416</td><td>0.454</td><td>0.446</td><td>0.225</td><td>0.264</td><td>0.288</td><td>0.289</td></tr><tr><td>5</td><td>0.595</td><td>0.606</td><td>0.634</td><td>0.614</td><td>0.461</td><td>0.475</td><td>0.468</td><td>0.453</td></tr><tr><td>10</td><td>0.622</td><td>0.647</td><td>0.677</td><td>0.649</td><td>0.542</td><td>0.526</td><td>0.524</td><td>0.520</td></tr><tr><td>2000</td><td>0.697</td><td>0.731</td><td>0.740</td><td>0.731</td><td>0.669</td><td>0.698</td><td>0.702</td><td>0.718</td></tr></table>

As reported in Tab. 4, the quality

of the few-shot learned representations increases as  $\tau$  gets large, which aligns with our observation in the previous section that the expressiveness of the learned object base representations highly correlates with  $\tau$ . However, note that there is a slight drop in performance when  $\tau$  increases from 0.7 to 0.8 (fourth and fifth column). The reason is that as  $\tau$  gets really large, it also becomes much easier to omit objects that can not pass the quality test. As a result, the hypernetwork, which translates the representation to network weights, also gets less trained. Thus, when tested on new objects, the performance may not match that of the trained objects for the same set of base representations, suggesting again that a moderate  $\tau$  is needed to balance between the succinctness and generalization of the learned base representations.

The above observation does not hold for the representations learned over the full space. Moreover, when comparing the performance within the low data regime, we can see that those object representations found on the manifold outperform those found in the entire space by a large margin. For example, the new object representations found with the learned bases under  $\tau = 0.7$  outperform their counterparts by  $57.6\%$ ,  $35.5\%$ ,  $29.2\%$  for the 1-shot, 5-shot, and 10-shot settings, respectively. This demonstrates the potential of using the learned base representations to help reduce the supervision needed to learn a new object. Also, it confirms that the learned base representations are meaningful since the manifold spanned by them provides a good regularity for learning unseen objects.

# 4.3 ORDER OF TRAINING OBJECTS

The proposed object pursuit algorithm learns object representations in a stream, so we also check how the learning dynamics vary when the order of training objects changes. We fix  $\tau$  to 0.6 and run ten pursuit processes with random training object order. We reported the mean and standard deviation of the metrics proposed in Tab. 3. As observed, the pursuit process is robust to the training object order.

Table 5: Pursuit dynamics under random training object order.  

<table><tr><td></td><td>|z|/N</td><td>|μ|/N</td><td>Re</td><td>Rf</td><td>Aμ</td></tr><tr><td>mean</td><td>0.43</td><td>0.50</td><td>0.10</td><td>0.15</td><td>0.76</td></tr><tr><td>std-dev</td><td>0.02</td><td>0.03</td><td>0.04</td><td>0.04</td><td>0.01</td></tr></table>

# 4.4 FORGETTING PREVENTION

In this section, we want to check if the forgetting prevention term in Eq. 4 is effective and how it affects the pursuit dynamics. We run pursuit processes with different values of the coefficient  $\beta$ , where the quality measure  $\tau$  and training object order is fixed. Segmentation accuracy  $\mathcal{A}_{\mu}$  and forgetting rate  $\gamma_{f}$  (i.e., how many objects falls under the quality measure after the process is finished) in Tab. 6 demonstrate the effectiveness of the forgetting prevention term: when  $\beta$  decreases, the segmentation accuracy drops, and the forgetting rate increases; when  $\beta$  vanishes, the forgetting rate reaches  $97\%$ , which means that the hypernetwork almost forgets all the object representations it previously learned. We can also observe that both  $|\mathbf{z}| / N$  and  $|\pmb{\mu}| / N$  increase

Table 6: Pursuit dynamics under different forgetting prevention constraints.  

<table><tr><td>β</td><td>0.0</td><td>0.02</td><td>0.04</td><td>0.1</td></tr><tr><td>|z|/N</td><td>0.61</td><td>0.46</td><td>0.42</td><td>0.39</td></tr><tr><td>|μ|/N</td><td>0.88</td><td>0.61</td><td>0.58</td><td>0.54</td></tr><tr><td>Re</td><td>0.13</td><td>0.14</td><td>0.21</td><td>0.21</td></tr><tr><td>Rf</td><td>0.19</td><td>0.14</td><td>0.18</td><td>0.21</td></tr><tr><td>Aμ</td><td>0.02</td><td>0.67</td><td>0.71</td><td>0.72</td></tr><tr><td>γf</td><td>0.97</td><td>0.04</td><td>0.02</td><td>0.02</td></tr></table>

when  $\beta$  decreases. This is due to the fact that when the hypernetwork forgets what are learned, any incoming object will be unlikely to be considered as seen, nor to be expressed by current bases. So the hypernetwork tends to learn them as new base objects, which causes  $|\mathbf{z}| / N$  to increase. This is also evidenced by the drop in  $\mathcal{R}_e$ , which is rate of new objects that are certified as on the object manifold. Furthermore, without the constraint of the forgetting prevention term, it is more likely to get higher accuracy in learning a new object, which decreases the number of unqualified objects. Since the number of redundant objects and unqualified objects both drop when  $\beta$  decreases,  $|\pmb{\mu}| / N$  increases. Thus, in order to reduce computational cost and enforce learning meaningful representations, one would like to apply a relatively large  $\beta$  during the pursuit process.

We can also observe that as  $\beta$  changes from 0.02 to 0.1,  $\mathcal{R}_f$  increases monotonically, this is because the forgetting prevention constraint affects the quality of the learned representations, since less freedom is available when  $\beta$  is extremely large. Consequently, fewer objects will be qualified with a good representation measured by  $\tau$ . On the other hand,  $\mathcal{R}_f$  is also high when  $\text{beta}$  is set to 0. The reason is that when learning a new object without the constraint of the forgetting prevention term, the hypernetwork tends to overfit, thus making it easier for this new object to be considered as redundant, i.e., it can be expressed by the existing base representations, even though the learned representation will be forgotten by the network right after the current learning episode.

# 4.5 MORE RESULTS

In the appendix, we also provide ablation studies on how the sparsity constraints in Eq. 3 and Eq. 4 affect the learning dynamics and the quality of the learned representations. By examining the most relevant base objects for a novel object that can be expressed by the base representations (Fig. ??), we can qualitatively see that high-level concepts are learned within the representation space as objects that share similar geometry or appearance will be more correlated than others. For curiosity, we also test the usefulness of the base object representations on real-world video objects. As demonstrated in Fig. 5, the learned base representations can capture well the representations of real-world objects with a single learning example even if they are trained on synthetic data.

# 5 CONCLUSION

We demonstrate that the proposed object pursuit framework can be used for continuously learning object-centric representations from data collected by manipulating a single object. The key designs, e.g., object re-identification, forgetting prevention, and redundancy check, all contribute to the quality of the learned base object representations. We also show the potential of using the learned object-centric representations for tasks at a low-annotation regime. Especially, the learned object manifold provides a meaningful and effective prior on objects, which can facilitate downstream tasks that require object-level reasoning. As inspired by an initial attempt on the real-world data (Fig. 5), we would also like to check the proposed object pursuit algorithm in the real world. For example, we can train an autonomous agent to collect data from the natural 3D environment with a more efficient interaction policy, and then test the learned object representations on real-world compositional visual reasoning tasks. These are in our future research agenda.

Ethics Statement. The proposed object pursuit framework aims at learning object representations for object-centric visual reasoning tasks. Currently, the experiments are performed in simulation, which is publicly available and comes with a proper license. However, how to use the learned representations could be an issue, and we will explicitly state the guideline on how to use our code and trained models ethically. In our future research, when data collection in the real world is involved, we will consult the university ethics review committee for advice. However, in the current form, we do not observe any significant concerns.

Reproducibility Statement. Our code, training data, and learned models will be made publicly available after the acceptance of our paper. We will add detailed comments in the code so that the implementation can be easily understood. For a preview of the implementation, please refer to the attached code in the supplementary materials.

# REFERENCES

Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection for online continual learning. In Advances in neural information processing systems, pp. 11816-11825, 2019.  
Yogesh Balaji, Mehrdad Farajtabar, Dong Yin, Alex Mott, and Ang Li. The effectiveness of memory replay in large scale continual learning. arXiv preprint arXiv:2010.02418, 2020.  
Daniel Bear, Chaofei Fan, Damian Mrowca, Yunzhu Li, Seth Alter, Aran Nayebi, Jeremy Schwartz, Li F Fei-Fei, Jiajun Wu, Josh Tenenbaum, et al. Learning physical graph representations from visual scenes. Advances in Neural Information Processing Systems, 33, 2020.  
Luca Bertinetto, João F Henriques, Jack Valmadre, Philip Torr, and Andrea Vedaldi. Learning feedforward one-shot learners. In Advances in neural information processing systems, pp. 523-531, 2016.  
Christopher P Burgess, Loic Matthey, Nicholas Watters, Rishabh Kabra, Irina Higgins, Matt Botvinick, and Alexander Lerchner. Monet: Unsupervised scene decomposition and representation. arXiv preprint arXiv:1901.11390, 2019.  
Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. In Advances in neural information processing systems, 2020.  
Hyuntak Cha, Jaeho Lee, and Jinwoo Shin. Co2l: Contrastive continual learning. arXiv preprint arXiv:2106.14413, 2021.  
Arslan Chaudhry, Albert Gordo, Puneet Kumar Dokania, Philip Torr, and David Lopez-Paz. Using hindsight to anchor past knowledge in continual learning. arXiv preprint arXiv:2002.08165, 2020.  
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pp. 801-818, 2018a.  
Ricky TQ Chen, Xuechen Li, Roger Grosse, and David Duvenaud. Isolating sources of disentangle-ment in variational autoencoders. arXiv preprint arXiv:1802.04942, 2018b.  
Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704, 2016.  
Matthias Delange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Greg Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.  
Timothy J Draelos, Nadine E Miner, Christopher C Lamb, Jonathan A Cox, Craig M Vineyard, Kristofor D Carlson, William M Severa, Conrad D James, and James B Aimone. Neurogenesis deep learning: Extending deep networks to accommodate new classes. In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 526-533. IEEE, 2017.

Martin Engelcke, Oiwi Parker Jones, and Ingmar Posner. Reconstruction bottlenecks in object-centric generative models. arXiv preprint arXiv:2007.06245, 2020a.  
Martin Engelcke, Adam R. Kosiorek, Oiwi Parker Jones, and Ingmar Posner. Genesis: Generative scene inference and sampling with object-centric latent representations. In International Conference on Learning Representations, 2020b. URL https://openreview.net/forum?id=BkxfaTVFwH.  
SM Eslami, Nicolas Heess, Theophane Weber, Yuval Tassa, David Szepesvari, Geoffrey E Hinton, et al. Attend, infer, repeat: Fast scene understanding with generative models. Advances in Neural Information Processing Systems, 29:3225-3233, 2016.  
Chelsea Finn, Aravind Rajeswaran, Sham Kakade, and Sergey Levine. Online meta-learning. In International Conference on Machine Learning, pp. 1920-1930. PMLR, 2019.  
Tomer Galanti and Lior Wolf. On the modularity of hypernetworks. arXiv preprint arXiv:2002.10006, 2020.  
Klaus Greff, Sjoerd Van Steenkiste, and Jurgen Schmidhuber. Neural expectation maximization. arXiv preprint arXiv:1708.03498, 2017.  
Klaus Greff, Raphaël Lopez Kaufman, Rishabh Kabra, Nick Watters, Christopher Burgess, Daniel Zoran, Loic Matthey, Matthew Botvinick, and Alexander Lerchner. Multi-object representation learning with iterative variational inference. In International Conference on Machine Learning, pp. 2424-2433. PMLR, 2019.  
David Ha, Andrew Dai, and Quoc V Le. Hypernetworks. arXiv preprint arXiv:1609.09106, 2016.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. 2016.  
Jindong Jiang, Sepehr Janghorbani, Gerard De Melo, and Sungjin Ahn. Scalar: Generative world models with scalable object representations. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SJxrKgStDH.  
Rishabh Kabra, Daniel Zoran, Goker Erdogan, Loic Matthey, Antonia Creswell, Matthew Botvinick, Alexander Lerchner, and Christopher P Burgess. Simone: View-invariant, temporally-abstracted object representations via unsupervised video decomposition. arXiv preprint arXiv:2106.03849, 2021.  
Nitin Kamra, Umang Gupta, and Yan Liu. Deep generative dual memory network for continual learning. arXiv preprint arXiv:1710.10368, 2017.  
Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4401-4410, 2019.  
Ronald Kemker and Christopher Kanan. Fearnet: Brain-inspired model for incremental learning. arXiv preprint arXiv:1711.10563, 2017.  
Hyunjik Kim and Andriy Mnih. Disentangling by factorising. In International Conference on Machine Learning, pp. 2649-2658. PMLR, 2018.  
David A Klindt, Lukas Schott, Yash Sharma, Ivan Ustyuzhaninov, Wieland Brendel, Matthias Bethge, and Dylan Paiton. Towards nonlinear disentanglement in natural data with temporal sparse coding. In International Conference on Learning Representations, 2021.  
Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv, 2017.  
Adam R Kosiorek, Hyunjik Kim, Ingmar Posner, and Yee Whye Teh. Sequential attend, infer, repeat: Generative modelling of moving objects. arXiv preprint arXiv:1806.01794, 2018.

Adam R Kosiorek, Sara Sabour, Yee Whye Teh, and Geoffrey E Hinton. Stacked capsule autoencoders. arXiv preprint arXiv:1906.06818, 2019.  
David Krueger, Chin-Wei Huang, Riashat Islam, Ryan Turner, Alexandre Lacoste, and Aaron Courville. Bayesian hypernetworks. arXiv preprint arXiv:1710.04759, 2017.  
Zhiyuan Li, Jaideep Vitthal Murkute, Prashnna Kumar Gyawali, and Linwei Wang. Progressive learning and disentanglement of hierarchical representations. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SJxpsxrYPS.  
Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935-2947, 2017.  
Francesco Locatello, Michael Tschannen, Stefan Bauer, Gunnar Ratsch, Bernhard Schölkopf, and Olivier Bachem. Disentangling factors of variations using few labels. In International Conference on Learning Representations, 2020a. URL https://openreview.net/forum?id=SygagpEKwB.  
Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. arXiv preprint arXiv:2006.15055, 2020b.  
David Lopez-Paz and Marc'Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30:6467-6476, 2017.  
Jonathan Lorraine and David Duvenaud. Stochastic hyperparameter optimization through hypernetworks. arXiv preprint arXiv:1802.09419, 2018.  
Benno Lüders, Mikkel Schlager, and Sebastian Risi. Continual learning through evolvable neural tuning machines. In Nips 2016 workshop on continual learning and deep networks (cldl 2016), 2016.  
Tianyu Ma, Adrian V Dalca, and Mert R Sabuncu. Hyper-convolution networks for biomedical image segmentation. arXiv preprint arXiv:2105.10559, 2021.  
F Milletari, N Navab, SA Ahmadi, and V-net. Fully convolutional neural networks for volumetric medical image segmentation. In Proceedings of the 2016 Fourth International Conference on 3D Vision (3DV), pp. 565-571.  
Tsendsuren Munkhdalai and Hong Yu. Meta networks. In International Conference on Machine Learning, pp. 2554-2563. PMLR, 2017.  
Michael Niemeyer and Andreas Geiger. Giraffe: Representing scenes as compositional generative neural feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11453-11464, 2021.  
Yuval Nirkin, Lior Wolf, and Tal Hassner. Hyperseg: Patch-wise hypernetwork for real-time semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4061-4070, 2021.  
German I Parisi, Jun Tani, Cornelius Weber, and Stefan Wermter. Lifelong learning of spatiotemporal representations with dual-memory recurrent self-organization. Frontiers in neurorobotics, 12: 78, 2018.  
German I Parisi, Ronald Kemker, Jose L Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. Neural Networks, 113:54-71, 2019.  
Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, and Davide Maltoni. Latent replay for real-time continual learning. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 10203-10209. IEEE, 2020.  
Ori Press, Tomer Galanti, Sagie Benaim, and Lior Wolf. Emerging disentanglement in auto-encoder based unsupervised image content transfer. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=BylE1205Fm.

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001-2010, 2017.  
Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro. Learning to learn without forgetting by maximizing transfer and minimizing interference. arXiv preprint arXiv:1810.11910, 2018.  
Andrei A Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. arXiv preprint arXiv:1807.05960, 2018.  
Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep generative replay. arXiv preprint arXiv:1705.08690, 2017.  
Vincent Sitzmann, Eric R Chan, Richard Tucker, Noah Snavely, and Gordon Wetzstein. Metasdf: Meta-learning signed distance functions. arXiv preprint arXiv:2006.09662, 2020.  
Yi Tay, Zhe Zhao, Dara Bahri, Donald Metzler, and Da-Cheng Juan. Hypergrid transformers: Towards a single model for multiple tasks. In International Conference on Learning Representations, 2020.  
Johannes von Oswald, Christian Henning, João Sacramento, and Benjamin F Grewe. Continual learning with hypernetworks. arXiv preprint arXiv:1906.00695, 2019.  
Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Meta-learning to detect rare objects. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9925-9934, 2019.  
Jiajun Wu, Joshua B Tenenbaum, and Pushmeet Kohli. Neural scene de-rendering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 699-707, 2017.  
Yanchao Yang, Yutong Chen, and Stefano Soatto. Learning to manipulate individual objects in an image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6558-6567, 2020.  
Shunyu Yao, Tzu Ming Harry Hsu, Jun-Yan Zhu, Jiajun Wu, Antonio Torralba, William T Freeman, and Joshua B Tenenbaum. 3d-aware scene manipulation via inverse graphics. arXiv preprint arXiv:1808.09351, 2018.  
Sharon Zhou, Eric Zelikman, Fred Lu, Andrew Y. Ng, Gunnar E. Carlsson, and Stefano Ermon. Evaluating the disentanglement of deep generative models through manifold topology. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=djws0m4Ft_A.
