# The Policy-gradient Placement and Generative Routing Neural Networks for Chip Design

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Placement and routing are two critical yet time-consuming steps of chip design in modern VLSI systems. Distinct from traditional heuristic solvers, this paper on one hand proposes an RL-based model for mixed-size macro placement, which differs from existing learning-based placers that often consider the macro by coarse grid-based mask. While the standard cells are placed via gradient-based GPU acceleration. On the other hand, a one-shot conditional generative routing model, which is composed of a special-designed input-size-adapting generator and a bi-discriminator, is devised to perform one-shot routing to the pins within each net, and the order of nets to route is adaptively learned. Combining these techniques, we develop a flexible neural pipeline, which to our best knowledge, is the first joint placement and routing network without involving any traditional heuristic solver. Experimental results on chip design benchmarks showcase the effectiveness of our approach, with code that will be made publicly available.

# 1 Introduction

The scale of integrated circuits (ICs) has been enlarged dramatically, posing a challenge to the scalability of existing Electronic Design Automation (EDA) technologies. The increasing circuit density raises additional issues for VLSI placers and routers as the component size of modern VLSI design continues to drop and on-chip connectivity becomes increasingly sophisticated. Due to growing on-chip connectivity, concentrated needs and restricted resources, modern designs are prone to congestion issues and wirelength minimization, which has become a critical task at every stage of the chip design process. Accordingly, placement and routing that physically arranges the locations and the routes of nets become more crucial in modern VLSI systems.

Placement and global routing $^{1}$  are two critical and time-consuming steps in chip design. On the one hand, the components of a netlist, including macros and standard cells, are mapped to positions on the chip layout by placement, with standard cells being basic logic cells (e.g. logic gates) and macros as functional blocks (e.g. SRAMs). Moreover, the increasingly extensive use of intellectual property (IP) modules and pre-designed macro blocks makes mixed-size placement an indispensable part of physical design. The goal of placement is to optimize power, performance, and area (PPA) metrics meanwhile obeying the constraints e.g. placement density and routing congestion.

Global routing, on the other hand, creates routing channels inside a layout based on the placement assignment by connecting pins of positioned IC components to a net while adhering to technological limits. It is tightly coupled with the placement task, as an excellent placement solution can result in better chip area utilization, timing performance, and routability etc. The objective of routing is to minimize total wirelength without violating the limits of congestion and critical timing.

Differing from the traditional learning-free solvers for placement and routing, learning-based models are recently explored and applied to (macro) placement [2, 1] and routing [3], which are mostly based on reinforcement learning (RL). Like in vision, language and other domains, an end-to-end network is often welcomed for the possibility for global optimization of the whole system, while such pure neural networks for joint placement and routing remains unresolved in literature. In particular, there are two standing issues worth further study in this paper: i) the macros are of mixed-size with varying width/height [4], bringing difficulty to the discrete grid-based placement by (current reinforcement) learning; ii) the scale of grid size in routing dataset e.g. ISPD-07 [5] can be intractable for RL as currently its action has to be designed at the local grid point level [6] rather than generate the whole routing across grids in one shot. In this paper, we aim to develop a flexible (mixed-size friendly) approach. More ambitiously, we further aim to enable a pure neural pipeline with both learnable placement net and routing net. Specifically, the placement part is fulfilled by a policy gradient based RL method for macros by considering their sizes, followed with gradient-based optimization for placing stand cells. The routing part involves a conditional generative model to finish the routing across pins at net level. The highlights of this work are:

1) We propose a conditional generative routing network to perform routing, in one-shot for each net. In contrast, existing RL-based solvers [6] need to perform routing step by step at grid point level within each net, in a sequential and inefficient manner. Moreover, the order for nets to route is adaptively optimized by learning instead of a pre-fixed one as performed in existing routers.  
2) We propose an RL-based model for learning mixed-size macro placement, which differs from existing learning-based placers [1, 2, 7] that often consider the macro by coarse grid-based mask. As such, the placement results are more realistic and require less post-processing to resolve collision.  
3) Combining these techniques, we develop a neural network-based pipeline for placement and routing, which to our best knowledge, is the first pure neural networks for placement and routing<sup>2</sup>.  
4) Experimental results on benchmarks show the relatively cost-efficiency (compared with RL-based routing solver [6]) and competitive performance. Source code will be made publicly available.

# 2 Related Work

For the inter-discipline nature of this paper, we briefly introduce the necessary background and related work to properly position our work with different communities from EDA to machine learning. Due to page limit, classical methods for placement and routing are presented in Appendix A.1.

Learning-based Solvers for Placement. Machine learning has recently been introduced which may help reduce the heuristics. [7] devises a cyclic framework between the reinforcement learning (RL) and SA modules, in which the RL module alters the relative spatial sequence between circuit components, while SA searches the solution space based on RL initialization. Following the seminal work [2] of learning sequential decision-making for macro placement, the method called DeepPlace in [1] proposes a joint learning technique for the placement and subsequent routing via RL.

Learning-based Solvers for Routing. The recent work [8] presents an attention-based RL method for obtaining pin order within each net (rather than net order in this paper which is of much larger size), followed by a classical pattern router. A DQN agent [6] is developed to decide on the routing direction on a grid graph at each step. It makes up a simple 12-element vector to represent the state of the environment. However, the model is trained on synthesized  $8 \times 8$  and  $16 \times 16$  grids, with no more than 50 nets that consist of 2 or 3 pins. To make the learning more scalable, a two-page preliminary work [9] formulates the routing of a net as an image-to-image translation and uses a variational auto-encoder (VAE) [10] model to generate the solution. However, the model is merely capable of handling a net with no more than three pins on a  $64 \times 64$  grid. Compared with classic routing solvers, existing RL-based methods can be much more time-consuming, making the end-to-end learning of both placement and routing nets very hard. Moreover, compared with [8] learning the pin order inside a net, we try to learn the order for routing at the net level which is new in literature.

Generative Models for Placement and Routing. There are emerging works on introducing generative models for chip design. [11] proposes a generative adversarial network (GAN) [12] guided

well generation framework to mimic experts' behavior from high-quality manually-crafted layouts. [13] adopts a GAN for generating wells and guides the placement in analog circuit layout synthesis. [14] proposes a generative model for the placement optimization of analog integrated circuit basic blocks. ThermGAN [15] treats the thermal map estimation problem as an image-generation problem using the generative model. [16] uses GAN to predict the congestion heatmap to assist classical routers. The work [17] proposes a conditional GAN to solve the multi-terminal path-finding task. In [18] the generative models are adopted to synthesize diverse layout patterns. More broadly speaking, generative models have also been recently applied in combinatorial optimization, specifically via a latent space learning and search scheme as done in CVAE-Opt [19]. In this paper, we adopt cGAN to generate routes in one shot for each net by regarding the layout to be routed as an image. To our best knowledge, no generative model has been successfully devised and applied to the pin routing problem for each net, which in fact involves complicated and constrained routing.

# 3 Methodology

**Approach Overview.** Given a netlist as input, the goal is to place the macros and standard cells on the chip canvas, ideally with a minimized overlapping areas. Based on the placement results, routing is performed in general to minimize the total wirelength while not violating the constraints. For placement, we aim to flexibly and efficiently handle the mixed-size macros via an RL scheme, and meanwhile optimize the net order for routing. For routing, we propose a conditional generative model to obtain the routes in one-shot instead of performing sequential pin connection as done in previous learning-based [6] as well as classic routers [20, 21]. The stacking RL placement network and generative routing network can be learned via gradient back propagation in an end-to-end manner, which meanwhile completes the whole task in line with the state-of-the-art learning-based placement and routing solvers [1, 2]. Note our work is pure learning based in contrast to [1] that still involves unlearnable classic routers in the whole pipeline. The pure neural architecture of our model implies our model has the potential to enjoy higher capacity by using larger model for further improvement.

# 3.1 Reinforcement Learning for Mixed-size Placement

A natural idea for classical placers to address the mixed-size issue for placement is to adopt a hierarchical approach based on partitioning. However, it sacrifices the solution quality since each sub-problem is solved independently. Meanwhile, DeepPlace as a learning-based placer assumes that each macro only occupies one cell in the grid graph and ignores pre-placed macros, which leads to a severe overlap issue in the final placement result. This motivates us to extend the formulation of DeepPlace by considering the real size of macros as well as initial placement information. We still adopt [22] as the CNN backbone and GCN [23] as the GNN backbone which consists of three layers that contain 16, 32 and 16 feature channels in policy network which is updated by Proximal Policy Optimization (PPO) [24], in line with the effective setting adopted in [1]. The revised elements of the Markov Decision Processes (MDPs) for mixed-size placement are defined as:

- State  $s_t$ : the state representation still consists of global image  $I$  portrayed the layout and netlist graph  $H$  contains detailed position of placed macros, except that the initial state of  $I$  is no longer a zero matrix  $I_{n \times n} = \mathbf{0}$ . Instead, our model preprocesses the positions of fixed macros in the dataset and sets  $I_{xy}$  as 1 if  $(x, y)$  has already been occupied before placement.  
- Action  $a_{t}$ : the action of RL agent is to find the central of current macro, and position  $(x_{o},y_{o})$  is available if all points  $p$  in the region  $\mathbf{R}$  satisfy  $I_{p} = 0$ , where  $\mathbf{R} = \{(x,y)\mid |x - x_o|\leq \frac{h}{2},|y - y_o|\leq \frac{w}{2}\}$  and  $h,w$  denote the height and width of the current macro respectively.  
- Reward  $r_t$ : to further control the overlap in the final placement, the reward at the end of episode is a negative weighted sum of wirelength, routing congestion and overlapping area from the final solution:  $R_E = -L_{wl} - \lambda_1 \cdot L_{cg} - \lambda_2 \cdot L_{ol}$  as weighted by  $\lambda_1$  and  $\lambda_2$ .

# 3.2 Conditional Generative Learning for Pin Routing in Net

The Global Routing Grid Protocol. In global routing, the physical chip is usually divided into rectangular areas, as shown in the left part of Fig. 1. Each area is called a global routing cell (Gcell), which corresponds to a node  $v_{i} \in V$  in the grid graph  $G(V,E)$  on the right side. While each edge  $e_{ij} \in E$  represents the joint boundary between abutting Gcells  $v_{i}$  and  $v_{j}$ . For edge  $e$ , its capacity  $c_{e}$  is the allowed maximum number of wires that can cross  $e$ , and the usage  $u_{e}$  is the actual number of

wires that  $e$  has been assigned. The overflow  $o_e$  denotes the amount of wires beyond  $c_e$  defined as  $\max(0, u_e - c_e)$ . Each route conforms to the rectilinear Steiner tree structure [25].

Grid-based Conditional Generative Routing. The overall routing task is composed of numerous nets whose routing can be independent to each other. A single net consists of a series of pins placed in the nodes of the grid graph. Thus in this paper, we mainly consider the routing over multiple pins in a net. Given a single net, we formulate the one-net routing problem as the mapping from a one-net routing image  $x$  to the corresponding route layout image  $y$ , where  $x$  contains three channels: 1) the locations of pins,

![](images/4ede55927ddf30d485e59508a05e29871602ecdd222ae814904d8c4f4c38bf82.jpg)  
Figure 1: A chip partitioned into global cells with mixed-size macros, and its grid graph layout.

![](images/2d84b32a3eaeccd093938f28b8172ff2a61249deb9fda5c5d6a4640e27362d41.jpg)

2) the availability of horizontal grid edges, and 3) the availability of vertical grid edges, and one for  $y$ . As to the output, the value of each pixel of  $y$  denotes the likelihood of whether this grid point belongs to the route or not. Hence, the synthesis of routes can also be viewed as a binary classification of the pixels in  $y$ . The pixels whose likelihood is higher than the threshold are subsequently collected to form the route. In case of disconnected routes, we apply maze routing to refine the outcomes. In the initial design of our model, we do not adopt the random noise  $z$ , which mainly leads to producing fairly stochastic outputs, since the routing task barely requires stochasticity.

Our routing model adopts the conditional generative adversarial framework, which has shown effectiveness in image generation. The generator is composed of a basic generator for the input size of  $64 \times 64$  or below (the smaller ones are padded to resize into  $64 \times 64$ ) and an extension for the input size of larger than  $64 \times 64$ . The discriminator consists of two sub-discriminators to estimate routes from validity and realness. We, furthermore, design an enhanced loss to improve the performance of our model. The structure of the generative model is visualized in Fig. 2.

# 3.2.1 Layout Input-size-Adapting Generator

During routing, the physical chips are decomposed into Gcells in terms of various widths and heights as shown in Fig. 1, causing the diversity of the scale of the corresponding grid graphs. To make it more tractable, we develop an input-size-adapting generator to handle various grid graphs.

First, we construct a basic generator,  $G_{base}$ , to solve grids not larger than  $64 \times 64$  as the chip is divided into  $64 \times 64$  tiles in the macro placement stage. The architecture proposed by [26] is partly adopted as the backbone of  $G_{base}$ , which has been proven successful in generative tasks. Our basic generator contains four components: 1) a convolutional front-end, 2) a series of residual blocks, 3) a transposed convolutional component, and 4) a convolutional layer to generate the output.

Second, to handle larger grids, we establish another generator  $G_{large}$ , which is composed by two sub-networks:  $G_{inner}$  and  $G_{outer}$  ( $G_i$  and  $G_o$ , for simplicity).  $G_i$  and  $G_o$  are termed as the guiding network and the filling network, respectively. The guiding network consists of the first three parts that  $G_{base}$  owns. In contrast, the components of filling network is similar to  $G_{base}$ , and correspondingly we use  $G_o^k (k = 1,2,3,4)$  to denote them. We feed the input grid to  $G_o^1$  to obtain a feature map, and downsample the input grid to feed  $G_i$  to acquire another feature map.  $G_o^2$  takes in the element-wise sum of these two feature maps and integrates the guiding information into  $G_o$ , and the hybrid feature map then is converted into the output. The architecture of our generator is illustrated in Fig. 2(left). We can further incrementally stack additional sub-networks on  $G_{large}$ , and model compression techniques can be used to help keep the inner network neat. While training the networks, we first pre-train the sub-networks separately, and then we jointly train them to fine-tune the whole network.

Remarks. The CNN-based generator coincides with routing: 2-D neighborhood structure, translation equivariance and locality. Amid routing, chips are formulated as grids, which are further transformed into images. Routing also exhibits translation equivariance since translating a whole net with the context will not change the routing result. Moreover, for each grid node, the convolution kernel gathers the information from locally adjacent vertices, especially those directly connected to it, to form local routing features. The holistic route is then produced. Rather than simply stacking layers to handle long-range dependencies, the well-trained guiding network provides global information equivalent to long-range dependencies.

![](images/eb2ddaac4ecf76c342f7989aca6e403bb5918cfed68eb64f9320ce60fc598b8d.jpg)  
Figure 2: Architecture of the proposed generative routing model using cGAN. Left Top: the standard generator to handle grid graphs layout as input no bigger than  $64 \times 64$ . We pad black pixels to those smaller than  $64 \times 64$ . Left Bottom: the generator for larger grids. In the input layout, the red pixels represent pins, and the black blocks denote edges without capacity. We append  $G_{large}$  to  $G_{base}$ , and the two networks are jointly trained on large grids. The element-wise sum of the feature map of  $G_{large}$  and the feature map from  $G_{base}$  is fed to the residual blocks of  $G_{large}$  as the input. Right: The architecture of the bi-discriminator with two branches for connectivity and realness scoring. These two branches are trained with connectivity labels and realness labels, respectively.

![](images/8ab12d4e3124a330bc4b01cc6b813ba72bdcf42644d9cbf29a0b87eba84baba6.jpg)

# 3.2.2 Bi-Discriminator to Consider both Realness and Connectivity

Routing problems have an inherent constraint that all pins should be connected. Therefore we devise a discriminator to evaluate the connectivity of the output. To effectively train the connectivity discriminator, we develop an algorithm to accurately figure out the connectivity of each fake and real route, and then we employ the results as labels. Connectivity alone is not sufficient to evaluate the authenticity of the output, so we adopt another discriminator to estimate the realizness of the output, as the original discriminator in GAN. Overall, the adversarial loss of our model can be expressed as

$$
\mathcal {L} _ {a d v} (G, D) = \sum_ {i = 1, 2} \lambda_ {i} \left(\mathbb {E} _ {x, y} [ \log D _ {i} (x, y) ] + \mathbb {E} _ {x} [ \log (1 - D _ {i} (x, G (x))) ]\right), \tag {1}
$$

where  $D_{1}$  and  $D_{2}$  denote the connectivity discriminator and the realness discriminator, respectively, and  $\lambda_{1}$  and  $\lambda_{2}$  represent the corresponding weights s.t.  $\lambda_{1} + \lambda_{2} = 1$ . The discriminators share the convolutional front-end and a stack of  $L = 3$  convolutional and residual blocks, and then they make evaluations from different angles as depicted in the right part of Fig. 2.

# 3.2.3 Enhanced Model Loss

With the cGAN objective mixed with a traditional loss, such as L1 and L2 loss, training is inefficient as most grid points are easy negatives that cannot yield effective learning signals. In addition, tons of trivial negatives impair the training and give rise to a degraded model, and the output, thus, inclines to converge to an empty route. To bridge the gap between easy negatives and scarce positives, we apply the focal loss [27] and modify it to fit our task:

$$
\mathcal {L} _ {F L} (G) = - \mathbb {E} _ {x, y} \left[ \frac {1}{N} \sum_ {i = 1} ^ {N} \alpha \left[ y _ {i} (1 - g _ {i}) ^ {\gamma} \log g _ {i} + (1 - y _ {i}) g _ {i} ^ {\gamma} \log (1 - g _ {i}) \right] \right], \tag {2}
$$

where  $i = 1, \dots, N$  represents grid points, and  $y_{i}$  and  $g_{i}$  respectively denotes the real and generated value of corresponding grid point. We also incorporate the L2 loss into our objective to approach the real routes, and because it has been found beneficial to the synthesis [28, 29].

The introduction of the connectivity discriminator improves the correctness of the results, but at the same time, it may slightly increase the wirelength. Since the wirelength has an accurate theoretical lower bound, i.e. half-perimeter wirelength (HPWL) of the bounding box of a net, we take the

![](images/2f359f7b59341cf61429029165aa2659c5b35822aaca94f62c7c8d2842cf8045.jpg)  
Figure 3: Our neural macro placement and routing pipeline. Given netlist as input, our mixed-size agent sequentially places macros on the chip layout. Generative router is then adopted to route the net chosen by net order agent. Inspired by EM algorithm, we update generative router by placement result from mixed-size agent, then placement and net order agents are learned jointly in a whole reinforcement learning framework to minimize wirelength calculated by trained generative model.

difference between the length of the generated route and the HPWL as a regularization term to limit the wire length. We use  $\mathcal{L}_r(G) = \mathbb{E}_x\left[\| l(G(x)) - h(x)\| _1\right]$  to represent the regularization term, where  $l(G(x))$  denotes the length of the generated route, and  $h(x)$  denotes the HPWL of the net.

The overall enhanced objective of our model gathers the above losses:

$$
\min  _ {G} \left(\left(\max  _ {D} \mathcal {L} _ {a d v} (G, D)\right) + \mu_ {F L} \mathcal {L} _ {F L} (G) + \mu_ {L 2} \mathcal {L} _ {L 2} (G) + \mu_ {r} \mathcal {L} _ {r} (G)\right), \tag {3}
$$

where  $\mu_{FL},\mu_{L2}$  and  $\mu_r$  are defined as the factors of  $\mathcal{L}_{FL}$ ,  $\mathcal{L}_{L2}$  and  $\mathcal{L}_r$ , respectively.

# 3.3 Neural Macro Placement and Routing Pipeline

Combining the RL-based model for learning mixed-size macro placement with one-shot generative routing network to perform routing as we introduce above, we propose a pure neural pipeline for macro placement and routing. Fig. 3 shows the flow of our mixed-size macro placer with adaptive reward function between coarse HPWL estimation to wirelength from the neural router. Given the circuit information, our mixed-size agent sequentially places the macros on the chip layout, after which the generative model for routing is adopted to connect the net chosen by net order agent and calculate wirelength as feedback. Inspired by EM algorithm, we first update the generative router using placement result from mixed-size agent (similar to E step), then placement and net order agents are learned jointly in a whole reinforcement learning framework to minimize wirelength calculated by trained generative model (corresponding to M step) following a recursive pattern.

# 3.3.1 Reward Adaptation between Coarse HPWL and Router's WL

For classical placers, HPWL is a common metric for estimating the true wirelength decided by routing. In our neural pipeline, however, we apply a one-shot generative routing network to route all the nets directly, which reduces bias in the reward signal. Nevertheless, it is worth noting that the untrained policy network for placement would start with random weights so that placement results are of low quality. As a result, the distribution of pins in a single net will spread out, which is difficult for a generative model-based router to produce accurate route layout images. To tackle this problem, we propose an adaptive scheme to calculate wirelength for our placement agent, integrating HPWL and neural router's output simultaneously. We introduce variable  $\lambda$  to scale two values and define the smoothed wirelength  $WL_{s}$  as follows:

$$
W L _ {s} = \lambda \cdot W L _ {n} + (1 - \lambda) \cdot H P W L \tag {4}
$$

where  $WL_{n}$  is the feedback of neural router. In each iteration, variable  $\lambda$  is updated by function  $1 - e^{-0.01\cdot n_{iter}}$ . Initially,  $\lambda$  begins with 0 so that the wirelength is determined by HPWL that serves as a coarse reward signal. As the training proceeds, the feedback of neural router gradually becomes a prominent factor to provide a more accurate objective for the placement agent.

# 3.3.2 Learning Net Order to Route

The order in which nets are routed is one of the most critical factors that affects the routing quality [30]. Most classical routers determine the net order by heuristics, e.g., routing smaller nets earlier [31] due

to the flexibility of finding free path. However, there are diverse definitions for "smaller net", none of which is proved to be optimal. What makes the situation worse, the complexity of real routing procedure requires us to change the net order dynamically, which is hard to implement in such a complicated system. Fortunately, our neural router divides the routing task into a series of one-net routing problems and then solves them independently, making it convenient to learn the net order.

We build net order learning module upon the neural router by developing a RL agent to determine which net to route next. Inspired by the structure of placement agent, the state of net order problem consists of routing image  $R$  as a representation of current routing layout, and graph  $G$  indicating the connectivity between nets. There are three channels of  $R$ : the locations of routed net, the capability of horizontal and vertical grid edges. Graph  $G$  is an edge-to-vertex dual of netlist graph  $H$ , whose vertices denote nets (i.e., hyperedges of  $H$ ) and edges denote common cells between nets (i.e., nodes of  $H$ ). Note that the state of placement and net ordering tasks are quite similar while both seek to minimize the total wirelength, we combine them into a whole RL framework by adopting same policy network to generate feature embeddings for two tasks respectively. The united structure without heuristic solver reflects the strongly coupled relationship between placement and routing, which differs from [1] that merely applies router as a black box to calculate the reward.

# 4 Experiments

# 4.1 Protocols and Setup

Experiments are conducted on a server with RTX 3090 GPUs and AMD 3970X 32-Core CPU, and implemented by PyTorch. We term our whole approach for placement and routing as PRNet.

Benchmarks & Datasets. For placement, we validate our RL agent for mixed-size macro placement using ISPD-2005 benchmark [32] after pre-processing, such that most fixed macros are exchanged for movable ones in line with [1]. For routing<sup>3</sup>, we choose the ISPD-07 [5] benchmarks to produce routing instances and use the routes generated by the strong classic router [33] as training labels.

In the ISPD-07 benchmarks, some nets can contain up to hundreds of pins, but the average amount of pins in a single net is still about 4. In other words, massive nets contain no more than 4 pins. Therefore, the routing model should have sufficient ability to route the easy nets. From around  $750\mathrm{K}$  routing instances, we collect  $30\mathrm{K}64\times 64$  nets as the Route-small-4 dataset whose instance contains up to 4 pins,  $80\mathrm{K}64\times 64$  samples as Route-small, and  $100\mathrm{K}128\times 128$  samples as Route-large to evaluate the model's ability. Each of the three datasets is randomly divided into a training set  $(80\%)$  and a test set  $(20\%)$ . These three training sets are used to train generative routing models and pick up the best one according to their performance on the test sets. Then we continue to train the best  $64\times 64$  model with additional  $200\mathrm{K}64\times 64$  instances and train the  $128\times 128$  model with additional  $400\mathrm{K}128\times 128$  instances, which is used to perform experiments on the ISPD-98 routing benchmarks [34].

Training. To train the placement and net ordering RL agent, we use PPO [24] to update the policy network and Adam optimizer [35] is utilized with a learning rate of  $2.5 \times 10^{-4}$ . For training the routing models, we use Adam with learning rate of  $2 \times 10^{-4}$ ,  $\beta_{1} = 0.5$ ,  $\beta_{2} = 0.999$  and a weight decay of 0.01. We employ a batch size of 64. A linear learning rate decay is also applied.

Evaluation. In mixed-size placement, we adopt HPWL as the proxy of wirelength and overlapping area to evaluate both methods while we introduce wirelength (WL) and routing congestion (RC) [36] in overall placement and routing. For routing, since there is little metric for generative routing model, we introduce the metrics correctness rate (CrrtR) and wirelength ratio (WLR) to evaluate the generated results of generative models on the datasets. CrrtR signifies the ratio of the amount of connected overflow-free routes to the number of all routes, or in short the accuracy of generated result. WLR represents the ratio of the total wirelength of connected overflow-free routes to the total wirelength of the corresponding real routes. Lower WLR indicates that the route requires fewer wires. In the experiments on the ISPD-98 routing benchmarks, wirelength, overflow and runtime are used.

# 4.2 Results on Mixed-size Placement

We compare the total wirelength together with overlapping area of our mixed-size approach with the state-of-the-art and open-sourced method called DeepPlace [1] as shown in Table 1. Both methods

![](images/d457fbeb78659727874e60be4dc2465125a0bca46eba9f8bf275db9ba8458c89.jpg)  
(a) Result of DeepPlace [1].

![](images/b5e57b67642e690d0a04af095d345d8b739d55f95b6fcd3fcaaa7a933e1f45ed.jpg)  
Figure 4: Visualization of macro (in orange) /standard cell (in blue) placement by DeepPlace [1] and our mixed-size placer on circuit bigblue1. Our placer tends to place large macros in the center of canvas to avoid overlapping, while they are close to each other on the boundary for DeepPlace.  
(b) Result of our mixed-size placer.

Table 1: Comparison on mixed-size placement task on the eight circuits from ISPD-2005.  

<table><tr><td rowspan="2">Circuit</td><td rowspan="2"># Cells</td><td rowspan="2"># Mov.</td><td colspan="2">Mixed-size technique (ours)</td><td colspan="2">DeepPlace [1]</td></tr><tr><td>Wirelength ↓</td><td>Overlap Area↓</td><td>Wirelength ↓</td><td>Overlap Area↓</td></tr><tr><td>adaptec1</td><td>211K</td><td>514</td><td>82783826</td><td>12606828</td><td>80117232</td><td>66608273</td></tr><tr><td>adaptec2</td><td>255K</td><td>542</td><td>123307824</td><td>19485631</td><td>123265964</td><td>47085963</td></tr><tr><td>adaptec3</td><td>451K</td><td>710</td><td>232373680</td><td>58588016</td><td>241072304</td><td>140272759</td></tr><tr><td>adaptec4</td><td>496K</td><td>1309</td><td>234008876</td><td>73075220</td><td>236391936</td><td>169853555</td></tr><tr><td>bigblue1</td><td>278K</td><td>551</td><td>141020208</td><td>2041890</td><td>140435296</td><td>3519755</td></tr><tr><td>bigblue2</td><td>558K</td><td>948</td><td>144803296</td><td>70702107</td><td>140465488</td><td>103663199</td></tr><tr><td>bigblue3</td><td>1097K</td><td>1227</td><td>468632064</td><td>39664931</td><td>450633360</td><td>574956948</td></tr><tr><td>bigblue4</td><td>2177K</td><td>659</td><td>1001315712</td><td>67794270</td><td>951984128</td><td>87630042</td></tr><tr><td>ratio</td><td>-</td><td>-</td><td>1.000</td><td>1.0</td><td>0.987</td><td>3.9</td></tr></table>

generate intermediate macro placement via RL, and then adopt gradient-based optimization placer as used in [37] to obtain complete placement solution. With only a slight increase of the total wirelength (within  $1.3\%$  difference on average), our mixed-size macro placer achieves approximately  $4\times$  reduction over DeepPlace on the overlapping area, stressing the importance of modeling macro's shape in state space. Moreover, the reduced overlapping area requires less post-processing to resolve collision, which facilitates improvement of wirelength in the long term. Examples of our mixed-size placer and DeepPlace on circuit bigblue1 are visualized in Fig. 4.

# 4.3 Results on Routing

Comparison of Generative Backbones. We compare our model with a VAE based router [9] and use VAE* to denote it. We then combine the VAE* with a vanilla discriminator and our bi-discriminator. In addition, we implement a U-Net based cGAN following pix2pix [29] and use U-Net*-cGAN to denote it. Then we remove the discriminator to obtain a U-Net [38] generator and define it as U-Net*. We further replace the discriminator of U-Net*-cGAN with our bi-discriminator. We also try to train the RL agent following the work of [6], but it fails to converge after 2-week training. Table 2 shows the results. The ResNet-based [39] models outperform the counterparts based on VAE* and U-Net*. Our model achieves approximately  $2 \times$  correctness rate,  $14.7\%$  improvement of the wirelength on Route-small-4 and  $2.4\%$  on Route-small over VAE*. The vanilla cGAN discriminator slightly improves VAE* and U-Net* on one side while sacrificing the other side, and it debases the ResNet generator. However, the bi-discriminator strengthens the generators except for the VAE*.

Ablation Studies. We conduct ablation experiments to investigate the contributions of the design choices in our model. In Table 2, we compare the full version with ResNet-based cGAN, as well as the pure ResNet generator. The ResNet generator outdoes the cGAN, but the bi-discriminator significantly improves the generator. Moreover, the enhanced loss improves the wirelength at the marginal expense of correctness. Appendix A.4.1 contains further details of comparison among loss functions, and Appendix A.4.2 shows the effectiveness of the input-size-adapting network.

Table 2: Evaluation of different backbones w.r.t. correctness rate (CrrtR) and wirelength ratio (WLR) for the routing on: Route-small-4 and Route-small. cGAN: the vanilla cGAN model with a single realizness discriminator; bcGAN: the bi-discriminator version. EL: enhanced loss in Eq. 3.  
Table 3: Evaluation of wirelength (WL) and routing congestion (RC) for overall placement and routing pipeline on ISPD-05 benchmark. "GR": our generative router; "NOL": net order learning.  

<table><tr><td rowspan="2">our router w/ different generative models</td><td colspan="2">Route-small-4</td><td colspan="2">Route-small</td></tr><tr><td>CrrtR↑</td><td>WLR↓</td><td>CrrtR↑</td><td>WLR↓</td></tr><tr><td>VAE* [9]</td><td>0.414±0.020</td><td>1.179±0.033</td><td>0.397±0.008</td><td>1.042±0.006</td></tr><tr><td>VAE*-cGAN [29]</td><td>0.557±0.065</td><td>1.292±0.108</td><td>0.439±0.021</td><td>1.315±0.015</td></tr><tr><td>VAE*-bcGAN</td><td>0.474±0.048</td><td>1.525±0.029</td><td>0.488±0.007</td><td>1.241±0.012</td></tr><tr><td>U-Net* [38]</td><td>0.724±0.001</td><td>3.306±0.266</td><td>0.524±0.005</td><td>1.232±0.016</td></tr><tr><td>U-Net*-cGAN</td><td>0.602±0.009</td><td>1.028±0.001</td><td>0.532±0.011</td><td>1.286±0.022</td></tr><tr><td>U-Net*-bcGAN</td><td>0.721±0.012</td><td>1.134±0.055</td><td>0.552±0.007</td><td>1.104±0.054</td></tr><tr><td>ResNet [39]</td><td>0.783±0.002</td><td>1.023±0.003</td><td>0.594±0.004</td><td>1.030±0.007</td></tr><tr><td>ResNet-cGAN</td><td>0.698±0.010</td><td>1.073±0.011</td><td>0.568±0.020</td><td>1.320±0.151</td></tr><tr><td>ResNet-bcGAN</td><td>0.804±0.021</td><td>1.035±0.013</td><td>0.738±0.005</td><td>1.036±0.002</td></tr><tr><td>ResNet-bcGAN+EL (full version of our router)</td><td>0.814±0.001</td><td>1.010±0.000</td><td>0.735±0.010</td><td>1.018±0.004</td></tr></table>

Table 4: Evaluation of wirelength (WL) and runtime (Time) with three classical routers on ISPD-98 routing benchmarks. Note that the overflow (OF) is all zero for all methods.  

<table><tr><td rowspan="2">variants of our PRNet</td><td colspan="2">adaptec1</td><td colspan="2">adaptec3</td></tr><tr><td>WL↓</td><td>RC↓</td><td>WL↓</td><td>RC↓</td></tr><tr><td>RL-based Placer (i.e. DeepPlace [1])</td><td>6149</td><td>10.565</td><td>30154</td><td>62.751</td></tr><tr><td>RL-based Placer + GR</td><td>5940</td><td>10.464</td><td>29711</td><td>73.324</td></tr><tr><td>RL-based Placer + GR + NOL (full version of PRNet)</td><td>5787</td><td>9.386</td><td>29462</td><td>43.207</td></tr></table>

<table><tr><td rowspan="2">Circuits</td><td colspan="2">Our router</td><td colspan="2">NTHU-Route 2.0 [30]</td><td colspan="2">BoxRouter 2.0 [40]</td><td colspan="2">FastRoute 3.0 [41]</td></tr><tr><td>WL↓</td><td>Time(s)↓</td><td>WL↓</td><td>Time(s)↓</td><td>WL↓</td><td>Time(s)↓</td><td>WL↓</td><td>Time(s)↓</td></tr><tr><td>ibm01</td><td>62337</td><td>59.2</td><td>62498</td><td>1.54</td><td>62659</td><td>33</td><td>64221</td><td>0.64</td></tr><tr><td>ibm02</td><td>170270</td><td>179.9</td><td>170531</td><td>3.15</td><td>171110</td><td>36</td><td>172223</td><td>0.85</td></tr><tr><td>ibm03</td><td>146362</td><td>194.6</td><td>146458</td><td>1.49</td><td>146634</td><td>18</td><td>146753</td><td>0.49</td></tr><tr><td>ibm04</td><td>165874</td><td>254.4</td><td>166452</td><td>3.81</td><td>167275</td><td>116</td><td>170146</td><td>2.7</td></tr></table>

Test Results. We test our conditional generative routing model on the ISPD-98 benchmarks and compare the wirelength, overflow and runtime with three classical routers that perform best on ISPD-98 benchmarks. Table 4 shows the results. Our generative routing model presents competitive consequences on wirelength, while it takes a longer time to accomplish the routing task, compared with strong heuristic baselines [30, 40, 41]. Our model takes an image encoded from the whole grid with a net as the input and sequentially solve each net in a one-shot manner, while classical routers only consider the local area, which may obtain fewer wirelength yet consume more time. However, it is still much more efficient and easier to train than the RL-based router [6] from our empirical experience. We leave the speedup of our generative model for future work.

# 4.4 Results of Overall Placement and Routing with Ablation Study on Net Order Learning

We compare our PRNet with DeepPlace, along with an ablation study to verify the impact of net order learning. The circuits used for evaluation are the same as in mixed-size placement, and we concentrate on macros only for simplicity. Note that the real shape of macros is ignored and the grid-based mask is coarser in [1], hence the results shown in Table 3 are not identical to those in the original paper [1]. For all test cases, our neural placement and routing pipeline outperforms the other two methods in terms of both wirelength (WL) and routing congestion (RC). The significant difference in routing congestion without net order learning indicates that net order agent is able to arrange the sequence of routing efficiently, especially on circuit adaptec3. As a result, it is easy for every net to find free routing path while keeping away from congested area. In addition, training placement model with generative neural router in an end-to-end manner further improves the final wirelength, despite a little degradation of routing congestion if we discard the net order agent.

# 5 Conclusion and Discussion

We have presented a neural mixed-size placement and routing pipeline. The routing is achieved by one-shot generation of the whole path, with our devised net order learning module to dynamically adjust the routing order. Experimental results show the effectiveness of our approach.

# References

[1] R. Cheng and J. Yan, "On joint learning for solving placement and routing in chip design," NeurIPS, 2021.  
[2] A. Mirhoseini, A. Goldie, M. Yazgan, J. W. Jiang, E. Songhori, S. Wang, Y.-J. Lee, E. Johnson, O. Pathak, A. Nazi et al., "A graph placement methodology for fast chip design," Nature, 2021.  
[3] J. Liu, G. Chen, and E. F. Young, "Rest: Constructing rectilinear steiner minimum tree via reinforcement learning," in DAC, 2021.  
[4] Y.-L. Chuang, G.-J. Nam, C. J. Alpert, Y.-W. Chang, J. Roy, and N. Viswanathan, "Design-hierarchy aware mixed-size placement for routability optimization," in ICCAD, 2010.  
[5] G.-J. Nam, M. Yildiz, D. Pan, and P. Madden, "Ispd 2007 global routing contest," 2007.  
[6] H. Liao, W. Zhang, X. Dong, B. Poczos, K. Shimada, and L. Burak Kara, "A deep reinforcement learning approach for global routing," Journal of Mechanical Design, 2020.  
[7] D. Vashisht, H. Rampal, H. Liao, Y. Lu, D. Shanbhag, E. Fallon, and L. B. Kara, "Placement in integrated circuits using cyclic reinforcement learning and simulated annealing," arXiv preprint arXiv:2011.07577, 2020.  
[8] H. Liao, Q. Dong, X. Dong, W. Zhang, W. Zhang, W. Qi, E. Fallon, and L. B. Kara, “Attention routing: track-assignment detailed routing using attention-based reinforcement learning,” in International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. American Society of Mechanical Engineers, 2020.  
[9] D. Utyamishev and I. Partin-Vaisband, “Late breaking results: A neural network that routes ics,” in DAC. IEEE, 2020.  
[10] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," in ICLR, 2014.  
[11] B. Xu, Y. Lin, X. Tang, S. Li, L. Shen, N. Sun, and D. Z. Pan, "Wellgan: Generative-adversarial-network-guided well generation for analog/mixed-signal circuit layout," in 2019 56th ACM/IEEE Design Automation Conference (DAC). IEEE, 2019.  
[12] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in NIPS, 2014.  
[13] K. Zhu, H. Chen, M. Liu, X. Tang, W. Shi, N. Sun, and D. Z. Pan, "Generative-adversarial-network-guided well-aware placement for analog circuits," in 2022 27th Asia and South Pacific Design Automation Conference (ASP-DAC). IEEE, 2022.  
[14] A. Gusmão, R. Póvoa, N. Horta, N. Lourenço, and R. Martins, “Deepplacer: A custom integrated opamp placement tool using deep models,” Applied Soft Computing, 2022.  
[15] W. Jin, S. Sadiqbatcha, J. Zhang, and S. X.-D. Tan, “Full-chip thermal map estimation for commercial multi-core cpus with generative adversarial learning** this work is supported in part by nsf grants under no. ccf-1816361, in part by nsf grant under no. ccf-2007135 and no. oise-1854276.” in 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD). IEEE, 2020.  
[16] Z. Zhou, Z. Zhu, J. Chen, Y. Ma, and A. Ivanov, "Congestion-aware global routing using deep convolutional generative adversarial networks," in 2019 ACM/IEEE 1st Workshop on Machine Learning for CAD (MLCAD), 2019.  
[17] D. Utyamishev and I. Partin-Vaisband, "Multiterminal pathfinding in practical vlsi systems with deep neural networks," 2022.  
[18] H. Yang, P. Pathak, F. Gennari, Y. C. Lai, and B. Yu, "Deeppattern: Layout pattern generation with transforming convolutional auto-encoder," in the 56th Annual Design Automation Conference 2019, 2019.  
[19] A. Hottung, B. Bhandari, and K. Tierney, "Learning a latent search space for routing problems using variational autoencoders," in International Conference on Learning Representations, 2020.  
[20] M. D. Moffitt, “Maizerouter: Engineering an effective global router,” TCAD, 2008.  
[21] M. M. Ozdal and M. D. Wong, “Archer: A history-based global routing algorithm,” TCAD, 2009.

[22] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., "Human-level control through deep reinforcement learning," nature, vol. 518, no. 7540, pp. 529-533, 2015.  
[23] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," arXiv preprint arXiv:1609.02907, 2016.  
[24] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.  
[25] M. R. Garey and D. S. Johnson, “The rectilinear steiner tree problem is np-complete,” SIAM Journal on Applied Mathematics, vol. 32, no. 4, pp. 826-834, 1977.  
[26] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and superresolution,” in European conference on computer vision. Springer, 2016, pp. 694–711.  
[27] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object detection," in Proceedings of the IEEE international conference on computer vision, 2017.  
[28] D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros, “Context encoders: Feature learning by inpainting,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.  
[29] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," CVPR 2017, pp. 5967-5976, 2017.  
[30] Y.-J. Chang, Y.-T. Lee, J.-R. Gao, P.-C. Wu, and T.-C. Wang, "Nthu-route 2.0: a robust global router for modern designs," TCAD, 2010.  
[31] Y. Xu, Y. Zhang, and C. Chu, "Fastroute 4.0: Global router with efficient via minimization," in ASP-DAC. IEEE, 2009.  
[32] G.-J. Nam, C. J. Alpert, P. Villarrubia, B. Winter, and M. Yildiz, “The ispd2005 placement contest and benchmark suite,” in Proceedings of the 2005 international symposium on Physical design, 2005, pp. 216–220.  
[33] W.-H. Liu, W.-C. Kao, Y.-L. Li, and K.-Y. Chao, "Nctu-gr 2.0: Multithreaded collision-aware global routing with bounded-length maze routing," TCAD, 2013.  
[34] C. J. Alpert, “The ispd98 circuit benchmark suite,” in Proceedings of the 1998 international symposium on Physical design, 1998, pp. 80–85.  
[35] P. D. Kingma and L. J. Ba, "Adam: A method for stochastic optimization," international conference on learning representations, 2015.  
[36] C.-K. Cheng, A. B. Kahng, I. Kang, and L. Wang, "Replace: Advancing solution quality and routability validation in global placement," TCAD, 2018.  
[37] Y. Lin, Z. Jiang, J. Gu, W. Li, S. Dhar, H. Ren, B. Khailany, and D. Z. Pan, “Dreamplace: Deep learning toolkit-enabledgpu acceleration for modern vlsi placement,” TCAD, 2020.  
[38] R. O, F. P, and B. T, “U-net: Convolutional networks for biomedical image segmentation,” in MICAAI, 2015.  
[39] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in CVPR, 2016.  
[40] M. Cho, K. Lu, K. Yuan, and D. Z. Pan, "Boxrouter 2.0: A hybrid and robust global router with layer assignment for routability," TODAES, 2009.  
[41] Y. Zhang, Y. Xu, and C. Chu, "Fastroute3.0: a fast and high quality global router based on virtual capacity," in ICCAD. IEEE, 2008.  
[42] M. A. Breuer, “A class of min-cut placement algorithms,” in DAC, 1977.  
[43] C. M. Fiduccia and R. M. Mattheyses, “A linear-time heuristic for improving network partitions,” in DAC. IEEE, 1982.  
[44] A. R. Agnihotri, S. Ono, and P. H. Madden, "Recursive bisection placement: Feng shui 5.0 implementation details," in ISPD, 2005.  
[45] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by simulated annealing," Science, 1983.

[46] P. Spindler, U. Schlichtmann, and F. M. Johannes, “Kraftwerk2—a fast force-directed quadratic placement approach using an accurate net model,” TCAD, 2008.  
[47] T.-C. Chen, Z.-W. Jiang, T.-C. Hsu, H.-C. Chen, and Y.-W. Chang, "Ntuple3: An analytical placer for large-scale mixed-size designs with preplaced blocks and density constraints," TCAD, 2008.  
[48] A. B. Kahng and Q. Wang, "Implementation and extensibility of an analytic placer," TCAD, 2005.  
[49] J. Lu, P. Chen, C.-C. Chang, L. Sha, D. J.-H. Huang, C.-C. Teng, and C.-K. Cheng, "eplace: Electrostatics-based placement using fast fourier transform and nesterov's method," TODAES, 2015.  
[50] T. Taghavi, X. Yang, and B. Choi, "Dragon2005: Large-scale mixed-size placement tool," in ISPD, 2005.  
[51] H. Chen, Y. Chuang, Y.-W. Chang, and Y. Chang, "Constraint graph-based macro placement for modern mixed-size circuit designs," in ICCAD, 2008.  
[52] M. Cho and D. Z. Pan, "Boxrouter: A new global router based on box expansion and progressive ip," TCAD, 2007.  
[53] T.-H. Wu, A. Davoodi, and J. T. Linderoth, "Grip: Scalable 3d global routing using integer programming," in DAC, 2009.  
[54] T.-H. Wu, A. Davoodi, and J. T. Linderoth, "A parallel integer programming approach to global routing," in Design Automation Conference. IEEE, 2010.  
[55] C. Chu and Y.-C. Wong, "Flute: Fast lookup table based rectilinear steiner minimal tree algorithm for vlsi design," TCAD, 2007.  
[56] C. Y. Lee, "An algorithm for path connections and its applications," IRE Transactions on Electronic Computers, 1961.  
[57] R. Kastner, E. Bozorgzadeh, and M. Sarrafzadeh, "Pattern routing: Use and theory for increasing predictability and avoiding coupling," TCAD, 2002.  
[58] H.-Y. Chen, C.-H. Hsu, and Y.-W. Chang, "High-performance global routing with fast overflow reduction," in ASP-DAC. IEEE, 2009.
