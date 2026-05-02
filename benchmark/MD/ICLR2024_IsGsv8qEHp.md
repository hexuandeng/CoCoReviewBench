# HUMAN-ORIENTED REPRESENTATION LEARNING FOR ROBOTIC MANIPULATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Humans inherently possess generalizable visual representations that empower them to efficiently explore and interact with the environments in manipulation tasks. We advocate that such a representation automatically arises from simultaneously learning about multiple simple perceptual skills that are critical for everyday scenarios (e.g., hand detection, state estimate, etc.) and is better suited for learning robot manipulation policies compared to current state-of-the-art visual representations purely based on self-supervised objectives. We formalize this idea through the lens of human-oriented multi-task fine-tuning on top of pre-trained visual encoders, where each task is a perceptual skill tied to human-environment interactions. We introduce Task Fusion Decoder as a plug-and-play embedding translator that utilizes the underlying relationships among these perceptual skills to guide the representation learning towards encoding meaningful structure for what's important for all perceptual skills, ultimately empowering learning of downstream robotic manipulation tasks. Extensive experiments across a range of robotic tasks and embodiments, in both simulations and real-world environments, show that our Task Fusion Decoder improves the representation of three state-of-the-art visual encoders including R3M, MVP, and EgoVLP, for downstream manipulation policy-learning. More demos, datasets, models, and code can be found at our anonymous webpage.

# 1 INTRODUCTION

In the fields of robotics and artificial intelligence, imbuing machines with the ability to efficiently interact with their environment has long been a challenging problem. While humans can effortlessly explore and manipulate their surroundings with very high generalization, robots often fail even when faced with basic manipulation tasks, particularly in unfamiliar environments. These representations empower us to perceive and interact with our environment, effectively learning complex manipulation skills. How to learn generalizable representations for robotic manipulations thus has drawn much attention.

Existing representation learning for robotics can be generally divided into three streams. 1) Traditionally representations were hand-crafted (e.g., key point detection (Das et al., 2021) inspired by biological studies (Johansson, 1973)). They provide strong inductive bias from human engineers, but encode a limited understanding of what matters about human behavior. 2) Modern state-of-the-art methods (Chen et al., 2016; Higgins et al., 2016; He et al., 2020; Chen et al., 2020; He et al., 2022; Nair et al., 2022) propose to automatically discover generalizable representations from data, e.g., by masked image modeling and contrastive learning techniques. Though general-purpose or language semantic-based representations can be learned, they fail to grasp human behavior biases and motion cues, e.g., hand-object interaction, for robotic manipulation tasks. 3) Recent human-in-the-loop methods (Bajcsy et al., 2018; Bobu et al., 2022; 2023a) attempt to disentangle and guide aspects of the representation through additional human feedback. However, they are limited to learning from low-dimensional data (e.g., physical state trajectories) due to the huge amount of human labels that are required. Each of these approaches comes with its own set of drawbacks, which lead to suboptimal performance in robotic manipulations.

In this work, we propose that a robust and generalizable visual representation can be automatically derived from the simultaneous acquisition of multiple simple perceptual skills that mirror those crit-

![](images/2483d5a8cfd77b3856dbc276d5d52a23a617e0d3652a399e1605466fc24cb77c.jpg)  
Figure 1: Left: human-oriented representation learning as a multi-task learner. Right: robots leverage the human-oriented representation to learn various manipulation tasks.

ical to human-environment interactions, as shown in Fig. 1. This concept aligns with insights from cognitive science (Kirkham et al., 2002), which posits that humans learn to extract a generalizable behavioral representation from perceptual input by mastering a multitude of simple perceptual skills, such as spatial-temporal understanding and hand-object contact estimation, all of which are critical for everyday scenarios. Centered on these human-inspired skills, we introduce Task Fusion Decoder (TFD) as a plug-and-play multitask learner to learn human-oriented representation for robotic manipulation. Unlike current state-of-the-art visual representations, which primarily rely on self-supervised objectives, our approach harnesses the power of these human-inspired perceptual skills with low-cost human priors.

Task Fusion Decoder is carefully designed with the following considerations. 1) It learns perceptual skills on the largest ego-centric video dataset Ego4D (Grauman et al., 2022) with three representative tasks that capture how humans manipulate objects: object state change classification (OSCC), point-of-no-return temporal localization (PNR), and state change object detection (SCOD). In this way, the robot manipulation representation space is learned and distilled from real-world human experience. 2) It takes advantage of its inside self- and cross-attention mechanisms to establish information flow across tasks through the attention matrix and learn inherent task relationships automatically through end-to-end training. The underlying relationships between these perceptual skills are utilized to guide the representation learning towards encoding meaningful structure for manipulation tasks. 3) It is plug-and-play and can be directly built on previous foundational backbones with an efficient fine-tuning strategy, which enables it to be easily generalized and transferred to novel settings and models. We will show it improves the performance of various state-of-the-art models on various robot manipulation benchmarks and tasks.

Our contributions are three-fold. 1) We introduce an efficient and unified framework, Task Fusion Decoder, tailored as a human-oriented multitask learner aimed at cultivating representations guided by human-inspired skills for robotic manipulations. 2) The plug-and-play nature of our framework ensures flexibility, allowing it to seamlessly adapt to different base models and simulation environments. To demonstrate its real-world applicability, we also collect and open-source a real-world robot manipulation dataset, comprising 17 kinds of tasks featuring expert demonstrations. 3) Extensive experiments across various model backbones (i.e., MVP (Xiao et al., 2022), R3M (Nair et al., 2022), and EgoVLP (Qinghong Lin et al., 2022)), benchmarks (i.e., Franka Kitchen (Gupta et al., 2019), MetaWorld (Yu et al., 2020), Adroit (Rajeswaran et al., 2017), and real-world manipulations), and diverse settings (e.g., different cameras and evaluation metrics) demonstrate our effectiveness.

# 2 RELATED WORK

Representation learning for robotic learning. Representation learning, with the goal of acquiring effective visual encoders (Nair et al., 2022; Mu et al., 2023a; Hansen et al., 2022; Ze et al., 2023; Parisi et al., 2022; Yen-Chen et al., 2020; Shridhar et al., 2022; Khandelwal et al., 2022; Shah & Kumar, 2021; Seo et al., 2022), is crucial to computer vision and robotic learning tasks. Recently, it has been dominated by unsupervised and self-supervised methods (Chen et al., 2016; Higgins et al., 2016; He et al., 2020; Chen et al., 2020; He et al., 2022; Nair et al., 2022; Ma et al., 2022; Brohan et al., 2022; Alakuijala et al., 2023; Karamcheti et al., 2023; Mu et al., 2023b; Jing et al., 2023;

![](images/4f8bb17837ec6876372a7599e584daf17abede82e346a0a144d8e33ad42b6cd8.jpg)  
Figure 2: The pipeline for the finetuning framework by using task fusion network. The task fusion decoder which includes the cross-attention and self-attention, can adjust the video encoder representation and fuse different tasks information.

Majumdar et al., 2023). These methods try to learn disentangled representations from large datasets (Russakovsky et al., 2015; Goyal et al., 2017; Damen et al., 2018; Shan et al., 2020; Grauman et al., 2022). Though requiring little human cost, these methods purposefully bypass human input, consequently, the learned representations are prone to spurious correlations and do not necessarily capture the attributes that are important for downstream tasks (LeCun, 2022; Bobu et al., 2023b). For example, Xiao et al. (Xiao et al., 2022) propose using masked autoencoders (MAE) to learn a mid-level representation for robot learning of human motor skills (e.g., pick and place). However, the MAE representation is tailored for reconstructing pixel-level image structure and does not necessarily encode essential high-level behavior cues such as hand-object interaction. To mitigate this, another line of works attempts to leverage human priors by explicitly involving a human in the learning loop to iteratively guide the representation towards human-orientated representations (Bobu et al., 2021; Katz et al., 2021; Bobu et al., 2022; 2023a). However, these methods do not scale when learning from raw pixels due to the laborious human costs. Our idea fills the gap between unsupervised/self-supervised and human-guided representation learning. Our human-oriented representation arises from simultaneously learning about multiple perceptual skills from large and well-labeled video datasets that already capture human priors. Through this, we can effectively capture important attributes that are important for human motor skills in everyday scenarios in a human-oriented but label-efficient way.

Multitask learning. Multitask representation learning uses proxy tasks to instill human's intuition on important attributes about the downstream task in representation learning (Brown et al., 2020; Yamada et al., 2022). The hope is that by learning a shared representation optimized for all the tasks, robots can effectively leverage these representations for novel but related tasks. Tasks have inherent relationships and encoding their relationships into the learning process can promote generalizable representations that achieve efficient learning and task transfer (e.g., Taskonomy (Zamir et al., 2018) and Cross-Task (Zamir et al., 2020)). However, learning the underlying relationship between tasks remains a challenge. Previous methods use a computational approach to identify task relationship by manually sampling feasible task relationships, training and evaluating the benefit of each sampled task relationship (Zamir et al., 2018; 2020). However, their scalability remains a serious issue as they require running the entire training pipeline for each candidate task relationship. (Bahl et al., 2023) adopts a multi-task structure for affordance. Compared with directly predicting affordance, the visual representation learning method is more flexible to fit various kinds of robot learning tasks with observation space. We advance multi-task learning by enabling the model to automatically learn the task relationship during training. Our method explicitly helps each task to learn to query useful information from other tasks.

# 3 METHODOLOGY

In recent advancements within the field of visual-motor control, there has been a growing emphasis on harnessing the remarkable generalization capabilities of machine learning models to develop unique representations for robot learning. As representatives, R3M (Nair et al., 2022) proposes a large vision-language alignment model based on ResNet (He et al., 2016) for behavior cloning; MVP (Xiao et al., 2022) leverages masked modeling on Vision Transformer (ViT) (Dosovitskiy et al., 2020) to extract useful visual representation for reinforcement learning; EgoVLP (Qinghong Lin et al., 2022) learns video representations upon a video transformer (Bain

et al., 2021). To leverage their successes, we proposed to cultivate better representations for robotic manipulation by fine-tuning these vision backbones with human-oriented guidance from diverse human action related tasks. In the following sections, we introduce our Task Fusion Decoder, which is a general-purpose decoder that can work with any existing encoder networks. We then detail its training for multi-task structure. For the human-oriented tasks selection, we leverage three mutually related tasks in the hand object interaction benchmark from the Ego4D dataset for joint training. We describe them as follows.

The object state change classification (OSCC) task is to classify if there is a state change in the video clip; the point-of-no-return temporal localization (PNR) task is to localize the keyframe with state change in the video clip; the state change object detection (SCOD) task is to localize the hand object positions during the interaction process.

# 3.1 TASK FUSION DECODER

Previous works primarily incorporate high-level information from the entire visual scene, often overlooking the vital influence of human motion within the representation. However, human knowledge such as hand-object interactions in the environments is important for robotic manipulations. To gather different human pre-knowledge concurrently, it is crucial to incorporate different temporal and spatial tasks simultaneously into a single representation. Also, different vision tasks should have information interaction, for the human-like synesthesia. To achieve this, we design a decoder-only network structure Task Fusion Decoder, which can both induce task-specific information and integrate different tasks.

Task Fusion Decoder is a multitask learner (see Figure 15) aiming to learn three human-oriented tasks which are originally from the ego-centric video dataset Ego4D (Grauman et al., 2022): object state change classification (OSCC), point-of-no-return temporal localization (PNR), and state change object detection (SCOD). The definition for the three tasks can be found in Figure 3. It is also designed to work with various vision backbones, such as ResNet (He et al., 2016), ViT (Dosovitskiy et al., 2020), and Timesformer (Bain et al., 2021). Given a video, we denote its number of input frames as  $T$ , the outputted number of patches (for ViT) or feature map size (for ResNet) per frame as  $P$ , and the representation dimension for the encoder as  $D$ . In this way, we can have: (1) the global feature  $h_{cls} \in \mathbb{R}^{1 \times D}$  representing the whole video sequence, e.g., the class token for ViT or final layer feature for ResNet; and (2)  $h_{total} \in \mathbb{R}^{(P \times T) \times D}$  as dense features with spatial and temporal information preserved.

For time-related tasks, representation  $h_t$  for the whole video sequence is required for learning. We choose  $h_{cls}$  as  $h_t$  and adopt a time positional embedding to localize the frame. For spatial-related tasks, representation  $h_s$  for capturing the localization of one specific action, so we adopt a frame pre-selection strategy to select the keyframe that only covers the state change frame from  $h_{total}$ . In this case,  $h_s \in \mathbb{R}^{P \times D}$  denotes the representation of the state change frame. Similarly, we adopt a positional encoding for  $h_s$  before feeding into the decoder network. For ResNet, we append an additional transformer encoder network to adapt the convolutional feature to the patch-wise feature.

Within Task Fusion Decoder, we define 10 task tokens  $z_{i}^{k}$  as the input of the  $k_{th}$  decoder layer, where  $1 \leq k \leq N$ ,  $z_{1}^{k}$  and  $z_{2}^{k}$  are object state change classification(OSCC) task token and temporal localization(PNR) task token, respectively;  $z_{3}^{k} - z_{10}^{k}$  are state change object detection(SCOD) task tokens, which provide nominated bounding boxes for hand and object detection. The  $k_{th}$  layer of the decoder structure can be formulated as:

$$
\left\{f _ {i} ^ {k} \right\} _ {i} = \text {S e l f - A t t e n t i o n} \left(\left\{z _ {i} ^ {k} \right\} _ {i}\right) \tag {1}
$$

$$
\left\{z _ {i} ^ {k + 1} \right\} _ {i} = \text {C r o s s - A t t e n t i o n} \left(h _ {t}, \left\{f _ {i} ^ {k} \right\} _ {i}\right), i \in \{1, 2 \} \tag {2}
$$

$$
\left\{z _ {i} ^ {k + 1} \right\} _ {i} = \text {C r o s s - A t t e n t i o n} \left(h _ {s}, \left\{f _ {i} ^ {k} \right\} _ {i}\right), 3 \leq i \leq 1 0 \tag {3}
$$

where  $f_{i}^{k}$  is the feature after interacting between task tokens,  $z_{i}^{k + 1}$  is the feature of next layer decoder input. Self-attention can perform task fusion for each layer. For the last layer of the decoder network, we adapt 10 MLP layers for 10 different task tokens as translators for the tasks with human pre-knowledge.

# 3.2 JOINT MULTITASK TRAINING

For the OSCC task, there is a binary label to represent whether a state changes or not. The decoder output is the probability of containing a state change in the input video sequence. The loss of OSCC task  $L_{osc}$  is thus a cross-entropy loss as a two-category classification problem.

For the PNR task, the label  $D_{pnr}$  is a distribution with the length of number frames  $T$ , where the label of the state change frame is 1, and others are 0. For video clips without state change, all label is set to  $1 / T$ . We mimic the assigned distribution with KL-divergence loss as follows:

$$
L _ {p n r} = \mathrm {K L} \left(f \left(z _ {2} ^ {N}\right), D _ {p n r}\right) \tag {4}
$$

where  $f(z_2^N)$  is the decoder output probability ground truth state change frame distribution.

![](images/62fcd0cf8c96601e108ea4a63201e8878da146c65dcc3d0a1c5eeee6a337c385.jpg)  
Figure 3: Definition for the three spatial and time related tasks and the formulation of the loss function.

For the SCOD task, we formulate it an object detection task following DETR (Carion et al., 2020), which uses the Hungarian algorithm (Kuhn, 1955) to select the most nominated bounding boxes for hands and objects. The decoder outputs are logits for bounding-box positions and object classes. We get the  $L_{scod}$  by a bounding box localization loss and a classification loss.

For joint training of the three multi-tasks, we propose to balance the three losses by adding weighted terms as a variance constraint (Kendall et al., 2018) for them:

$$
L = \frac {1}{2 \sigma_ {1} ^ {2}} L _ {\text {o s c c}} + \frac {1}{2 \sigma_ {2} ^ {2}} L _ {\text {p n r}} + \frac {1}{2 \sigma_ {3} ^ {2}} L _ {\text {s c o d}} + \log \left(\sigma_ {1} \sigma_ {2} \sigma_ {3}\right), \tag {5}
$$

where  $\sigma_{i}$  is a learnable variance. By leveraging such a constraint, the three tasks are automatically learned in a balanced manner.

# 4 EXPERIMENTS

# 4.1 IMPLEMENTATION DETAILS

We leverage our Task Fusion Decoder to finetune three backbone models that are frequently used in robotics tasks: R3M, MVP, and EgoVLP. The FHO slice of the Ego4D dataset is used. The

Table 1: Success rate evaluation on R3M model. We indicate performance decrease in Blue and performance increase in Red.  

<table><tr><td colspan="2">env</td><td>R3M (%)</td><td>R3M+ours (%)</td></tr><tr><td rowspan="6">kitchen</td><td>sdoor-open</td><td>64.00</td><td>79.00 (+15.00)</td></tr><tr><td>ldoor-open</td><td>38.33</td><td>29.00 (-9.33)</td></tr><tr><td>light-on</td><td>75.00</td><td>77.34 (+2.34)</td></tr><tr><td>micro-open</td><td>27.34</td><td>28.67 (+1.33)</td></tr><tr><td>knob-on</td><td>61.34</td><td>58.00 (-3.34)</td></tr><tr><td>average</td><td>53.20</td><td>54.40 (+1.20)</td></tr><tr><td rowspan="6">metaworld</td><td>assembly</td><td>93.67</td><td>98.67 (+5.00)</td></tr><tr><td>bin-pick</td><td>44.67</td><td>56.33 (+11.66)</td></tr><tr><td>button-press</td><td>56.34</td><td>62.67 (+6.33)</td></tr><tr><td>hammer</td><td>92.67</td><td>86.34 (-6.33)</td></tr><tr><td>drawing-open</td><td>100.00</td><td>100.00 (+0.00)</td></tr><tr><td>average</td><td>77.47</td><td>80.80 (+3.33)</td></tr><tr><td rowspan="3">adroit</td><td>pen</td><td>67.33</td><td>70.00 (+2.67)</td></tr><tr><td>relocate</td><td>63.33</td><td>66.22 (+2.89)</td></tr><tr><td>average</td><td>65.33</td><td>68.11 (+2.78)</td></tr></table>

Figure 4: Tasks defined in Kitchen, MetaWorld and Adroit environments from different views.  

<table><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Table 2: Success rate evaluation on the EgoVLP and MVP models.  

<table><tr><td colspan="2">env</td><td>EgoVLP (%)</td><td>EgoVLP+ours (%)</td><td>MVP (%)</td><td>MVP+ours (%)</td></tr><tr><td rowspan="6">kitchen</td><td>sdoor-open</td><td>43.00</td><td>44.00 (+1.00)</td><td>32.00</td><td>44.00 (+12.00)</td></tr><tr><td>ldoor-open</td><td>4.00</td><td>7.00 (+3.00)</td><td>9.00</td><td>11.00 (+2.00)</td></tr><tr><td>light-on</td><td>19.00</td><td>12.00 (-7.00)</td><td>18.00</td><td>15.00 (-3.00)</td></tr><tr><td>micro-open</td><td>11.00</td><td>16.00 (+5.00)</td><td>4.00</td><td>7.00 (+3.00)</td></tr><tr><td>knob-on</td><td>11.00</td><td>14.00 (+3.00)</td><td>6.00</td><td>4.00 (-2.00)</td></tr><tr><td>average</td><td>17.60</td><td>18.60 (+1.00)</td><td>13.80</td><td>16.20 (+2.40)</td></tr><tr><td rowspan="6">metaworld</td><td>assembly</td><td>10.67</td><td>21.33 (+10.66)</td><td>14.67</td><td>27.33 (+12.66)</td></tr><tr><td>bin-pick</td><td>4.67</td><td>12.00 (+7.33)</td><td>3.33</td><td>4.00 (+0.67)</td></tr><tr><td>button-press</td><td>24.00</td><td>15.33 (-8.67)</td><td>40.67</td><td>32.00 (-8.67)</td></tr><tr><td>hammer</td><td>58.00</td><td>81.33 (+23.33)</td><td>98.67</td><td>97.33 (-1.34)</td></tr><tr><td>drawing-open</td><td>62.67</td><td>88.67 (+26.00)</td><td>40.67</td><td>44.00 (+3.33)</td></tr><tr><td>average</td><td>32.00</td><td>43.73 (+11.73)</td><td>39.60</td><td>40.93 (+1.33)</td></tr><tr><td rowspan="3">adroit</td><td>pen</td><td>67.33</td><td>69.33 (+2.00)</td><td>60.67</td><td>62.00 (+1.33)</td></tr><tr><td>relocate</td><td>26.67</td><td>32.00 (+5.33)</td><td>16.00</td><td>19.33 (+3.33)</td></tr><tr><td>average</td><td>47.00</td><td>50.67 (+3.67)</td><td>38.34</td><td>40.67 (+2.33)</td></tr></table>

training dataset contains 41,000 video clips and the validation dataset contains 28,000 video clips. We randomly sample 16 frames from each video clip as the input. The image resolution is  $224 \times 224$ . We adopt the training code base in (Qinghong Lin et al., 2022). For all training experiments, we set the learning rate to  $3 \times 10^{-5}$  and the batch size to 66. The training takes three days on 5 A6000 GPUs with AdamW optimizer used.

# 4.2 EXPERIMENTAL RESULTS IN SIMULATORS

In this section, we verify that our finetuning strategy yields representation that improves the robot's imitation learning ability compared with directly using pretrained backbones in three simulation environments: Franka Kitchen, MetaWorld, and Adroit, shown in Fig. 4. In Kitchen and MetaWorld, the state is the raw perceptual input's embedding produced by the visual representation model. In Adroit, the state contains the proprioceptive state of the robot along with the observation embedding.

For R3M (Nair et al., 2022), we follow its evaluation procedure (Nair et al., 2022) to test our representation under the behavior cloning setting. We train an actor policy that maps a state to robot action over a total of 20,000 steps with the standard action prediction loss. The number of demonstrations used for training imitation policies in the three environments is 50, 25, and 100, respectively. During the evaluation process, we evaluate the policy every 1000 training steps and report the three best evaluation results from different visual views. The results are shown in Tab. 1. For EgoVLP and MVP, the number of demonstrations used for training imitation policies in the three environments is 10, 50, and 100, respectively. We evaluate policy every 5000 training steps and report the best result from different visual views. The results are shown in Tab. 2.

From Tab. 1 and Tab. 2, we observe that our fine-tuning strategy improves the policy success rate compared to directly using the backbones, indicating our method can help capture human-oriented and important representation for manipulation tasks.

# 4.3 ABLATION STUDY

In this section, we evaluate the success rate results with ablations on temporal-related tasks and spatial-related tasks to understand the benefits of inducing perceptual skills in the model and the necessity of different perceptual skills for different tasks. We use R3M as the base model and re-implement the training on the model with only time-related tasks and the model with only spatial-related tasks. We select five environments from Franka Kitchen, MetaWorld, and Adroit.

As shown in Tab. 3, in most environments, robotics require both spatial and temporal perceptual skills to enhance the representation of observations. However, in several environments, only one perceptual skill is sufficient, and the other may have a negative effect. In the 'door' environment, we believe that time information plays a leading role because capturing state changes over time can

Table 3: Ablation study about time-related tasks and spatial-related tasks.  

<table><tr><td>env</td><td>R3M</td><td>R3M+time</td><td>R3M+spatial</td><td>Ours(R3M+spatial+time)</td></tr><tr><td>micro</td><td>23.00</td><td>25.00</td><td>26.00</td><td>28.00</td></tr><tr><td>light</td><td>67.00</td><td>75.00</td><td>70.00</td><td>83.00</td></tr><tr><td>ldoor</td><td>41.00</td><td>46.00</td><td>23.00</td><td>32.00</td></tr><tr><td>assembly</td><td>84.00</td><td>84.67</td><td>83.33</td><td>92.67</td></tr><tr><td>relocate</td><td>36.67</td><td>37.33</td><td>40.00</td><td>36.67</td></tr></table>

Table 4: The OSCC and PNR task results on the Ego4D benchmark.  

<table><tr><td>Model</td><td>Video-Text Pretrained</td><td>OSCC ACC% (↑)</td><td>PNR ERR (seconds) (↓)</td></tr><tr><td>TimeSformer</td><td>Imagenet Init.</td><td>70.3</td><td>0.616</td></tr><tr><td>TimeSformer</td><td>EgoVLP</td><td>73.9</td><td>0.622</td></tr><tr><td>Ours</td><td>EgoVLP</td><td>76.3</td><td>0.616</td></tr></table>

be challenging. In the 'relocate' environment, spatial perception takes the lead as objects in the manipulation scene are readily apparent.

# 4.4 REAL-WORLD ROBOT EXPERIMENT

Dataset. We collect a Fanuc Manipulation dataset for robot behavior cloning, including 17 manipulation tasks and 450 expert demonstrations, as shown in Fig. 5. We employ a FANUC LRMate 200iD/7L robotic arm outfitted with an SMC gripper. The robot is manipulated using operational space velocity control. Demonstrations were collected via a human operator interface, which utilized a keyboard to control the robot's end effector. We established a set of seven key bindings to facilitate 3D translational, 3D rotational, and 1D gripper actions for robot control. During these demonstrations, we recorded camera images, robot joint angles, velocities, and expert actions.

In the training phase of behavior cloning, we concatenate the robot's joint angles with encoded image features to form the input state. Rather than directly imitating expert actions in the robot's op

![](images/6be6e96fe9d1dd148688dd238e23376ce1b2c3b0a49faebcd8443fa0dc1816c1.jpg)  
Figure 5: The distribution of our real-world robot dataset in a Fanuc robot, which covers many kinds of actions.

erational space (Nair et al., 2022), we opt to imitate the joint velocities derived from the collected joint trajectories. This approach allows for manipulation learning at a control frequency different from that of the human demonstrations, thereby offering flexibility in the network's inference time.

Fig. 6 presents experimental results for four representative tasks: pushing a box, closing a laptop, opening a drawer, and moving a cube to a specified location. During both training and evaluation, the robot arm's initial states and objects' initial states are randomized. We benchmark our approach against three existing methods: R3M, MVP, and EgoVLP. Our method outperforms most of these baselines across multiple tasks.

# 4.5 EVALUATION OF PERCEPTUAL TASKS ON EGO4D

To validate whether the multi-task network structure can capture task relationships and enhance computer vision representation, we employ our Task Fusion Decoder on the Ego4D Hand and Object Interactions benchmark. Due to label limitations, we re-implement our model using only time-related tasks, specifically OSCC and PNR. Subsequently, we evaluate the accuracy of object state change classification and temporal localization error in absolute seconds.

![](images/ff2887557ab05e1214d56e8dc1dd8a3743fbe7de4cf80960f0cce676795af4ff.jpg)  
(a) push box

![](images/25df0f596ba5dfe6d28550b45005e1563fc76c2143af9264454f6a77905ff5e6.jpg)  
(b) close laptop

![](images/385ffdb01c28546c28d90a041ecd74e6c14e11ec9645cc443fccb67a17621c6d.jpg)  
(c) open drawer

![](images/31e6e38803e1d775066efbfa811e2dc8fd0ccaf7a416e2e9ada540001ff1ba8f.jpg)  
(d) push cube

![](images/ba36c46de8e7380870d53c03aff95db3e6c94debdbbbd5c5e238f627653daed9.jpg)

![](images/015ba30c7244049b838db5e730f534aa459bc683956e213205af36b8aa541a31.jpg)

![](images/6b47ae54a91c6de0d17d7cc901f3bbcbaa47a885a479db44319a0114985ab6d7.jpg)

![](images/871c835a0b5e92828a5dced6dd73ffdb8711606f0631f0a04415795b9e0f83f9.jpg)

![](images/ffd28aadec7549c3524e9365899ebaab5e3bd9a56f00975442fd4d57e023eb2b.jpg)

![](images/df923cbf1241b83590c1be6845e7ef35be10cdf4c3a62df543128b96712d4735.jpg)

![](images/1f6d79c5782267ba99135a9f01616ee96d1dd912fe739ec8dbd014702f2ad461.jpg)

![](images/90888c7b9872d6870f123206651901bc0f4fb84e4f22cd5cfee6755096191ed1.jpg)

![](images/395325a3ffe5f4e4ec4cfea3c655771c8c808a801369e3dcdbbe693726746e9c.jpg)  
Figure 6: The result of our real robot experiments. The tasks are push the box, close the laptop, open the drawer, and push the cube from left to right.

![](images/bf7e063fcc210a2df992be5467734b2c9cc6e6f0cb058d2ac61ab2ed9fcafb7b.jpg)

![](images/4ee3d99855ed04bcff861d31deec353cb803377c245fa6d03dee52bcd0014777.jpg)

![](images/f01ca36a655fde0a337a614d7627e2fbf86418848fa8a3ad0417c5236e83d657.jpg)

From the results in Tab. 4, we observe that our model improves OSCC accuracy by  $2.4\%$  and reduces the PNR error by 0.006 seconds compared to the trained EgoVLP model. When compared to the ImageNet initialization model, our approach achieves a  $6\%$  improvement in OSCC accuracy while maintaining nearly identical PNR task performance. The strong result of these vision tasks verifies that our task fusion model can capture the task relationship hence making them benefit each other, showing effectiveness in learning a multi-task joint representation.

# 5 REPRESENTATION ANALYSIS

In this section, to demonstrate the effectiveness of our method, we first analyze the attention map in the manipulation scene to observe the impact of the spatial-related task. We then visualize the frame distribution at different times using a t-SNE figure (Van der Maaten & Hinton, 2008) to assess the effect of keyframe prediction.

# 5.1 ATTENTION MAP VISUALIZATION

The initial goal of the spatial-related task we designed is to capture the interaction between hands and manipulated objects and transfer it to the field of robotics manipulation. Therefore, we aim to demonstrate that our method places greater emphasis on the manipulation area while filtering out redundant information from the entire task area.

To validate our training strategy, we visualize the attention map of the last layer for R3M (ResNet) by Grad-CAM (Selvaraju et al., 2017). We separately visualize the attention maps for the original model, our fine-tuned model, and the ablative model, which includes only the time-related task, as shown in Fig. 7. We can see that: in both real robot scenes and simulation scenes, after the manipulation occurs, our method adjusts the representation to focus more on the action area, while the base model does not exhibit such an effect. Additionally, even with the time-related task, our method still cannot concentrate on the manipulation's local area, which confirms the effectiveness of the spatial-related task design in our network.

![](images/5a2c37ee78191f6d4c208a07237089bb5cfc53f25bc3f6783f0eb14650898fd4.jpg)

![](images/588dfb54842fc2098ffbfdf160817e9965578dd17f5f7e4d49b79da6c7a4dfa3.jpg)  
(a) Original

![](images/5fb7ae2ab1905686c3d4951db6ed954a60e30f3bc60fc8fa890818475b54aa3e.jpg)

![](images/0f1b3b8c751792683fe5a51fc75f593d8095117975592bb0197f3ed268bc81f0.jpg)  
(b) R3M

![](images/531a6a2fda8f7650c9b3558e0db619a080ef4a57471cc82eff0878b1d150ab57.jpg)

![](images/1135d11edfc20fc8681edb0eb2f5ec25f372acda92e5579bd692ae3a1943d215.jpg)  
(c) Ours w/o spatial

![](images/24b47188d5cc299878613ef03e3256bc6c9dce370fb862ed274f2ee3270687f5.jpg)

![](images/e9ad0def330544b229b65793d7c1c36ebdde9925e2f2e3266230184d07910705.jpg)  
Figure 7: The attention map visualization in different scenes. The pictures are original picture, base R3M model result and ours spatial ablation model result and ours model result from left to right.  
(d) Ours

# 5.2 T-SNE VISUALIZATION OF RPRESENTATIONS

In this section, we plot the t-SNE figure for the representations of the whole sequence of the manipulation task in four kitchen environments at the same time. Because we add OSCC and PNR tasks for the human pre-knowledge for the model, which can capture the state change and predict the state change frame, the model will change the distribution for the representations of a manipulation task sequence.

As shown in Fig. 8, we classify each action sequence into before manipulation action and after manipulation

![](images/48c286c826492b37116e331be75f40a649a613b42d1f358dea8bfffb3d0f9c88.jpg)  
Figure 8: Left: the t-SNE figure for R3M model Right: the t-SNE figure for our model. Our model has a stronger ability to capture the state change

action. In more tasks, our model can have a bigger gap for representation in temporal, and get a clearer relationship between before-action and after-action representations.

# 6 CONCLUSION AND DISCUSSION

In conclusion, this work introduces a novel paradigm in the field of robot representation learning, emphasizing the importance of human-oriented perceptual skills for achieving robust and generalizable visual representations. By leveraging the simultaneous acquisition of multiple simple perceptual skills critical to human-environment interactions, we propose a plug-and-play module Task Fusion Decoder, which acts as an embedding translator, guiding representation learning towards encoding meaningful structures for robotic manipulation. We demonstrate its versatility by improving the representation of various state-of-the-art visual encoders across a wide range of robotic tasks, both in simulation and real-world environments. Furthermore, we introduce a real-world dataset with expert demonstrations to support our findings.

Future work and broader impact. In the future, we will explore the incorporation of a feedback loop or reward function into a joint visual representation learning and policy learning framework. Our approach has no ethical or societal issues on its own, except those inherited from robot learning.

# REFERENCES

Minttu Alakuijala, Gabriel Dulac-Arnold, Julien Mairal, Jean Ponce, and Cordelia Schmid. Learning reward functions for robotic manipulation by observing humans. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 5006-5012. IEEE, 2023. 2  
Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, and Deepak Pathak. Affordances from human videos as a versatile representation for robotics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13778-13790, 2023. 3  
Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1728-1738, 2021. 3, 4  
Andrea Bajcsy, Dylan P Losey, Marcia K O'Malley, and Anca D Dragan. Learning from physical human corrections, one feature at a time. In Proceedings of the 2018 ACM/IEEE International Conference on Human-Robot Interaction, pp. 141-149, 2018. 1  
Andreea Bobu, Marius Wiggert, Claire Tomlin, and Anca D Dragan. Feature expansive reward learning: Rethinking human input. In Proceedings of the 2021 ACM/IEEE International Conference on Human-Robot Interaction, pp. 216-224, 2021. 3  
Andreea Bobu, Marius Wiggert, Claire Tomlin, and Anca D Dragan. Inducing structure in reward learning by learning features. The International Journal of Robotics Research, pp. 02783649221078031, 2022. 1, 3  
Andreea Bobu, Yi Liu, Rohin Shah, Daniel S Brown, and Anca D Dragan. Sirl: Similarity-based implicit representation learning. arXiv preprint arXiv:2301.00810, 2023a. 1, 3  
Andreea Bobu, Andi Peng, Pulkit Agrawal, Julie Shah, and Anca D Dragan. Aligning robot and human representations. arXiv preprint arXiv:2302.01928, 2023b. 3  
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2  
Daniel Brown, Russell Coleman, Ravi Srinivasan, and Scott Niekum. Safe imitation learning via fast bayesian reward inference from preferences. In International Conference on Machine Learning, pp. 1165-1177. PMLR, 2020. 3  
Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pp. 213-229. Springer, 2020. 5  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597-1607. PMLR, 2020. 1, 2  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Info-gan: Interpretable representation learning by information maximizing generative adversarial nets. Advances in neural information processing systems, 29, 2016. 1, 2  
Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European conference on computer vision (ECCV), pp. 720-736, 2018. 3  
Neha Das, Sarah Bechtle, Todor Davchev, Dinesh Jayaraman, Akshara Rai, and Franziska Meier. Model-based inverse reinforcement learning from visual demonstrations. In Conference on Robot Learning, pp. 1930-1942. PMLR, 2021. 1  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.3,4

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. "The" something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pp. 5842-5850, 2017. 3  
Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonio Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18995-19012, 2022. 2, 3, 4  
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman. Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning. arXiv preprint arXiv:1910.11956, 2019. 2  
Nicklas Hansen, Zhecheng Yuan, Yanjie Ze, Tongzhou Mu, Aravind Rajeswaran, Hao Su, Huazhe Xu, and Xiaolong Wang. On pre-training for visuo-motor control: Revisiting a learning-from-scratch baseline. arXiv preprint arXiv:2212.05749, 2022. 2  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016. 3, 4  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729-9738, 2020. 1, 2  
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dólár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000-16009, 2022. 1, 2  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. 2016. 1, 2  
Ya Jing, Xuelin Zhu, Xingbin Liu, Qie Sima, Taozheng Yang, Yunhai Feng, and Tao Kong. Exploring visual pre-training for robot manipulation: Datasets, models and methods. arXiv preprint arXiv:2308.03620, 2023. 2  
Gunnar Johansson. Visual perception of biological motion and a model for its analysis. Perception & psychophysics, 14(2):201-211, 1973. 1  
Siddharth Karamcheti, Suraj Nair, Annie S Chen, Thomas Kollar, Chelsea Finn, Dorsa Sadigh, and Percy Liang. Language-driven representation learning for robotics. arXiv preprint arXiv:2302.12766, 2023. 2  
Sydney M Katz, Amir Maleki, Erdem Biryik, and Mykel J Kochenderfer. Preference-based learning of reward function features. arXiv preprint arXiv:2103.02727, 2021. 3  
Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7482-7491, 2018. 5  
Apoory Khandelwal, Luca Weihs, Roozbeh Mottaghi, and Aniruddha Kembhavi. Simple but effective: Clip embeddings for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14829-14838, 2022. 2  
Natasha Z Kirkham, Jonathan A Slemmer, and Scott P Johnson. Visual statistical learning in infancy: Evidence for a domain general learning mechanism. Cognition, 83(2):B35-B42, 2002. 2  
Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83-97, 1955. 5  
Yann LeCun. A path towards autonomous machine intelligence. preprint posted on openreview, 2022.3

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030, 2022. 2  
Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Yecheng Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Pieter Abbeel, Jitendra Malik, et al. Where are we in the search for an artificial visual cortex for embodied intelligence? arXiv preprint arXiv:2303.18240, 2023. 3  
Yao Mu, Shunyu Yao, Mingyu Ding, Ping Luo, and Chuang Gan. Ec2: Emergent communication for embodied control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6704-6714, 2023a. 2  
Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. arXiv preprint arXiv:2305.15021, 2023b. 2  
Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022. 1, 2, 3, 6, 7  
Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, and Abhinav Gupta. The unsurprising effectiveness of pre-trained vision models for control. In International Conference on Machine Learning, pp. 17359-17371. PMLR, 2022. 2  
Kevin Qinghong Lin, Alex Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Zhong-cong Xu, Difei Gao, Rongcheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. arXiv e-prints, pp. arXiv-2206, 2022. 2, 3, 6  
Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087, 2017. 2  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211-252, 2015. 3  
Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pp. 618-626, 2017. 8  
Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel. Reinforcement learning with action-free pre-training from videos. In International Conference on Machine Learning, pp. 19561-19579. PMLR, 2022. 2  
Rutav Shah and Vikash Kumar. Rrl: Resnet as representation for reinforcement learning. arXiv preprint arXiv:2107.03380, 2021. 2  
Dandan Shan, Jiaqi Geng, Michelle Shu, and David F Fouhey. Understanding human hands in contact at internet scale. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9869-9878, 2020. 3  
Mohit Shridhar, Lucas Manuelli, and Dieter Fox. *Cliport: What and where pathways for robotic manipulation*. In *Conference on Robot Learning*, pp. 894–906. PMLR, 2022. 2  
Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008. 8  
Tete Xiao, Ilija Radosavovic, Trevor Darrell, and Jitendra Malik. Masked visual pre-training for motor control. arXiv preprint arXiv:2203.06173, 2022. 2, 3  
Jun Yamada, Karl Pertsch, Anisha Gunjal, and Joseph J Lim. Task-induced representation learning. arXiv preprint arXiv:2204.11827, 2022. 3

Lin Yen-Chen, Andy Zeng, Shuran Song, Phillip Isola, and Tsung-Yi Lin. Learning to see before learning to act: Visual pre-training for manipulation. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 7286-7293. IEEE, 2020. 2  
Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pp. 1094-1100. PMLR, 2020. 2  
Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3712-3722, 2018. 3  
Amir R Zamir, Alexander Sax, Nikhil Cheerla, Rohan Suri, Zhangjie Cao, Jitendra Malik, and Leonidas J Guibas. Robust learning through cross-task consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11197-11206, 2020. 3  
Yanjie Ze, Ge Yan, Yueh-Hua Wu, Annabella Macaluso, Yuying Ge, Jianglong Ye, Nicklas Hansen, Li Erran Li, and Xiaolong Wang. Gnfactor: Multi-task real robot learning with generalizable neural feature fields. arXiv preprint arXiv:2308.16891, 2023. 2
