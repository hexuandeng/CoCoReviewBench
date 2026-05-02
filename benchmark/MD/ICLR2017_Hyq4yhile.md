# LEARNING INVARIANT FEATURE SPACES TO TRANSFER SKILLS WITH REINFORCEMENT LEARNING

Abhishek Gupta†*, Coline Devin†*, YuXuan Liu†, Pieter Abbeel†‡, Sergey Levine†

† UC Berkeley, Department of Electrical Engineering and Computer Science ‡ OpenAI {abhigupta, coline, svlevine}@eecs.berkeley.edu {yuxuanliu}@berkeley.edu {pieter}@openai.com

# ABSTRACT

People can learn a wide range of tasks from their own experience, but can also learn from observing other creatures. This can accelerate acquisition of new skills even when the observed agent differs substantially from the learning agent in terms of morphology. In this paper, we examine how reinforcement learning algorithms can transfer knowledge between morphologically different agents (e.g., different robots). We introduce a problem formulation where two agents are tasked with learning multiple skills by sharing information. Our method uses the skills that were learned by both agents to train invariant feature spaces that can then be used to transfer other skills from one agent to another. The process of learning these invariant feature spaces can be viewed as a kind of "analogy making," or implicit learning of partial correspondences between two distinct domains. We evaluate our transfer learning algorithm in two simulated robotic manipulation skills, and illustrate that we can transfer knowledge between simulated robotic arms with different numbers of links, as well as simulated arms with different actuation mechanisms, where one robot is torque-driven while the other is tendon-driven.

# 1 INTRODUCTION

People can learn large repertoires of motor skills autonomously from their own experience. However, learning is accelerated substantially when the learner is allowed to observe another person performing the same skill. In fact, human infants learn faster when they observe adults performing a task, even when the adult performs the task differently from the child, and even when the adult performs the task incorrectly (Meltzoff, 1999). Clearly, we can accelerate our own skill learning by observing a novel behavior, even when that behavior is performed by an agent with different physical capabilities or differences in morphology. Furthermore, evidence in neuroscience suggests that the parts of the brain in monkeys that respond to the pose of the hand can quickly adapt to instead respond to the pose of the end-effector of a tool held in the hand (Umitla et al., 2008). This suggests that the brain learns an invariant feature space for the task (e.g., reaching with a tool) that is independent of the morphology of the limb performing that task. Mirror neurons also fire both when the animal performs a task and when it observes another animal performing it (Rizzolatti & Craighero, 2004; Ferrari et al., 2005). Can we enable robots and other autonomous agents to transfer knowledge from other agents with different morphologies by learning such invariant representations?

In robotics and reinforcement learning, prior works have considered building direct isomorphisms between state spaces, as discussed in Section 2. However, most of these methods require specific domain knowledge to determine how to form the mapping, or operate on simple, low-dimensional environments. For instance, Taylor et al. (2008) find a mapping between state spaces by searching through all possible pairings. Learning state-to-state isomorphisms involves an assumption that the two domains can be brought into correspondence, which may not be the case for morphologically

different agents. Some aspects of the skill may not be transferable at all, in which case they must be learned from scratch, but we would like to maximize the information transferred between the agents.

In this paper, we formulate this multi-agent transfer learning problem in a setting where two agents are learning multiple skills. Using the skills that have been already acquired by both agents, each agent can construct a mapping from their states into an invariant feature space. Each agent can then transfer a new skill from the other agent by projecting the executions of that skill into the invariant space, and tracking the corresponding features through its own actions. This provides a well-shaped reward function to the learner that allows it to imitate those aspects of the "teacher" agent that are invariant to differences in their morphology, while ignoring the parts of the state that cannot be imitated. Since the mapping from the state spaces of each agent into the invariant feature space might be complex and nonlinear, we use deep neural networks to represent the mappings, and we present an algorithm that can learn these mappings from the shared previously acquired skills.

The main contributions of our work are a formulation of the multi-skill transfer problem, a definition of the common feature space, and an algorithm that can be used to learn the maximally informative feature space for transfer between two agents (e.g., two robots with different morphologies). To evaluate the efficiency of this transfer process, we use a reinforcement learning algorithm to transfer skills from one agent to another through the invariant feature space. The agents we consider may differ in state-space, action-space, and dynamics. We evaluate our transfer learning method in two simulated robotic manipulation tasks, and illustrate that we can transfer knowledge between simulated robotic arms with different numbers of links, as well as simulated arms with different actuation mechanisms, where one robot is torque-driven while the other is tendon-driven.

# 2 RELATED WORK

Transfer learning has long been recognized as an important direction in robotics and reinforcement learning (Taylor & Stone (2009)). Konidaris & Barto (2006) learned value functions on subsets of the state representation that were shared between tasks, providing a shaping reward in the target task. Taylor et al. (2007) manually construct a function to map a  $Q$ -function from one Markov decision process (MDP) to another. Ammar & Taylor (2012) manually define a common feature space between the states of two MDPs, and use this feature space to learn a mapping between states.

More recent work in deep learning has also looked at transferring policies by reusing policy parameters between environments (Rusu et al., 2016a;b; Braylan et al., 2015; Daftry et al., 2016), using either regularization or novel neural network architectures, though this work has not looked at transfer between agents with structural differences in state, such as different dimensionalities. Our approach is largely orthogonal to policy transfer methods, since our aim is not to directly transfer a skill policy, which is typically impossible in the presence of substantial morphological differences, but rather to learn a shared feature space that can be used to transfer information about a skill that is shared across robots, while ignoring those aspects that are not shared. Our own recent work has looked at morphological differences in the context of multi-agent and multi-task learning (Devin et al., 2016), by reusing neural network components across agent/task combinations. In contrast to that work, which transferred components of policies, our present work aims to learn common feature spaces in situations where we have just two agents. We do not aim to transfer parts of policies themselves, but instead look at shared structure in the states visited by optimal policies, which can be viewed as a kind of analogy making across domains.

Learning invariant feature spaces has also been studied in the domain of computer vision as a mechanism for domain adaptation. Past work by Chopra et al. (2005) used Siamese networks to learn a feature space where paired images are brought close together and unpaired images and pushed apart. This enables a semantically meaningful metric space to be learned with only pairs as labels. Later work on domain adaptation by Tzeng et al. (2015) and Ganin et al. (2016) use an adversarial approach to learn an image embedding that is useful for classification and invariant to the input image's domain. We use the idea of learning a metric space from paired states, though the adversarial approach could also be used with our method as an alternative objective function in future work.

# 3 PROBLEM FORMULATION AND ASSUMPTIONS

We formalize our transfer problem in a general way by considering a source domain and a target domain, denoted  $D_{S}$  and  $D_{T}$ , which each correspond to Markov decision processes (MDPs)  $D_{S} = (\mathcal{S}_{S},\mathcal{A}_{S},T_{S},R_{S})$  and  $D_{T} = (\mathcal{S}_{T},\mathcal{A}_{T},T_{T},R_{T})$ , each with its own state space  $\mathcal{S}$ , action space  $\mathcal{A}$ , dynamics or transition function  $T$ , and reward function  $R$ . In general, the state and action spaces in the two domains might be completely different. Correspondingly, the dynamics  $T_{S}$  and  $T_{T}$  also differ, often dramatically. However, we assume that the reward functions share some structural similarity, in that the state distribution of an optimal policy in the source domain will resemble the state distribution of an optimal policy in the target domain when projected into some common feature space. For example, in one of our experimental tasks,  $D_{S}$  corresponds to a robotic arm with 3 links, while  $D_{T}$  is an arm with 4 links. While the dimensionalities of the states and action are completely different, the two arms are performing the same task, with a reward that depends on the position of the end-effector. Although this end-effector is a complex nonlinear function of the state, the reward is structurally similar for both agents.

# 3.1 COMMON FEATURE SPACES

We can formalize this common feature space assumption as following: if  $\pi_S(s_S)$  denotes the state distribution of the optimal policy in  $D_{S}$ , and  $\pi_T(s_T)$  denotes the state distribution of the optimal policy in  $D_{T}$ , it is possible to learn two functions,  $f$  and  $g$ , such that  $p(f(s_S)) = p(g(s_T))$  for  $s_S \sim \pi_S$  and  $s_T \sim \pi_T$ . That is, the images of  $\pi_S$  under  $f$  and  $\pi_T$  under  $g$  correspond to the same distribution. This assumption is trivially true if we allow lossy mappings  $f$  and  $g$  (e.g. if  $f(s_S) = g(s_T) = 0$  for all  $s_S$  and  $s_T$ ). However, the less information we lose in  $f$  and  $g$ , the more informative the shared feature will be for the purpose of transfer. So while we might not in general be able to fully recover  $\pi_T$  from the image of  $\pi_S$  under  $f$ , we can attempt to learn  $f$  and  $g$  to maximize the amount of information contained in the shared space.

# 3.2 LEARNING WITH MULTIPLE SKILLS

In order to learn the common feature space, we need examples from both domains. While both agents could in principle learn a common feature space through direct exploration, in this work we instead assume that the agents have prior knowledge about each other, in the form of other skills that they have both learned. This assumption is reasonable, since many practical use-cases of transfer involve two agents that already have competence in a range of simple settings, and wish to transfer the competence of one agent in a new setting to another one. For example, we might wish to transfer a particular cooking skill from one home robot to another one, in a setting where both robots have already learned some basic manipulation behaviors that can allow us to build a common feature space between the two robots. Humans similarly leverage their extensive prior knowledge to aid in transfer, by recognizing limbs and hands and understanding their function.

To formalize the setting where the two agents can perform multiple tasks, we divide the state space in each of the two domains into an agent-specific state  $s_r$  and a task-specific state  $s_{\mathrm{env}}$ . A similar partitioning of the state variables was previously discussed by Devin et al. (2016), and is closely related to the agent-space proposed by Konidaris (2006). For simplicity, we will consider a case where there are just two skills: one proxy skill that has been learned by both agents, and one test skill that has been learned by the source agent in the domain  $D_S$  and is currently being transferred to the target agent in domain  $D_T$ . We will use  $D_{Sp}$  and  $D_{Tp}$  to denote the proxy task domains for the source and target agents. We assume that  $D_S$  and  $D_{Sp}$  (and similarly  $D_T$  and  $D_{Tp}$ ) differ only in their reward functions and task-specific states, with the agent-specific state spaces  $\mathcal{S}_r$  and action spaces being the same between the proxy and test domains. For example  $D_{Sp}$  might correspond to a 3-link robot pushing an object, while  $D_S$  might correspond to the same robot opening a drawer, and  $D_{Tp}$  and  $D_T$  correspond to a completely different robot performing those tasks. Then, we can learn functions  $f$  and  $g$  on the robot-specific states of the proxy domains, and use them to transfer knowledge from  $D_S$  to  $D_T$ .

The idea in this setup is that both agents will have already learned the proxy task, and we can compare how they perform this task in order to determine the common feature space. This is a natural problem setup for many robotic transfer learning problems, as well as other domains where multiple

distinct agents might need to each learn a large collection of skills, exchanging their experience and learning which information they can and cannot transfer from each other. In a practical scenario, each robot might have already learned a large number of basic skills, some of which were learned by both robots. These skills are candidate proxy tasks that the robots can use to learn their shared space, which one robot can then use to transfer knowledge from the other one and more quickly learn skills that it does not yet possess.

# 4 LEARNING COMMON FEATURE SPACES FOR SKILL TRANSFER

In this section, we will discuss how the shared space can be learned by means of the proxy task. We will then describe how this shared space can be used for knowledge transfer for a new task, and finally present results that evaluate transfer on a set of simulated robotic control domains.

We wish to find functions  $f$  and  $g$  such that, for states  $s_{S}p$  and  $s_{T}p$  along the optimal policies  $\pi_{S}p^{*}$  and  $\pi_{T}p^{*}$ ,  $f$  and  $g$  approximately satisfy  $p(f(s_{Sp,r})) = p(g(s_{Tp,r}))$ . If we can find the common feature space by learning  $f$  and  $g$ , we can optimize  $\pi_{T}$  by directly mimicking the distribution over  $f(s_{Sp,r})$ , where  $s_{Sp,r} \sim \pi_{S}$ .

# 4.1 LEARNING THE EMBEDDING FUNCTIONS FROM A PROXY TASK

To approximate the requirement that  $p(f(s_{Sp,r})) = p(g(s_{Tp,r}))$ , we assume a weak pairing  $P$  of states in the proxy domains. As  $f$  and  $g$  are parametrized as neural networks, we can optimize them using the similarity loss metric introduced by Chopra et al. (2005):

$$
\mathcal {L} _ {\mathrm {s i m}} \left(s _ {S p}, s _ {T p}; \boldsymbol {\theta} _ {f}, \boldsymbol {\theta} _ {g}\right) = \sum_ {i, j \in P} | | f \left(s _ {S p, r} ^ {i}; \boldsymbol {\theta} _ {f}\right) - g \left(s _ {T p, r} ^ {i}; \boldsymbol {\theta} _ {g}\right) | | _ {2}.
$$

Where  $\theta_{f}$  and  $\theta_{g}$  are the function parameters.

However, as described in Section 3, if this is the only objective for learning  $f$  and  $g$ , we can easily end up with uninformative degenerate mappings, such as the one where  $f(s_{Sp,r}) = g(s_{Tp,r}) = 0$ . Intuitively, a good pair of mappings  $f$  and  $g$  would be as close as possible to being invertible, so as to preserve as much of the information about the source domain as possible. We therefore train a second pair of decoder networks with the goal of optimizing the quality of the reconstruction of  $s_{Sp,r}$  and  $s_{Tp,r}$  from the shared feature space, which encourages  $f$  and  $g$  to preserve the maximum amount of domain-invariant information. We define decoders  $\mathrm{Dec}_S(f(s_{Sp,r}))$  and  $\mathrm{Dec}_T(g(s_{Tp,r}))$  that map from the feature space back to their respective states. Note that, compared to conven-

![](images/e115a306c2dbfc7b822dbe3ad4f7d43f92feb9079c02485a5d576d26a4af0feb.jpg)  
Figure 1: The two embedding functions  $f$  and  $g$  are trained with a contrastive loss between the domains, along with decoders that optimize autoencoder losses.

tional Siamese network methods, the weights between  $f$  and  $g$  are not tied, and in general the networks have different dimensional inputs. The objectives for these are:

$$
\mathcal {L} _ {\mathrm {A E} _ {S}} \left(s _ {S p, r}; \boldsymbol {\theta} _ {f}, \boldsymbol {\theta} _ {\operatorname {D e c} _ {S}}\right) = \sum_ {i} | | s _ {S p, r} ^ {(i)} - \operatorname {D e c} _ {S} \left(f \left(s _ {S p, r} ^ {(i)}; \boldsymbol {\theta} _ {f}\right); \boldsymbol {\theta} _ {\operatorname {D e c} _ {S}}\right) | | _ {2},
$$

$$
\mathcal {L} _ {\mathrm {A E} _ {\mathrm {T}}} \left(s _ {T p, r}; \boldsymbol {\theta} _ {g}, \boldsymbol {\theta} _ {\mathrm {D e c} _ {T}}\right) = \sum_ {i} | | s _ {T p, r} ^ {(i)} - \mathrm {D e c} _ {T} \left(g \left(s _ {T p, r} ^ {(i)}; \boldsymbol {\theta} _ {g}\right); \boldsymbol {\theta} _ {\mathrm {D e c} _ {T}}\right) | | _ {2},
$$

where  $\theta_{\mathrm{Dec}_S}$  and  $\theta_{\mathrm{Dec}_T}$  are the decoder weights. We train the entire network end-to-end using backpropagation, where the full objective is

$$
\min  _ {\theta_ {f}, \theta_ {g}, \theta_ {\mathrm {D e c} _ {S}}, \theta_ {\mathrm {D e c} _ {T}}} \mathcal {L} _ {\mathrm {A E} _ {S}} (s _ {3, r}; \theta_ {f}, \theta_ {\mathrm {D e c} _ {S}}) + \mathcal {L} _ {\mathrm {A E} _ {\mathrm {T}}} (s _ {4, r}; \theta_ {g}, \theta_ {\mathrm {D e c} _ {T}}) + \mathcal {L} _ {\mathrm {s i m}} (s _ {3, r}, s _ {4, r}; \theta_ {f}, \theta_ {g})
$$

A diagram of this learning approach is shown in Figure 1. This procedure requires a reasonable estimate of the correspondences  $P$  for the contrastive loss. These correspondences could be obtain through an unsupervised alignment procedure or an EM-like algorithm, but in our method we take a simpler approach and exploit the fact that the skills we consider are episodic. In such episodic skills, a reasonable approximate alignment can be obtained by assuming that the two agents will perform each task at roughly the same rate, and we can therefore simply pair the states that are visited in the same time step in the two proxy domains.

# 4.1.1 USING THE COMMON EMBEDDING FOR KNOWLEDGE TRANSFER

The functions  $f$  and  $g$  learned using the approach described above establish an invariant space across the two domains. However, because these functions need not be invertible, directly mapping from a state in the source domain to a state in the target domain is not feasible.

Instead of attempting direct policy transfer, we match the distributions of optimal trajectories across the domains. Given  $f$  and  $g$  learned from the network described in Section 4, and the distribution  $\pi_S^*$  of optimal trajectories in the source domain, we can incentivize the distribution of trajectories in the target domain to be similar to the source domains under the mappings  $f$  and  $g$ . Ideally, we would like the distributions  $p(f(s_{S,r}))$  and  $p(g(s_{T,r}))$  to match as closely as possible. However, it may still be necessary for the target agent to learn some aspects of the skill from scratch, since not all intricacies will transfer in the presence of morphological differences. We therefore use a reinforcement learning algorithm to learn  $\pi_T$ , but with an additional term added to the reward function that provides guidance via  $f(s_{S,r})$ . This term has following form:

$$
r _ {\text {t r a n s f e r}} \left(s _ {T, r} ^ {(t)}\right) = \alpha | | f \left(s _ {S, r} ^ {(t)}; \theta_ {f}\right) - g \left(s _ {T, r} ^ {(t)}; \theta_ {g}\right) | | _ {2},
$$

where  $s_{S,r}^{(t)}$  is the agent-specific state along the optimal policy in the source domain at time step  $t$ , and  $s_{T,r}^{(t)}$  is the agent-specific state along the current policy that is being learned in the target domain at time step  $t$ , and  $\alpha$  is a weight on the transfer reward that controls its importance relative to the overall task goal. In essence, this additional reward provides a form of reward shaping, which gives additional learning guidance in the target domain. In sparse reward environments, task performance is highly dependent on directed exploration, and this additional incentive to match trajectory distributions in the embedding space provides strong guidance for task performance.

In tasks where the pairs mapping  $\mathcal{P}$  is imperfect, the transfer reward may sometimes interfere with learning when the target domain policy is already very good, though it is usually very helpful in the early stages of learning. We therefore might consider gradually reducing the weight  $\alpha$  as learning progresses in the target domain. We use this technique for our second experiment, which learns a policy for a tendon-driven arm.

![](images/267f88fb00bca528e9e40f8f6b72dfc6edbbad4ae41089c929554abb9483a981.jpg)  
Figure 2: The 3 and 4 link robots performing the button pressing task, which we use to evaluate the performance of our transfer method. Each task is trained on multiple conditions where the objects start in different locations.

# 5 EXPERIMENTS

Our experiments aim to evaluate how well common feature space learning can transfer skills between morphologically different agents. The experiments were performed in simulation using the MuJoCo physics simulator (Todorov et al., 2012), in order to explore a variety of different robots and actuation mechanisms. The embedding functions  $f$  and  $g$  in our experiments are 3 layer neural networks with 60 hidden units each and ReLu non-linearities. They are trained end-to-end with standard backpropagation using the ADAM opti

mizer (Kingma & Ba, 2015). Videos of our experiment will be available at https://sites.google.com/site/invariantfeaturetransfer/ For details of the reinforcement learning algorithm used, refer to Appendix A.

# 5.1 TRANSFER BETWEEN ROBOTS WITH DIFFERENT NUMBERS OF LINKS

In our first experiment, we evaluate our method on transferring information from a 3-link robot to a 4-link robot. These robots have similar size but different numbers of links and actuators, making the representation needed for transfer non-trivial to learn. In order to evaluate the effectiveness of our method, we consider tasks with sparse or delayed rewards, which are difficult to learn quickly without the use of prior knowledge, large amounts of experience, or a detailed shaping function to guide exploration. For transfer between the 3 link and 4 link robots, we evaluate our method on a button pressing task as shown in Figures 2 and 7. The goal of this task is to reach through a narrow opening and press the white button to the red goal marker indicated in the figure. The caveat is that the reward signal tells the arms nothing about where the button is, but only penalizes

distance between the white button and the red goal. Prior work has generally used well-shaped reward functions for tasks of this type, with terms that reward the arm for approaching the object of interest (Lillicrap et al., 2015; Devin et al., 2016). Without the presence of a directed reward shaping guiding the arm towards the button, it is very difficult for the task to be performed at all in the target domain, as seen from the performance of learning from scratch ("baseline") in the target domain in Figure 5. This is indicative of how such a task might be learned in the real world, where it is hard to provide anything but very sparse feedback by using a sensor on the button.

![](images/089524ea7ccddb58183260731f746821f77fffa750c16f068163632861f67c87.jpg)  
Figure 3: The 4-link robot pushing the button. Note that the reward function only tells the agent how far the button has been depressed, and provides no information to indicate that the arm should reach for the button.

![](images/574b77a6b886a68cca36b87d4e1801ac5a1d61a1f18a371b948e5a6324e8e6df.jpg)

![](images/5da9cf33615ad4d82b4c7e53aa7542ff13e54ff7855190220d835e3334477843.jpg)

![](images/3082d813ed7b2ab58d54d50eed862cbb598caa0f8d71e2579ad4c7201348e76c.jpg)

![](images/a09ab97efd71d6e8f30cc5bb837211c21d4af001e2f608dbaa487668d5c49193.jpg)

![](images/b0ff8d73c51aee051eb648b7100b2bd5f047de9d48c024fb5ee7ffe2a8dfa3bf.jpg)

![](images/a8f6465f8498abf338a9472acbd863369fb55713ef6dcfc6a958b679075706d0.jpg)

![](images/b572a7ca633644f33523d58fb002f10fc8bcac6fc2f0d15122e75353c8d41496.jpg)

For this experiment, we compare the quality of transfer when using different proxy tasks: reaching a target, moving a white block to the red goal, and inserting a peg into a slot near the robot, as shown in Figure 4. These tasks are significantly easier than the sparse reward button pressing task. Collecting successful trajectories from the proxy task, we train the functions  $f$  and  $g$  as described in Section 4. Note that the state in both robots is just the joint angles and joint velocities. Learning a suitable common feature space therefore requires the networks to understand how to map from joint angles to end-effectors for both robots.

![](images/0a9af6b746de3a3c91e232dcb303442a4e12379f5fa7ad610f5e6b010f746bf8.jpg)  
Figure 4: The 3 and 4 link robots performing each of the three proxy tasks we consider: target reaching, peg insertion, and block moving. Our results indicate that using all three proxy tasks to learn the common feature space improves performance over any single proxy task.

![](images/db65128705ab15b03a0c9918bd28dd164a35cc1567eb331773d6f6e3f817a7dd.jpg)

![](images/b2f063f4b9c70132e7964b431077fe49bd0d7a65998cc66e2c2d7e61860e8061.jpg)

We consider the 3-link robot pressing the button as the source domain and the 4-link robot pressing the button as the target domain. We allow the domain with the 3-link robot to have a well-shaped cost function which has 2 terms: one for bringing the arm close to the button, and one for the distance of the button from the red goal position. The performance of our method is shown in Figure 5. We see that the agent trained with our method performs more directed exploration and actually succeeds at learning the task, as compared to the baseline of learning from scratch.

![](images/09906178ae0d1ef9b1fd9057e958abce85d9ae03fe99d75ecfdc2c267a7e01e8.jpg)  
Figure 5: Performance of 4-link arm on the sparse reward button pushing task described in Section 5.1. The baseline is learning without any transfer reward; it does not learn to perform the task. The "peg," "push," and "reach" proxy ablations indicate the performance when using embedding functions learned from those proxy tasks. The embedding improves significantly when learned from all three proxy tasks, indicating that our method benefits from additional prior experience.

![](images/28bd05293614b6453bad782306b65604bf03a1973ddfec001a157bfb4d966c0d.jpg)  
Figure 7: The tendon-driven robot pulling the block. Note that the reward function only tells the agent how far the block is from the red goal and provides no information to indicate that the arm should reach around the block in order to pull it. The block is restricted to move only towards the red goal, but the agent needs to move under and around the block to pull it.

# 5.2 TRANSFER BETWEEN TORQUE CONTROLLED AND TENDON CONTROLLED MANIPULATORS

In order to illustrate the ability of our method to transfer across vastly different actuation mechanisms and learn representations that are hard to specify by hand, we consider transfer between a torque driven arm and a tendon driven arm, both with 3 links. These arms are pictured in Figure 6. The torque driven arm has motors at each of its joints that directly control its motion, and the state includes joint angles and joint velocities. The tendon driven arm, illustrated in Figure 6, uses three tendons to actuate the joints. The first tendon spans both the shoulder and the elbow, while the second and third control the elbow and wrist individually. The last tendon has a variable-length lever arm, while the first two have fixed-length lever arms, corresponding to tendons that conform to the arm as it bends. This coupled system uses tendon lengths and tendon veloc

![](images/74cca165ca00355de4621c76178feaf19f15090595b762459f8c118e34353b30.jpg)  
Figure 6: The top images show the source and target domain robots: the robot on the left is torque driven at the joints and the one on the right is tendon driven. The tendons are highlighted in the image; the green tendon has a variable-length lever arm, while the yellow tendons have fixed-length lever arms. Note that the first tendon couples two joints. The bottom images show two variations of the test task.

ities as the state representation, without direct access to joint angles or end-effector positions.

The state representations of the two robots are dramatically different, both in terms of units, dimensionality, and semantics. Therefore, learning a suitable common feature space represents a considerable challenge. In our evaluation, the torque driven arm is the source robot, and the tendon driven arm is the target robot. The task we require both robots to perform is a block pulling task indicated in Figure 6. This involves pulling a block in the direction indicated, which is non-trivial because it requires moving the arm under and around the block, which is restricted to only move in the directions indicated in Figure 6. With random exploration, the target robot is unable to perform directed exploration to get the arm to actually pull the block in the desired direction, as shown in Figure 8.

We use one proxy task in the experiment, which involves both arms reaching to various locations. With embedding functions  $f$  and  $g$  trained on optimal trajectories from the proxy task, we see that the transfer reward from our method enables the task to actually be performed with a tendon driven arm. The baseline, which again corresponds to attempting to learn the task with the target tendon-driven arm from scratch, fails completely. These results indicate that learning the common feature subspace can enable substantially accelerated learning in the target domain, and in fact can allow the target agent to learn a task that it fails to learn without any transfer rewards.

![](images/4225e7efd9bd3c9fc36501055923eb8c6ea46ad1a496caca41220ed06617fe86.jpg)  
Figure 8: Performance of tendon-controlled arm on block pulling task. While the environment's reward is too sparse to succeed at the task in a reasonable time, using our method to match feature space state distributions enables effective faster learning.

# 6 TRANSFER THROUGH IMAGE FEATURES

A compelling use-case for learned common embeddings is in learning vision-based policies. In this experimental setup, we evaluate our method on learning embeddings from raw pixels instead of from robot state. Enabling transfer from extra high dimensional inputs like images would allow significantly more natural transfer across a variety of robots without restrictive assumptions about full state information.

We evaluate our method on transfer across a 3-link and a 4-link robot as in Section 5.1, but use images instead of state. Because images from the source and target domains are the same size and the same "type", we let  $g = f$ . We

![](images/cce7bbc46fc8967f2c085955d7e72afeb95aa643d6588c5679c2bc63b1f05c2c.jpg)  
Figure 9: Performance of 4-link robot on block pushing task for transfer using raw images. We transfer from the 3-link robot by learning a feature space from raw pixels of both domains, enabling effective faster learning. The baseline is unable to succeed because of the reward signal is too sparse without transfer.

parametrize  $f$  as 3 convolutional layers with 5x5 filters and no pooling. A spatial softmax (Levine et al., 2016) is applied to the output of the third layer such that  $f$  outputs normalized pixel indices of feature points on the image. These "feature points" form the latent representation that we compare across domains. Intuitively the common "feature points" embeddings should represent parts of the robots which are common across different robots.

Embeddings between the domains are built using a proxy task of reaching to a point, similar to the one described in the previous experiments. The test task in this case is to push a white block to a red target as shown in Figure 10, which suffers from sparse rewards because the reward only accounts for the distance of the block from the goal. Unless the robot knows that it has to touch the block, it receives no reward and has unguided exploration. As shown in Figure 9, our method is able to transfer meaningful information from source to target robot directly from raw images and successfully perform the task even in the presence of sparse rewards.

![](images/29cda3c53b1a598590724d9e6569f6bba33eaedc1e418d9bf4baf9b81c2d0e43.jpg)  
Figure 10: The 3-link robot demonstrating the task. The yellow triangles mark the locations of the feature points output by  $f$  applied to the image pixels. We then use the feature points to transfer the skill to the 4-link robot.

![](images/a4461c2031df9956ae54e27768bec65fe97ae519cd53374af8900a9b8201562f.jpg)

![](images/81bc320f0b3e894d818cc99801b5861cf63e4940f4118d52e1cf34fc5e991923.jpg)

# 7 DISCUSSION AND FUTURE WORK

We presented a method for transferring skills between morphologically different agents using invariant feature spaces. The formulation of our transfer problem corresponds to a setting where two agents (e.g. two different robots) have each learned a collection of skills, with some skills known to just one of the agents, and some shared by both. A shared skill can be used to learn a space that implicitly brings the agents into correspondence, without assuming that an explicit state space isomorphism can be constructed. By then mapping into this space a skill that is known to only one of the agents, the other agent can substantially accelerate its learning of this skill by transferring the shared structure. We present an algorithm for learning the shared feature spaces using a shared proxy task, and experimentally illustrate that we can use this method to transfer manipulation skills between different simulated robotic arms. Our experiments include transfer between arms with different numbers of links, as well as transfer from a torque-driven arm to a tendon-driven arm.

A promising direction for future work is to explicitly handle situations where the two (or more) agents must transfer new skills by using a large collection of prior behaviors, with different degrees of similarity between the agents. In this case, constructing a shared feature space involves not only mapping the skills into a single space, but deciding which skills should or should not be combined. For example, a wheeled robot might share manipulation strategies with a legged robot, but should not attempt to share locomotion behaviors.

In a large-scale lifelong learning domain with many agents and many skills, we could also consider using our approach to gradually construct more and more detailed common feature spaces by transferring a skill from one agent to another, using that new skill to build a better common feature space, and then using this improved feature space to transfer more skills. Automatically choosing which skills to transfer when in order to minimize the training time of an entire skill repertoire is an interesting and exciting direction for future work.

# REFERENCES

Haitham Bou Ammar and Matthew E. Taylor. Reinforcement learning transfer via common subspaces. In Adaptive and Learning Agents: International Workshop, 2012.  
Alexander Braylan, Mark Hollenbeck, Elliot Meyerson, and Risto Miikkulainen. Reuse of neural modules for general video game playing. CoRR, abs/1512.01537, 2015.  
Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on, volume 1, pp. 539-546. IEEE, 2005.  
Shreyansh Daftry, J. Andrew Bagnell, and Martial Hebert. Learning transferable policies for monocular reactive MAV control. In International Symposium on Experimental Robotics (ISER), 2016.  
Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, and Sergey Levine. Learning modular neural network policies for multi-task and multi-robot transfer. arXiv preprint arXiv:1609.07088, 2016.  
P. F. Ferrari, S. Rozzi, and L. Fogassi. Mirror neurons responding to observation of actions made with tools in monkey ventral premotor cortex. Journal of Cognitive Neuroscience, 17(2), 2005.  
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Francois Laviolette, Mario Marchand, and Victor Lempitsky. Journal of Machine Learning Research, 17, 2016.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.  
George Konidaris. A framework for transfer in reinforcement learning. In ICML-06 Workshop on Structural Knowledge Transfer for Machine Learning, 2006.  
George Konidaris and Andrew Barto. Autonomous shaping: knowledge transfer in reinforcement learning. In International Conference on Machine Learning (ICML), pp. 489-496, 2006.  
Sergey Levine and Pieter Abbeel. Learning neural network policies with guided policy search under unknown dynamics. In Advances in Neural Information Processing Systems, 2014.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research, 17:1-40, 2016.  
Weiwei Li and Emanuel Todorov. Iterative linear quadratic regulator design for nonlinear biological movement systems. In ICINCO (1), 2004.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. CoRR, abs/1509.02971, 2015.  
Andrew Meltzoff. Born to learn: What infants learn from watching us. Skillman, NJ: Pediatric Institute Publication, 1999.

Giacomo Rizzolatti and Laila Craighero. The mirror neuron system. Annual Review of Neuroscience, 27:169-192, 2004.  
Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. CoRR, abs/1606.04671, 2016a.  
Andrei A Rusu, Matej Vecerik, Thomas Rothörl, Nicolas Heess, Razvan Pascanu, and Raia Hadsell. Sim-to-real robot learning from pixels with progressive nets. arXiv preprint arXiv:1610.04286, 2016b.  
Matthew Taylor, Peter Stone, and Yaxin Liu. Transfer learning via inter-task mappings for temporal difference learning. Journal of Machine Learning Research, 8(1):2125-2167, 2007.  
Matthew E. Taylor and Peter Stone. Transfer learning for reinforcement learning domains: A survey. Journal of Machine Learning Research, 10:1633-1685, 2009.  
Matthew E. Taylor, Nicholas K. Jong, and Peter Stone. Transferring instances for model-based reinforcement learning. In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), 2008.  
E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. In International Conference on Intelligent Robots and Systems (IROS), 2012.  
Eric Tzeng, Judy Hoffman, Trevor Darrell, and Kate Saenko. Simultaneous deep transfer across domains and tasks. In International Conference in Computer Vision (ICCV), 2015.  
M. A. Umitla, L. Escola, I. Intskirveli, F. Grammont, M. Rochat, F. Caruana, A. Jezzini, V. Gallese, and G. Rizzolatti. When piers become fingers in the monkey motor system. Proceedings of the National Academy of Sciences, 105(6):2209-2213, 2008.
