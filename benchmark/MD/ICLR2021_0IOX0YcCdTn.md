# BUTLER: BUILDING UNDERSTANDING IN TEXTWORLD VIA LANGUAGE FOR EMBodied REASONING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Given a simple request (e.g., Put a washed apple in the kitchen fridge), humans can reason in purely abstract terms by imagining action sequences and scoring their likelihood of success, prototypicality, and efficiency, all without moving a muscle. Once we see the kitchen in question, we can update our abstract plans to fit the scene. Embodied agents require the same abilities, but existing work does not yet provide the infrastructure necessary for both reasoning abstractly and executing concretely. We address this limitation by introducing ALFWorld, a simulator that enables agents to learn abstract, text-based policies in TextWorld (Côté et al., 2018) and then execute goals from the ALFRED benchmark (Shridhar et al., 2020) in a rich visual environment. ALFWorld enables the creation of a new BUTLER agent whose abstract knowledge, learned in TextWorld, corresponds directly to concrete, visually grounded actions. In turn, as we demonstrate empirically, this fosters better agent generalization than training only in the visually grounded environment. BUTLER's simple, modular design factors the problem to allow researchers to focus on models for improving every piece of the pipeline (language understanding, planning, navigation, visual scene understanding, and so forth).

# 1 INTRODUCTION

Consider helping a friend prepare dinner in an unfamiliar house: when your friend asks you to clean and slice an apple for an appetizer, how would you approach the task? Intuitively, one could reason abstractly: (1) find an apple (2) wash the apple in the sink (3) put the clean apple on the cutting board (4) find a knife (5) slice the apple with the knife (6) put the slices in a bowl. Even in an unfamiliar setting, abstract reasoning can help accomplish the goal by leveraging semantic priors. Priors like locations of objects - apples are commonly found in the kitchen, as are implements for cleaning and slicing, object affordances - a sink is useful for washing an apple, a refrigerator is not, pre-conditions - better to wash an apple before slicing it, rather than the converse. We hypothesize that, learning to solve tasks using abstract language, unconstrained by the particulars of the physical world, enables agents to complete embodied tasks in novel physical environments by leveraging the kinds of semantic priors that are exposed by abstraction.

![](images/d94f0dfff63f25214522c17092d85dc65e3d9d3174e19e37a6e568b616f2a856.jpg)  
Welcome!

You are in the middle of the room. Looking around you, you see a diningtable, a stove, a microwave, and a cabinet.

Your task is to: Put a pan on the dinetable.

> goto the cabinet

You arrive at the cabinet. The cabinet is closed.

> open the cabinet

The cabinet is empty.

> goto the stove

You arrive at the stove. Near the stove, you see a pan, a pot, a bread loaf, a lettuce, and a winebottle.

> take the pan from the stove

You take the pan from the stove.

> goto the diningtable

You arrive at the diningtable.

> put the pan on the diningtable

you put the pan on the diningtable.

![](images/712a91ce92e3893de4b79da738a9fe7bb49f2ad4b18878b6a58e4c108c8d0ef5.jpg)

![](images/6be109915cc2a5e3a49ebee692bc9147e2041b4db37981594ac1637b5ea300f3.jpg)

![](images/182b7ca55b51af58286683264f17ba1b47113be7737404f643a517e0c89646ac.jpg)

![](images/922292d31a11591e080d73a07acc1b09ff6f280cb18a5d0a2e630c06828d05a3.jpg)

![](images/7c342ba50ebaafdae9625db084efdc6fd72debfae68e032c7e9b6fa698de459c.jpg)  
Figure 1: ALFWorld: Interactive aligned text and embodied worlds. An example with high-level text actions (left) and low-level physical actions (right).

To test this hypothesis, we have created the novel ALFWorld framework, the first interactive, parallel environment that aligns text descriptions and commands with physically embodied robotic simulation. We build ALFWorld by extending two prior works: TextWorld (Côté et al., 2018) - an engine for interactive text-based games, and ALFRED (Shridhar et al., 2020) - a large scale dataset for vision-language instruction following in embodied environments. ALFWorld provides two views of the same underlying world and two modes by which to interact with it: TextWorld, an abstract, text-based environment, generates textual observations of the world and responds to high-level text actions; ALFRED, the embodied simulator, renders the world in high-dimensional images and responds to low-level physical actions as from a robot (Figure 1). Unlike prior work on instruction following (MacMahon et al., 2006; Anderson et al., 2018a), which typically uses a fixed corpus of cross-modal expert demonstrations, we argue that aligned parallel environments like ALFWorld offer a distinct advantage: they allow agents to explore, interact, and learn in the abstract environment of language before encountering the complexities of the embodied environment.

While fields such as robotic control use simulators like MuJoCo (Todorov et al., 2012) to provide infinite data through interaction, there has been no analogous mechanism – short of hiring a human around the clock – for providing linguistic feedback and annotations to an embodied agent. TextWorld addresses this discrepancy by providing programmatic and aligned linguistic signals during agent exploration. This facilitates the first work, to our knowledge, in which an embodied agent learns the meaning of complex multi-step policies, expressed in language, directly through interaction.

Empowered by the ALFWorld framework, we introduce BUTLER, an agent that first learns to perform abstract tasks in TextWorld using Imitation Learning (IL) and then transfers the learned policies to embodied tasks in ALFRED. When operating in the embodied world, BUTLER leverages the abstract understanding gained from TextWorld to generate text-based actions; these serve as high-level subgoals that facilitate physical action generation by a low-level controller. Broadly, we find that BUTLER is capable of generalizing in a zero-shot manner from TextWorld to unseen embodied tasks and settings. Our results show that training first in the abstract text-based environment is not only  $7 \times$  faster, but also yields better performance than training from scratch in the embodied world. These results lend credibility to the hypothesis that solving abstract language-based tasks can help build priors that enable agents to generalize to unfamiliar embodied environments.

Our contributions are as follows:

$\S 2$  ALFWorld environment: The first parallel interactive text-based and embodied environment.  
§ 3 BUTLER architecture: An agent that learns high-level policies in language that transfer to low-level embodied executions, and whose modular components can be independently upgraded.  
§ 4 Generalization: We demonstrate empirically that BUTLER, trained in the abstract text domain, generalizes better to the embodied setting than agents trained from corpora of demonstrations or from scratch in the embodied world.

# 2 ALIGNING ALFRED AND TEXTWORLD

The ALFRED dataset (Shridhar et al., 2020), set in the THOR simulator (Kolve et al., 2017), is a benchmark for learning to complete embodied household tasks using natural language instructions and egocentric visual observations. ALFRED involves a wide variety of 3D interactive environments and compositional tasks. As shown in Figure 1 (right), ALFRED tasks pose challenging interaction and navigation problems to an agent in a high-fidelity simulated environment. Tasks come annotated with a goal instruction that describes the objective (e.g., "put a pan on the dining table"). The dataset provides both template-based and human-annotated goals (see Appendix E). Agents observe the

world through high-dimensional pixel images and interact using low-level action primitives: MOVEAHEAD, ROTATELEFT/RIGHT, LOOKUP/DOWN, PICKUP, PUT, OPEN, CLOSE, and TOGGLEON/OFF.

Table 1: Six ALFRED task types with heldout seen and unseen evaluation sets.  

<table><tr><td>task-type</td><td># train</td><td># seen</td><td># unseen</td></tr><tr><td>Pick &amp; Place</td><td>790</td><td>35</td><td>24</td></tr><tr><td>Examine in Light</td><td>308</td><td>13</td><td>18</td></tr><tr><td>Clean &amp; Place</td><td>650</td><td>27</td><td>31</td></tr><tr><td>Heat &amp; Place</td><td>459</td><td>16</td><td>23</td></tr><tr><td>Cool &amp; Place</td><td>533</td><td>25</td><td>21</td></tr><tr><td>Pick Two &amp; Place</td><td>813</td><td>24</td><td>17</td></tr><tr><td>All</td><td>3,553</td><td>140</td><td>134</td></tr></table>

While ALFRED also provides low-level step-by-step language instructions on how to complete a particular goal, we tackle the challenge of completing tasks with only high-level goal descriptions. This task is harder than the instruction-following challenge posed in ALFRED, since the agent begins without any information about object locations or a sequential plan for solving the task.

Our aligned ALFWorld framework adopts six ALFRED task-types (Table 1) of various difficulty levels. These typically involve first finding a particular object, which often requires the agent to open and search receptacles like drawers or cabinets. Subsequently, all tasks other than Pick & Place require some interaction with the object such as heating (place object in microwave and start it) or cleaning (wash object in a sink). To conclude, the object must be placed in the designated location.

Within each task category there is significant variation: the embodied environment includes 120 rooms (30 kitchens, 30 bedrooms, 30 bathrooms, 30 living rooms), each dynamically populated with a set of portable objects (e.g., apple, mug), and static receptacles (e.g., microwave, fridge). For each task type we construct a larger train set, as well as seen and unseen validation evaluation sets: (1): seen consists of known task tuples {task-type, object, receptacle, room} in rooms seen during training, but with different instantiations of object locations, quantities, and visual appearances (e.g. two blue pencils on a shelf instead of three red pencils in a drawer seen in training). (2): unseen consists of new task tuples with known or unknown object-receptacle pairs, but always in an unseen room with different receptacles and scene layouts than in training tasks.

The seen set is designed to measure in-distribution generalization, whereas the unseen set measures out-of-distribution generalization. The scenes in ALFRED are visually diverse, so even the same task tuple can lead to very distinct tasks, e.g., involving differently colored apples, shaped statues, or textured cabinets. For this reason, purely vision-based agents often struggle to generalize to unseen environments and objects (see unimodal baselines in Section 5).

The TextWorld framework (Côté et al., 2018) procedurally generates text-based environments for training and evaluating language-based agents. We extend TextWorld to create text-based analogs of each ALFRED environment. Aligning text and embodied environments necessitates a common latent structure representing the state of the simulated world. ALFWorld uses PDDL - Planning Domain Definition Language (McDermott et al., 1998) to describe each scene from ALFRED and to construct an equivalent text game using the TextWorld engine. The dynamics of each game are defined by the PDDL domain (see Appendix C for additional details). We generate text that serves as a stand-in for visual observations by filling templates sampled from a context-sensitive grammar designed for the ALFRED environments. For interaction, TextWorld environments use the following high-level actions:

```txt
goto{recep} take{obj}from{recep} put{obj}in/on{recep}   
open{recep} close{recep} toggle{obj}/{recep}   
clean{obj}with{recep} heat{obj}with{recep} cool{obj}with{recep}
```

where  $\{\mathrm{obj}\}$  and  $\{\mathrm{recep}\}$  correspond to objects and receptacles. Note that heat, cool, clean, and goto are high-level actions that correspond to several low-level embodied actions.

Since TextWorld is an abstract representation of the world, transferring a TextWorld-trained agent to an embodied setting involves dealing with some domain gaps. For example, it is not possible to place objects inside a receptacle that is already full. Similarly, the physical size of objects and receptacles must be respected – it is not possible to put a large pot inside the microwave. The agent is also subject to visual challenges like occluded objects, misdetections, and inaccurate object relations.

# 3 INTRODUCING BUTLER: AN EMBodied MULTI-TASK AGENT

We investigate learning in the abstract language modality before generalizing to the embodied setting. This requires an agent capable of spanning both modalities. BUTLER uses three components: BUTLER::BRAIN - the abstract text agent, BUTLER::VISION - the language state estimator, and BUTLER::BODY - the low-level controller. An overview of BUTLER is shown in Figure 2.

![](images/c84a7f8fb14adff2492f5823ff35d12fb1ac22b63b730c9d13704aa4157ee52d.jpg)  
Figure 2: BUTLER Agent consists of three modular components. 1) BUTLER::BRAIN: a text agent pre-trained with the TextWorld engine (indicated by the dashed yellow box) which simulates an abstract textual equivalent of the embodied world. It is then fine-tuned or directly evaluated on new embodied tasks. 2) BUTLER::VISION: a state estimator that translates, at each time step, the visual frame  $v_{t}$  from the embodied world into a textual observation  $o_{t}$  using a pre-trained Mask R-CNN detector. The text agent uses the current observation  $o_{t}$ , the initial observation  $o_{0}$ , and the task goal  $g$  to predict the next high-level action  $a_{t}$ . 3) BUTLER::BODY: a controller that translates the high-level action  $a_{t}$  into a sequence of low-level actions in the embodied environment.

# 3.1 BUTLER::BRAIN (TEXT AGENT):  $o_0, o_t, g \to a_t$

BUTLER::BRAIN is a novel text-based game agent that generates high-level text actions in a token-by-token fashion akin to Natural Language Generation (NLG) approaches for dialogue (Sharma et al., 2017) and summarization (Gehrmann et al., 2018). An overview of the agent's architecture is shown in Figure 3. At game step  $t$ , the encoder takes the initial text observation  $o_0$ , current observation  $o_t$ , and the goal description  $g$  as input and generates a context-

![](images/6c0662bb088a3b19d93729840112ea004ce53255e11e5d58b4e073b9f9e22fe4.jpg)  
Figure 3: BUTLER::BRAIN: The text agent takes the initial/current observations  $o_0 / o_t$ , and goal  $g$  to generate a textual action  $a_t$  token-by-token.

aware representation of the current observable game state. Here  $o_0$  explicitly lists all the navigable receptacles in the scene. Since games are partially observable, the agent only has access to the observation describing the effects of its previous action and its present location. Therefore, we incorporate two memory mechanisms to imbue the agent with history: (1) a recurrent aggregator, adapted from Yuan et al. (2018), combines the encoded state with recurrent state  $h_{t-1}$  from the previous game step; (2) an observation queue feeds in the  $k$  most recent, unique textual observations. The decoder generates an action sentence  $a_t$  token-by-token to interact with the game. The encoder and decoder are based on a Transformer Seq2Seq model with pointer softmax mechanism (Gulcehre et al., 2016). We leverage pre-trained BERT embeddings (Sanh et al., 2019), and tie output embeddings with input embeddings (Press and Wolf, 2016). The agent is trained in an imitation learning setting with DAgger (Ross et al., 2011) using expert demonstrations. See Appendix A for complete details.

When playing a game, an agent might get stuck at certain states due to various failures (e.g., action is grammatically incorrect, wrong object name). The observation for a failed action does not contain any useful feedback, so a fully deterministic model tends to produce the same (wrong) action repeatedly. Since our decoder generates token-by-token and does not rely on templates, BUTLER::BRAIN is fully capable of leveraging search heuristics such as Beam Search (Reddy et al., 1977). During evaluation, BUTLER::BRAIN uses Beam Search to generate alternative action sentences in the event of a failed action, but otherwise greedily picks a sequence of best words.

# 3.2 BUTLER::VISION (STATE ESTIMATOR):  $v_{t} \rightarrow o_{t}$

At test time, agents in the embodied world must operate purely from visual input without any PDDL-based scene descriptions. To this end, BUTLER::VISION's language state estimator functions as a captioning module that translates visual observations  $v_{t}$  into textual descriptions  $o_{t}$ .

Table 2: Generalization within TextWorld environments: We independently train BUTLER::BRAIN on each type of TextWorld task and evaluate on heldout scenes of the same type. Respectively, tn/sn/un indicate success rate on train/seen/unseen tasks. All sn and un scores are computed using the random seeds (from 8 in total) producing the best final training score on each task type. BUTLER is trained with DAgger and performs beam search during evaluation. Without beam search, BUTLER $_g$  decodes actions greedily and gets stuck repeating failed actions. Further removing DAgger and training the model in a Seq2Seq fashion leads to worse generalization. Note that tn scores for BUTLER are lower than sn and un as they were computed without beam search.  

<table><tr><td rowspan="2"></td><td colspan="3">Pick &amp; Place</td><td colspan="3">Examine in Light</td><td colspan="3">Clean &amp; Place</td><td colspan="3">Heat &amp; Place</td><td colspan="3">Cool &amp; Place</td><td colspan="3">Pick Two &amp; Place</td><td colspan="3">All Tasks</td></tr><tr><td>tn</td><td>sn</td><td>un</td><td>tn</td><td>sn</td><td>un</td><td>tn</td><td>sn</td><td>un</td><td>tn</td><td>sn</td><td>un</td><td>tn</td><td>sn</td><td>un</td><td>tn</td><td>sn</td><td>un</td><td>tn</td><td>sn</td><td>un</td></tr><tr><td>BUTLER</td><td>54</td><td>61</td><td>46</td><td>59</td><td>39</td><td>22</td><td>37</td><td>44</td><td>39</td><td>60</td><td>81</td><td>74</td><td>46</td><td>60</td><td>100</td><td>27</td><td>29</td><td>24</td><td>16</td><td>40</td><td>37</td></tr><tr><td>\( \text{BUTLER}_g \)</td><td>54</td><td>43</td><td>33</td><td>59</td><td>31</td><td>17</td><td>37</td><td>30</td><td>26</td><td>60</td><td>69</td><td>70</td><td>46</td><td>50</td><td>76</td><td>27</td><td>38</td><td>12</td><td>16</td><td>19</td><td>22</td></tr><tr><td>Seq2Seq</td><td>31</td><td>26</td><td>8</td><td>44</td><td>31</td><td>11</td><td>34</td><td>30</td><td>42</td><td>36</td><td>50</td><td>30</td><td>27</td><td>32</td><td>33</td><td>17</td><td>8</td><td>6</td><td>9</td><td>10</td><td>9</td></tr></table>

Specifically, we use a pre-trained Mask R-CNN detector (He et al., 2017) to detect objects in the visual frame. The detector is trained separately in a supervised setting with random frames from ALFRED training scenes (see Appendix F). For each frame  $v_{t}$ , the detector generates  $N$  detections  $\{(c_1, m_1), (c_2, m_2), \ldots, (c_N, m_N)\}$ , where  $c_{n}$  is the predicted object class, and  $m_{n}$  is a pixel-wise object mask. These detections are formatted into a sentence using a template e.g., On table 1, you see a mug 1, a tomato 1, and a tomato 2. To handle multiple instances of objects, each object is associated with a class  $c_{n}$  and a number ID e.g., tomato 1. Commands goto, open, and examine generate a list of detections, whereas all other commands generate affirmative responses if the action succeeds e.g.,  $a_{t}$ : put mug 1 on desk 2 →  $o_{t+1}$ : You put mug 1 on desk 2, otherwise produce Nothing happens to indicate failures or no state-change. See Appendix D for a full list of templates. While this work presents preliminary results with template-based descriptions, future work could generate more descriptive observations using pre-trained image captioning models (Johnson et al., 2016), video-action captioning frameworks (Sun et al., 2019), or scene-graph parsers (Tang et al., 2020).

# 3.3 BUTLER::BODY (CONTROLLER):  $v_{t}, a_{t} \rightarrow \{\hat{a}_{1}, \hat{a}_{2}, \dots, \hat{a}_{L}\}$

The controller translates a high-level text action  $a_{t}$  into a sequence of  $L$  low-level physical actions  $\{\hat{a}_1,\hat{a}_2,\dots ,\hat{a}_L\}$  that are executable in the embodied environment. The controller handles two types of commands: manipulation and navigation. For manipulation actions, we use the ALFRED API to interact with the simulator by providing an API action and a pixel-wise mask based on Mask R-CNN detections  $m_{n}$  that was produced during state-estimation. For navigation commands, each episode is initialized with a pre-built grid-map of the scene, where each receptacle instance is associated with a receptacle class and an interaction viewpoint  $(x,y,\theta ,\phi)$  with  $x$  and  $y$  representing the 2D position,  $\theta$  and  $\phi$  representing the agent's yaw rotation and camera tilt. The goto command invokes an A* planner to find the shortest path between two viewpoints. The planner outputs a sequence of  $L$  displacements in terms of motion primitives: MOVEAHEAD, ROTATERIGHT, ROTATELEFT, LOOKUP, and LOOKDOWN, which are executed in an open-loop fashion via the ALFRED API. We note that a given pre-built grid-map of receptacle locations is a strong prior assumption, but future work could incorporate existing models from the vision-language navigation literature (Anderson et al., 2018a; Wang et al., 2019) for map-free navigation

# 4 EXPERIMENTS

We design experiments to answer the following questions: (1) Is it possible to learn robust generalizing policies in TextWorld that can solve a large variety of tasks? (2) Can these abstract policies provide suitable guidance to help agents solve physically embodied tasks? (3) In contrast to directly training in the embodied world, do abstract textual policies enable better task completion and generalization?

# 4.1 BUTLER::BRAIN (TEXT AGENT) PRE-TRAINING

To answer the first question, we train BUTLER::BRAIN in abstract TextWorld environments spanning the six tasks in Table 1, as well as All Tasks, a simple union of all 6. Because of the strong diversity

Table 3: Zero-shot Domain Transfer. Left: Success percentages of best-performing BUTLER::BRAIN agents evaluated in TextWorld. Mid-Left: Success percentages after zero-shot transfer to embodied environments. Mid-Right: Success percentages of BUTLER with an oracle state-estimator and controller, an upper-bound. Right: Success percentages of BUTLER with human-annotated goal descriptions, an additional source of generalization difficulty. Successes are averaged across three evaluation runs. Goal-condition success rates (Shridhar et al., 2020) are given in parentheses.  

<table><tr><td rowspan="2">task-type</td><td colspan="2">TextWorld</td><td colspan="2">Seq2Seq</td><td colspan="2">BUTLER</td><td colspan="2">BUTLER-ORACLE</td><td colspan="2">Human Goals</td></tr><tr><td>seen</td><td>unseen</td><td>seen</td><td>unseen</td><td>seen</td><td>unseen</td><td>seen</td><td>unseen</td><td>seen</td><td>unseen</td></tr><tr><td>Pick &amp; Place</td><td>69</td><td>50</td><td>28 (28)</td><td>17 (17)</td><td>30 (30)</td><td>24 (24)</td><td>53 (53)</td><td>31 (31)</td><td>20 (20)</td><td>10 (10)</td></tr><tr><td>Examine in Light</td><td>69</td><td>39</td><td>5 (13)</td><td>0 (6)</td><td>10 (26)</td><td>0 (15)</td><td>22 (41)</td><td>12 (37)</td><td>2 (9)</td><td>0 (8)</td></tr><tr><td>Clean &amp; Place</td><td>67</td><td>74</td><td>32 (41)</td><td>12 (31)</td><td>32 (46)</td><td>22 (39)</td><td>44 (57)</td><td>41 (56)</td><td>18 (31)</td><td>22 (39)</td></tr><tr><td>Heat &amp; Place</td><td>88</td><td>83</td><td>10 (29)</td><td>12 (33)</td><td>17 (38)</td><td>16 (39)</td><td>60 (66)</td><td>60 (72)</td><td>8 (29)</td><td>5 (30)</td></tr><tr><td>Cool &amp; Place</td><td>76</td><td>91</td><td>2 (19)</td><td>21 (34)</td><td>5 (21)</td><td>19 (33)</td><td>41 (49)</td><td>27 (44)</td><td>7 (26)</td><td>17 (34)</td></tr><tr><td>Pick Two &amp; Place</td><td>54</td><td>65</td><td>12 (23)</td><td>0 (26)</td><td>15 (33)</td><td>8 (30)</td><td>32 (42)</td><td>29 (44)</td><td>6 (16)</td><td>0 (6)</td></tr><tr><td>All Tasks</td><td>40</td><td>35</td><td>6 (15)</td><td>5 (14)</td><td>19 (31)</td><td>10 (20)</td><td>37 (46)</td><td>26 (37)</td><td>8 (17)</td><td>3 (12)</td></tr></table>

across task types, the All Tasks setting shows the extent to which a single policy can learn and generalize on the large set of 3,553 different text-based tasks. After finding that current reinforcement learning approaches were not successful on our set of training tasks (see Appendix I), we turned to DAgger (Ross et al., 2011) assisted by a rule-based expert (detailed in Appendix G). BUTLER::BRAIN is trained for 50K episodes using data collected by interacting with the set of training games.

Results in Table 2 show (i) Training success rate varies from  $16 - 60\%$  depending on the category of tasks, illustrating the challenge of solving hundreds to thousands of training tasks within each category. (ii) Transferring from training to heldout test games typically reduces performance, with the unseen rooms leading to the largest performance drops. Notable exceptions include heat and cool tasks where unseen performance exceeds training performance. (iii) Beam search is a key contributor to test performance; its ablation causes a performance drop of  $21\%$  on the seen split of All Tasks. (iv) Further ablating the DAgger strategy and directly training a Sequence-to-Sequence (Seq2Seq) model with pre-recorded expert demonstrations causes a bigger performance drop of  $30\%$  on seen split of All Tasks. These results suggest that online interaction with the environment, as facilitated by DAgger learning and beam search, is essential for recovering from mistakes and sub-optimal behavior.

# 4.2 TEXTWORLD TO EMBodied GENERALIZATION

To understand whether abstract policies can provide guidance for agents to solve physically embodied tasks, we study the zero-shot domain transfer of BUTLER to novel tasks in embodied environments. Table 3 presents results for agents trained independently on individual tasks and also jointly on all 6 tasks. For each category of task, we select the agent with best evaluation performance in TextWorld (from 8 random seeds). This is done separately for each split: seen and unseen. These best-performing agents are then evaluated on the heldout seen and unseen ALFRED tasks.

The Seq2Seq baseline is trained in TextWorld from pre-recorded expert demonstrations using standard supervised learning. BUTLER is our main model using the Mask R-CNN detector and  $\mathrm{A}^*$  navigator. BUTLER-ORACLE uses an oracle state-estimator with ground-truth object detections and an oracle controller that directly teleports between locations. In Human Goals, instead of templated goal descriptions, we evaluate BUTLER using human-annotated ALFRED goals, which contain 66 unseen verbs (e.g., 'wash', 'grab', 'chill') and 189 unseen nouns (e.g., 'rag', 'lotion', 'disc'; see Appendix E for full list). For embodied evaluations, we also report goal-condition success rates, a metric proposed in ALFRED (Shridhar et al., 2020) to measure partial goal completion<sup>3</sup>.

Overall, TextWorld training generalizes well to unseen embodied tasks. The drop in performance from TextWorld to BUTLER-ORACLE is often a result of the inability of TextWorld-trained agents to understand physical constraints and infeasibilities, e.g., placing a plate inside a full microwave. Future works could address this issue by trying to reduce the domain gap between the two environments, or fine-tuning the agent in the embodied setting with reinforcement learning. The further drop in performance with BUTLER is a result of misdetections from Mask R-CNN and navigation failures caused by collisions. The Mask R-CNN detector struggles with unseen environments which are

visually very distinct from training scenes. Finally, even though the agents were trained only with a templated language, they are able to handle some human-annotated goals in Human Goals.

The supplementary video contains qualitative examples of the BUTLER agent solving tasks in unseen environments. It showcases 3 successes and 1 failure of a TextWorld-only agent trained on All Tasks. In “put a watch in the safe”, the agent has never seen the ‘watch’-‘safe’ combination as a goal.

# 4.3 TRAINING STRATEGIES

Given the domain gap between TextWorld and the embodied world, a natural question is Why not eliminate this gap by training from scratch in the embodied world? To answer this question, we investigate three training strategies: (i) EMBodied-ONLY: pure embodied training, (ii) TW-ONLY: pure TextWorld training followed by zero-shot

Table 4: Training Strategy Success. Trained on Pick & Place Tasks for 50K episodes with embodied evaluations using an oracle state-estimator and controller.  

<table><tr><td>Training Strategy</td><td>train (succ %)</td><td>seen (succ %)</td><td>unseen (succ %)</td><td>train speed (eps/s)</td></tr><tr><td>EMBodied-ONLY</td><td>36.5</td><td>48.6</td><td>41.7</td><td>0.9</td></tr><tr><td>TW-ONLY</td><td>58.7</td><td>57.1</td><td>62.5</td><td>6.1</td></tr><tr><td>HYBRID</td><td>31.0</td><td>42.9</td><td>41.7</td><td>0.7</td></tr></table>

embodied transfer and (iii) HYBRID training that switches between the two environments with  $75\%$  probability for TextWorld and  $25\%$  for embodied world. Table 4 presents success rates for these agents trained and evaluated on the Pick & Place task. All evaluations were conducted with an oracle state-estimator and controller. For a fair comparison, each agent is trained for 50K episodes and training speed is recorded for each strategy. We report peak performance for each split.

Results indicate that TW-ONLY training has higher performance and better generalization to unseen environments than HYBRID or EMBodied-ONLY. We hypothesize that the abstract TextWorld environment allows the agent to focus on quickly learning tasks without having to deal execution-failures and expert-failures caused by physical constraints inherent to embodied environments. TextWorld training is also  $7 \times$  faster since it does not require running a rendering or physics engine like the embodied setting.

# 5 ABLATIONS

Unimodal Baselines: Table 5 presents results for unimodal baseline comparisons to BUTLER. For all baselines, the action space and controller are fixed, but the state space is substituted with different modalities. To study the agents' capability of learning a single policy that generalizes across various tasks, we train and evaluate on All Tasks. In VISION (RESNET18), the textual observation from the state-estimator is replaced with ResNet-18 fc7 features (He

Table 5: Unimodal Baselines. Trained on All Tasks with 50K episodes and evaluated in the embodied environment.  

<table><tr><td>Agent</td><td>seen (succ %)</td><td>unseen (succ %)</td></tr><tr><td>BUTLER</td><td>18.8</td><td>10.1</td></tr><tr><td>VISION (RESNET18)</td><td>10.0</td><td>6.0</td></tr><tr><td>VISION (MCNN-FPN)</td><td>11.4</td><td>4.5</td></tr><tr><td>ACTION-ONLY</td><td>0.0</td><td>0.0</td></tr></table>

et al., 2016) from the visual frame. Similarly, VISION (MCNN-FPN) uses the pre-trained Mask R-CNN from the state-estimator to extract FPN layer features for the whole image. ACTION-ONLY acts without any visual or textual feedback. We report peak performance for each split.

The visual models tend to overfit to seen environments and generalize poorly to unfamiliar environments. Operating in text-space allows better transfer of policies without needing to learn state representations that are robust to visually diverse environments. The zero-performing ACTION-ONLY baseline indicates that memorizing action sequences is an infeasible strategy for agents.

Model Ablations Figure 4 illustrates more factors that affect the performance of BUTLER::BRAIN. The three rows of plots show training curves, evaluation curves in seen and unseen settings, respectively. All experiments are run on the Pick & Place task with 8 random seeds.

In the first column, we show the effect of using different observation queue lengths  $k$  as described in Section 3.1, in which size 0 refers to not providing any observation information to the agent. In the second column, we examine the effect of explicitly keeping the initial observation  $o_0$ , which lists all the receptacles in the scene. Keeping the initial observation  $o_0$  facilitates the pointer softmax mechanism in the decoder by guiding it to generate receptacle words more accurately.

The third column suggests that the recurrent component in our aggregator is helpful in making history-based decisions when the current observation contains insufficient information. Finally, in the fourth column, we see that using more training games can lead to better generalizability in both seen and unseen settings. Fewer training games achieve high training scores by quickly overfitting, which lead to zero evaluation scores.

![](images/e60f6f0d6fb038c626aff8920973be58e462f82751216f7fd91a2d475cf129c9.jpg)  
Figure 4: Ablation study. x-axis: 0 to 50k episodes; y-axis: normalized success from 0 to 100.

# 6 RELATED WORK

The longstanding goal of grounding language learning in embodied settings has lead to substantial work on interactive environments. ALFWorld extends that work with fully-interactive aligned environments that parallel textual interactions with photo-realistic renderings and physical interactions.

Interactive Text-Only Environments: We build on the work of text-based environments like TextWorld (Côté et al., 2018) and Jericho (Hausknecht et al., 2020). While these environments allow for textual interactions, they are not grounded in visual or physical modalities.

Vision and language: While substantial work exists on vision-language representation learning e.g., MAttNet (Yu et al., 2018b), CMN (Hu et al., 2017), VQA (Antol et al., 2015), CLEVR (Johnson et al., 2017), ViLBERT (Lu et al., 2019), they lack embodied or sequential decision making.

Embodied Language Learning: To address language learning in embodied domains, a number of interactive environments have been proposed: BabyAI (Chevalier-Boisvert et al., 2019), Room2Room (Anderson et al., 2018b), ALFRED (Shridhar et al., 2020), InteractiveQA (Gordon et al., 2018), EmbodiedQA (Das et al., 2018), and NetHack (Küttler et al., 2020). These environments use language to communicate instructions, goals, or queries to the agent, but not as a fully interactive modality.

Language for State and Action Representation: Others have used language for more than just goal-specification. Schwartz et al. (2019) use language as a state representation for VizDoom. Hu et al. (2019) use a natural language instructor to command a low-level executor, and Jiang et al. (2019) use language as an abstraction for hierarchical RL. However these works do not feature an interactive text environment, for pre-training the agent in an abstract textual space. Zhu et al. (2017) use high-level commands similar to ALFWorld to solve tasks in THOR with IL and RL-finetuning methods, but the policy only generalizes to a small set of tasks due to the vision-based state representation.

Game Engines as World Models: The concept of using TextWorld as a "game engine" to represent the world is broadly related to inverse graphics (Kulkarni et al., 2015) and inverse dynamics (Wu et al., 2017) where abstract visual or physical models are used for reasoning and future predictions.

# 7 CONCLUSION

We introduced ALFWorld, the first interactive text environment with aligned embodied worlds. ALFWorld allows agents to explore, interact, and learn abstract polices in a textual environment. Pre-training our novel BUTLER agent in TextWorld, we show zero-shot generalization to embodied tasks in the ALFRED dataset. The results indicate that reasoning in textual space allows for better generalization to unseen scenes and also faster training, compared to other modalities like vision.

BUTLER is designed with modular components which can be upgraded in future work. Examples include the template-based state-estimator and the A* navigator which could be replaced with learned modules, enabling end-to-end training of the full pipeline. Another avenue of future work is to learn "textual dynamics models" through environment interactions, akin to vision-based world models (Ha and Schmidhuber, 2018). Such models would facilitate construction of text-engines for new domains, without requiring access to symbolic state descriptions like PDDL. Overall, we are excited by the challenges posed by aligned text and embodied environments for better cross-modal learning.

# REFERENCES

Adhikari, A., Yuan, X., Côté, M.-A., Zelinka, M., Rondeau, M.-A., Laroche, R., Poupart, P., Tang, J., Trischler, A., and Hamilton, W. L. (2020). Learning dynamic belief graphs to generalize on text-based games. In Neural Information Processing Systems (NeurIPS).  
Ammanabrolu, P. and Hausknecht, M. (2020). Graph constrained reinforcement learning for natural language action spaces. In International Conference on Learning Representations.  
Anderson, P., Wu, Q., Teney, D., Bruce, J., Johnson, M., Sunderhauf, N., Reid, I., Gould, S., and van den Hengel, A. (2018a). Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.  
Anderson, P., Wu, Q., Teney, D., Bruce, J., Johnson, M., Sunderhauf, N., Reid, I., Gould, S., and van den Hengel, A. (2018b). Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).  
Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., and Parikh, D. (2015). VQA: Visual Question Answering. In International Conference on Computer Vision (ICCV).  
Ba, L. J., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. CoRR, abs/1607.06450.  
Chevalier-Boisvert, M., Bahdanau, D., Lahlou, S., Willems, L., Saharia, C., Nguyen, T. H., and Bengio, Y. (2019). BabyAI: First steps towards grounded language learning with a human in the loop. In International Conference on Learning Representations.  
Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).  
Côté, M.-A., Kádár, A., Yuan, X., Kybartas, B., Barnes, T., Fine, E., Moore, J., Tao, R. Y., Hausknecht, M., Asri, L. E., Adada, M., Tay, W., and Trischler, A. (2018). Textworld: A learning environment for text-based games. CoRR, abs/1806.11532.  
Das, A., Datta, S., Gkioxari, G., Lee, S., Parikh, D., and Batra, D. (2018). Embodied Question Answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).  
Gehrmann, S., Deng, Y., and Rush, A. (2018). Bottom-up abstractive summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.  
Gordon, D., Kembhavi, A., Rastegari, M., Redmon, J., Fox, D., and Farhadi, A. (2018). Iqa: Visual question answering in interactive environments. In Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on.  
Gulcehre, C., Ahn, S., Nallapati, R., Zhou, B., and Bengio, Y. (2016). Pointing the unknown words. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).  
Ha, D. and Schmidhuber, J. (2018). Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems 31.  
Hausknecht, M. and Stone, P. (2015). Deep recurrent q-learning for partially observable mdps. arXiv preprint arXiv:1507.06527.  
Hausknecht, M. J., Ammanabrolu, P., Côté, M.-A., and Yuan, X. (2020). Interactive fiction games: A colossal adventure. In AAAI.  
He, K., Gkioxari, G., Dollár, P., and Girshick, R. (2017). Mask r-cnn. In Proceedings of the IEEE international conference on computer vision.

He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition.  
Helmert, M. (2006). The Fast Downward planning system. Journal of Artificial Intelligence Research.  
Hu, H., Yarats, D., Gong, Q., Tian, Y., and Lewis, M. (2019). Hierarchical decision making by generating and following natural language instructions. In Advances in Neural Information Processing Systems.  
Hu, R., Rohrbach, M., Andreas, J., Darrell, T., and Saenko, K. (2017). Modeling relationships in referential expressions with compositional modular networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.  
Jiang, Y., Gu, S. S., Murphy, K. P., and Finn, C. (2019). Language as an abstraction for hierarchical deep reinforcement learning. In Advances in Neural Information Processing Systems.  
Johnson, J., Hariharan, B., van der Maaten, L., Fei-Fei, L., Zitnick, C. L., and Girshick, R. (2017). Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR.  
Johnson, J., Karpathy, A., and Fei-Fei, L. (2016). Densecap: Fully convolutional localization networks for dense captioning. In Proceedings of the IEEE conference on computer vision and pattern recognition.  
Kingma, D. P. and Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.  
Kolve, E., Mottaghi, R., Han, W., VanderBilt, E., Weihs, L., Herrasti, A., Gordon, D., Zhu, Y., Gupta, A., and Farhadi, A. (2017). Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474.  
Kulkarni, T. D., Whitney, W. F., Kohli, P., and Tenenbaum, J. (2015). Deep convolutional inverse graphics network. In Advances in neural information processing systems.  
Küttler, H., Nardelli, N., Miller, A. H., Raileanu, R., Selvatici, M., Grefenstette, E., and Rocktäschel, T. (2020). The nethack learning environment.  
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. (2014). Microsoft coco: Common objects in context. In European conference on computer vision.  
Lu, J., Batra, D., Parikh, D., and Lee, S. (2019). Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Advances in Neural Information Processing Systems.  
MacMahon, M., Stankiewicz, B., and Kuipers, B. (2006). Walk the talk: Connecting language, knowledge, and action in route instructions. In Proceedings of the 21st National Conference on Artificial Intelligence (AAAI-2006).  
McDermott, D., Ghallab, M., Howe, A., Knoblock, C., Ram, A., Veloso, M., Weld, D., and Wilkins, D. (1998). Pddl-the planning domain definition language.  
Press, O. and Wolf, L. (2016). Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859.  
Reddy, D. R. et al. (1977). Speech understanding systems: A summary of results of the five-year research effort. Department of Computer Science. Carnegie-Mell University, Pittsburgh, PA, 17.  
Ross, S., Gordon, G., and Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics.  
Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.  
Schwartz, E., Tennenholtz, G., Tessler, C., and Mannor, S. (2019). Language is power: Representing states using natural language in reinforcement learning.

Sharma, S., Asri, L. E., Schulz, H., and Zumer, J. (2017). Relevance of unsupervised metrics in task-oriented dialogue for evaluating natural language generation. arXiv preprint arXiv:1706.09799.  
Shridhar, M., Thomason, J., Gordon, D., Bisk, Y., Han, W., Mottaghi, R., Zettlemoyer, L., and Fox, D. (2020). Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10740-10749.  
Sun, C., Myers, A., Vondrick, C., Murphy, K., and Schmid, C. (2019). Videobert: A joint model for video and language representation learning. In Proceedings of the IEEE International Conference on Computer Vision.  
Tang, K., Niu, Y., Huang, J., Shi, J., and Zhang, H. (2020). Unbiased scene graph generation from biased training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
Todorov, E., Erez, T., and Tassa, Y. (2012). Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems.  
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems 30.  
Wang, X., Huang, Q., Celikyilmaz, A., Gao, J., Shen, D., Wang, Y.-F., Wang, W. Y., and Zhang, L. (2019). Reinforced cross-modal matching and self-supervised imitation learning for vision-language navigation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.  
Wu, J., Lu, E., Kohli, P., Freeman, B., and Tenenbaum, J. (2017). Learning to see physics via visual de-animation. In Advances in Neural Information Processing Systems.  
Yu, A. W., Dohan, D., Le, Q., Luong, T., Zhao, R., and Chen, K. (2018a). Fast and accurate reading comprehension by combining self-attention and convolution. In International Conference on Learning Representations.  
Yu, L., Lin, Z., Shen, X., Yang, J., Lu, X., Bansal, M., and Berg, T. L. (2018b). Mattnet: Modular attention network for referring expression comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.  
Yuan, X., Côté, M.-A., Sordoni, A., Laroche, R., Combes, R. T. d., Hausknecht, M., and Trischler, A. (2018). Counting to explore and generalize in text-based games. arXiv preprint arXiv:1806.11525.  
Zhu, Y., Gordon, D., Kolve, E., Fox, D., Fei-Fei, L., Gupta, A., Mottaghi, R., and Farhadi, A. (2017). Visual semantic planning using deep successor representations. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017.
