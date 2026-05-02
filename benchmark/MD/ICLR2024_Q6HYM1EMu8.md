# LARG², LANGUAGE-BASED AUTOMATIC REWARD AND GOAL GENERATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Robotic tasks currently addressed with reinforcement learning such as locomotion, navigation, and manipulation are challenged with the problem of defining reward functions to maximize and goals to reach. Alternative methodologies, like imitation learning, often require labor-intensive human annotations to produce datasets of task descriptions associated with trajectories. As a response, this paper introduces "Language-based Automatic Reward and Goal Generation"  $(\mathrm{LARG}^2)$ , a framework that harnesses code generation capabilities of LLMs to enables the conversion of text-based task descriptions into corresponding reward and goal-generation functions. We leverages Chain-of-thought mechanisms and the common-sense knowledge embedded in Large Language Models (LLMs) for this purpose. It is complemented by automatic error discovery and correction mechanisms. We validate the effectiveness of  $\mathrm{LARG}^2$  by conducting extensive experiments in the context of robotic manipulation demonstrating its ability to train and execute without human annotation of any kind.

# 1 INTRODUCTION

The statistical learning approach to robot control has emerged with the potential of revolutionizing various industries, spanning from manufacturing to healthcare. Various preliminary approaches, such as imitation learning (Tai et al., 2016; Kumar et al., 2022), transfer learning (Stüber et al., 2018; Wiese et al., 2021; Weng et al., 2020), and interactive learning (Kelly et al., 2018; Chisari et al., 2021; Faulkner et al., 2020), have been proposed for that matter.

In the field of robotic manipulation, decision models are currently evolving from the traditional optimal control approaches towards policy learning through Multi-task and Goal-Conditioned Reinforcement Learning (Huang et al., 2022b). Following this line of work, multi-modal task definition (Jiang et al., 2022; Shah et al., 2022), associated with reasoning and action planning abilities facilitated by Large Language Models (LLMs) (Huang et al., 2022a), have enabled agents to adapt to real-world uncertainty which is hardly handled with traditional robotic control. However, the difficulties of connecting textual descriptions of tasks with their associated goals and reward functions have led to unscalable solutions involving labor-intensive annotation practices.

Motivated by these observations, we introduce  $\mathrm{LARG}^2$ , Language-based Automatic Reward and Goal Generation. For a given sequential decision task described using natural language, our method automates the generation of either goals or associated reward functions depending on the learning scheme. It leverages the common-sense and reasoning capabilities offered by recent LLMs in terms of text understanding and source code generation. In the context of robotic manipulation, our approach samples goals conditioned by a task description to train a corresponding policy using Goal-Conditioned Reinforcement Learning (GCRL). Following this idea, we generate executable reward functions to train corresponding policies using Multi-Task Reinforcement Learning (MTRL), assuming task descriptions are given as input to the policy. Finally, we evaluate these two settings of  $\mathrm{LARG}^2$  over a set of language-formulated tasks in a tabletop manipulation scenario.

# 2 PRELIMINARIES: REINFORCEMENT LEARNING FOR ROBOTIC MANIPULATION

Reinforcement Learning deals with an agent performing sequences of actions in a given environment to maximize a cumulative sum of rewards. Such problem is commonly framed as Markov Decision Processes (MDPs):  $M = \{S, A, T, \rho_0, R\}$  (Sutton & Barto, 2005; Mnih et al., 2016; Lillicrap et al., 2016). The agent and its environment, as well as their interaction dynamics, are defined by the first components  $\{S, A, T, \rho_0\}$ , where  $s \in S$  describes the current state of the agent-environment interaction and  $\rho_0$  is the distribution over initial states. The agent interacts with the environment through actions  $a \in A$ . The transition function  $T$  models the distribution of the next state  $s_{t+1}$  conditioned with the current state and action  $T: p(s_{t+1}|s_t, a_t)$ . Then, the objective of the agent is defined by the remaining component of the MDP,  $R: S \to \mathbb{R}$ . Solving a Markov decision process consists in finding a policy  $\pi: S \to A$  that maximizes the cumulative sum of discounted rewards accumulated through experiences.

In the context of robotic manipulation, a task commonly consists in altering the environment into a targeted state through selective contact (Gu et al., 2017). Naturally, tasks are expressed as  $g = (c_{g}, R_{G})$  pair where  $c_{g}$  is a goal configuration such as Cartesian coordinates of each element composing the environment or a textual description of it, and  $R_{G}: S \times G \to \mathbb{R}$  is a goal-achievement function that measures progress towards goal achievement and is shared across goals. A goal-conditioned MDP is defined as:  $M_{g} = \{S, A, T, \rho_{0}, c_{g}, R_{G}\}$  with a reward function shared across goals. In multi-task reinforcement learning settings, an agent solves a possibly large set of tasks jointly. It is trained on a set of rewards associated with each task. Finally, goals are defined as constraints on one or several consecutive states that the agent seeks to satisfy (Plappert et al., 2018; Nair et al., 2018; OpenAI et al., 2021).

# 3 RELATED WORK

# 3.1 CHALLENGES OF REWARD DEFINITION AND SHAPING

A sequential decision task which is not solved through imitation but reinforcement requires defining an informative reward function to enable the learning paradigm. Reward shaping consists in manually designing a function incorporating elements from domain knowledge to guide policy search algorithms. Formally, this can be defined as  $R' = R + F$ , where  $F$  is the shaping reward function, and  $R'$  is the modified reward function Dorigo & Colombetti (1994); Randlov & Alstrom (1998). As a main limitation, a reward function needs to be crafted for each task. For instance Brohan et al. (2022) leveraged large number of human demonstrations and specific handcrafted definitions of tasks to train a robotic transformer. However, as MTRL aims at dealing with a large set of goals and tasks to implement, such an approach becomes hardly scalable. In this work, we study how to leverage the common-sense and prior knowledge embedded in LLMs to automate the textual paraphrasing of task description and the generation of associated reward functions.

# 3.2 LARGE LANGUAGE MODELS FOR CONTROL

The use of Large Language Models to control autonomous agents has recently started to be investigated. Shah et al. (2022) has combined a text encoder, a visual encoder, and visual navigation models, to provide text-based instructions to a navigating agent. This idea has been further developed in Huang et al. (2022b) using LLM capabilities to support action planning, reasoning, and internal dialogue among models for manipulation tasks. Similarly, Liang et al. (2022) proposes to use LLMs to transform textual instructions into a code-based policy. Unfortunately, it involves an interactive design process for a hard-coded policy, rather than a task-conditional learning process. In contrast, our method, which also relies on a specific prompt design, allows the agent to learn new skills through goal generation and automatic reward shaping.

Along this line, Colas et al. (2020b) proposes to derive goals from a textual description of the task. However, the language remains limited to the logical descriptions of the expected configuration of the scene and the goal is reduced to a finite set of eligible targets. In contrast, our approach allows using natural language beyond logical forms, grounded with reasoning capabilities and enriched with common-sense captured in large pre-trained language models. Also related, Colas et al. (2020a)

proposes to train a conditional variational auto-encoder to create a language-conditioned goal generator. However, it assumes the existence of pre-trained goal-conditioned policies and no LLM is considered to achieve this objective.

# 3.3 IMPROVING GENERATION WITH CHAIN OF THOUGHT

To address LLMs limitations such as hallucination, lack of consistency or lack of grounding several works attempt to enhance the alignment of generated answers with expected behavior or constraints. The "Chain-of-Thought" (CoT) approach (Wei et al., 2023) aims to influence text generation by using a sequence of intermediate reasoning steps as part of the LLM prompt, thereby promoting a consistent generation path. It involves providing examples of expected reasoning behavior as part of the prompt (Wei et al., 2022; Wang et al., 2023b; Wu et al., 2023; Diao et al., 2023). This approach has recently shown successes, particularly in handling complex queries such as mathematics reasoning questions (Imani et al., 2023).

In a subsequent study (Wang et al., 2023a), the authors conducted an analysis of the influence of example composition on reasoning consistency and accuracy. They underscored the importance of example relevance in provided reasoning steps to achieve accurate answers.

In our approach, we hypothesize that CoT presents a promising mechanism to enhance LLM's code generation capabilities. To this end, we leverage existing code repositories to support reasoning for goal and reward function generation.

# 3.4 CONCURRENT WORK

Recently, a method for the generation of reward functions in the context of robotic skill learning as been introduced in Yu et al. (2023). However, this concurrent work exclusively focuses on generating goal poses to complement existing reward functions in the context of robotic manipulation and quadruped's pose control. The generation of reward functions is not addressed in this work. Furthermore, neither code correction nor Chain-of-though mechanism are considered to guide and possibly fix code generation.

# 4 LARG², LANGUAGE-BASED AUTOMATIC REWARD AND GOAL GENERATION

Our method, illustrated in figure 1, translates textual task descriptions into both goal and reward functions to enable scalable training of goal conditioned (GCRL) and multi-tasks reinforcement learning (MTRL) policies. It is composed of three sequential steps. The initial one is responsible for gathering input for the second step, which carries out code generation. The final step assesses and, if necessary, correct generated functions through a feedback loop. Once validated, the code is used with standard off-the-shelf GCRL or MTRL frameworks. For the MTRL scenario, the second step also encodes textual task descriptions into an embedding vector which is appended to the state vector to align policies with task definitions.

# 4.1 ELEMENTS OF THE PROMPT

Our first step consists in collecting inputs for building a dedicated prompt  $(P_{2})$  to condition code generation. One main element is the task definition  $(T)$  so to automate the production of a large training set we leverage paraphrasing capabilities of a pre-trained LLM  $(L1)$  to generate variations from a single description. We use a prompt  $(P_{1})$  such as: "Generate n paraphrases for the task bellow:" It produces a set of semantically similar tasks without handcrafting a whole collection of tasks such as,  $L1(P,T)\to \{T_1,\dots,T_n\}$ .

Supplemental code examples  $(X)$  can also be collected to complement the main prompt  $(P_{2})$  enabling a Chain-of-Thought (CoT) mechanism to guide the code production process. For this, we assume the availability of code repositories such as  $\mathrm{Gibthub}^{1}$ . These repositories need to possess adequate documentation, and the code should be commented. Naturally, it is preferable for this code

![](images/fc7eab3ec75bfe7f4563c24db595005acda69cab901522ebb19ca5da04c42943.jpg)  
Figure 1: LARG<sup>2</sup> transforms a textual task description into either 1) a goal to be used as input of a given reward function for GCRL, or 2) a reward function for MTRL. We use pre-trained and instructed LLMs with dedicated prompts for our generation procedures. For GCRL, the goal is appended to the state description given as input to the policy. For MTRL, the text-based task description is encoded using a pre-trained language model to complement the state vector. Optionally, supplemental code examples can be searched and retrieved to complement the prompt therefore leveraging a Chain-of-Through mechanism. A code validation loop is provided to ensure that generated functions can be properly executed within GCRL or MTRL frameworks.

![](images/8ac8f1808ac95524afa89f6be95bb42dfe27275f0c80715b061e57559a144d87.jpg)  
Search & Retrieve pipeline  
Figure 2: LARG<sup>2</sup> leverages context, environment, guidelines and task descriptions to query either the parametric memory of a LLM or a code database to retrieve function examples. It serves as additional context, enriching a dedicated prompt used to convert textual task descriptions into goal poses or reward functions.

to be correct, although Wang et al. (2023a) has shown that even invalid examples used in a CoT mechanism can still yield valid answers.

As illustrated in figure 15, we propose two viable options. Either code repositories are part of a dataset used to train a LLM, in which case the information is embedded within and accessible from the model's parametric memory, or they are independently indexed. The latter option allows to extend the LLM's background knowledge with external information, which may be more relevant for a specific application.

In the following description, we focus on the latter approach although we also test parametric memory during our experiments. First, we segment each code file from a repository into a set of functions and each function is indexed individually. This indexing process  $(I)$  combines information from multiple sources, including theREADME.md file  $(R)$ , the function's signature  $(S)$ , its docstring  $(D)$ , and its code  $(C)$ . This aggregation can be represented as:  $R, S, D, C \to F$ , where  $F$  represents the indexed function. The result is encoded into a collection of embeddings and stored within a Vector database for semantic retrieval.

# 4.2 GENERATING GOAL AND REWARD FUNCTIONS

The second stage uses a dedicated prompt  $(P_{2})$  with ad-hoc parameters to query a LLM  $(L_{2})$  for generating either goal or reward functions. This prompt, illustrated in figure 3, is composed of  $\{T,G,E,X\}$  where  $T$  and  $G$  are provided by the user, or through paraphrasing,  $E$  from pre

![](images/2c79630c6c515d511bbc2aff3e754b1fc97de2356661c49a7a159bbfa79dc3ec.jpg)  
Figure 3: The prompt is composed of a set of parts describing the general context which is highlighted in grey, supporting code examples in blue, the environment description in orange, the task description in green, guidelines in yellow and the description of the function signature to be used as template for code generation.

![](images/b3e7248101838ab3d43945c7b230d62d168731c696089635248be7332451e8ec.jpg)

requisite dataset and, optionally,  $X$  either from the parametric memory of a LLM or from queries to an ad-hoc code database.

$C$  is the high level description of the objective such as "We aim to develop a Python function for generating goals for a Franka-Move tabletop rearrangement task within IsaacGym".  $T$  is the task description and  $E$  provides critical information defining the action space. It includes details such as the dimensions, and locations of objects involved in the experiments.  $X$  is an optional list of code examples to guide intermediate reasoning steps of the LLM. Guidelines,  $G$ , reflect a comprehensive summary referencing preceding sections. It consolidates the list of elements or constraints that must be taken into account when generating the code. For CoT, its purpose is to provide the reasoning schema required to generate a more relevant function.  $S$  is the signature of the function that needs to be completed, along with its docstring. This specification ensures that the generated function aligns with specific requirements, enabling it to be executed seamlessly within a larger GCRL or MTRL framework. Figure 15 illustrates the search and retrieval process for supplemental examples  $(X)$ .

# 4.2.1 AUTOMATIC GENERATION OF GOALS FOR GCRL APPLICATIONS

In the context of tabletop manipulation scenarios, a task consists in re-arranging a set of objects composing the scene. In such a case, goals are objects' Cartesian coordinates. In a GCRL settings, these goals parameterize a reward function which, for instance, incorporates environment-dependent reward terms and Euclidian distance between the current pose of the objects and the target pose. Therefore, goals generated by  $\mathrm{LARG}^2$  are used to compute the reward signal at each step. The prompt  $p$ , described in previous section, allows to generate a function  $F$  such as  $L2(\{T,G,E,X\},P_2)\to F$  to set goal values.

# 4.2.2 AUTOMATIC GENERATION OF REWARD FUNCTIONS FOR MTRL APPLICATIONS

The second utilization of  $\mathrm{LARG}^2$  generates the implementation of a reward function. While Large Language Models can support the full generation of complex reward functions  $(R)$ , we propose to simplify the generation by identifying different parts in such a function, some being task-independent  $(I)$  and others closely related to the task definition  $(D)$  so that  $R$  is a composition of both parts,  $R = I + D$ . In robotic manipulation, common task-independent components address bonuses for lifting the objects or penalties for the number of actions to achieve a given purpose. Task-dependent components, which are driven by the textual task description, align constraints with penalties  $(N)$  and guidelines with bonuses  $(B)$ . Both components are combined in a global reward function.

To compose this global reward function, we consider the existence of predefined categories of tasks with their environments, formalized using languages such as YAML  $^{2}$  or Python, providing independent reward components  $(I)$  available in repositories like IsaacGym  $^{3}$ . In that respect, the search and retrieval step allows to collect reward components as examples to support full reward generation.

For the task dependant part of the reward, we leverage the generation capability of the LLM (L2) to map task descriptions into bonuses  $(B)$  and penalties  $(N)$  so that:

$$
R = I + \sum_ {i = 1} ^ {n} \alpha_ {i}. B _ {i} + \sum_ {j = 1} ^ {m} \beta_ {j}. N _ {j}
$$

Weights  $(\alpha$  and  $\beta)$  associated with these parameters could be adjusted in an optimization loop.

# 4.2.3 TASK-ENCODING AND POLICY

For GCRL, the input of the policy is composed with the environment state and the goal generated by  $\mathrm{LARG}^2$ . For MTRL, the goal of each task is replaced by a textual description of the task. We use Google T5 (Raffel et al., 2020a;b) as pre-trained text encoder to encode the text into an embedding vector. This vector is added to the state vector, along with proprioception and exteroception data, in the training phase to label tasks. This approach allows to use textual descriptions of tasks as input to neural policies such as what is proposed by (Jiang et al., 2022).

# 4.3 CODE VALIDATION AND AUTO-CORRECTION

Naturally, the generated code can not be guarantee in terms of code validity or outcomes. As a consequence, we automate iterations, emphasizing the elements that need to be modified until the result converges toward expectations. The errors which are commonly encounter correspond to under-specified elements in the original prompt or from LLM limitations such as hallucinations (Ji et al., 2022). So, we finalize the code generation with an automatic validation step described in Figure 4 which exploits the output of the Python interpreter.

![](images/1bfecd4747e9755e0d55f48ece37f603aed5c0e93e334f531ad9372de05f7472.jpg)  
Figure 4: The code correction loop uses the exceptions raised during execution to request modifications. Then, a functional test is generated before moving to the learning loop.

For validation purposes, we execute the generated code using placeholder variables. If the code fails, we catch the exceptions raised by the Python interpreter filtering the thread of exceptions to keep the latest stack and use the error message to fill a prompt requesting code modifications. As illustrated in Figure 5, our prompt contains (1) a header which requests the LLM to fix the raised exception, (2) the text of the raised exception, and (3) the code of the incorrect function. Several iterations can be performed until the code can be properly executed.

Once the generated function satisfies the code correction step, we use another prompt to generate a functional test to evaluate this first function as detailed in section A.1.5 of the appendix. This step filters out potentially incorrect code prior to running the training loop. This prompt, illustrated in Figure 6, is composed of (1) a header requesting the LLM to generate a functional test, (2) a list of guidelines conditioning the test, and (3) the generated function.

```python
Could you please fix the error: 'line 38, in compute_franka Reward   
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! in the following function implementation: import torch from torch import Tensor from typing import Tuple def compute_franka Reward(object_pos: Tensor, lfinger_grasp_pos: Tensor, rfinger grasp_pos: Tensor) -> Tuple[Tensor, Tensor]: ...
```

Figure 5: Prompt for the automatic code correction step which contains the error message and the code to be improved in blue.

```python
Update the following python script with functional tests for the reward function 'compute_franka Reward'.   
Rewards tests should only validate cases when they should be positive  $(>0)$  or negative  $(<  0)$    
Succes should be tested against 1 or O values.   
Do no add any explanation text.   
Return the same script plus what you have inserted..   
import torch   
from torch import Tensor   
from typing import Tuple   
def compute_franka Reward(object_pos: Tensor,lfinger_grasp_pos: Tensor,rfinge r_grasp_pos: Tensor) -> Tuple[Tensor, Tensor]:
```

Figure 6: Prompt requesting the generation of a functional test for the provided function, in blue.

# 5 EXPERIMENTS

Three experiments are designed in the context of robotic manipulation using a table top scenario to answer the following questions: Can we generate valid goal positions from textual task descriptions for GCRL settings; Can we automatically generate reward functions to train multi-task policies; How code examples enhance the relevance of generated functions?

In the GCRL case, we evaluate goal generation on a series of 8 tasks involving a single object, and 4 tasks involving a set of 3 objects. The list of tasks is detailed in Table 2. In the MTRL case, we address the generation of reward functions for 9 manipulation tasks detailed in Table 5 of the appendix. As a source of predefined environments and reward functions for the GCRL experiment, we use Pick and Place scenarios and repositories defined in the IsaacGym repository designed for a Franka Emika Panda robot arm<sup>4</sup>.

# 5.1 CHAIN-OF-THOUGHT FOR CODE GENERATION

As a first experiment, we test the compliance of the generated poses against specifications provided in task descriptions. Also, we evaluate the impact of the additional code examples on the relevance of generated goals. For this, we prompt a pretrained LLM to generate a short set of supplemental python functions. A list of these prompts is provided in section A.2.4 of the appendix.

In this evaluation, we use three LLMs: GPT4 (GPT)  $^{5}$ , Hyper Clova X (HCX)  $^{6}$ , and StarCoder (SC)  $^{7}$ . These models are used in the  $\mathrm{LARG}^2$  pipeline either in a straightforward manner without supplemental examples in the prompt, or with retrieved functions (RF) provided as examples.

Table 1: Performance comparison between three LLMs: GPT4 (GPT), HyperClovaX (HCX), and StarCoder (SC).  

<table><tr><td>Task</td><td>GPT</td><td>GPT+RF</td><td>HCX</td><td>HCX+RF</td><td>SC</td><td>SC+RF</td></tr><tr><td>Move a cube in the top right corner of the table.</td><td>0.8</td><td>0.75</td><td>1</td><td>0.25</td><td>0</td><td>0</td></tr><tr><td>Lift the cube 15cm above the table.</td><td>0.8</td><td>0.9</td><td>0.8</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Take the cube and move it to the left side of the table.</td><td>1</td><td>0.75</td><td>1</td><td>0.5</td><td>0.25</td><td>0</td></tr><tr><td>Take the cube and move it closer to the robotic arm.</td><td>0.4</td><td>0.5</td><td>0</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Move the cube 20cm to the left of its initial position.</td><td>0.5</td><td>0.75</td><td>0.5</td><td>0</td><td>0</td><td>0</td></tr></table>

Inspired by the results presented in table 1 which highlight a positive impact of supplemental examples using the GPT4 model, we apply this model in subsequent experiments to test LARG² for Goal-Conditioned and Multi-Task Reinforcement Learning.

# 5.2 LARG² FOR GOAL-CONDITIONED REINFORCEMENT LEARNING

For this experiment, we use a neural policy trained beforehand using Proximal Policy Optimization (Schulman et al., 2017). The policy takes as input the position and velocity of each joint of the robot and the respective pose of the objects composing the scene. The policy triggers joint displacement

in a  $\mathbb{R}^7$  action space. The goal information, generated by  $\mathrm{LARG}^2$  is used as additional input to the policy.

Regarding prompting, we create a dedicated code database to support search and retrieval for supplemental examples using The Stack  $^{8}$  which is a database that contains 6TB of source code files covering 358 programming languages build as part of the BigCode project  $^{9}$ . For the sake of performance, we keep only Python files from repositories related to robot learning for manipulation tasks. We use text-based information found in markdown files associated with each repository for this filtering process. Once filtered, we index and store this dataset in a vector database, ChromaDB  $^{10}$ . Repository descriptions, comments and function names are encoded using SentenceTransformer  $^{11}$ .

To evaluate the benefit of the search and retrieval process, we test the influence of two parameters: the number of provided examples and the alignment, or lack of, between the names of these functions and the name of the targeted one as defined in the signature  $(S)$  part of the prompt.. Specifically, we include either one, two, or three functions as examples and explore modifications of the supplemental function names to match the name of the expected function. This particular modification is inspired by an observation highlighted in Wang et al. (2023a) which underscores the importance of name coherence within the Chain-of-Thought mechanism.

In Table 2, we evaluate the validity of generated goal poses with respect to textual task descriptions. We compare LARG $^2$  without supplemental examples (L) with the retrieval augmented version including 2 and 3 top ranked functions (I_2 and I_3), only the best function (I_b) and a random selection among the top 4 excluding the top one. Similarly we replicate the experiment with modifications of supplemental function names to match the targeted function (M_2, M_3, M_b, M_r).

Table 2: Evaluation of LARG $^2$ performance for goal pose generation according to various configurations of the code example part of the prompt.  

<table><tr><td>Task</td><td>L</td><td>I.b</td><td>I.r</td><td>I.2</td><td>I.3</td><td>M.b</td><td>M.r</td><td>M.2</td><td>M.3</td></tr><tr><td>Move a cube in the top right corner of the table.</td><td>0.75</td><td>0.7</td><td>0.9</td><td>0.9</td><td>0.4</td><td>0.5</td><td>0.9</td><td>0.9</td><td>0.25</td></tr><tr><td>Lift the cube 15cm above the table.</td><td>1.0</td><td>1.0</td><td>0.8</td><td>0.9</td><td>0.8</td><td>0.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Take the cube and move it to the left side of the table.</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.3</td><td>0.6</td><td>1.0</td></tr><tr><td>Take the cube and move it closer to the robotic arm.</td><td>0.5</td><td>0.6</td><td>0.6</td><td>0.4</td><td>0.3</td><td>0.3</td><td>0.7</td><td>0.3</td><td>0.7</td></tr><tr><td>Lift the cube 20cm above the table and 15 cm ahead.</td><td>0.5</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Push a cube 10cm to the right and 10cm backward.</td><td>0.2</td><td>0.5</td><td>0.2</td><td>0.2</td><td>0.5</td><td>0.6</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Grab a cube and lift it a bit and move it a bit ahead.</td><td>1.0</td><td>0.5</td><td>0.7</td><td>0.8</td><td>0.7</td><td>0.7</td><td>0.7</td><td>0.8</td><td>0.9</td></tr><tr><td>Move the cube at 20cm to the left of its initial position.</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>0.7</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Move one cube to the left side of the table, another one to the right side of the table, and put the last cube at the center of the table.</td><td>0.9</td><td>0.7</td><td>0.8</td><td>0.5</td><td>0.4</td><td>1.0</td><td>0.6</td><td>0.7</td><td>0.7</td></tr><tr><td>Move the three cubes so they are 10 cm close to one another.</td><td>0.9</td><td>1.0</td><td>1.0</td><td>0.3</td><td>0.2</td><td>1.0</td><td>1.0</td><td>0.2</td><td>0.2</td></tr><tr><td>Move the three cubes on the table so that at the end they form a right-angled triangle.</td><td>1.0</td><td>1.0</td><td>0.9</td><td>0.2</td><td>0.1</td><td>1.0</td><td>1.0</td><td>0.3</td><td>0.2</td></tr><tr><td>Reposition the three cubes on the table such that they create a square, with the table&#x27;s center serving as one of the square&#x27;s corners.</td><td>0.8</td><td>0.2</td><td>0.9</td><td>0.2</td><td>0.1</td><td>0.8</td><td>0.9</td><td>0.9</td><td>0.8</td></tr></table>

This experiment demonstrates both the capability of  $\mathrm{LARG}^2$  to generate goals that match requirements as defined in task descriptions and the positive influence of additional code samples on the accuracy of generated functions. Furthermore, it reveals that the naming of functions has minimal impact on performance.

# 5.3 LARG² FOR MULTI TASK REINFORCEMENT LEARNING

In this experiment, we train a policy using Proximal Policy Optimization with default Franka Move parameters. The policy takes as input the task description which is encoded using a pre-trained Google T5-small language model Raffel et al.. For each task, we use the [CLS] token embedding computed by the encoder layer of the model which is defined in  $\mathbb{R}^{512}$ . We concatenate this embedding with the state information of our manipulation environment defined in  $\mathbb{R}^7$  and feed it into a stack of fully connected layers used as policy. This policy is composed of 3 layers using respectively,  $\{512, 128, 64\}$  hidden dimensions. Alternately, as suggested by Jiang et al. (2022), we tested feeding the token embedding into each layer of the stack instead of concatenating it as input but we did not observe improvements.

For the reward generation process, we first set the task-independent reward component leveraging rewards available from IsaacGym for pick and place manipulation. This component handles gripper finger distance to the object, bonuses for lifting the object and penalties for the number of actions to reach the objective. This component, which is therefore common to each task, is not generated. It is added to the task dependant reward generated by  $\mathrm{LARG}^2$  for each task. Details about this process are further discussed in section A.1.3 of the appendix. Reward functions apply goal poses generated according to the task to compute related scores. Figures 7 and 8, respectively present generated goal positions for 9 manipulation tasks, detailed in the appendix, and the success rate of subsequently trained policies.

![](images/79eac620acdb564e20aff2404be4862e663e312720e7c6ccb7b404633721d955.jpg)  
Figure 7: Generated goal position for 9 manipulation tasks.

![](images/976671f929648b654b48ee1ac958829ff89ad537706324dba599799dfd9bd546.jpg)  
Figure 8: Success rate evaluations of MTRL over automatic reward generation.

As a summary,  $\mathrm{LARG}^2$  demonstrates its capability of producing valid reward functions to successfully train and execute MTRL policies from textual task descriptions.

# 6 LIMITATIONS AND FUTURE WORKS

Our experiments have highlighted limitations in LLM reliability to convert user instructions into executable and valid code. Even though our experiments involved highly structured information such as function signature and docstring, which limits the effect of hallucination, the risk of semantic errors cannot be ruled out. To address these limitations, the auto-correction loop described in our paper seems an effective option to be further investigated.

# 7 CONCLUSION

In this paper, we introduce  $\mathrm{LARG}^2$  which enables scalable task-conditioned reinforcement learning from textual descriptions. Our method leverages the in-context learning and code-generation capabilities of large language models to complete or fully generate goal-sampling and reward functions from textual descriptions of tasks. For this purpose, our method incorporates automatic code validation and functional testing. Additionally, our approach augments the contextual information provided to the LLM with supplemental functions to activate a Chain-of-Thought mechanism to further increase the relevance of generated code. We evaluate the capability of our method to translate a series of text-based task descriptions into actionable objectives for GCRL and to generate rewards functions to train MTRL policies for robotic manipulation. Our experiment confirms the benefit of  $\mathrm{LARG}^2$  for aligning textual task descriptions with generated goal and reward functions. We believe it opens a novel and scalable direction for training RL-based policies for robots on the basis of textual instructions. Still, further work remains to address reward generation for long horizon objectives as well as for improvements in supplemental function retrieval using for instance a learning to rank approach.

# REFERENCES

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J. Joshi, Ryan C. Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael S. Ryoo, Grecia Salazar, Pannag R. Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Anand Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Ho Vuong, F. Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale. ArXiv, abs/2212.06817, 2022.  
Eugenio Chisari, Tim Welschehold, Joschka Boedecker, Wolfram Burgard, and Abhinav Valada. Correct me if i am wrong: Interactive learning for robotic manipulation. IEEE Robotics and Automation Letters, 7:3695-3702, 2021.  
Cédric Colas, Ahmed Akakzia, Pierre-Yves Oudeyer, Mohamed Chetouani, and Olivier Sigaud. Language-conditioned goal generation: a new approach to language grounding for RL. CoRR, abs/2006.07043, 2020a. URL https://arxiv.org/abs/2006.07043.  
Cédric Colas, Ahmed Akakzia, Pierre-Yves Oudeyer, Mohamed Chetouani, and Olivier Sigaud. Language-conditioned goal generation: a new approach to language grounding for rl. ArXiv, abs/2006.07043, 2020b.  
Shizhe Diao, Pengcheng Wang, Yong Lin, and Tong Zhang. Active prompting with chain-of-thought for large language models. ArXiv, abs/2302.12246, 2023.  
Marco Dorigo and Marco Colombetti. Robot shaping: Developing autonomous agents through learning. Artificial intelligence, 71(2):321-370, 1994.  
Taylor A. Kessler Faulkner, Elaine Schaertl Short, and Andrea Lockerd Thomaz. Interactive reinforcement learning with inaccurate feedback. 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 7498-7504, 2020.  
Shixiang Gu, Ethan Holly, Timothy Lillicrap, and Sergey Levine. Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. In 2017 IEEE International Conference on Robotics and Automation (ICRA), pp. 3389-3396, 2017. doi: 10.1109/ICRA.2017.7989385.  
Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. 2022a. doi: 10.48550/ARXIV.2201.07207. URL https://arxiv.org/abs/2201.07207.  
Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Noah Brown, Tomas Jackson, Linda Liu, Sergey Levine, Karol Hausman, and Brian Ichter. Inner monologue: Embodied reasoning through planning with language models. 2022b. doi: 10.48550/ARXIV.2207.05608. URL https://arxiv.org/abs/2207.05608.  
Shima Imani, Liang Du, and H. Shrivastava. Mathprompter: Mathematical reasoning using large language models. In Annual Meeting of the Association for Computational Linguistics, 2023. URL https://api.sementicscholar.org/CorpusID:257427208.  
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. CoRR, abs/2202.03629, 2022. URL https://arxiv.org/abs/2202.03629.  
Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. 2022. doi: 10.48550/ARXIV.2210.03094. URL https://arxiv.org/abs/2210.03094.

Michael Kelly, Chelsea Sidrane, K. Driggs-Campbell, and Mykel J. Kochenderfer. Hg-dagger: Interactive imitation learning with human experts. 2019 International Conference on Robotics and Automation (ICRA), pp. 8077-8083, 2018.  
Aviral Kumar, Joey Hong, Anika Singh, and Sergey Levine. Should i run offline reinforcement learning or behavioral cloning? In International Conference on Learning Representations, 2022.  
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Munoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder: may the source be with you!, 2023.  
Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. 2022. doi: 10.48550/ARXIV.2209.07753. URL https://arxiv.org/abs/2209.07753.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. CoRR, abs/1509.02971, 2016.  
Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In ICML, 2016.  
Ashvin Nair, Vitchyr H. Pong, Murtaza Dalal, Shikhar Bahl, Steven Lin, and Sergey Levine. Visual reinforcement learning with imagined goals. In NeurIPS, 2018.  
OpenAI OpenAI, Matthias Plappert, Raul Sampedro, Tao Xu, Ilge Akkaya, Vineet Kosaraju, Peter Welinder, Ruben D'Sa, Arthur Petron, Henrique Pondé de Oliveira Pinto, Alex Paino, Hyeonwoo Noh, Lilian Weng, Qiming Yuan, Casey Chu, and Wojciech Zaremba. Asymmetric self-play for automatic goal discovery in robotic manipulation. ArXiv, abs/2101.04882, 2021.  
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. ArXiv, abs/2203.02155, 2022.  
Matthias Plappert, Marcin Andrychowicz, Alex Ray, Bob McGrew, Bowen Baker, Glenn Powell, Jonas Schneider, Joshua Tobin, Maciek Chociej, Peter Welinder, Vikash Kumar, and Wojciech Zaremba. Multi-goal reinforcement learning: Challenging robotics environments and request for research. ArXiv, abs/1802.09464, 2018.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. URL http://arxiv.org/abs/1910.10683.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2020a.  
Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. ArXiv, abs/1910.10683, 2020b.

Jette Randlov and Preben Alstrom. Learning to drive a bicycle using reinforcement learning and shaping. In Proceedings of the 15th International Conference on Machine Learning (ICML'98), pp. 463-471, 1998.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv, abs/1707.06347, 2017.  
Dhruv Shah, Blazej Osinski, Brian Ichter, and Sergey Levine. Lm-nav: Robotic navigation with large pre-trained models of language, vision, and action. 2022. doi: 10.48550/ARXIV.2207.04429. URL https://arxiv.org/abs/2207.04429.  
Jochen Stüber, Marek Kopicki, and Claudio Zito. Feature-based transfer learning for robotic push manipulation. 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 1-5, 2018.  
Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction. IEEE Transactions on Neural Networks, 16:285-286, 2005.  
Lei Tai, Jingwei Zhang, Ming Liu, Joschka Boedecker, and Wolfram Burgard. A survey of deep network solutions for learning control in robotics: From reinforcement to imitation. arXiv: Robotics, 2016.  
Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters, 2023a.  
Hongru Wang, Rui Wang, Fei Mi, Zezhong Wang, Rui-Lan Xu, and Kam-Fai Wong. Chain-of-thought prompting for responding to in-depth dialogue questions with llm. ArXiv, abs/2305.11792, 2023b.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. ArXiv, abs/2201.11903, 2022.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.  
Thomas Weng, Amith Pallankize, Yimin Tang, Oliver Kroemer, and David Held. Multi-modal transfer learning for grasping transparent and specular objects. IEEE Robotics and Automation Letters, 5:3796-3803, 2020.  
Mats Wiese, Gundula Runge-Borchert, Benjamin-Hieu Cao, and Annika Raatz. Transfer learning for accurate modeling and control of soft actuators. 2021 IEEE 4th International Conference on Soft Robotics (RoboSoft), pp. 51-57, 2021.  
Skyler Wu, Eric Meng Shen, Charumathi Badrinath, Jiaqi Ma, and Himabindu Lakkaraju. Analyzing chain-of-thought prompting in large language models via gradient-based feature attributions. ArXiv, abs/2307.13339, 2023.  
Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, Montse Gonzalez Arenas, Hao-Tien Lewis Chiang, Tom Erez, Leonard Hasenclever, Jan Humplik, Brian Ichter, Ted Xiao, Peng Xu, Andy Zeng, Tingnan Zhang, Nicolas Heess, Dorsa Sadigh, Jie Tan, Yuval Tassa, and Fei Xia. Language to rewards for robotic skill synthesis, 2023.
