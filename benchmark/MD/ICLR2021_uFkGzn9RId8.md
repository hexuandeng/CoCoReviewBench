# THE ACT OF REMEMBERING: A STUDY IN PARTIALLY OBSERVABLE REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Reinforcement Learning (RL) agents typically learn memoryless policies—policies that only consider the last observation when selecting actions. Learning memoryless policies is efficient and optimal in fully observable environments. However, some form of memory is necessary when RL agents are faced with partial observability. In this paper, we study a lightweight approach to tackle partial observability in RL. We provide the agent with an external memory and additional actions to control what, if anything, is written to the memory. At every step, the current memory state is part of the agent's observation, and the agent selects a tuple of actions: one action that modifies the environment and another that modifies the memory. When the external memory is sufficiently expressive, optimal memoryless policies yield globally optimal solutions. Unfortunately, previous attempts to use external memory in the form of binary memory have produced poor results in practice. Here, we investigate alternative forms of memory in support of learning effective memoryless policies. Our novel forms of memory outperform binary and LSTM-based memory in well-established partially observable domains.

# 1 INTRODUCTION

Reinforcement Learning (RL) agents learn policies (i.e., mappings from observations to actions) by interacting with an environment. RL agents usually learn memoryless policies, which are policies that only consider the last observation when selecting the next action. In fully observable environments, learning memoryless policies proves to be efficient and optimal. However, when faced with partially observable environments, RL agents require some form of memory in order to learn optimal behaviours. This is usually accomplished using k-order memories (Mnih et al., 2015), recurrent networks (Hausknecht & Stone, 2015), or memory-augmented neural networks (Oh et al., 2016).

In this paper, we study a lightweight alternative approach to tackle partially observability in RL. The approach consists of providing the agent with an external memory and extra actions to control it (as shown in Figure 1). The resulting RL problem is still partially observable, but if the external memory is sufficiently expressive, then optimal memoryless policies will also yield globally optimal solutions. Previous works that explored this idea using external binary or continuous memories produced poor results with standard RL methods (Peshkin et al., 1999; Zhang et al., 2016). Our work shows that the main issue is with the type of memory they were using, and that RL agents are capable of learning effective strategies to utilize external memories when structured appropriately.

This paper proposes a general framework for studying agents that learn to control external memory in service of partially observable RL. In addition to binary memory, we propose two novel forms of external memory, called Ok and OAk. We study the theory behind learning memoryless policies that jointly decide what to do and what to remember. Finally, we present empirical results that show that Ok and OAk memories are usually more sample efficient than LSTM memories, can solve problems that were unsolvable using LSTMs, and are faster to train. These results suggest interesting avenues for future work in the theory and practice of RL in partially observable environments.

# 2 PRELIMINARIES

RL agents learn how to act by interacting with an environment. Often these environments are modelled as a Markov Decision Process (MDP). An MDP is a tuple  $\mathcal{M} = \langle S, A, R, p, \gamma, \mu \rangle$ , where  $S$  is

![](images/10e9c689615834cfac62189d8800c8e133cfcda3aa356b310db11de1165f4e45.jpg)  
Figure 1: A diagram of a Memory-Augmented Environment.

a finite set of states,  $A$  is a finite set of actions,  $R$  is the finite set of possible rewards,  $p(s', r|s, a)$  defines the dynamics of the MDP,  $\gamma$  is the discount factor, and  $\mu$  is the initial state distribution. We focus on the case where an interaction is divided into episodes. At the beginning of an episode, the environment is set to an initial state  $s_0$ , sampled from  $\mu$ . Then, at time step  $t$ , the agent observes the current state  $s_t \in S$  and executes an action  $a_t \in A$ . In response, the environment returns the next state  $s_{t+1}$  and immediate reward  $r_t$  sampled from  $p(s_{t+1}, r_t | s_t, a_t)$ . The process then repeats.

Agents select actions according to a policy  $\pi(a|s)$ —which is a probability distribution from states to actions. The prediction task is to estimate how "good" a policy is, where the policy is evaluated according to the expected discounted return in any state. This can be done by estimating the action-value function  $q_{\pi}$  of policy  $\pi$ , where  $q_{\pi}(s,a)$  represents the expected discounted return when executing action  $a$  in state  $s$  and following  $\pi$  thereafter.  $q_{\pi}$  is usually estimated using Monte Carlo samples (Barto & Duff, 1994) or TD methods (Sutton, 1988). The control task involves finding the optimal policy  $\pi^{*}$ . This is the policy that maximizes the expected discounted return in every state. To do so, most RL methods rely on the policy improvement theorem, which we discuss in Section 5.

We use a Partially Observable Markov Decision Process (POMDP) formulation to model partial observability. A POMDP is a tuple  $\mathcal{P} = \langle S,O,A,R,p,\omega ,\gamma ,\mu \rangle$  , where  $S,A,R,p,\gamma ,$  and  $\mu$  are defined as in an MDP,  $O$  is a finite set of observations, and  $\omega (o|s)$  is the observation probability distribution. Interacting with a POMDP is similar to an MDP. The environment starts from a sampled initial state  $s_0\sim \mu$  . At time step  $t$  , the agent is in state  $s_t\in S$  , executes an action  $a_{t}\in A$  , receives an immediate reward  $r_t$  , and moves to  $s_{t + 1}$  according to  $p(s_{t + 1},r_t|s_t,a_t)$  . However, the agent does not observe  $s_t$  directly. Instead, the agent observes  $o_t\in O$  , which is linked to  $s_t$  via  $\omega (o_t|s_t)$

# 3 RELATED WORK

Early attempts to perform RL in partially observable domains focused on learning memoryless policies. Jaakkola et al. (1995) identified an RL algorithm that was guaranteed to converge to locally optimal memoryless policies, and similar guarantees have been given in the POMDP literature (Li et al., 2011). Unfortunately, Singh et al. (1994) showed that an optimal memoryless policy  $\pi^{*}(a_{t}|o_{t})$  can be arbitrarily worse than the optimal history-based policy  $\pi^{*}(a_{t}|o_{0},a_{0},\ldots ,o_{t})$  for POMDPs.

Different approaches have been proposed to learn history-based policies  $\pi^{*}(a_{t}|o_{0},\ldots ,o_{t})$  using some form of state-approximation technique. For example, model-based RL methods learn a state representation of histories that enables Markovian prediction of the next observation and immediate reward, and then learns policies over that representation (Littman et al., 2002; Poupart & Vlassis, 2008; Doshi-Velez et al., 2013; Ghavamzadeh et al., 2015; Zhang et al., 2019; Toro Icarte et al., 2019). The focus of our work is on model-free methods, which are the state of the art for solving partially observable problems from low-level inputs (such as images). In model-free RL, history-based policies are approximated using recurrent neural networks (Hausknecht & Stone, 2015; Mnih et al., 2016; Wang et al., 2016; Schulman et al., 2017; Jaderberg et al., 2016), or some form of memory-augmented neural network (Oh et al., 2016; Khan et al., 2017; Hung et al., 2018). They are usually trained using policy gradient methods. These approaches are computationally expensive because they require the backpropagation of gradients through the history of observations and actions for learning history-based policies. In comparison, our approach is much more lightweight - being faster to train than LSTMs and generally having better sample complexity.

On the other hand, it is possible to learn memoryless policies that optimally solve POMDPs. The trick is to give the agent a (large enough) memory and extra actions to write to it. From the agent's perspective, it learns a standard memoryless policy from observations to actions, but the observations now include the state of the memory, and the actions include options for how to alter the memory.

The main purpose of our work is to resurrect this simple idea by understanding why previous work were unable to make it work. We also proposed a unified framework to study agents with external memories and two novel memories that outperform existing forms of external memories.

Concretely, the idea of providing some form of external memory to an agent and actions to modify it goes back to Littman (1993), who discussed a hypothetical agent that could learn to control an external binary memory in support of solving partially observable tasks. Peshkin et al. (1999) reported the first empirical results using tabular RL to learn memoryless policies over such binary memories. While the results were promising in some small environments that required only one bit of external memory to be solved, they did not scale to more complex domains. After Peshkin et al. (1999), there was not much work trying to push this line of research forward. We believe that the reason is that RL agents cannot reliably learn to control binary memories (as shown in our results). That said, there is one recent work that has further explored the idea of modifying external memories using actions. Zhang et al. (2016) proposed to use continuous memories, where each element in the array was a floating point number, instead of binary memories. However, they learned the memoryless policies using imitation learning and pointed out that standard RL methods did not work because the reward signal was insufficient supervision for the agent to understand how to appropriately control the memory. One contribution of our work is to advance our understanding of methods that provide external memory to standard RL agents, and to show that they can work well in practice.

# 4 AGENTS WITH EXTERNAL MEMORY

In this section, we formally define what it means to provide external memory to an agent, and describe several forms of external memory. We will use the following problem to aid explanation:

Example 4.1 (the gravity domain (Toro Icarte et al., 2019)). The gravity domain, shown in Figure 2, consists of an agent (purple triangle), a cookie, and a button. The agent can move in the four cardinal directions and receives a reward of 1 when it eats the cookie. Doing so ends the episode. There is an external force pulling the agent down—i.e., the outcome of the "move-up" action is a downward movement with probability 0.9—which can be turned off (or back on) by pressing the button. Every episode begins with the agent in the bottom left corner and the external force on.

The optimal policy for this problem is to first press the button and then go to the cookie. Since the agent cannot observe the force, optimal behaviour requires memory of the state of the button, meaning that no memoryless policies can solve this problem optimally. However, suppose that the agent was given a single bit that they could write to on every step using the special actions write-1 and write-0. This memory can then be used to record the state of the button, and so an optimal memoryless policy for this augmented problem will optimally solve the gravity domain.

Figure 1 shows a generalization of this idea. From the agent's perspective, they are, as usual, performing actions in an environment and receiving observations and rewards in return. However, they are now interacting with a memory-augmented environment—which consists of a sub-environment (i.e., the original POMDP environment) and a memory. The memory receives an action  $w$  (selected by the agent) and local information coming from the sub-environment  $(o, a, r, o')$  to update its internal state to  $m'$ . We formalize these external memory modules as follows:

Definition 4.1 (external memories). Let  $\mathcal{P} = \langle S,O,A,R,p,\omega ,\gamma ,\mu \rangle$  be a POMDP. An external memory for  $\mathcal{P}$  is a tuple  $\mathcal{M}_{\mathcal{P}} = \langle M,W,\Gamma ,\eta \rangle$ , where  $M$  is a finite set of memory-states,  $W$  is a finite set of memory-writing actions,  $\Gamma (m^{\prime}|m,w,o,a,r,o^{\prime})$  is the memory-writing distribution, and  $\eta$  is the initial memory-state distribution.

An external memory module defines the set of possible memory configurations  $(M)$  and how the agent can manipulate that memory ( $W$  and  $\Gamma$ ). In the one-bit example for the gravity domain,  $M$  consists of the two possible values of the bit (0 or 1),  $W$  consists of the two possible write options of the bit (0 or 1), and the memory-writing distribution  $\Gamma$  updates the bit of memory to 0 or 1 depending on which action was selected. We now define a memory-augmented environment as follows:

Definition 4.2 (memory-augmented environments). A memory-augmented environment is a tuple  $\mathcal{E} = \langle \mathcal{P},\mathcal{M}_{\mathcal{P}}\rangle$  where  $\mathcal{P}$  is a POMDP and  $\mathcal{M}_{\mathcal{P}}$  is an external memory for  $\mathcal{P}$ .

The interaction between an agent and a memory-augmented environment  $\mathcal{E} = \langle \mathcal{P},\mathcal{M}_{\mathcal{P}}\rangle$  is the same as with the original environment, just with an augmented observation and action space. At the beginning of each episode, an initial state  $s_0$ , observation  $o_0$ , and memory state  $m_0$ , are sampled

![](images/3e0bd8e889f0a4fedb0f9750a2c1b6697677c6e5536fdf2ad3d8d87d6eafdf42.jpg)  
Figure 2: Experiments in the gravity domain. We reported the avg. reward per 100 steps.

![](images/2f7c7908a4d0adf15402adc38e008c12f4497c3b0e7bdca7c16bfd3758861d57.jpg)

![](images/3d1cac2269f2594ba871bc78e64509e25b0c75dc0c4520331d34a5a525dadffe.jpg)

![](images/5debbc77584cfa481bc46f570417adad6d55aaf866f0363943d5a11f87064cb7.jpg)

according to  $s_0 \sim \mu$ ,  $o_0 \sim \omega(o_0 | s_0)$ , and  $m_0 \sim \eta$ , respectively. At time step  $t$ , the agent observes  $\bar{o}_t = \langle o_t, m_t \rangle$  and executes an action  $\bar{a}_t = \langle a_t, w_t \rangle \in A \times W$  in  $\mathcal{E}$ . Consequently, the sub-environment samples an immediate reward  $r_t$  and the next state  $s_{t+1}$  according to  $p(s_{t+1}, r_t | s_t, a_t)$ . The sub-environment also samples the next observation  $o_{t+1} \sim \omega(o_{t+1} | s_{t+1})$ . The memory state is then updated to  $m_{t+1}$  according to  $\Gamma(m_{t+1} | m_t, w_t, o_t, a_t, r_t, o_{t+1})$ . Finally, the agent receives the immediate reward  $r_t$  and the next observation  $\bar{o}_{t+1} = \langle o_{t+1}, m_{t+1} \rangle$ , and the process repeats.

Any standard RL algorithm can be used to find a memoryless policy for a given memory-augmented environment  $\mathcal{E} = \langle \mathcal{P},\mathcal{M}_{\mathcal{P}}\rangle$ . We note that the optimal memoryless policy for  $\mathcal{E}$  must be at least as good as the optimal memoryless policy for the original POMDP  $\mathcal{P}$ . This is because  $\mathcal{E}$  and  $\mathcal{P}$  share a reward function, and the agent can always choose to ignore the memory. That said, if the external memory module  $\mathcal{M}_{\mathcal{P}}$  is "expressive enough," then optimal memoryless policies for  $\langle \mathcal{P},\mathcal{M}_{\mathcal{P}}\rangle$  will be just as good as the optimal policy for  $\mathcal{P}$ . This is shown formally in Appendix A.2.

# 4.1 EXTERNAL MEMORY MODULES

Let us now consider several examples of external memory modules. We begin by showing how binary memories (Littman, 1993; 1994; Peshkin et al., 1999) can be expressed using this formalism. We use the notation  $\mathsf{B}k$  to refer to a binary memory of  $k$  bits:

Definition 4.3 (Bk memories). Let  $\mathcal{P} = \langle S,O,A,R,p,\omega ,\gamma ,\mu \rangle$  be a POMDP. A Bk memory for  $\mathcal{P}$  is a  $k$  -bit external memory  $\mathcal{M}_{\mathcal{P}} = \langle M,W,\Gamma ,\eta \rangle$  , where  $M = \{0,1\} ^k$ $W = \{0,1\} ^k$ $\eta (0^{k}) = 1$  (zero otherwise), and  $\Gamma (m^{\prime}|m,w,o,a,r,o^{\prime}) = 1$  if and only if  $m^{\prime} = w$  (zero otherwise).

Bk memories are especially attractive given how flexible and expressive they are. Unfortunately, learning to control Bk memories is difficult. Figure 2 shows the performance of tabular q-learning (Watkins & Dayan, 1992) and 5-step actor-critic (Grondman et al., 2012) in the gravity domain using different types of external memories. In the figure, None represents not using any external memory, and K1, O1, and OA1 are explained below. Notice that neither q-learning nor 5-step actor-critic were able to understand how to use the B1 memory to consistently solve the gravity domain.

There are two main problems with  $\mathrm{Bk}$  memories. First, the action space grows exponentially with  $k$ . Second,  $\mathrm{Bk}$  memories can be too flexible in that the agent can modify the memory arbitrarily and irrespective of what has actually happened, and thereby completely alter what the agent believes about its current situation. For example, recall that in the gravity domain, the agent should use the memory to record whether gravity is on (0) or off (1). However, if the agent incorrectly decides to record that the gravity is off prematurely (i.e., before touching the button), it will believe it has transitioned from a state with low expected reward (where it first has to go to the button) to a state with high expected reward (where the agent wrongly believes that it can go directly to the cookie without any opposition from gravity). This can lead to an unstable learning process, as shown below.

The main motivation behind our proposed  $Ok$  memories is to alleviate these issues.  $Ok$  memories are a generalization of  $k$ -order memories, which are buffers of a fixed size that contain the last  $k$  observations. We refer to  $k$ -order memories as  $Kk$  memories, where the second  $k$  indicates the size of the buffer. We formally describe them as external memories in Appendix A.3. Note that K1 represents a 2-order memory since actions are taken over  $\langle o, m \rangle$ . The main disadvantage of  $k$ -order memories is that they do not allow the agent to remember events that occurred more than  $k$  steps in the past.  $Ok$  memories solve this issue by letting the agent decide whether to push the current

![](images/be437b67bea0a04b54dbc2cc5125e48e9d90907822f5c6519c311e3a09b00d5d.jpg)  
Figure 3: Performance of a greedy policy every 10,000 training steps in the recall task.

![](images/e31992d49e618289839ecf590ce91774dd1cb1165f226d2b9ec9f8169509d12f.jpg)

![](images/0431bbdb45ffda147b38ba6fb82a7d39d09e2642c67c70f7707a6cbfaa4d8363.jpg)

![](images/960bbfa725025be4215cfdd91921015c5a050c845dc3a7f639d2d8f0c9450840.jpg)

observation into the k-order buffer or not. Note that, since the agent can only push into the buffer observations that did occur, Ok memories are unable to imagine events that have not yet happened.

Definition 4.4 (Ok memories). Let  $\mathcal{P} = \langle S,O,A,R,p,\omega ,\gamma ,\mu \rangle$  be a POMDP. An Ok memory for  $\mathcal{P}$  is a memory buffer (of size  $k$ )  $\mathcal{M}_{\mathcal{P}} = \langle M,W,\Gamma ,\eta \rangle$ , where  $M = (O\cup \{\emptyset \})^k$ ,  $W = \{\top, \bot\}$ ,  $\eta (\emptyset^k) = 1$  (zero otherwise), and  $\Gamma(m'|m,w,o,a,r,o') = 1$  if  $w = \bot$  and  $m' = m$ , or  $w = \top$ ,  $m = \langle o^1,o^2,\dots,o^k\rangle$ , and  $m' = \langle o^2,\dots,o^k,o\rangle$  (zero otherwise).

Ok memories have strong empirical performance in the gravity domain (see Figure 2), outperforming B1 and K1. That said, Ok memories are insufficient in domains where the history of actions matters. For such domains, we propose OAk memories. An OAk memory is similar to an Ok memory but when the agent chooses to push to its buffer, the information includes the current observation and the action that is executed in the sub-environment. OAk memories are defined in Appendix A.3.

We note that optimal memoryless policies over  $\mathrm{B}k$ ,  $\mathrm{O}A k$ ,  $\mathrm{O}k$ , and  $\mathrm{K}k$  will optimally solve the original POMDP for some value of  $k$ , under some assumptions. This is shown in Appendix A.4.

# 5 LEARNING POLICIES IN MEMORY-AUGMENTED ENVIRONMENTS

The objective of this section is to understand the theory behind learning memoryless policies over memory-augmented environments and to provide insights into why Ok and OAk memories tend to perform better than Bk memories. We begin with the following example.

Example 5.1 (a recall task). The recall task is a partially observable environment with only one possible observation,  $o$  (i.e., all states appear the same), and three actions,  $a_1$ ,  $a_2$ , and  $a_3$ . The episode ends after performing three actions. If the agent executes actions  $a_1$ ,  $a_2$ , and  $a_3$  (in that order), it gets a reward of 1; otherwise it gets a reward of 0.

The purpose of the recall task is to show that even if a memoryless policy for a memory-augmented environment is globally optimal, the memory-augmented environment itself might not be an MDP. Figure 3 shows a transition diagram for the recall task using an OA1 memory. Since the observation is always the same, the different states that the agent encounters only differ by the state in the memory. In the diagram, nodes represent the memory states and the transitions show how the memory is updated by the agent's actions. Note that node  $i$  represents that the memory buffer contains  $\langle o, a_i \rangle$  and that the buffer starts empty  $(\emptyset)$ . For the action labels, the first number indicates the action number in the sub-environment (1, 2, or 3). The second character represents the memory action. For instance,  $2^{\top}$  represents that the agent executed  $a_2$  in the sub-environment and saved that action into the memory buffer. The label o/w stands for otherwise. The blue arrows show a deterministic memoryless policy that optimally solves this problem. That is, execute  $1^{\top}$ , then  $2^{\top}$ , and finally  $3^{\top}$ .

Notice that this memory-augmented environment has a memoryless policy that is optimal for the original POMDP, but it is not an MDP. The reason is that the reward given by the bottom transition  $3^{\top}$  will be 0 or 1 depending on the history. If the agent follows the blue path, it will give a reward of one. In contrast, if the agent follows the red arrow, it will get a reward of zero. Something similar occurs when using B2 memory, which is the smallest Bk memory that can encode an optimal policy for the recall task. This "non-Markovianess" impacts the performance of RL agents that explicitly exploit the Markovian assumption. For example, if we run q-learning and evaluate the performance of the greedy policy (i.e., without exploration) every 10,000 steps, we see that q-learning does not converge. Instead, q-learning jumps between an optimal policy and a zero reward policy, as shown in Figure 3.

![](images/71b000464cae106ecdbfa425dbf0e46255a8b77c072d72bc65f054f4a5c0c0ed.jpg)  
Figure 4: Experiments in the gravity domain. We reported the avg. reward per 100 steps.

![](images/c065d0f02d94808ef96a27f37d375e7d6501c046e59c50380ce3eee87f56dd22.jpg)

![](images/4567946638346ce05a71c82048c72d4cdee45d60596e23650e599c7ec163de0b.jpg)

Now that we know that memory-augmented environments are not MDPs, we focus on proving that they are POMDPs. Such a proof can be found in Appendix A.1 and has important repercussions. In particular, all the theory for learning memoryless policies for POMDPs (Littman, 1994; Singh et al., 1994; Jaakkola et al., 1995; Li et al., 2011) also applies to memory-augmented environments. We explore this further in two parts: the prediction problem and the control problem.

# 5.1 HOW TO EVALUATE POLICIES IN MEMORY-AUGMENTED ENVIRONMENTS

For a given POMDP  $\mathcal{P} = \langle S,O,A,R,p,\omega ,\gamma ,\mu \rangle$  and a memoryless policy  $\pi (a|o)$ , the policy prediction problem consists of estimating  $q_{\pi}(o,a)$ . The POMDP theory shows that Monte-Carlo estimates are guaranteed to converge to the real values of  $q_{\pi}(o,a)$ , though they do have high variance. In contrast, TD estimates have lower variance but might not converge to  $q_{\pi}(o,a)$  (Singh et al., 1994).

Failing to correctly estimate  $q_{\pi}(o, a)$  is the reason behind q-learning's instability in the recall task (Figure 3). For instance, let  $\pi$  be the optimal policy represented by blue arrows in OA1, then the real q-value for the red arrow  $q_{\pi}(\emptyset, 2\top)$  is zero (the agent gets no reward if it executes  $a_2$  in the first action). However, a one-step TD estimate would converge to  $q_{\pi}(\emptyset, 2\top) = 0 + \gamma q_{\pi}(2, 3\top) = \gamma$ . This is a problem since now  $q_{\pi}(\emptyset, 2\top) > q_{\pi}(\emptyset, 1\top) = \gamma^2$  (for  $\gamma \in (0, 1)$ ), and so q-learning will move from the current optimal policy  $\pi$  to the zero reward policy that executes  $2\top$  in  $\emptyset$ . We refer to these types of transitions as non-Markovian shortcuts. Note that, as Figure 3 shows, the B2 memory has more non-Markovian shortcuts than OA1. This is why q-learning over B2 is more unstable than q-learning over OA1 in this domain. More generally, we would expect that  $\mathrm{Bk}$  memories introduce more non-Markovian shortcuts than OAK memories since they are more flexible, which could partially explain the better empirical performance of OAK and Ok memories.

There are two approaches that can mitigate this problem. The first is to use  $n$ -step TD estimates, with a large enough value of  $n$ . As Figure 4 shows, the performance of 20-step actor-critic in the gravity domain is far superior to 5-step actor-critic. The second is to increase the size of the memory, since doing so tends to remove non-Markovian shortcuts. This is also shown in Figure 4, as 5-step actor-critic performs better when using O2 or OA2, than when using O1 or OA1.

# 5.2 HOW TO IMPROVE POLICIES IN MEMORY-AUGMENTED ENVIRONMENTS

We now focus our attention on the second part of the problem: how to use q-value estimates to find better policies. To do so, most RL algorithms exploit the policy improvement theorem. For MDPs, this theorem guarantees that updating the current policy  $\pi$  by any amount towards the greedy policy  $\tau(s) = \arg \max_{a \in A} q_{\pi}(s, a)$  will lead to better policies (Watkins, 1989; Sutton & Barto, 2018).

When learning memoryless policies for POMDPs, it is known that the policy improvement theorem only works locally (Jaakkola et al., 1995). To see why, recall that we are estimating q-values over observations  $o \in O$  and not over states  $s \in S$ . Formally, the q-value  $q_{\pi}(o,a)$  is defined as follows:  $q_{\pi}(o,a) = \sum_{s \in S} P_{\pi}(s|o) q_{\pi}(s,a)$ , for all  $o \in O, a \in A$ . Here,  $P_{\pi}(s|o)$  is the probability of being in state  $s$  given that the observation is  $o$ , when following policy  $\pi$ . Intuitively, the policy improvement theorem does not work generally here because moving  $\pi(o)$  towards  $\tau(o) = \arg \max_{a \in A} q_{\pi}(o,a)$  increases the expectation over  $q_{\pi}(s,a)$  without considering how  $P_{\pi}(s|o)$  might change. Conversely, the policy improvement theorem works locally because updating  $\pi$  by a small amount will also

![](images/8bebe8051e65dec231e6060be2b6b2b75b49e18e40a401bf5c68aabf059dac6d.jpg)  
Figure 5: Tabular experiments in a variation of the recall task (details in Appendix B).

![](images/d023378d3fb2795e0d3a7f9d9fe43dbcba5a7ecd2f7ea471b752316759e29d85.jpg)

only have a small effect on  $P_{\pi}(s|o)$ , making such a difference insignificant. Therefore, a policy learning method that takes small update steps is guaranteed to converge to locally optimal memoryless policies—explaining why actor-critic converges smoothly in the gravity domain (Figure 2). Unfortunately, convergence to optimal memoryless policies is not guaranteed for general POMDPs.

Since memory-augmented environments are a form of POMDP, this local convergence guarantee also applies to them. As such, if the memory can represent the optimal policy, then that solution will be stable given accurate q-value estimations. This raises the question of what conditions for memory would guarantee convergence to a globally optimal memoryless policy. To investigate this topic, we considered an idealized version of Jaakkola et al. (1995)'s approach. This agent starts from a random policy  $\pi$  and uses an oracle to compute  $q_{\pi}(o,a)$  for all  $o\in O$  and  $a\in A$ . Then, it moves  $\pi (o)$  towards  $\arg \max_{a\in A}q_{\pi}(o,a)$  a small step  $\delta$  for all  $o\in O$ , and repeats.

Figure 5 shows the behaviour of this algorithm on a variant of the recall task. The rewards were selected to encourage convergence to suboptimal solutions (more details in Appendix B). In this environment, OA1 and B1 are enough to encode a memoryless policy that is globally optimal. However, note that OA1 converges to a suboptimal solution. Therefore, memory-augmented environments might converge to suboptimal solutions even if the memory is expressive enough to encode globally optimal policies. We do note that this problem vanishes as we increase the size of the memory in this domain. Unfortunately, convergence to an optimal memoryless policy cannot be guaranteed, even for memories that can model the belief states, as we prove for  $\mathsf{Bk}$  in Appendix C.

# 5.3 SUMMARY: FROM THEORY TO PRACTICE

The theory suggests that the best approaches for learning effective memoryless policies in memory-augmented environments are methods that exploit the policy improvement theorem locally and evaluate policies using Monte-Carlo estimates (or n-step TD methods), such as n-step actor-critic, A3C (Mnih et al., 2016), or PPO (Schulman et al., 2017). Our empirical evidence also suggests the use of Ok or OAk memories over Bk memories. While the core of our experimental analysis uses PPO, we also tested pure TD methods, including Sarsa(λ) (Seijen & Sutton, 2014) and DDQN (Van Hasselt et al., 2016). Those results, which are shown in Appendix D.6, also favor Ok memories. Finally, note that integrating external memories into existing RL toolkits is trivial. For instance, it takes less than 40 lines of code to integrate each external memory into OpenAI gym (Brockman et al., 2016).

# 6 EXPERIMENTAL EVALUATION

We ran experiments on a variety of environments with different types of external memory, including our new Ok and OAk memories, as well as the existing k-order memories (Kk) (Mnih et al., 2015) and binary memories (Bk) (Littman, 1993). Below, we present results when using PPO (Schulman et al., 2017) and these memories. We also experimented with using no memory (None) and when using an LSTM. Figure 6 shows the results. Each line is the average reward per episode over 30 runs and the shadow area represents half a standard deviation. Details of the domains, hyperparameters, and network architectures can be found in Appendix D. We will release our code upon publication.

The left column shows results in the Hallway environments. These environments have been shown to be difficult for PPO with LSTM-based memory in previous work (Toro Icarte et al., 2019) and, indeed, we were also unable to get PPO with LSTMs to perform any better than a random policy. In contrast, PPO with OA6 and O6 memories is able to solve these tasks.

![](images/f54eb199e3d07b924a9847f465efe234a33d637915ce3775a8a9443bbaebdfd8.jpg)

![](images/64ba57ecb36e18ae1e0f6aea02999fb4af9f0f7e5639daf8457b33a8ee6a8c07.jpg)

![](images/af8af6c6280b1f87ea0de6da937636dda87faf06399b0cc9abb7c62ec24cf8e0.jpg)

Figure 6: Results over partially observable benchmarks using PPO and different memories.  
![](images/c5c3edffe63807086ea6b62adccc75ca78faf0e7e39a2693dfc976ad0869041e.jpg)  
K6 B6 O6 OA6 None K3 B3 OA3 LSTM

![](images/a6cdb22f7a7897a55f53da28859130aded6c2bb841e78cdd2c1aff36fcd752d7.jpg)

![](images/ab1e6d9d705df0bcaa354b5bafe5c7639aa8206f27b4d13fa87574e3668c6cbd.jpg)

The middle column shows results in the MiniGrid environment (Chevalier-Boisvert & Willems, 2018). We experimented with the RedBlueDoors and MemoryS7 environments because they were specifically designed to test the agent's memory capabilities. We also decreased the agent's field of view from  $8 \times 8$  to  $3 \times 3$  cells to make these problems more challenging. In both cases, O3 and OA3 perform best, as they consistently converge to good solutions on all runs. In contrast, the LSTM performance was unreliable: we note that around half of the LSTM runs converged to poor policies.

The previous results used feed-forward networks for function approximation in grid-like domains. To test our approach in visually complex domains using convolutional networks, we also experimented with two Atari games: Pong and Seaquest. For these domains, we only gave the agent one frame of the game at a time (aside from the current memory state) and followed Machado et al. (2018)'s recommendations for making the environment stochastic. These domains are almost fully-observable, so it is unreasonable to expect Ok, Bk, or OAk to outperform a k-order memory. Still, O3 has comparable performance to K3 in Pong and outperformed LSTMs in Seaquest. This shows that Ok memories can work well in visually complex domains. Note that OA3 performs well in Atari when trained by DDQN (see Appendix D.6) but it does not when using PPO.

Finally, we note that learning memoryless policies is usually faster than learning history-based policies. In fact, training PPO with an Ok memory was between 1.06 to 9.85 times faster than training PPO with an LSTM when using CPUs and between 1.71 to 2.94 times faster when using GPUs. The complete list of speedups can be found in Appendix D.5.

# 7 CONCLUDING REMARKS

This work presented a lightweight approach to tackling partially observable RL. We provided the agent with an external memory and extra actions to write to it, and then used RL to learn a memoryless policy that jointly decides what to do and what to remember. This idea has been around since the 90s, but this is the first work to show how to make it work well in practice. The key step was to study the theory behind memory-augmented environments and to use that theory to propose novel forms of memories that support learning. Experimental results confirmed the effectiveness of our approach. Using the same RL agent, our external memories outperformed LSTM memories while being also faster to train and trivial to implement. Our results suggests a broad array of topics for future exploration from exploring the effectiveness of other forms of memories to furthering the theory of RL in memory-augmented environments.

# REFERENCES

Andrew Barto and Michael Duff. Monte Carlo matrix inversion and reinforcement learning. In Proceedings of the 7th Conference on Advances in Neural Information Processing Systems (NIPS), pp. 687-694, 1994.  
Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253-279, 2013.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI gym. CoRR, abs/1606.01540, 2016. URL http://arxiv.org/abs/1606.01540.  
Maxime Chevalier-Boisvert and Lucas Willems. Minimalistic gridworld environment for OpenAI gym. https://github.com/maximecb/gym-minigrid, 2018.  
Finale Doshi-Velez, David Pfau, Frank Wood, and Nicholas Roy. Bayesian nonparametric methods for partially-observable reinforcement learning. IEEE transactions on pattern analysis and machine intelligence, 37(2):394–407, 2013.  
Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, Aviv Tamar, et al. Bayesian reinforcement learning: A survey. Foundations and Trends in Machine Learning, 8(5-6):359-483, 2015.  
Ivo Grondman, Lucian Busoniu, Gabriel A. D. Lopes, and Robert Babuska. A survey of actor-critic reinforcement learning: Standard and natural policy gradients. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 42(6):1291-1307, 2012.  
Matthew Hausknecht and Peter Stone. Deep recurrent q-learning for partially observable MDPs. In AAAI Fall Symposium on Sequential Decision Making for Intelligent Agents (AAAI-SDMIA15), 2015.  
Christopher Hesse, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. OpenAI baselines. https://github.com/openai/baselines, 2017.  
Chia-Chun Hung, Timothy Lillicrap, Josh Abramson, Yan Wu, Mehdi Mirza, Federico Carnevale, Arun Ahuja, and Greg Wayne. Optimizing agent behavior over long time scales by transporting value. CoRR, abs/1810.06721, 2018. URL http://arxiv.org/abs/1810.06721.  
Tommi Jaakkola, Satinder P. Singh, and Michael I. Jordan. Reinforcement learning algorithm for partially observable Markov decision problems. In Proceedings of the 8th Conference on Advances in Neural Information Processing Systems (NIPS), pp. 345-352, 1995.  
Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z. Leibo, David Silver, and Koray Kavukcuoglu. Reinforcement learning with unsupervised auxiliary tasks. arXiv preprint arXiv:1611.05397, 2016.  
Arbaaz Khan, Clark Zhang, Nikolay Atanasov, Konstantinos Karydis, Vijay Kumar, and Daniel D. Lee. Memory augmented control networks. arXiv preprint arXiv:1709.05706, 2017.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Yanjie Li, Baoqun Yin, and Hongsheng Xi. Finding optimal memoryless policies of POMDPs under the expected average reward criterion. European Journal of Operational Research, 211(3):556-567, 2011.  
Michael L. Littman. An optimization-based categorization of reinforcement learning environments. In From Animals to Animats 2: Proceedings of the Second International Conference on Simulation of Adaptive Behavior, pp. 262-270, 1993.  
Michael L. Littman. Memoryless policies: Theoretical limitations and practical results. In Proceedings of the 3rd International Conference on Simulation of Adaptive Behavior (SAB), pp. 238-245, 1994.

Michael L. Littman, Richard S. Sutton, and Satinder Singh. Predictive representations of state. In Proceedings of the 15th Conference on Advances in Neural Information Processing Systems (NIPS), pp. 1555-1561, 2002.  
Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents. Journal of Artificial Intelligence Research, 61:523-562, 2018.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), pp. 1928-1937, 2016.  
Junhyuk Oh, Valliappa Chockalingam, Satinder Singh, and Honglak Lee. Control of memory, active perception, and action in apache. In Proceedings of the 33rd International Conference on Machine Learning (ICML), pp. 2790-2799, 2016.  
Leonid Peshkin, Nicolas Meuleau, and Leslie Pack Kaelbling. Learning policies with external memory. In Proceedings of the 16th International Conference on Machine Learning (ICML), pp. 307-314, 1999.  
Pascal Poupart and Nikos Vlassis. Model-based Bayesian reinforcement learning in partially observable domains. In Proceedings of the 10th International Symposium on Artificial Intelligence and Mathematics (ISAIM), pp. 1-2, 2008.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Harm Seijen and Richard S. Sutton. True online TD (lambda). In Proceedings of the 31st International Conference on Machine Learning (ICML), pp. 692-700, 2014.  
Satinder P. Singh, Tommi Jaakkola, and Michael I. Jordan. Learning without state-estimation in partially observable Markovian decision processes. In Proceedings of the 11th International Conference on Machine Learning (ICML), pp. 284-292, 1994.  
Richard S. Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3(1):9-44, 1988.  
Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.  
Rodrigo Toro Icarte, Ethan Waldie, Toryn Q. Klassen, Rick Valenzano, Margarita P. Castro, and Sheila A. McIlraith. Learning reward machines for partially observable reinforcement learning. In Proceedings of the 32nd Conference on Advances in Neural Information Processing Systems (NeurIPS), pp. 15523-15534, 2019.  
Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI), pp. 2094-2100, 2016.  
Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. Sample efficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224, 2016.  
Christopher J. C. H. Watkins and Peter Dayan. Q-learning. Machine learning, 8(3-4):279-292, 1992.  
Christopher John Cornish Hellaby Watkins. Learning from delayed rewards. PhD thesis, King's College, Cambridge, 1989.

Lucas Willems. RL starter files. https://github.com/lcswillems/rl-starter-files, 2019.

Amy Zhang, Zachary C. Lipton, Luis Pineda, Kamyar Azizzadenesheli, Anima Anandkumar, Laurent Itti, Joelle Pineau, and Tommaso Furlanello. Learning causal state representations of partially observable environments. arXiv preprint arXiv:1906.10437, 2019.

Marvin Zhang, Zoe McCarthy, Chelsea Finn, Sergey Levine, and Pieter Abbeel. Learning deep neural network policies with continuous memory states. In Proceedings of the 2016 IEEE International Conference on Robotics and Automation (ICRA), pp. 520-527, 2016.
