# ANALYSIS OF MEMORY ORGANIZATION FOR DYNAMIC NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

An increasing number of neural memory networks have been developed, leading to the need for a systematic approach to analyze and compare their underlying memory structures. Thus, in this paper, we first create a framework for memory organization and then compare four popular dynamic models: vanilla recurrent neural network, long short term memory, neural stack and neural RAM. This analysis helps to open the dynamic neural networks' black box from the memory usage prospective. Accordingly, a taxonomy for these networks and their variants is proposed and proved using a unifying architecture. With the taxonomy, both network architectures and learning tasks are classified into four classes. And a one-to-one mapping is built between them to help practitioners select the appropriate architecture. To exemplify each task type, four synthetic tasks with different memory requirements are developed. Moreover, we use two natural language processing applications to evaluate the methodology in a realistic setting.

# 1 INTRODUCTION

Recurrent neural networks have been extensively studied and enjoy their success in a lot of sequence learning problems. Elman and Jordan propose the first classic version of recurrent network (RNN) which introduces memory by adding a feedback from the hidden layer to itself for sequence recognition. Elman (1990) Jordan (1986). They are often referred to as vanilla RNN (vRNN) nowadays. Although vRNN is theoretically Turing complete if well-trained Doya (1993), it's usually ineffective when the sequence is long.

Many dynamic neural networks have emerged recently to improve the vRNN architecture. (Recurrent, dynamic, and memory neural network are used interchangeably in this paper.) Some of them adopt internal memory; some adopt external memory; some adopt logic gates; while others adopt an attention mechanism. As expected, all of them have advantages for some specific tasks, but it's hard to decide which one is optimal for a new task unless we have a clear understanding of the functions of all memory networks' components. Intuitively, we all know that if the network possesses more components, it can make use of more information, but what kind of the extra information they are using and how useful this extra information is, are still not fully understood in the current literatures. Thus, the major goal of this paper is to open the recurrent neural networks' black box from the memory usage prospective. We illustrate the role and importance of memory by first principles, which is indispensable to continue developing better memory architectures, and that can also help debug these networks. At least in this respect we think that the message of this paper is clear and important for the neural network community. A secondary goal is to summarize all these popular models in a systematic manner and employ the knowledge gained from the different characteristics of these memory structures to help users select the type of memory network given the type of problem. We do so by proposing a taxonomy and connecting models' relative expressive power to the memory requirement of different tasks.

# 2 RELATED WORK

Among the abundance of recurrent network papers, very few focus on understanding and analysis. Omlin & Giles (1996) discussed how vRNN behaves like deterministic finite-state automata. Gers & Schmidhuber (2001) Rodriguez (2001) Schmidhuber et al. (2002) compared LSTM Hochreiter &

Schmidhuber (1997) and vRNN's performance on some context-free/sensitive language. Collins et al. (2016) studied capacity of recurrent nets and how difficult they are to train. Karpathy et al. (2015) visualized long-term interactions and representations learned by recurrent networks. Greff et al. (2017) empirically studied the importance of various computational components of LSTM. Jozefowicz et al. (2015) evaluated a variety of recurrent neural network architectures and tried to find the best one. Chung et al. (2014) evaluated GRU Cho et al. (2014) compared to LSTMs. These works usually study the performance of networks based on the output error, our work focuses more on how these networks encoded information in order to solve a problem. What's more, these works are based on the basic RNNs and gated recurrent networks such as LSTM and GRU, while our work includes generalized recurrent networks such as neural stack Sun (1993) Joulin & Mikolov (2015), neural Turing machine Graves et al. (2014), etc., which completes and updates the literature.

# 3 MEMORY STRUCTURE ANALYSIS

In this section, we will analyze memory structures of four popular recurrent neural networks: vRNN, LSTM, neural stack and neural RAM. Attention is paid to how their underlying memory organizations lead to different features and expressive power.

# 3.1 VRNN

The vRNN network Jordan (1986) is composed of three layers: input, hidden recurrent and output layer. Besides all the feed forward connections, there is a feedback connection from the hidden layer to itself. The architecture of it is shown in Fig.1(a). The dynamics of the hidden layer can be written as,

$$
\mathbf {h} _ {t} = f \left(\mathbf {w} _ {x h} ^ {\mathrm {T}} \mathbf {x} _ {t} + \mathbf {w} _ {h h} ^ {\mathrm {T}} \mathbf {h} _ {t - 1} + \mathbf {b} _ {h}\right), \tag {1}
$$

$$
\mathbf {o} _ {t} = f \left(\mathbf {w} _ {h o} ^ {\mathrm {T}} \mathbf {h} _ {t} + \mathbf {b} _ {o}\right), \tag {2}
$$

where  $\mathbf{x}_t$ ,  $\mathbf{h}_t$  and  $\mathbf{o}_t$  are the input, hidden state and output vector at time  $t$ . We use  $\mathbf{w}$  and  $\mathbf{b}$  to represent weight and bias of corresponding sizes in this paper.  $f(x)$  is the nonlinear activation function.

![](images/dfd930cad20b4dd23151635a3fb73831341c6c046a8640aeedd3d8f499415f48.jpg)  
(a) Network architecture

![](images/a6316a3eb289d147443f4f7ad1cb5e36bfecc2cc0cb45cceb2101d5a2c4215d2.jpg)  
(b) Memory visualization  
Figure 1: vRNN

vRNN induces memory by encoding the past information in its hidden state units  $\mathbf{h}_t$ . Thus, the memory of vRNN is called state memory or internal memory. Fig.1(b) shows the state transition diagram of vRNN, where  $s0, s1, \ldots, s4$  represent the state at time  $t_0, t_1, \ldots, t_4$  respectively. The arrows show the variables' dependency relationship. Here state  $s1$  is decided only by  $s0$ ,  $s2$  is decided only by  $s1$  and so on. (All the memory visualization figures in this paper ignore the current input.) As the number of hidden units is limited in practice, there is always a compromise between memory depth and memory resolution in the vRNN De Vries & Principe (1992). For long memory depths sequences, vRNN needs a very large number of hidden units to achieve an acceptable accuracy. If the sequences are composed by symbols or discrete numbers, this can also be understood from Markov transition model prospective. To be specific, vRNN tries to learn a first order Markov transition model (with transition probability 1) where the current state is decided only by current input and the state at one previous step. Thus for first order Markov sequences, since the state space is not very large (the number of state is less than the size of input symbols' alphabet), vRNN always performs well. However, for higher order Markov sequences or sequences that don't have Markov property, vRNN still tries its best to build a first order Markov state model, which will result in a very large state space (it has to combine several old states into a new state). The compromise between memory depth and memory resolution (which are related to the number and temporal resolution of the states) would make vRNN not suitable for these kinds of sequences.

# 3.2 LSTM

LSTM was proposed to deal with the vanishing gradient problem of vRNN. In this section, we will analyze how LSTM provides more flexibility from the memory usage prospective. Different from vRNN, in the classical LSTM as shown in Fig.2(a), the feedback connection of hidden layer has to go through an external memory  $\mathbf{m}_t$ ,

$$
\mathbf {c} _ {t} = f \left(\mathbf {w} _ {h c} ^ {\mathrm {T}} \mathbf {h} _ {t} + \mathbf {b} _ {c}\right), \tag {3}
$$

$$
\mathbf {m} _ {t} = g _ {i, t} \mathbf {c} _ {t} + g _ {f, t} \mathbf {m} _ {t - 1}, \tag {4}
$$

$$
\mathbf {r} _ {t} = \mathbf {m} _ {t}, \tag {5}
$$

$$
\mathbf {h} _ {t} = f \left(\mathbf {w} _ {x h} ^ {\mathrm {T}} \mathbf {x} _ {t} + \mathbf {w} _ {r h} ^ {\mathrm {T}} g _ {o, t} \mathbf {r} _ {t - 1} + \mathbf {b} _ {h}\right), \tag {6}
$$

where  $\mathbf{h}_t$  (or  $\mathbf{c}_t$ ) is the state of the network. The external memory  $\mathbf{m}_t$  is a combination of  $\mathbf{m}_{t-1}$  and current state  $\mathbf{c}_t$ . If  $g_{i,t} = 0$  and  $g_{f,t} = 1$  for several successive time steps, the content saved in the external memory  $\mathbf{m}_t$  would be the long term memory of the system.

This external memory  $\mathbf{m}_t$  adds more flexibility to the state transition diagram. As shown in Fig.2(b), the current state  $s_t$  (represented by hidden state  $\mathbf{h}_t$ ) depends on either the previous one state  $s_{t-1}$  or the external memory  $\mathbf{m}_{t-1}$  (if forget gate  $g_{f,t} = 0$ , input gate  $g_{i,t} = 1$ ,  $s_t$  depends on  $s_{t-1}$ ; if  $g_{f,t} = 1$ ,  $g_{i,t} = 0$ ,  $s_t$  depends on  $\mathbf{m}_{t-1}$ ; if  $0 < g_{f,t} < 1$ ,  $0 < g_{i,t} < 1$ ,  $s_t$  depends on both  $\mathbf{m}_{t-1}$  and  $s_{t-1}$ ). Calculation details of these gates are in Appendix A.1). For example,  $s1$  depends on one previous state  $s0$  illustrated by the blue arrows,  $s7$ ,  $s8$  depend on the long term memory  $M00$  illustrated by the yellow arrows,  $s9$  depends on both the previous state  $s8$  and

the long term memory  $M00$ . The introduced external memory circumvents the compromise between the memory depth versus memory resolution that is always present in the state memory in vRNN. For instance, for a 10th order binary Markov sequence whose state dependence relationship is  $s_t = f(s_{t-1}, s_{t-10})$ , vRNN has to learn a state space with  $2^{10}$  state (it has to combine 10 states into a new state), however, LSTM only needs to learn a state model with 2 states and an external memory storing the state information 10 steps before. By constructing this short path between long term memory and current state, LSTM works much better than vRNN for sequences that skip intermediate values of time dependencies.

Although LSTM is more effective than vRNN, we have to know its limitations. For example, if there is no skip in time dependence, i.e.,  $s_t = f(s_{t-1}, s_{t-2}, \ldots, s_{t-10})$ , LSTM and vRNN have the same expressive power. This also tells us the argument that "LSTM is always better than vRNN" is not correct. Another drawback of LSTM is its transient storage of the long term memory. In other words, if the long term memory is updated, its old value is erased. For example, in Fig.2(b), at time  $t_9$ , when M00 is updated to M01, M00 is erased. Thus, the future states don't have access to memory M00 any more. According to this property, this architecture is extremely useful when the previous states don't need to be addressed again after they are updated.

# 3.3 NEURAL STACK

Neural stack refers to neural networks using a stack as its external memory. The stack is controlled by either a feedforward network or a vRNN. One property of stack is that only the topmost content of the stack can be read or written. Writing to the stack is implemented by three operations: push, adding an element to the top of the stack; pop, removing the topmost element of the stack; no-operation, keeping the stack unchanged.

The diagram for the neural stack network is shown in Fig.3(a)(Here, we use the architecture in Joulin & Mikolov (2015)). Elements in the stack would be updated as follows,

$$
\mathbf {s} _ {t} (i) = \left\{ \begin{array}{c c} d _ {t} ^ {p u s h} \mathbf {c} + d _ {t} ^ {p o p} \mathbf {s} _ {t - 1} (1) + d _ {t} ^ {n o - o p} \mathbf {s} _ {t - 1} (0), & i f i = 0, \\ d _ {t} ^ {p u s h} \mathbf {s} _ {t - 1} (i - 1) + d _ {t} ^ {p o p} \mathbf {s} _ {t - 1} (i + 1) + d _ {t} ^ {n o - o p} \mathbf {s} _ {t - 1} (i), & o t h e r w i s e, \end{array} \right. \tag {7}
$$

$\mathbf{s}_t(i)$  is the content of the stack at time  $t$  in position  $i$ .  $s_t(0)$  is the topmost content at time  $t$ ,  $\mathbf{c}$  is the candidate content to be pushed onto the stack,  $d_t^{push}$ ,  $d_t^{pop}$  and  $d_t^{no-op}$

![](images/807e4e98f07eed3b11167a410ee2425acee881e8572eabaf6e99ad43e368a553.jpg)  
(a) Network architecture

![](images/34c08da7206b255fb96621c774410ef928e7524c29924a54730c8ff9406ed19b.jpg)  
(b) Memory visualization  
Figure 2: LSTM: A blue belt named  $M0$  represents the external memory: at  $t_1$ , memory  $M00$  is generated and stored, at time  $t_9$ ,  $M00$  is updated to  $M01$ . The black dash arrows represent the effect of the current state on the external memory. The state index is also the time index.

are push, pop and no-operation signals. In order to train the network with BPTT, all operations have to be implemented by continuous functions over a continuous domain. The calculation details of the stack contents and corresponding operators are in Appendix A.2. Since the recurrence is introduced by the stack memory, the dynamics of the model are,

$$
\mathbf {h} _ {t} = g \left(\mathbf {w} _ {x h} ^ {\mathrm {T}} \mathbf {x} _ {t} + \mathbf {w} _ {r h} ^ {\mathrm {T}} \mathbf {r} _ {t} + \mathbf {b} _ {h}\right), \tag {8}
$$

where  $\mathbf{r}_t$  is the read vector at time  $t$ ,

$$
\mathbf {r} _ {t} = g _ {o} \mathbf {s} _ {t} (0). \tag {9}
$$

Although the architecture of neural stack looks very different from vRNN and LSTM. There are some underlying similarities between them from the memory organization prospective. Fig.3(b) shows the memory space for the neural stack. Different from LSTM, neural stack can store more than one useful content in its external memory bank. For example, at time  $t_0$ ,  $M00$  is saved in memory belt  $M0$ , at time  $t_2$ ,  $M10$  is saved in belt  $M1$ . A black arrow on the left of the memory content is used to point the top of stack at each time step. All these contents can be addressed when they are needed. For example,  $M10$  is used again at time  $t_4$  after popping out  $M20$  in belt  $M2$ . With this external memory, all the useful information of the input is retained. Different from the state memory, the content of past is not altered, it is stored in its original form or the transformation form. As the content and the operations on the past are separate, we can efficiently select the useful content from this structured memory other than using the mixture of all the content before.

LSTM can be seen as a special case of the neural stack. In the neural stack, if all the contents in the

stack below the topmost element will never be addressed again, only one memory belt is enough. In this case, neural stack degrades to LSTM as shown in Fig.3(b) with a green dash box. The stack operators (push, pop, no-op) in neural stack have the same function as the input and forget gates in LSTM: deciding how to revise the memory contents. The problem for LSTM is that the previous memories are erased after they are updated, which also happens continuously with the vRNN state. Hence both learning models have difficulties to accomplish some simple memorization tasks such as reversing a sequence. However, the external memory bank in neural stack can help to solve this problem by online storing and extracting more than one content.

Although the neural stack can go back to the previous memory, it has two constraints. Firstly, it can not jump to any memory position, the previous memory should be addressed and updated sequentially. For example, as shown in the second line in Fig.3(b), if we want to go back to the memory in the belt  $M1$ , we have to pass memory in belt  $M2$  first. Secondly, after the memory content is popped out of the stack, it will be forgotten. For example, at time  $t_4$ , memory in belt  $M2$  is popped out, so in the future time steps, content in belt  $M2$  can not be accessed and updated any more.

From the state transition analysis above we can draw the conclusion that, for the tasks where the previous memory needs to be addressed sequentially (first in last out), the stack neural network is our first choice.

# 3.4 NEURAL RAM

Recently, some dynamic neural networks with an external random access memory have been studied. In these networks, all the contents in the memory bank can be randomly accessed. Neural Turing machine Graves et al. (2014) (NTM) is one example. Its network architecture is shown in Fig.4(a).

The challenge of this network is that all the memory addresses are discrete in nature. In order to learn read and write addresses by error backpropagation, they have to be extended to continuous domain.

![](images/070998d445c50f0ae775d74173b8691e44b05b6f135c69dfb1ea5a527389959f.jpg)  
(a) Network architecture

![](images/97b5030f4564bc40802d21d3e65fee44617e53738dc57a76e0cc62e4d82e310a.jpg)  
(b) Memory visualization  
Figure 3: Neural stack: the network first saves state  $M00$  in belt  $M0$  and updates it to  $M01$ . At time  $t_2$ , instead of replacing  $M01$  with a new state  $M10$ , a new belt  $M1$  is created to save  $M10$ . In this way, both  $M01$  and  $M10$  are kept. Similarly, at time  $t_5$ ,  $M30$  is saved in another belt  $M3$ . In time  $t_5$ , the content in the stack is  $M01$ ,  $M11$ ,  $M30$  and  $M30$  is the topmost element.

A solution to this difficulty is to read from and write to all the memory slots with different strengths. These strengths can also be explained as the probabilities of each slot to be read from and written to. To be specific, the reading vector at time step  $t$  is,

$$
\mathbf {r} _ {t} = \sum_ {i = 0} ^ {M - 1} w _ {t} ^ {r} (i) \mathbf {m} _ {t} (i), \tag {10}
$$

$\mathbf{m}$  is the memory bank with  $M$  memory slots,  $w_{t}^{r}(i)$  is the normalized reading weight for  $i$ th slot at time  $t$  which satisfying,  $\sum_{i} w_{t}^{r}(i) = 1, 0 \leq w_{t}^{r}(i) \leq 1$ . In the writing process, each memory slot is updated as,

$$
\mathbf {c} _ {t} = f \left(\mathbf {w} _ {h c} ^ {\mathrm {T}} \mathbf {h} _ {t} + \mathbf {b} _ {c}\right), \tag {11}
$$

$$
\mathbf {m} _ {t} (i) = w _ {t} ^ {w} (i) \mathbf {c} _ {t} (i) + e _ {t} (i) \mathbf {m} _ {t - 1} (i), \forall i \tag {12}
$$

here  $w_{t}^{r}(i)$  is the writing weight and  $e_t(i)$  is the erasing weight for memory slot  $i$  at time  $t$ . The calculation details of these weights are in Appendix A.3. The dynamics of the hidden layer are,

$$
\mathbf {h} _ {t} = f \left(\mathbf {w} _ {x h} ^ {\mathrm {T}} \mathbf {x} _ {t} + \mathbf {w} _ {r h} ^ {\mathrm {T}} \mathbf {r} _ {t - 1} + \mathbf {b} _ {h}\right), \tag {13}
$$

![](images/cc9ce9496aae1bc51068bc7c10fddeda4869e7f544f881266ab940d50d89964a.jpg)  
(a) Network architecture

![](images/379c8f26471bcf4fb9b0f017439baa65f176cfd2364aa105d95a035ae8cb112e.jpg)  
(b) Memory visualization  
Figure 4: Neural RAM

Fig.4(b) shows its memory structure. The RAM network can be seen as an improvement of the neural stack in the sense that all the contents in the memory bank can be read from and written to multiple times. And there is no requirement for the order of storing, updating and accessing memory elements. For example, in Fig.4(b), at time  $t_0$ , memory  $M00$  is stored in belt  $M0$ , at time  $t_1$ , system control can directly jump to belt  $M2$  to store  $M20$ . What's more, the reading and writing slots can be different. For example, at  $t_1$ , the network writes to belt  $M2$  and reads the content in  $M0$ . The black arrows on the left of the contents in external memory represent the reading contents. This neural RAM network can degrade to neural stack if the memory accessing order is restricted. Similarly, it can degrade to LSTM if only one memory belt is used. From the analysis above, it is not hard to see that neural RAM is the most powerful network among all the models d

# 4 A MEMORY NETWORK TAXONOMY

From the analysis in the above section we can draw a conclusion that, the innovation of LSTM versus the vRNN is the incorporation of an external memory and three gates to balance the external memory and internal memory; the innovation of neural stack is to extend one external memory to several external memories and to propose a method to visit the memory slots in a certain order; the innovation of neural RAM is to remove the constraint of the memory

![](images/bd19be04dde04816d0d80aa20410ac164036c41e67a728495fb53078b6c1cb08.jpg)  
Figure 5: Memory Network Taxonomy

slotting order, which allows any memory slot to be visited at any time and any number of times. The different memory organizations make these networks have different expressive power. In this section, a taxonomy of recurrent neural network is proposed to classify all these popular models into four classes ordered by a rigorous inclusion relationship, as shown in Fig.5, i.e., vRNN  $\subseteq$  LSTM  $\subseteq$  neural stack  $\subseteq$  neural RAM. Some classes are named after a typical model. For example, vRNN class also includes IRNN Le et al. (2015), highway network Srivastava et al. (2015), LSTM class also includes GRU Cho et al. (2014), peephole network Gers & Schmidhuber (2000). Neural stack class includes the architecture in Sun (1993); Sun et al. (2017); Joulin & Mikolov (2015); Grefenstette et al. (2015); Neural RAM class includes NTM Graves et al. (2014), DNC Graves et al. (2016), enhanced LSTM Graves (2013) and Weston et al. (2014), etc. The classification of these four types of networks are based on the their memory characteristics, i.e., internal memory; one external memory slot; external memory slots with a restricted visiting order; external memory slots without restricted visiting order. For instance, LSTM and GRU belong to the same class since both of them have one external memory slot, though their gate calculations are different.

Table 1: Mapping from network types to task types  

<table><tr><td>networks</td><td>memory requirements of tasks</td></tr><tr><td>vRNN</td><td>only state memory, memory is forced to be used all the time</td></tr><tr><td>LSTM</td><td>state memory and memory of a single external event</td></tr><tr><td>Neural stack</td><td>memory of multiple events, information of each event should be used sequentially, only one event is accessible at each time step</td></tr><tr><td>Neural RAM</td><td>memory of multiple events, all are accessible at each time step, no restriction on how many times they are used</td></tr></table>

In the following subsections, we will first prove the inclusion relationship mathematically and then show how to link the property of different memory structures to the memory requirement of different tasks, which can help practitioners select the most parsimonious model for a specific task.

# 4.1 INCLUSION RELATIONSHIP DERIVATIONS

Theorem 1. Neural RAM can be degraded to a neural stack if,

i) all the reading weights except that for the topmost memory slot are set to zeros,  $w_{t}^{r}(i) = 0$ , if  $i \neq 0$ ;

ii) only the writing weight for the topmost memory slot is learned, all others are copied from it,  $w_{t}^{r}(i) = w_{t}^{r}(0), i \neq 0$ ;

iii) in the writing process, instead of learning all the contents to be written to the stack as in Eq.(11), only the content of  $M0$  is learned as,  $\mathbf{c}_t(0) = t(\mathbf{w}_{hc}^{\prime \mathrm{T}}\mathbf{h}_{t - 1} + b_c) + \gamma \mathbf{m}_{t - 1}(1)$ , all others are calculated as,  $\mathbf{c}_t(i) = \mathbf{m}_{t - 1}(i - 1) + \gamma \mathbf{m}_{t - 1}(i + 1)$ , if  $i \neq 0$ .;

iv) only the writing and erasing weights for the topmost element are learned, all others are just a copy of the topmost's values,  $w_{t}^{r}(i) = w_{t}^{r}(0)$ ,  $e_{t}(i) = e_{t}(0)$ .

Theorem 2. The neural stack can be degraded to the LSTM if the pop signal is zero,  $d_t^{pop} = 0$ . The  $d_t^{push}$  in neural stack works as the input gate in LSTM, and  $d_t^{no-op}$  in neural stack works as the forget gate in LSTM.

Theorem 3. The LSTM is degraded to the vRNN if, i) all three gates are set as constants,  $g_{o} = 0$ ,  $g_{i} = 1$  and  $g_{f} = 0$ ; ii) weight  $\mathbf{w}_{hc}$  and bias  $\mathbf{b}_c$  are set as constants  $\mathbf{w}_{hc} = \mathbf{I}$ ,  $\mathbf{b}_c = \mathbf{0}$ ; iii) the activation function  $t(x)$  is set as linear activation function  $t_1(x) = x$ .

Proof. All the proofs are in Appendix B.

![](images/91e012655d2a3e3bc643400908a3b5d2f2fa056d2f61a62aaf2d892a1de04d1b.jpg)

# 4.2 MAPPING FROM NETWORK TYPES TO TASK TYPES

It is not hard to see that the proposed taxonomy resembles the hierarchical organization of automata: vRNN  $\Leftrightarrow$  Finite state machine, neural stack  $\Leftrightarrow$  Deterministic pushdown automaton, neural RAM  $\Leftrightarrow$  Turing machine. Hence, if our task is sequence recognition or classification, the recognizable sequences for each network can be illustrated by the Chomsky hierarchy. However, these networks can do some more sequence learning tasks such as prediction. In this case, sequences do not need to satisfy the restrict grammars. For example, there is no need for the input sequences to always start from a start state and go back to an accepted state. Hence, in order to make our taxonomy fit into these more general sequences, we divide all the sequence learning tasks into four classes according to their memory requirements, as summarized in Table 1. This mapping can help practitioners select the most parsimonious architecture (we can always go for the most powerful model, but it needs more resources to train) for all sequence learning tasks if they know the memory requirement. In order to exemplify each task type, four tasks employing synthetic symbol sequences are developed: counting, counting with interference, reversing and repeat copying. We will analyze the memory requirements of them one by one.

Counting For the counting task, the input sequences are composed of  $a$ 's,  $b$ 's and  $c$ 's. The output sequence is trying to count the number of  $a$ 's. For instance, when the input sequence is aaabcaa, the output sequence would be 1233345. For this kind of sequences, a state variable is needed to remember the number of  $a$ 's. Once receiving an  $a$ , there is a state transition. In this problem, the

state space is not very large. A first order Markov state model is more than enough to describe it. Hence, as long as the network has one feedback loop, the counting task can be completed. "Task can be completed" in this paper means the output error is almost zero.

Counting with interference For the counting with interference task, the input sequences are the same as the counting task. We still want to count the number of  $a$ , but if encountering  $b$  or  $c$ , the output should also be  $b$  or  $c$ . For example, if the input is aabbaca, the output sequence is 12bb3c4. For this kind of problem, an external memory cache is required, because when  $b$  or  $c$  is encountered, the hidden layer's output (internal memory) will be over-written. If we want to recall the number of  $a$ 's, this value needs to be stored in an external memory for future use, and an input gate will be needed to keep the external memory unaffected when inputting  $b$  and  $c$ . ( $g_i = 1$  when input  $a$  and  $g_i = 0$  when input  $b$ ,  $c$ .) Thus LSTM, neural stack and neural RAM are capable of solving this problem. However, in vRNN, since the only memory is the state memory and the output is forced to be a function of this state memory, the interference of  $b$  and  $c$  would make vRNN unable to accomplish this task.

Reversing The third task is sequence reversing. For example, if the input sequence is abacde  $\delta x x x x x$ , the output sequence should be xxxxxxxedcaba.  $\delta$  is the delimiter symbol and  $x$  means any symbol. When encountering  $\delta$  in the input sequence, no matter what the following symbols are, the output would be the input symbols before  $\delta$  in a reverse order. For this task, all the useful past information should be stored and then retrieved in a reverse order. Hence, the memory should have the ability to store more than one element and the reading order is related to the writing order. Since vRNN does not have any memory bank and LSTM's memory is forgotten after it is updated, these two networks fail for this task. On the other hand, both neural stack and neural RAM can store more than one content and the task satisfies the "first in last out" principle, thus they can solve this task.

Repeat copying The hardest task is repeat copying, by which we mean the output sequence is several times repeated version of the input sequences. For example, if the input sequence is adbcexxxxxxxxxxxxxx, the output should be xxxxxxxxadcbcadbecd. That is, when encountering the repeating number symbol  $\varepsilon$ , the output will be the previous input sequence for  $\varepsilon$  times. For this kind of task, not only more than one past content need be stored, they should be retrieved more than one time, here the number is 3. Since all the saved information in neural stack is forgotten after being popped out, it is unable to learn the task. Thus, neural RAM is the only network that can handle this task.

This classification of tasks is very meaningful since it can guide the users in the right direction. If we select the wrong type of network, there will be an error and/or speed penalty no matter how we adjust the hyper parameters. As shown in our experiment part, for sequence reversing (which belongs to the third type of task), neural stack and neural RAM with 6 hidden neurons will converge to near zero error, but for vRNN and LSTM, even if we set the number of hidden neurons to 1000, their output will always fluctuate around a non-zero value.

# 5 EXPERIMENTS

In order to illustrate the impact of different memory organizations, we test the performance of the four networks on the synthetic tasks described in last section. We also use them to visualize how each network encodes information in order to solve a problem in AppendixD. Then, we used two natural language processing applications to elaborate how to employ the knowledge gained from the different characteristic of the memory structures to help users select the right type of network. The details about parameters settings are in AppendixC.

![](images/bb2d667cf82a5930caba4b71066cade164905a99f6a0675c14c815cc3c32c6f6.jpg)  
Figure 6: Learning curve for four synthetic tasks

![](images/0db88261dc80179830ac3062081262cc1ca4b69113634435f03cb732412a0044.jpg)

![](images/e56e50004e66c6ca45b76aa31da451216d9b124dee794203eaf57ec44f38863b.jpg)

![](images/60cf3e1c489e4516b248c2a35d75f0dbe3916567502216cb9b2a1e28971ff946.jpg)

Table 2: Average error for movie review  

<table><tr><td></td><td>vRNN</td><td>LSTM</td><td>neural Stack</td><td>neural RAM</td></tr><tr><td>error rate</td><td>31±5</td><td>19±2.5</td><td>23±10</td><td>20±9</td></tr></table>

Table 3: Average error for three tasks from bAbI tasks  

<table><tr><td>Task</td><td>vRNN</td><td>LSTM</td><td>neural Stack</td><td>neural RAM</td></tr><tr><td>task 1</td><td>52±1.5</td><td>28.4±1.5</td><td>41±2.0</td><td>9.0±12.6</td></tr><tr><td>task 2</td><td>79±2.5</td><td>56.0±1.5</td><td>75±6</td><td>39.2±20.5</td></tr><tr><td>task 3</td><td>85±2.5</td><td>51.3±1.4</td><td>78±6.4</td><td>39.6±16.4</td></tr></table>

Learning curves for the four tasks using different networks are shown in Fig.6. The performance is measured in MSE for first two tasks and output entropy for the other two tasks. We use the same number of units in all these architectures for a fair comparison. From the result we can observe that for counting, all the four networks can achieve an almost zero error; for counting with interference, all the networks except for vRNN can complete the task; for sequence reversing, neural stack and neural RAM are the suitable networks; and for repeat copying, neural RAM is the only network to solving the problem. We also tried some different parameter settings, for instance, setting the number of hidden units from 5 to 1000, the performances are same as Fig.6 except for a different non-zero error value when the network is not capable to accomplish the task.

# 5.2 NATURAL LANGUAGE PROCESSING

For synthetic problems it was clear cut to design problems that exemplify the expressive power of the different memory networks. For real world problems this task is more complex because sometimes it is hard to pin point the memory requirements or the problem may be a blend of classes. In this case all the networks may solve the problem to a certain extent. Hence, we will illustrate how to select the minimum network resources to accomplish the task with relatively better performance in this section.

Sentiment Analysis The first experiment is a sentiment analysis which will infer the emotional tone of the text as negative or positive. In order to judge the emotional tone of the text at the end, an external memory whose value would be affected by some key words is useful. And since the goal is to classify the emotional tone as either 1 or 0, the specific contents of the text are not very important here so there is no need to store all of them. Hence, a network with an external memory slot should perform better than the one without. But the memory bank which can store multiple contents does not show more advantages here. We test these networks' performance on lmdb movie review dataset Pennington et al. (2014). The results are in Table 2, which shows that vRNN performs worst. LSTM, neural stack, neural RAM have similar performances. Thus, our analysis is verified.

Question Answering Then we investigate the performance of these four networks on three question answering tasks from bAbI datasetWeston et al. (2015). The target is to give an answer after reading a little story followed by a question. For example, the story is, "Mary got the milk there. John moved to the bedroom. Sandra went back to the kitchen. Mary travelled to the hallway." And the question is, "Where is the milk?" The machine is expected to give the answer "hallway". For this problem, in order to give the right answer, the machine should memorize the facts that Mary got the milk and travelled to the hallway. What's more, since the machine doesn't know the question when reading the story, it has to store all the potential useful facts. Thus an external memory bank whose whichever content can be visited is useful here. According to our memory capability analysis, the neural RAM should perform the best here. From the results in Table 3, we can see that neural RAM indeed achieves the best performance.

# 6 CONCLUSION

In this paper, we analyze the memory structure for several recurrent networks and propose a taxonomy of them. We use four synthetic tasks and two natural language processing problems to illustrate utility of the taxonomy. Although we showed differences in performance in the experiments, it is too early to say that we presented all the tools to select parsimoniously the memory architecture for a given application. Because the user has to analyze the requirements of the application, which may not be trivial, more work is needed to create rules of thumb to help practitioners.

# REFERENCES

Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.  
Jasmine Collins, Jascha Sohl-Dickstein, and David Sussillo. Capacity and trainability in recurrent neural networks. arXiv preprint arXiv:1611.09913, 2016.  
Bert De Vries and Jose C Principe. The gamma model-a new neural model for temporal processing. Neural networks, 5(4):565-576, 1992.  
Kenji Doya. Universality of fully connected recurrent neural networks. 1993.  
Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179-211, 1990.  
Felix A Gers and E Schmidhuber. LSTM recurrent networks learn simple context-free and context-sensitive languages. IEEE Transactions on Neural Networks, 12(6):1333-1340, 2001.  
Felix A Gers and Jürgen Schmidhuber. Recurrent nets that time and count. In Neural Networks, 2000. IJCNN 2000, Proceedings of the IEEE-INNS-ENNS International Joint Conference on, volume 3, pp. 189–194. IEEE, 2000.  
Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. arXiv preprint arXiv:1410.5401, 2014.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626): 471-476, 2016.  
Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, and Phil Blunsom. Learning to transduce with unbounded memory. In Advances in Neural Information Processing Systems, pp. 1828-1836, 2015.  
Klaus Greff, Rupesh K Srivastava, Jan Koutnik, Bas R Steunebrink, and Jürgen Schmidhuber. Lstm: A search space odyssey. IEEE transactions on neural networks and learning systems, 2017.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Michael I. Jordan. Attractor dynamics and parallelism in a connectionist sequential machine. In Proceedings of the Eighth Annual Conference of the Cognitive Science Society, pp. 531-546. Hillsdale, NJ: Erlbaum, 1986.  
Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented recurrent nets. In Advances in neural information processing systems, pp. 190-198, 2015.  
Rafal Jozefowicz, Wojciech Zaremba, and Ilya Sutskever. An empirical exploration of recurrent network architectures. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15), pp. 2342-2350, 2015.  
Andrej Karpathy, Justin Johnson, and Li Fei-Fei. Visualizing and understanding recurrent networks. arXiv preprint arXiv:1506.02078, 2015.  
Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. arXiv preprint arXiv:1504.00941, 2015.

Christian W Omlin and C Lee Giles. Constructing deterministic finite-state automata in recurrent neural networks. Journal of the ACM (JACM), 43(6):937-972, 1996.  
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543, 2014. URL http://www.aclweb.org/anthology/D14-1162.  
Paul Rodriguez. Simple recurrent networks learn context-free and context-sensitive languages by counting. Neural computation, 13(9):2093-2118, 2001.  
Jürgen Schmidhuber, F Gers, and Douglas Eck. Learning nonregular languages: A comparison of simple recurrent networks and lstm. Neural Computation, 14(9):2039-2041, 2002.  
Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. Highway networks. arXiv preprint arXiv:1505.00387, 2015.  
Guo-Zheng Sun. Learning context-free grammar with enhanced neural network pushdown automaton. In Grammatical Inference: Theory, Applications and Alternatives, IEE Colloquium on, pp. P6-1. IET, 1993.  
GZ Sun, C Lee Giles, HH Chen, and YC Lee. The neural network pushdown automaton: Model, stack and learning simulations. arXiv preprint arXiv:1711.05738, 2017.  
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. CoRR, abs/1410.3916, 2014. URL http://arxiv.org/abs/1410.3916.  
Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart van Merrienboer, Armand Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. arXiv preprint arXiv:1502.05698, 2015.
