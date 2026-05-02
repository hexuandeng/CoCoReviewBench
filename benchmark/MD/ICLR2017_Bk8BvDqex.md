# METACONTROL FOR ADAPTIVE IMAGINATION-BASED OPTIMIZATION

Jessica B. Hamrick

UC Berkeley & DeepMind

jhamrick@berkeley.edu

Andrew J. Ballard

DeepMind

aybd@google.com

Razvan Pascanu

DeepMind

razp@google.com

Oriol Vinyals

DeepMind

vinyals@google.com

Nicolas Heess

DeepMind

heess@google.com

Peter W. Battaglia

DeepMind

peterbattaglia@google.com

# ABSTRACT

Many machine learning systems are built to solve the hardest examples of a particular task, which often makes them large and expensive to run—especially with respect to the easier examples, which might require much less computation. For an agent with a limited computational budget, this "one-size-fits-all" approach may result in the agent wasting valuable computation on easy examples, while not spending enough on hard examples. Rather than learning a single, fixed policy for solving all instances of a task, we introduce a metacounter which learns to optimize a sequence of "imagined" internal simulations over predictive models of the world in order to construct a more informed, and more economical, solution. The metacounter component is a model-free reinforcement learning agent, which decides both how many iterations of the optimization procedure to run, as well as which model to consult on each iteration. The models (which we call "experts") can be state transition models, action-value functions, or any other mechanism that provides information useful for solving the task, and can be learned on-policy or off-policy in parallel with the metacounter. When the metacounter, controller, and experts were trained with "interaction networks" (Battaglia et al., 2016) as expert models, our approach was able to solve a challenging decision-making problem under complex non-linear dynamics. The metacounter learned to adapt the amount of computation it performed to the difficulty of the task, and learned how to choose which experts to consult by factoring in both their reliability and individual computational resource costs. This allowed the metacounter to achieve a lower overall cost (task loss plus computational cost) than more traditional fixed policy approaches. These results demonstrate that our approach is a powerful framework for using rich forward models for efficient model-based reinforcement learning.

# 1 INTRODUCTION

While there have been significant recent advances in deep reinforcement learning (Mnih et al., 2015; Silver et al., 2016) and control (Lillicrap et al., 2015; Levine et al., 2016), most efforts train a network that performs a fixed sequence of computations. Here we introduce an alternative in which an agent uses a metacontroller to choose which, and how many, computations to perform. It "imagines" the consequences of potential actions proposed by an actor module, and refines them internally, before executing them in the world. The metacontroller adaptively decides which expert models to use to evaluate candidate actions, and when it is time to stop imagining and act. The learned experts may be state transition models, action-value functions, or any other function that is relevant to the task, and can vary in their accuracy and computational costs. Our metacontroller's learned policy can exploit the diversity of its pool of experts by trading off between their costs and reliability, allowing it to automatically identify which expert is most worthwhile.

We draw inspiration from research in cognitive science and neuroscience which has studied how people use a meta-level of reasoning in order to control the use of their internal models and allocation of their computational resources. Evidence suggests that humans rely on rich generative models of the world for planning (Gläscher et al., 2010), control (Wolpert & Kawato, 1998), and reasoning (Hegarty, 2004; Johnson-Laird, 2010; Battaglia et al., 2013), that they adapt the amount of computation they perform with their model to the demands of the task (Hamrick et al., 2015), and that they trade off between multiple strategies of varying quality (Lee et al., 2014; Lieder et al., 2014; Lieder & Griffiths, in revision; Kool et al., in press).

Our imagination-based optimization approach is related to classic artificial intelligence research on bounded-rational metareasoning (Horvitz, 1988; Russell & Wefald, 1991; Hay et al., 2012), which formulates a meta-level MDP for selecting computations to perform, where the computations have a known cost. We also build on classic work by Schmidhuber (1990a;b), which used an RL controller with a recurrent neural network (RNN) world model to evaluate and improve upon candidate controls online.

Recently Andrychowicz et al. (2016) used a fully differentiable deep network to learn to perform gradient descent optimization. Our work is also related to recent notions of "conditional computation" (Bengio, 2013; Bengio et al., 2015), which adaptively modifies network structure online, and "adaptive computation time" (Graves, 2016) which allows for variable numbers of internal "pondering" iterations to optimize computational cost.

Our work's key contribution is a framework for learning to optimize via a metacontroller which manages an adaptive, imagination-based optimization loop. This represents a hybrid RL system where a model-free metacontroller constructs its decisions using an actor policy to manage model-free and model-based experts. Our experimental results demonstrate that a metacontroller can flexibly allocate its computational resources on a case-by-case basis to achieve greater performance than more rigid fixed policy approaches, using more computation when it is required by a more difficult task.

# 2 MODEL

We consider a class of fully observed, one-shot decision-making tasks (i.e., continuous, contextual bandits). The performance objective is to find a control  $c \in \mathcal{C}$  which, given an initial state  $x \in \mathcal{X}$ , minimizes some loss function  $\mathcal{L}$  between a known future goal state  $x^{*}$  and the result of a forward process,  $f(x, c)$ . The performance loss  $L_{P}$  is the (negative) utility of executing the control in the world, and is related to the optimal solution  $c^{*} \in \mathcal{C}$  as follows:

$$
L _ {P} \left(x ^ {*}, x, c\right) = \mathcal {L} \left(x ^ {*}, f (x, c)\right), \tag {1}
$$

$$
c ^ {*} = \underset {c} {\arg \min } L _ {P} \left(x ^ {*}, x, c\right). \tag {2}
$$

However, (2) defines only the optimal solution—not how to achieve it.

# 2.1 OPTIMIZING PERFORMANCE

We consider an iterative optimization procedure that takes  $x^{*}$  and  $x$  as input and returns an approximation of  $c^{*}$  in order to minimize (1). The optimization procedure consists of a controller, which iteratively proposes controls, and an expert, which evaluates how good those controls are. On the  $n^{\mathrm{th}}$  iteration, the controller  $\pi^{C}: \mathcal{X} \times \mathcal{X} \times \mathcal{H} \rightarrow \mathcal{C}$  takes as input,  $x^{*}, x$ , and information about the history of previously proposed controls and evaluations  $h_{n-1} \in \mathcal{H}$ , and returns a proposed control  $c_{n}$  that aims to improve on previously proposed controls. An expert  $E: \mathcal{X} \times \mathcal{X} \times \mathcal{C} \rightarrow \mathcal{E}$  takes the proposed control and provides some information  $e_{n} \in \mathcal{E}$  about the quality of the control, which we call an opinion. This opinion is added to the history, which is passed back to the controller, and the loop continues for  $N$  steps, after which a final control  $c_{N}$  is proposed.

Standard optimization methods use principled heuristics for proposing controls. In gradient descent, for example, controls are proposed by adjusting  $c_{n}$  in the direction of the gradient of the reward with respect to the control. In Bayesian optimization, controls are proposed based on selection criteria such as "probability of improvement", or a meta-selection criterion for choosing among several basic selection criteria Hoffman et al. (2011); Shahriari et al. (2014). Rather than choosing one of several controllers, our work learns a single controller and instead focuses on selecting from

![](images/21d8de4809622a8b6c99520377a101cd65f60616a795afabf33993e715fd88c4.jpg)  
Figure 1: Metacontroller architecture and task. A: The manager takes the scene and history and determines which action to take (whether to execute or ponder, and with what expert to ponder with). The controller takes the scene and history and computes a control that is sent to the expert/world as determined by the manager. The outcome and reward from the expert, along with the history, action, and control, are fed into the memory, which produces the next history. B-C: Scenes consisted of a number of planets (depicted here by colored circles) of different masses as well as a spaceship (also with a variable mass). The task was to apply a force to the spaceship for one time step of simulation (depicted here as a solid red arrow) such that the resulting trajectory (dotted red arrow) would put the spaceship at a target (bullseye) after 11 steps of simulation. The white ring of the bullseye corresponds to a performance loss of 0.12-0.15, the black ring to a loss of 0.09-0.12, the blue ring to a loss of 0.06-0.09, the red ring to a loss of 0.03-0.06, and the yellow center to a loss of 0.03 or less. B depicts an easy, 1-planet scene, while C depicts a very difficult 5-planet scene.

![](images/7dc4a42dbe0b20ff55edb511241ee6a9f4da3981f45fe75e4841135ca70fb1be.jpg)

![](images/9a42c3232dfe753caf8ee984ab71ea190067fc401a81a56a71504b8454589cae.jpg)

multiple experts (see Sec. 2.2). In some cases  $f$  is known and inexpensive to compute, and thus the optimization procedure sets  $E \equiv f$ . However, in many real-world settings,  $f$  is expensive or non-stationary and so it can be advantageous to use an approximation of  $f$  (e.g., a state transition model),  $L_{P}$  (e.g., an action-value function), or any other quantity that gives some information about  $f$  or  $L_{P}$ .

# 2.2 OPTIMIZING COMPUTATIONAL COST

Given a controller and one or more experts, there are two important decisions to be made. First, how many optimization iterations should be performed? The approximate solution usually improves with more iterations, but each iteration costs computational resources. However, most traditional optimizers either ignore the cost of computation or select the number of iterations using simple heuristics. Because they do not balance the cost of computation against the performance loss, the overall effectiveness of these approaches is subject to the skill and preferences of the practitioners who use them. Second, which expert should be used on each step of the optimization? Some experts may be accurate but expensive to compute in terms of time, energy and/or money, while others may be crude, yet cheap. Moreover, the reliability of the experts may not be known a priori, further

limiting the effectiveness of the optimization procedure. Our use of a metacontroller address these issues by jointly optimizing over the choices of how many steps to take and which experts to use.

We consider a family of optimizers which use the same controller,  $\pi^C$ , but vary in their expert evaluators,  $\{E_1,\ldots ,E_K\}$ . Assuming that the controller and experts are deterministic functions, the number of iterations  $N$  and the sequences of experts  $\mathbf{k} = (k_{1},\dots,k_{N - 1})$  exactly determine the final control and performance loss  $L_{P}$ . This means we have transformed the performance optimization over  $c$  into an optimization over  $N$  and  $\mathbf{k}$ :  $(N,\mathbf{k})^{*} = \arg \min_{k,n}L_{P}(x^{*},x,c(N,\mathbf{k},x,x^{*}))$ , where the notation  $c(N,\mathbf{k},x,x^{*})$  is used to emphasize that the control is a function  $N$ ,  $\mathbf{k}$ ,  $x$ , and  $x^{*}$ .

If each optimizer has an associated computational cost  $\tau_{k}$ , then  $N$  and  $\mathbf{k}$  also exactly determine the computational resource loss of the optimization run,  $L_{R}(N,\mathbf{k}) = \sum_{n = 1}^{N - 1}\tau_{k_{n}}$ . The total loss is then the sum of  $L_{P}$  and  $L_{R}$ , each of which are functions of  $N$  and  $\mathbf{k}$ ,

$$
\begin{array}{l} L _ {T} \left(x ^ {*}, x, N, \mathbf {k}\right) = L _ {P} \left(x ^ {*}, x, c (N, \mathbf {k}, x, x ^ {*})\right) + L _ {R} (N, \mathbf {k}) (3) \\ = \mathcal {L} \left(x ^ {*}, f \left(x, \pi^ {C} \left(x ^ {*}, x, h _ {N - 1}\right)\right)\right) + \sum_ {n = 1} ^ {N - 1} \tau_ {k _ {n}}, (4) \\ \end{array}
$$

and the optimal solution is defined as  $(N,\mathbf{k})^{*} = \arg \min_{N,\mathbf{k}}L_{T}(x^{*},x,N,\mathbf{k})$ . Optimizing  $L_{T}$  is difficult because of the recursive dependency on the history,  $h_{N - 1}$ , and because the discrete choices of  $N$  and  $\mathbf{k}$  mean  $L_{T}$  is not differentiable.

To optimize  $L_{T}$  we recast it as an RL problem where the objective is to jointly optimize task performance and computational cost. As shown in Figure 1a, the metacontroller agent  $a^{M}$  is comprised of a controller  $\pi^{C}$ , a pool of experts  $\{E_{1},\ldots ,E_{K}\}$ , a manager  $\pi^{M}$ , and a memory  $\mu$ . The manager is a meta-level policy (Russell & Wefald, 1991; Hay et al., 2012) over actions indexed by  $k$ , which determine whether to terminate the optimization procedure  $(k = 0)$  or to perform another iteration of the optimization procedure with the  $k^{\mathrm{th}}$  expert. Specifically, on the  $n^{\mathrm{th}}$  iteration the controller produces a new control  $c_{n}$  based on the history of controls, experts, and evaluations. The manager, also relying on this history, independently decides whether to end the optimization procedure (i.e., to execute the control in the world) or to perform another iteration and evaluate the proposed control with the  $k_{n}^{\mathrm{th}}$  expert (i.e., to ponder, after Graves (2016)). The memory then updates the history  $h_n$  by concatenating  $k$ ,  $c_{n}$ , and  $e_n$  with the previous history  $h_{n - 1}$ . Coming back to the notion of imagination-based optimization, we suggest that this iterative optimization process is analogous to imagining what will happen (using one or more approximate world models) before actually executing that action in the world. For further details, see Appendix A.

We also define two special cases of the metacontroller for baseline comparisons. The iterative agent  $a^I$  does not have a manager and uses only a single expert. Its number of iterations are pre-set to a single  $N$ . The reactive agent,  $a^0$ , is a special case of the iterative agent, where the number of iterations is fixed to  $N = 0$ . This implies that proposed controls are executed immediately in the world, and are not evaluated by an expert.

# 2.3 NEURAL NETWORK IMPLEMENTATION

We use standard deep learning building blocks, e.g., multi-layer perceptrons (MLPs), RNNs, etc., to implement the controller, experts, manager, and memory, because they are effective at approximating complex functions via gradient-based and reinforcement learning, but other approaches could be used as well. In particular, we constructed our implementation to be able to make control decisions in complex dynamical systems, such as controlling the movement of a spaceship (Figure 1b-c), though we note that our approach is not limited to such physical reasoning tasks. Here we used mean-squared error (MSE) for our  $\mathcal{L}$  and Adam (Kingma & Ba, 2014) as the training optimizer.

Experts We implemented the experts as MLPs and "interaction networks" (INs) (Battaglia et al., 2016), which are well-suited to predicting complex dynamical systems like those in our experiments below. Each expert has parameters  $\theta^{E_k}$ , i.e.  $e_n = E_k(x^*, x, c_n; \theta^{E_k})$ , and may be trained either on-policy using the outputs of the controller (as is the case in this paper), or off-policy by any data that pairs states and controls with future states or reward outcomes. The objective  $L_{E_k}$  for each expert may be different depending on what the expert outputs. For example, the objective could be the loss between the goal and future states,  $L_{E_k} = \mathcal{L}\left(f(x, c), E_k(x^*, x, c; \theta^{E_k})\right)$ , which is what

we use in our experiments. Or, it could be the loss between  $L_{P}$  and an action-value function that predicts  $L_{P}$  directly,  $L_{E_k} = \mathcal{L}\left(L_P(x^*,x,c),E_k(x^*,x,c;\theta^{E_k})\right)$ . See Appendix B.1 for details.

Controller and Memory We implemented the controller as an MLP with parameters  $\theta^C$ , i.e.  $c_{n} = \pi^{C}(x^{*},x,h_{n - 1};\theta^{C})$ , and we implemented the memory as a Long Short-Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997) with parameters  $\theta^{\mu}$ . The memory embeds the history as a fixed-length vector, i.e.  $h_n = \mu (h_{n - 1},k_n,c_n,E_{k_n}(x^*,x,c_n);\theta^{\mu})$ . The controller and memory were trained jointly to optimize (1). However, this objective includes  $f$ , which is often unknown or not differentiable. We overcame this by approximating  $L_{P}$  with a differentiable critic analogous to those used in policy gradient methods (e.g. Silver et al., 2014; Lillicrap et al., 2015; Heess et al., 2015). See Appendices B.2 and B.3 for details.

Manager We implemented the manager as a stochastic policy that samples from a categorical distribution whose weights are produced by an MLP with parameters  $\theta^M$ , i.e.  $k_n \sim \text{Categorical}(k; \pi^M(x^*, x, h_{n-1}; \theta^M))$ . We trained the manager to minimize (3) using REINFORCE (Williams, 1992), but other deep RL algorithms could be used instead. See Appendix B.4 for details.

# 3 EXPERIMENTS

To evaluate our metacontroller agent, we measured its ability to learn to solve a class of physics-based tasks that are surprisingly challenging. Each episode consisted of a scene which contained a spaceship and multiple planets (Figure 1b-c). The spaceship's goal was to rendezvous with its mothership near the center of the system in exactly 11 time steps, but it only had enough fuel to fire its thrusters once. The planets were static but the gravitational force they exerted on the spacecraft induced complex non-linear dynamics on the motion over the 11 steps. The spacecraft's action space was continuous, up to some maximum magnitude, and represented the instantaneous Cartesian velocity vector imparted by its thrusters. Further details are in Appendix C.

We trained the reactive, iterative, and metacontroller agents on five versions of the spaceship task involving different numbers of planets. The iterative agent was trained to take anywhere from zero (i.e., the reactive agent) to ten ponder steps. The metacontroller was allowed to take a maximum of ten ponder steps. We considered three different experts which were all differentiable: an MLP expert which used an MLP to predict the final location of the spaceship, an IN expert which used an interaction network (Battaglia et al., 2016) to predict the full trajectory of the spaceship, and a true simulation expert which was the same as the world model. In some conditions the metacontroller could use exactly one expert and in others it was allowed to select between the MLP and IN experts. For experiments with the true simulation expert, we used it to backpropagate gradients to the controller and memory. For experiments with an MLP as the only expert, we used a learned IN as the critic. For experiments with an IN as one of its experts, the critic was an IN with shared parameters. We trained the metacontroller on a range of different ponder costs,  $\tau_{k}$ , for the different experts. Further details of the training procedure are available in Appendix D.

# 3.1 REACTIVE AND ITERATIVE AGENTS

Figure 2 shows the performance on the test set of the reactive and iterative agents for different numbers of ponder steps. The reactive agent performed poorly on the task, especially when the task was more difficult. With the five planets dataset, it was only able to achieve a performance loss of 0.583 on average (see Figure 1 for a depiction of the magnitude of the loss). In contrast, the iterative agent with the true simulation expert performed much better, reaching ceiling performance on the datasets with one and two planets, and achieving a performance loss of 0.0683 on the five planets dataset. The IN and MLP experts also improve over the reactive agent, with a minimum performance loss of 0.117 and 0.375 on the five planets dataset, respectively.

Figure 2 also highlights how important the choice of expert is. When using the true simulation and IN experts, the iterative agent performs well. With the MLP expert, however, performance is substantially diminished. But despite the poor performance of the MLP expert, there is still some benefit of pondering with it. With even just a few steps, the MLP iterative agent outperforms its

![](images/b8b087dc8a0d7713f17569ab227d62d1294c4218076811819c1d6e737e99f30f.jpg)  
Figure 2: Test performance of the reactive and iterative agents. Each line corresponds to the performance of an iterative agent (either the true simulation expert, the MLP expert, or the interaction net expert) trained for a fixed number of ponder steps on one of the five datasets; the line color indicates which dataset the controller was trained on. In all cases, performance refers to the performance loss,  $L_{P}$ . Left: the MLP expert struggles with the task due to its limited expressivity, but still benefits from pondering. Middle: the IN expert performs almost as well as the true simulation expert, even though it is not a perfect model. Right: The true simulation expert does quite well on the task, especially with multiple ponder steps.

![](images/3dfd48fc0064f21cadce666c1dbb52f88323baba649de7f03d152c5b3bb47ce9.jpg)

![](images/3f3d5c5bfd6f672591c851c67863cf2fe93641dfcefb2583d648a0f4b0b9617f.jpg)

reactive counterpart. However comparing the reactive agent with the  $N = 1$  iterative agent is somewhat unfair because the iterative agent has more parameters due to the expert and the memory. However, given that there tend to also be an increase in performance between one and two ponder steps (and beyond), it is clear that pondering—even with a highly inaccurate model—can still lead to better performance than a model-free reactive approach.

# 3.2 METACONTROLLER WITH ONE EXPERT

Though the iterative agents achieve impressive results, they expend more computation than necessary. For example, in the one and two planet conditions, the performances of the IN and true simulation iterative agents received little performance benefit from pondering more than two or three steps, while for the four and five planet conditions they required at least five to eight steps before their performance converged. When computational resources have no cost, the number of steps are of no concern, but when they have some cost it is important to be economical.

Because the metacontroller learns to choose its number of pondering steps, it can balance its performance loss against the cost of computation. Figure 3 (top row, middle and right subplots) shows that the IN and true simulation expert metacontroller take fewer ponder steps as  $\tau$  increases, tracking closely the minimum of the iterative agent's cost curve (i.e., the metacontroller points are always near the iterative agent curves' minima). This adaptive behavior emerges automatically from the manager's learned policy, and avoids the need to perform a hyperparameter search to find the best number of iterations for a given  $\tau$ .

The metacontroller does not simply choose an average number of ponder steps to take per episode: it actually tailors this choice to the difficulty of each episode. Figure 4 shows how the number of ponder steps the IN metacontroller chooses in each episode depends on that episode's difficulty, as measured by the episode's loss under the reactive agent. For more difficult episodes, the metacontroller tends to take more ponder steps, as indicated by the positive slopes of the best fit lines, and this proportionality persists across the different levels of  $\tau$  in each subplot.

The ability to adapt its choice of number of ponder steps on a per-episode basis is very valuable because it allows the metacontroller to spend additional computation only on those episodes which require it. The total costs of the IN and true simulation metacontrollers' are  $11\%$  and  $15\%$  lower (median) than the best achievable costs of their corresponding iterative agents, respectively, across the range of  $\tau$  values we tested (see Figure 7 in the Appendix for details).

There can even be a benefit to using a metacontroller when there are no computational resource costs. Consider the rightmost points in Figure 3 (bottom row, middle and right subplots), which show the performance loss for the IN and true simulation metacontrollers when  $\tau$  is low. Remarkably,

![](images/8835121d4b578ca07583bab8780fd1aa4ea69edda5a4215c908d11ae0ae55998.jpg)

![](images/ef61b407c6823bb5398eb368698622c621cec6551667642acbee809fe188e033.jpg)

![](images/9479f802804d44204d51b712572372afe7f9da5952ee3a1d7868f4df4fd4d64c.jpg)

![](images/1a2e520c23d95ffb5799e1509bb83b66691ab00aa94f92818e350f238a8e0c3b.jpg)  
Figure 3: Test performance of the metacontroller with a single expert on the five planets dataset. Top row: Here we show total cost rather than just performance on the task (i.e., including computation cost). Different colors (and curves) show the result for different  $\tau$ . The error bars (for the metacontroller) indicate  $2.5\%$  and  $97.5\%$  confidence intervals. When the point is below its corresponding curve, it means that the metacontroller was able to achieve a better speed-accuracy trade-off than that achievable by the iterative agent. Bottom row: Here we show just the performance loss (i.e., without computational cost). The lines indicate the performance of the iterative agents for different numbers of ponder, each subplot is a different expert. The points indicate the performance of the metacontroller, with each point corresponding to a different value of  $\tau$ . The  $x$ -coordinate of each point is an average across the number of ponder steps, and the  $y$ -coordinate is the average loss. The fact that the points are below the curve means the metacontroller agent learns to perform better than the iterative agent with the equivalent number of ponder steps.

![](images/eedada23d9493cfd7811d61030566ac373c2dafa3883190961588922ed9f0990.jpg)

![](images/e84a6b3c5be187ca2b944687590d3d0643d71777c8381fd57cbbab761ace459e.jpg)

![](images/3c1bcd0043b90249a054d5ffeaa6d94bff795f799f6ef31e572cd71e63cebab7.jpg)  
Figure 4: Relationship between the number of ponder steps and per-episode difficulty for the IN metacontroller. Each subplot's  $x$ -axis represents the episode difficulty, as measured by the reactive controller's loss. Each  $y$ -axis represents the number of ponder steps the metacontroller took. The points are individual episodes, and the line is the best fit regression line and  $95\%$  confidence intervals. The different subplots show different values of  $\tau$  (labeled in the title). In each case, there is a clear positive relationship between the difficulty of the task and the number of ponder steps, suggesting that the metacontroller learns to spend more time on hard problems and less time on easier problems.

![](images/619f36c5977284d556ba33f22ee8bee48b07b869b62b273f7c539908f9285d65.jpg)

![](images/cadd8dbe3ee7ac725d2407193f2a073dc0d37c31a625c9e26e79675f89fd41b5.jpg)

![](images/8beb5ed52295183724c368dfe7af94cb3f72d571dcb3bf63908a66356dfce01c.jpg)  
Figure 5: Test performance of the metacontroller with multiple experts on the five planets dataset. Left: The average number of total ponder steps, for different values of  $\tau$ . As with the single-expert metacontrollers, fewer ponder steps are taken when the cost is very high, and more are taken when the cost is low. Right: The fraction of ponder steps taken by the MLP expert relative to the IN expert. In the majority of cases, the metacontroller favors using the IN expert as it is much more reliable. The few exceptions (red squares) are cases when the cost of the IN expert is much higher relative to the cost of the MLP expert.

![](images/a6df3207fbe6314c0b34e38f449721b375d437bd0132ca2baf89bcea405a0fc9.jpg)

these points still outperform the best achievable iterative agents. This suggests that there can be an advantage to stopping pondering once a good solution is found, and more generally demonstrates that the metacontroller's learning process can lead to strategies that are superior to those available to less flexible agents.

The metacontroller with the MLP expert had very poor average performance and high variance on the five planet condition (Figure 3, top left subplot), which is why we restricted our focus in this section to how the metacontrollers with IN and true simulation experts behaved. The MLP's poor performance is crucial, however, for the following section (3.3) which analyzes how a multiple-expert metacontroller manages experts which vary greater in their reliability.

# 3.3 METACONTROLLER WITH TWO EXPERTS

When we allow the manager to additionally choose between two experts, rather than only relying on a single expert, we find a similar pattern of results in terms of the number of ponder steps (Figure 5, left). Additionally, the metacontroller is successfully able to identify the more reliable IN network and consequently uses it a majority of the time, except in a few cases where the cost of the IN network is extremely high relative to the cost of the MLP network (Figure 5, right). This pattern of results makes sense given the good performance (described in the previous section) of the metacontroller with the IN expert compared to the poor performance of the metacontroller with the MLP expert. The manager should not generally rely on the MLP expert because it is simply not a reliable source of information.

However, the metacontroller has more difficulty finding an optimal balance between the two experts on a step-by-step basis: the addition of a second expert did not yield much of an improvement over the single-expert metacontroller, with only  $9\%$  of the different versions (trained with different  $\tau$  values for the two experts) achieving a lower loss than the best iterative controller. We believe the mixed performance of the metacontroller with multiple experts is partially due to an entropy term which we used to encourage the manager's policy to be non-deterministic (see Appendix B.4). In particular, for high values of  $\tau$ , the optimal thing to do is to always execute immediately without pondering. However, because of the entropy term, the manager is encouraged to have a non-deterministic policy and therefore is likely to ponder more than it should—and to use experts that are more unreliable—even when this is suboptimal in terms of the total loss (3).

Despite the fact that the metacontroller with multiple experts does not result in a substantial improvement over that which uses a single expert, we emphasize that the manager is able to identify and use the more reliable expert the majority of the time. And, it is still able to choose a variable number of steps according to how difficult the task is (Figure 5, left). This, in and of itself, is an

improvement over more traditional optimization methods which would require that the expert is hand-picked ahead of time and that the number of steps are determined heuristically.

# 4 DISCUSSION

In this paper, we have presented an approach to adaptive, imagination-based optimization in neural networks. Our approach is able to flexibly choose which computations to perform as well as how many computations need to be performed, approximately solving a speed-accuracy trade-off that depends on the difficulty of the task. In this way, our approach learns to rely on whatever source of information is most useful and most efficient. Additionally, by consulting the experts on-the-fly, our approach allows agents to test out actions to ensure that their consequences are not disastrous before actually executing them.

While the experiments in this paper involve a one-shot decision task, our approach lays a foundation that can be built upon to support more complex situations. For example, rather than applying a force only on the first time step, we could turn the problem into one of trajectory optimization for continuous control by asking the controller to produce a sequence of forces. In the case of planning, our approach could potentially be combined with methods like Monte Carlo Tree-Search (MCTS) (Coulom, 2006), where our experts would be akin to having several different rollout policies to choose from, and our controller would be akin to the tree policy. While most MCTS implementations will run rollouts until a fixed amount of time has passed, our approach would allow the manager to adaptively choose the number of rollouts to perform and which policies to perform the rollouts with. Our method could also be used to naturally augment existing model-free approaches such as DQN (Mnih et al., 2015) with online model-based optimization by using the model-free policy as a controller and adding additional experts in the form of state-transition models. An interesting extension would be to compare our metacontroller architecture with a naive model-based controller that performs gradient-based optimization to produce the final control. We expect our metacontroller architecture might require fewer model evaluations and to be more robust to model inaccuracies compared to the gradient-based method, because our method has access to the full history of proposed controls and evaluations whereas traditional gradient-based methods do not.

Although we rely on differentiable experts in our metacontroller architecture, we do not utilize the gradient information from these experts. An interesting extension to our work would be to pass this gradient information through to the manager and controller (as in Andrychowicz et al. (2016)), which would likely improve performance further, especially in the more complex situations discussed here. Another possibility is to train some or all of the experts inline with the controller and metacontroller, rather than independently, which could allow their learned functionality to be more tightly integrated with the rest of the optimization loop, at the expense of their generality and ability to be repurposed for other uses.

To conclude, we have demonstrated how neural network-based agents can use metareasoning to adaptively choose what to think about, how to think about it, and for how long to think for. Our method is directly inspired by human cognition and suggests a way to make agents much more flexible and adaptive than they currently are, both in decision making tasks such as the one described here, as well as in planning and control settings more broadly.

# ACKNOWLEDGMENTS

We would like to thank Matt Hoffman, Andrea Tacchetti, Tom Erez, Nando de Freitas, Guillaume Desjardins, Joseph Modayil, Hubert Soyer, Alex Graves, David Reichert, Theo Weber, Jon Scholz, Will Dabney, and others on the DeepMind team for helpful discussions and feedback.

# REFERENCES

Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. arXiv:1606.04474, 2016.  
Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, and Koray Kavukcuoglu. Interaction networks for learning about objects, relations and physics. Advances in Neural Information Processing Systems, 2016.

Peter W. Battaglia, Jessica B. Hamrick, and Joshua B. Tenenbaum. Simulation as an engine of physical scene understanding. Proceedings of the National Academy of Sciences, 110(45):18327-18332, 2013.  
Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models. arXiv:1511.06297, 2015.  
Yoshua Bengio. Deep learning of representations: Looking forward. arXiv:1305.0445, 2013.  
Rémi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In International Conference on Computers and Games, pp. 72-83. Springer, 2006.  
Jan Glascher, Nathaniel Daw, Peter Dayan, and John P. O'Doherty. States versus rewards: Dissociable neural prediction error signals underlying model-based and model-free reinforcement learning. Neuron, 66(4):585-595, 2010.  
Alex Graves. Adaptive computation time for recurrent neural networks. arXiv:1603.08983, 2016.  
Jessica B. Hamrick, Kevin A. Smith, Thomas L. Griffiths, and Edward Vul. Think again? the amount of mental simulation tracks uncertainty in the outcome. In Proceedings of the 37th Annual Conference of the Cognitive Science Society, 2015.  
Nicholas Hay, Stuart J. Russell, David Tolpin, and Solomon Eyal Shimony. Selecting computations: Theory and applications. Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence, 2012.  
Nicolas Heess, Gregory Wayne, David Silver, Tim Lillicrap, Tom Erez, and Yuval Tassa. Learning continuous control policies by stochastic value gradients. Advances in Neural Information Processing Systems, 2015.  
Mary Hegarty. Mechanical reasoning by mental simulation. Trends in Cognitive Sciences, 8(6):280 - 285, 2004.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.  
Matthew W Hoffman, Eric Brochu, and Nando de Freitas. Portfolio allocation for Bayesian optimization. In Proceedings of the 27th Conference on Uncertainty in Artificial Intelligence, pp. 327-336, 2011.  
Eric J. Horvitz. Reasoning about beliefs and actions under computational resource constraints. In Uncertainty in Artificial Intelligence, Vol. 3, 1988.  
Philip N Johnson-Laird. Mental models and human reasoning. Proceedings of the National Academy of Sciences, 107(43):18243-18250, 2010.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv:1412.6980, 2014.  
Wouter Kool, Fiery A. Cushman, and Samuel J. Gershman. When does model-based control pay off? PLOS Computational Biology, in press.  
Sang Wan Lee, Shinsuke Shimojo, and John P. O'Doherty. Neural computations underlying arbitration between model-based and model-free learning. Neuron, 81:687-699, 2014.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research, 17:1-40, 2016.  
Falk Lieder and Thomas L. Griffiths. Strategy selection as rational metareasoning. in revision.  
Falk Lieder, Dillon Plunkett, Jessica B. Hamrick, Stuart J. Russell, Nicholas J. Hay, and Thomas L. Griffiths. Algorithm selection by rational metareasoning as a model of human strategy selection. 27:2870-2878, 2014.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv:1509.02971, 2015.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Stuart Russell and Eric Wefald. Principles of metareasoning. Artificial Intelligence, 49(1):361 - 395, 1991.  
Jürgen Schmidhuber. An on-line algorithm for dynamic reinforcement learning and planning in reactive environments. In Proceedings of the International Joint Conference on Neural Networks (IJCNN), 1990a.  
Jürgen Schmidhuber. Reinforcement learning in Markovian and non-Markovian environments. Advances in Neural Information Processing Systems, 1990b.  
Bobak Shahriari, Ziyu Wang, Matthew W Hoffman, Alexandre Bouchard-Côté, and Nando de Freitas. An entropy search portfolio for Bayesian optimization. arXiv:1406.4625, 2014.  
David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. Proceedings of the 31st International Conference on Machine Learning, 2014.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587):484-489, 2016.  
Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4):229-256, 1992.  
D.M. Wolpert and M. Kawato. Multiple paired forward and inverse models for motor control. Neural Networks, 11(78):1317 - 1329, 1998.
