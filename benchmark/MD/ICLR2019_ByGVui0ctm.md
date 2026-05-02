# GENERATIVE REPLAY WITH FEEDBACK CONNECTIONS AS A GENERAL STRATEGY FOR CONTINUAL LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Standard artificial neural networks suffer from the well-known issue of catastrophic forgetting, making continual or lifelong learning problematic. Recently, numerous methods have been proposed for continual learning, but due to differences in evaluation protocols it is difficult to directly compare their performance. To enable more meaningful comparisons, we identified three distinct continual learning scenarios based on whether task identity is known and, if it is not, whether it needs to be inferred. Performing the split and permuted MNIST task protocols according to each of these scenarios, we found that regularization-based approaches (e.g., elastic weight consolidation) failed when task identity needed to be inferred. In contrast, generative replay combined with distillation (i.e., using class probabilities as "soft targets") achieved superior performance in all three scenarios. In addition, we reduced the computational cost of generative replay by integrating the generative model into the main model by equipping it with generative feedback connections. This Replay-through-Feedback approach substantially shortened training time with no or negligible loss in performance. We believe this to be an important first step towards making the powerful technique of generative replay scalable to real-world continual learning applications.

# 1 INTRODUCTION

Current state-of-the-art deep neural networks can be trained to impressive performance on a wide variety of individual tasks. Learning multiple tasks in sequence, however, remains a substantial challenge for deep learning. When trained on a new task, standard neural networks forget most information related to previously learned tasks, a phenomenon referred to as "catastrophic forgetting".

In recent years, numerous methods for alleviating catastrophic forgetting have been proposed. However, due to the wide variety of experimental protocols used to evaluate them, many of these methods claim "state-of-the-art" performance (e.g., Kirkpatrick et al., 2017; Rebuffi et al., 2017; Nguyen et al., 2017; Masse et al., 2018; Kemker & Kanan, 2018; Wu et al., 2018). To obscure things further, some methods shown to perform well in some experimental settings are reported to dramatically fail in others: compare the performance of elastic weight consolidation in Kirkpatrick et al. (2017) and Zenke et al. (2017) with that in Kemker et al. (2017) and Kamra et al. (2017).

To enable a fairer and more structured comparison of methods for reducing catastrophic forgetting, as a first contribution this paper identifies three distinct continual learning scenarios of increasing difficulty. These scenarios are distinguished by whether at test time task identity is provided and, if it is not, whether task identity needs to be inferred. We show that such differences in experimental design can explain seemingly contradictory results reported in the recent literature: even for experimental protocols involving the relatively simple classification of MNIST-digits, methods that perform well in one continual learning scenario can completely fail in another.

Using these three scenarios, a second contribution of this paper is to provide an extensive comparison of recently proposed methods. These experiments reveal that generative replay, especially when combined with distillation techniques, has the capability to perform well on all three scenarios. An important disadvantage of this approach, however, is that it can be computationally very costly.

As a third contribution, this paper proposes a way to reduce these computational costs. Current approaches using generative replay train two separate models: a main model for solving the tasks

and a generative model for sampling examples representative of previous tasks. We merge the generative model into the main model by equipping it with feedback connections that are trained to have generative capability. We demonstrate that this substantially reduces training time, with no or negligible loss in performance.

# 2 CONTINUAL LEARNING SCENARIOS

We consider the continual learning problem in which a single model needs to sequentially learn a series of tasks, whereby it is not allowed to store raw data. This continual learning framework has been actively studied in recent years: many methods for alleviating catastrophic forgetting are being proposed, with almost as many different experimental protocols being used for their evaluation. We found that an important difference between these experimental protocols is whether at test time information about the task identity is available and—if it is not—whether the model is required to identify the identity of the task it has to solve. Yet, this crucial experimental design consideration is not always clearly stated and differences in this regard are sometimes not appreciated. For example, in Masse et al. (2018) a substantial improvement over state-of-the-art is reported, while their method assumes task identity is always available and the compared methods operate without this assumption. To enable more meaningful comparisons, we identify three distinct scenarios for continual learning.

In the first scenario, models are always informed about which task needs to be performed. This is the easiest continual learning scenario, and we refer to it as incremental task learning. Since task identity is always provided, it is possible to train models with task-specific components. A typical neural network architecture used in this scenario has a "multihead" output-layer, meaning that each task has its own output units but the rest of the network is (potentially) shared between tasks.

In the second scenario, which we refer to as incremental domain learning, task identity is not available at test time. Models however only need to solve the task at hand; they are not required to infer which task it is. Typical examples of this scenario are protocols whereby the structure of the tasks is always the same, but the input-distribution is changing. A classical example of such a task protocol is 'permuted MNIST' (Goodfellow et al., 2013), in which all tasks involve classifying MNIST-digits but with a different permutation applied to the pixels for each new task (Figure 2). Although permuted MNIST is most naturally performed according to the incremental domain learning scenario, it can be performed according to the other scenarios too (Table 2).

Finally, in the third scenario, models need to be able both to solve each task seen so far and to infer which task they are presented with. We refer to this scenario as incremental class learning, as it includes protocols in which new classes need to be learned incrementally. An example task protocol most naturally performed under this scenario is sequentially learning MNIST-digits ('split MNIST'; Figure 1), although this protocol has also been performed under the other two scenarios (Table 1).

# 3 CONTINUAL LEARNING STRATEGIES

A simple and intuitive explanation for catastrophic forgetting is that after a neural network is trained on a new task, its parameters are now optimized for the new task and no longer for the previous one(s). This formulation highlights two strategies for alleviating catastrophic forgetting: (1) not freely optimizing the entire network on each task, and (2) modifying the training data to make it more representative for previous tasks.

# 3.1 NOT OPTIMIZING ENTIRE NETWORK / REGULARIZED OPTIMIZATION

A straightforward way of not optimizing the full network on every task is to explicitly define a different sub-network for each task to be learned. A variety of recent papers have utilized this strategy, with different approaches as to how the parts of the network for each task are selected. A simple approach is to randomly and a priori assign which nodes will participate in each task (Context-dependent Gating [XdG]; Masse et al., 2018). Other approaches use evolutionary algorithms (Fernando et al., 2017) or gradient descent (Serrà et al., 2018) to learn which sets of units to employ for each task. By design, however, these approaches are limited to the incremental task learning scenario, as they require knowledge of task identity to select the correct task-specific components.

![](images/52caa0ad4cd4cda3a9c4bd946831f2088f6673fa24166ac26ce46ff866de009c.jpg)  
Figure 1: Schematic of the split MNIST task protocol.

![](images/bd5341d133d4975e18f9a2b81393060ede4ebbbd707e3eae2d73e638cdd3721e.jpg)

![](images/7151941c9e3046329397295d60f410e500b7460cf7dbe80c030232695844986b.jpg)

![](images/1a30bd314ea2affc259a985cdf57cf9abc0803f014c9cf4e01739473eda4b965.jpg)

![](images/3afecbad1aa100d7c8b0b57810360773116c4709fcf1360c6aac49b82c6987ca.jpg)

Table 1: The split MNIST task protocol according to each continual learning scenario.  

<table><tr><td>Incremental task learning</td><td>With task given, is it the first or second class? (e.g., ‘0’ or ‘1’)</td></tr><tr><td>Incremental domain learning</td><td>With task unknown, is it a first or second class?
(e.g., in ‘0’, ‘2’, ‘4’, ‘6’, ‘8’) or in ‘1’, ‘3’, ‘5’, ‘7’, ‘9’)</td></tr><tr><td>Incremental class learning</td><td>With task unknown, which digit is it? (choice from ‘0’ to ‘9’)</td></tr></table>

A modification to make this strategy applicable in the other scenarios is to preferentially train a different part of the network for each task, but to always use the entire network for execution. One way to do this is by differently regularizing the network's parameters during training on each new task, which is the approach of Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017) and Synaptic Intelligence (SI; Zenke et al., 2017). Both methods estimate for all parameters of the network how important they are for the previously learned tasks and penalize future changes to them accordingly (i.e., learning is slowed down for parts of the network important for previous tasks).

# 3.2 MODIFYING TRAINING DATA

A second strategy is to complement the training data for each new task to be learned with "pseudo-data" representative of the previous tasks. We refer to this strategy as replay. An early implementation of this strategy, called pseudo-rehearsal, generated completely random inputs as pseudo-data and labeled them based on the predictions of a copy of the model stored after finishing training on the previous task (Robins, 1995). This approach had some success with very simple, artificial inputs, but does not work with more complicated inputs (Atkinson et al., 2018).

An alternative is to take the input data of the current task, label them using the model trained on the previous tasks, and use the resulting input-target pairs as pseudo-data. This is the approach of Learning without Forgetting (LwF; Li & Hoiem, 2017). Another important aspect of this method is that instead of labeling the inputs to be replayed as the most likely category according to the previous tasks' model (i.e., "hard targets"), it pairs them with the by that model predicted probabilities for all target classes (i.e., "soft targets"). The objective for the replayed data is then to match the probabilities predicted by the model being trained to these target probabilities. The approach of matching predicted (and typically temperature-raised) probabilities of one network to those of another network had previously been used to compress (or "distill") information from one (large) network to another (smaller) network (Hinton et al., 2015).

Another option is to generate the input data to be replayed. For this, besides the main model for task performance (e.g., classification), a separate generative model is sequentially trained on all tasks to generate samples from their input data distributions. For the first application of this approach, which was called Deep Generative Replay (DGR), the generated input samples were paired with "hard targets" provided by the main model (Shin et al., 2017). We note that it is possible to combine LwF and DGR by replaying input samples from a generative model and pairing them with soft targets (see also Wu et al., 2018; Venkatesan et al., 2017). We include this hybrid method in our comparison under the name DGR+distill.

A final option is to store examples from previous tasks and replay those. Such "exact replay" can substantially boost performance (Rebuffi et al., 2017; Nguyen et al., 2017; Kemker & Kanan, 2018; see Appendix C), but due to privacy concerns or memory constraints, it is not always possible to do so. In this paper we restrict ourselves to the case where storing raw data is not allowed.

![](images/d153125a5bca310a764c9aa8762d1025dc62e3192554eca11b1adb3320ea9f8c.jpg)  
Figure 2: Schematic of the permuted MNIST task protocol.

![](images/10e5b298eda2117a5babdcd4f1efd6f06521f2e85dccb0e7ecc55be4fbe0ce0a.jpg)

![](images/065e6fadfe399794f3c2cb1e84474ab43ace608bdfd324958b7da69956f6f131.jpg)

Table 2: The permuted MNIST task protocol according to each continual learning scenario.  

<table><tr><td>Incremental task learning</td><td>Given permutation X was applied, which digit is it?</td></tr><tr><td>Incremental domain learning</td><td>With permutation unknown, which digit is it?</td></tr><tr><td>Incremental class learning</td><td>Which digit is it and which permutation was applied?</td></tr></table>

# 4 EXPERIMENTAL DETAILS

To compare the performance of the above discussed approaches, we used two different task protocols that were both performed according to all three continual learning scenarios defined in section 2.

# 4.1 TASK PROTOCOLS

The first task protocol was split MNIST (Zenke et al., 2017; Figure 1). For this protocol, the original MNIST-dataset was split into five tasks, where each task was a two digit classification. The original  $28 \times 28$  pixel grey-scale images were used without pre-processing. The standard training/test-split was used resulting in 60,000 training images ( $\sim 6000$  per digit) and 10,000 test images ( $\sim 1000$  per digit).

The second task protocol was permuted MNIST (Goodfellow et al., 2013; Figure 2). The tasks of this protocol were classifying MNIST-digits (every task now had all ten digits), whereby in each task the pixels of the MNIST-images were permuted in a different way. We used a sequence of ten such tasks. To generate the permuted images, the original images were first zero-padded to  $32 \times 32$  pixels. For each task, a random permutation was then generated and applied to these 1024 pixels. No other pre-processing was performed. Again the standard training/test-split was used.

# 4.2 METHODS

For a fair comparison, the same neural network architecture was used for all methods. For the split MNIST experiments, this was a multi-layer perceptron with 2 hidden layers of 400 nodes each, followed by a softmax output layer. ReLU non-linearities were used in all hidden layers. For the permuted MNIST experiments each hidden layer consisted of 1000 nodes.

All methods used the standard cross entropy classification loss for the model's predictions on the current task ( $\mathcal{L}_{\mathrm{current}} = \mathcal{L}_{\mathrm{classification}}$ ; see Appendix A.1.1). The regularization-based methods (i.e., EWC, online EWC and SI) added a regularization term to this loss, with regularization strength controlled by a hyperparameter:  $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{current}} + \lambda \mathcal{L}_{\mathrm{regularization}}$ . The value of this hyperparameter was set by a grid search, even though it could be argued that this is problematic in the context of continual learning (see Appendix B). The replay-based methods (i.e., LwF, DGR and DGR+distill) instead added a loss-term for the replayed data. In this case a hyperparameter could be avoided, as the loss for the current and replayed data could be weighted according to how many tasks the model has been trained on so far:  $\mathcal{L}_{\mathrm{total}} = \frac{1}{N_{\mathrm{tasks~so~far}}} \mathcal{L}_{\mathrm{current}} + \left(1 - \frac{1}{N_{\mathrm{tasks~so~far}}}\right) \mathcal{L}_{\mathrm{replay}}$ .

We compared the following approaches:

- None: The model was sequentially trained on all tasks in the standard way. This is also called fine-tuning, and can be seen as a lower bound.

- XdG: Following Masse et al. (2018), for each task a random subset of  $X\%$  of the units in each hidden layer was fully gated (i.e., their activations set to zero), with  $X$  a hyperparameter whose value was set by a grid search (see Appendix B). As this method requires availability of task identity at test time, it could only be used in the incremental task learning scenario.  
- EWC: The regularization term proposed in Kirkpatrick et al. (2017) was added to the loss, see Appendix A.2.1 for implementation details.  
- Online EWC: This is a modification of EWC proposed by Schwarz et al. (2018), with inspiration from Huszár (2018), that improves EWC's scalability by ensuring the computational cost of the regularization term does not grow with number of tasks (see Appendix A.2.2).  
- SI: The regularization proposed in Zenke et al. (2017) was added to the loss (see Appendix A.2.3).  
- LwF: Images of the current task were replayed with soft targets provided by a copy of the model stored after finishing training on the previous task (Li & Hoiem, 2017; see Appendix A.1.2).  
- DGR: A separate generative model was trained to generate the images to be replayed. Following Shin et al. (2017), the replayed images were labeled with the most likely category predicted by a copy of the main model stored after training on the previous task (i.e., hard targets).  
- DGR+distill: A separate generative model was trained to generate the images to be replayed, but these were then paired with soft targets (as in LwF) instead of hard targets (as in DGR).  
- Offline: The model was always trained using the data of all tasks so far. This is also called joint training, and was included as it can be seen as an upper bound.

For the split MNIST protocol, all models were trained for 2000 iterations per task using the ADAM-optimizer  $(\beta_{1} = 0.9, \beta_{2} = 0.999$ ; Kingma & Ba, 2014) with learning rate 0.001. The same optimizer was used for the permuted MNIST protocol, but with 5000 iterations and learning rate 0.0001. For each iteration,  $\mathcal{L}_{\mathrm{current}}$  (and  $\mathcal{L}_{\mathrm{regularization}}$ ) was calculated as average over 128 examples from the current task and—if replay was used—an additional 128 replayed examples (equally divided over all previous tasks) were used to calculate  $\mathcal{L}_{\mathrm{replay}}$ . Importantly, since the total number of replayed examples does not depend on the number of previous tasks, for our implementation of the replay-based methods the training time per task does not need to increase with number of tasks so far.

For DGR and DGR+distill, a separate generative model was sequentially trained on all tasks. A symmetric variational autoencoder (VAE; Kingma & Welling, 2013) was used as generative model, with 2 fully connected hidden layers of 400 (split MNIST) or 1000 (permuted MNIST) units and a stochastic latent variable layer of size 100. A standard normal distribution was used as prior. See Appendix A.1.3 for more details. Training of the generative model was also done with generative replay (provided by its own copy stored after finishing training on the previous task) and with the same hyperparameters (i.e., learning rate, optimizer, iterations, batch sizes) as for the main model.

# 5 RESULTS

For the split MNIST task protocol, we found a clear difference in difficulty between the three continual learning scenarios (see Table 3). Perhaps surprisingly, for all three scenarios with the split MNIST protocol, EWC and online EWC barely outperformed fine-tuning. SI performed better: it reduced catastrophic forgetting in the incremental task learning and incremental domain learning scenarios, but it also failed in the incremental class learning scenario. Strikingly, replaying images from the current task (LwF; e.g., replaying '2's and '3's in order not to forget how to recognize '0's and '1's), prevented the forgetting of previous tasks better than SI. Importantly, only the methods using generative replay retained good performance (above  $90\%$ ) in the incremental class learning scenario, and DGR+distill outperformed DGR in all scenarios.

For the permuted MNIST protocol (see Table 4), there was less difference between EWC, online EWC and SI: they performed reasonably well in the incremental task learning and incremental domain learning scenarios, but failed again in the incremental class learning scenario. While LwF had some success with the split MNIST protocol, this method did not work with the permuted MNIST protocol. The methods using generative replay were again the only ones successful in the incremental class learning scenario, and DGR+distill again always outperformed DGR.

Finally, although in the incremental task learning scenario XdG succeeded in reducing catastrophic forgetting on both task protocols, on both it was outperformed by SI (and thus by DGR+distill).

Table 3: Average test accuracy (over all tasks) on the split MNIST task protocol. Each experiment was performed 20 times with different random seeds, reported is the mean  $(\pm$  SEM) over these runs.  

<table><tr><td>Method</td><td>Incremental task learning</td><td>Incremental domain learning</td><td>Incremental class learning</td></tr><tr><td>None - lower bound</td><td>85.15 (± 1.00)</td><td>57.33 (± 1.66)</td><td>19.90 (± 0.02)</td></tr><tr><td>XdG</td><td>98.74 (± 0.31)</td><td>-</td><td>-</td></tr><tr><td>EWC</td><td>85.48 (± 1.20)</td><td>57.80 (± 1.61)</td><td>19.90 (± 0.02)</td></tr><tr><td>Online EWC</td><td>85.22 (± 1.06)</td><td>57.60 (± 1.66)</td><td>19.90 (± 0.02)</td></tr><tr><td>SI</td><td>99.14 (± 0.11)</td><td>63.77 (± 1.18)</td><td>20.04 (± 0.08)</td></tr><tr><td>LwF</td><td>99.60 (± 0.03)</td><td>71.02 (± 1.26)</td><td>24.17 (± 0.51)</td></tr><tr><td>DGR</td><td>99.47 (± 0.03)</td><td>95.74 (± 0.23)</td><td>91.24 (± 0.33)</td></tr><tr><td>DGR+distill</td><td>99.59 (± 0.03)</td><td>96.94 (± 0.14)</td><td>91.84 (± 0.27)</td></tr><tr><td>RtF</td><td>99.66 (± 0.03)</td><td>97.31 (± 0.11)</td><td>92.56 (± 0.21)</td></tr><tr><td>Offline - upper bound</td><td>99.64 (± 0.03)</td><td>98.41 (± 0.06)</td><td>97.93 (± 0.04)</td></tr></table>

Table 4: Idem as Table 3, except on the permuted MNIST task protocol.  

<table><tr><td>Method</td><td>Incremental task learning</td><td>Incremental domain learning</td><td>Incremental class learning</td></tr><tr><td>None - lower bound</td><td>81.79 (± 0.48)</td><td>78.51 (± 0.24)</td><td>17.26 (± 0.19)</td></tr><tr><td>XdG</td><td>91.40 (± 0.23)</td><td>-</td><td>-</td></tr><tr><td>EWC</td><td>94.74 (± 0.05)</td><td>94.31 (± 0.11)</td><td>25.04 (± 0.50)</td></tr><tr><td>Online EWC</td><td>95.96 (± 0.06)</td><td>94.42 (± 0.13)</td><td>33.88 (± 0.49)</td></tr><tr><td>SI</td><td>94.75 (± 0.14)</td><td>95.33 (± 0.11)</td><td>29.31 (± 0.62)</td></tr><tr><td>LwF</td><td>69.84 (± 0.46)</td><td>72.64 (± 0.52)</td><td>22.64 (± 0.23)</td></tr><tr><td>DGR</td><td>92.52 (± 0.08)</td><td>95.09 (± 0.04)</td><td>92.19 (± 0.09)</td></tr><tr><td>DGR+distill</td><td>97.51 (± 0.01)</td><td>97.35 (± 0.02)</td><td>96.38 (± 0.03)</td></tr><tr><td>RtF</td><td>97.31 (± 0.01)</td><td>97.06 (± 0.02)</td><td>96.23 (± 0.04)</td></tr><tr><td>Offline - upper bound</td><td>97.68 (± 0.01)</td><td>97.59 (± 0.01)</td><td>97.59 (± 0.02)</td></tr></table>

# 6 REPLAY-THROUGH-FEEDBACK (RTF)

Generative replay with distillation consistently outperformed the competing methods and even obtained excellent results in the challenging incremental class learning scenario. However, an important disadvantage of generative replay is that it is usually computationally expensive, among others because a separate generative model is trained. Indeed, in our experiments the training time for DGR and DGR+distill was roughly twice as long as for SI (see below). To reduce the computational cost of generative replay, we propose to integrate the generative model into the main model by equipping it with generative feedback connections.

![](images/5098ffd9bf1a48485dcce6196361845c2f5c09588223b11570bf5a572fe280b3.jpg)  
Figure 3: RtF schematic.

# 6.1 RTF: THEORY

To enable the main model to generate replay itself, we add (1) feedback connections that are trained to reconstruct inputs from their hidden representations and (2) a layer of stochastic latent variables  $\mathbf{z}$  that are trained to follow a known distribution from which it is easy to sample. In case of classification, the resulting network is for example a symmetrical VAE with an additional softmax classification layer from the final hidden layer of the encoder (Figure 3). Besides removing the need for a separate generative model, it is possible that regularization provided by the added generative objective helps to train a more robust classifier (Lasserre et al., 2006; Kingma et al., 2014).

![](images/adbf0ff98860f3f33d393ed60f50ddd6d6fc23550364c6dcb4eedbe7ad7830d3.jpg)  
Figure 4: Average test accuracy (over all tasks) on the split MNIST protocol plotted against training time. Each experiment was run 20 times: dots represent individual runs, stars indicate the mean.

![](images/80d86a7f41d8470078bcb7b65cc2a256c104930e5a207d47d427b97559db4f7d.jpg)

![](images/7c6ff33e4c7debdd7b8d8d496fd21a80a4f3bb475a6ade0383a3e905aa6fd020.jpg)

![](images/6e414f12656b941e9d9b6f764d4faf5db38974dbcf668d941469ed634acb485b.jpg)  
Figure 5: Idem as Figure 4, except on the permuted MNIST task protocol.

![](images/137baa021bae40fee81c5e1818076f2d437bf60651021fd50315ff7129c54a53.jpg)

![](images/6b3b439ab1fe8eff1db6fa65d74490c1d247e2610a8df38785c32f8a7b82faa0.jpg)

The loss function for the data of the current task now has two terms:  $\mathcal{L}_{\mathrm{current}} = \mathcal{L}_{\mathrm{generative}} + \mathcal{L}_{\mathrm{classification}}$ , whereby  $\mathcal{L}_{\mathrm{classification}}$  is the standard cross-entropy classification loss and  $\mathcal{L}_{\mathrm{generative}}$  is the VAE loss (see Appendix A.1.3). On later tasks, the training data of the current task is supplemented with replayed data. For the replayed data, as for LwF and DGR+distill, the classification term is replaced by a distillation term:  $\mathcal{L}_{\mathrm{replay}} = \mathcal{L}_{\mathrm{generative}} + \mathcal{L}_{\mathrm{distillation}}$ . The loss terms for the current and replayed data are again weighted according to how many tasks the model has seen so far.

# 6.2 RTF: RESULTS

For a fair comparison, the model we used for RtF also had 2 fully connected hidden layers with 400 (split MNIST) or 1000 (permuted MNIST) units. Similar to the VAE used for DGR and DGR+distill, the stochastic latent variable layer was of size 100 with a standard normal prior. Also the same hyperparameters (i.e., learning rate, optimizer, iterations, batch sizes) were used for training.

We found that RtF slightly outperformed DGR+distill on all experiments with the split MNIST protocol (Table 3), while it performed slightly less on the experiments with the permuted MNIST protocol (Table 4). These differences were relatively small and, similar to DGR+distill, RtF comfortably outperformed all other tested methods. To assess the extent to which RtF reduced the computational cost of generative replay, and to compare the resulting cost with that of the other methods, in Figures 4 and 5 we plotted for each method its performance against total training time on a NVIDIA GeForce GTX 1080 GPU. As expected training time was always longest for DGR and DGR+distill, and this time was substantially reduced—for most experiments almost halved—by RtF.

# 7 DISCUSSION

Catastrophic forgetting is a major obstacle to the development of artificial intelligence applications capable of true lifelong learning (Kumaran et al., 2016; Parisi et al., 2018), and enabling neural

![](images/3e4ad086797aeeabffd5179ff0cb30afd703e5acab557958d53fbf94d04f0434.jpg)  
Figure 6: Randomly selected samples from the generative model after finishing training on task 1, 4, 6 and 9 of the permuted MNIST protocol (i.e., examples of what is replayed during task 2, 5, 7 and 10). No permutation was applied for task 1. Because after task 6 (or 9),  $\sim 17\%$  (or  $\sim 11\%$ ) of samples should correspond to task 1 and should thus be unshuffled digits, this figure shows that at least the quality of those replayed images is far from perfect. Nevertheless, task 1 is not being forgotten.

![](images/06e937498622703d2b1cc4f383e6437e2794a4481ee7af98d6cda251041d1ede.jpg)

![](images/184c826f67b9c86ba8deffe338f5b8aaa70633070197dae795a10d8742ffd717.jpg)

![](images/ae083fad3a8082ab5555db67aef50500f7ebe2ce2e1e2a68469acee507c9406f.jpg)

networks to sequentially learn multiple tasks has become a topic of intense research. Despite its scope, this research field lacks common benchmarks—even though the same datasets tend to be used, which makes direct comparisons between published methods difficult. We found that an important difference between currently used experimental protocols is in whether task identity is provided and—if it is not—in whether it needs to be inferred. For each of the resulting three scenarios, we performed a comprehensive comparison of recently proposed methods. An important conclusion is that for the incremental class learning scenario (i.e., when task identity needs to be inferred), only replay-based methods are capable of producing acceptable results. In this scenario, even for relatively simple task protocols involving the classification of MNIST-digits, regularization-based methods such as EWC and SI completely failed. Moreover, also in the other scenarios, generative replay combined with distillation consistently outperformed all other tested methods. These results establish generative replay as a promising general strategy for lifelong learning.

However, an important limitation of the current study is that generating MNIST-digits is relatively easy. We leave it for future work to empirically address whether generative replay can scale to task protocols with more complicated inputs, but here we highlight several reasons why we believe this will be the case. First, with the permuted MNIST protocol, we observed that even when the quality of the replayed samples had substantially declined (see Figure 6), they still helped to prevent catastrophic forgetting. Second, under some conditions (e.g., with the split MNIST protocol), replaying inputs from the current task (i.e., LwF) works reasonably well, further indicating that the replayed samples need not be perfect and that "good enough" can suffice. We hypothesize that the use of distillation is especially important to make generative replay more robust to the quality of the replayed inputs. Finally, of course, the capabilities of generative models are improving at a rapid pace (e.g., Goodfellow et al., 2014; Oord et al., 2016; Rezende & Mohamed, 2015).

This last point however also warrants caution. Although the latest developments in for example generative adversarial networks, auto-regressive decoders or flow-based models enable training high quality generative models for increasingly complicated input distributions, this can come at high computational costs. Especially in a lifelong learning setting, where models continually need to be trained on new tasks and where training sometimes has to be in real-time, efficiency is important. We therefore emphasize that continual learning methods should not only be evaluated in terms of their performance, but also in terms of for example their training time (e.g., Figures 4 and 5; see also Farquhar & Gal, 2018). Here, we improved the efficiency of generative replay by merging the generator into the main model. We also want to highlight that in our implementation of replay the number of replayed examples did not increase with number of tasks. We hypothesize that a relatively small number of examples per task can be acceptable because information on the previous tasks is also contained in the initiation bias (i.e., training on each new task starts with a network that is already optimized for the previous tasks).

To conclude, we believe that generative replay brings more to the table than simply "shift[ing] the catastrophic forgetting problem to the training of the generative model" (Schwarz et al., 2018; p.3), and we envision that a small amount of good enough replay generated by the model's own feedback connections could become a valuable tool for real-world continual learning applications.

# REFERENCES

Craig Atkinson, Brendan McCane, Lech Szymanski, and Anthony Robins. Pseudo-recursal: Solving the catastrophic forgetting problem in deep neural networks. arXiv preprint arXiv:1802.03875, 2018.  
Carl Doersch. Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908, 2016.  
Sebastian Farquhar and Yarin Gal. Towards robust evaluations of continual learning. arXiv preprint arXiv:1805.09733, 2018.  
Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint arXiv:1701.08734, 2017.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks. arXiv preprint arXiv:1312.6211, 2013.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Ferenc Huszár. Note on the quadratic penalties in elastic weight consolidation. Proceedings of the National Academy of Sciences, 115(11):E2496-E2497, 2018.  
Nitin Kamra, Umang Gupta, and Yan Liu. Deep generative dual memory network for continual learning. arXiv preprint arXiv:1710.10368, 2017.  
Ronald Kemker and Christopher Kanan. Fearnet: Brain-inspired model for incremental learning. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=SJ1Xmf-Rb.  
Ronald Kemker, Angelina Abitino, Marc McClure, and Christopher Kanan. Measuring catastrophic forgetting in neural networks. arXiv preprint arXiv:1708.02072, 2017.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems, pp. 3581-3589, 2014.  
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, pp. 201611835, 2017.  
Dharshan Kumaran, Demis Hassabis, and James L McClelland. What learning systems do intelligent agents need? complementary learning systems theory updated. Trends in cognitive sciences, 20 (7):512-534, 2016.  
Julia A Lasserre, Christopher M Bishop, and Thomas P Minka. Principled hybrids of generative and discriminative models. In Computer Vision and Pattern Recognition, 2006 IEEE Computer Society Conference on, volume 1, pp. 87-94. IEEE, 2006.  
Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

James Martens. New insights and perspectives on the natural gradient method. arXiv preprint arXiv:1412.1193, 2014.  
Nicolas Y Masse, Gregory D Grant, and David J Freedman. Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. arXiv preprint arXiv:1802.01569, 2018.  
Cuong V Nguyen, Yingzhen Li, Thang D Bui, and Richard E Turner. Variational continual learning. arXiv preprint arXiv:1710.10628, 2017.  
Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759, 2016.  
German I Parisi, Ronald Kemker, Jose L Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. arXiv preprint arXiv:1802.07569, 2018.  
Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proc. CVPR, 2017.  
Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. arXiv preprint arXiv:1505.05770, 2015.  
Matthew Riemer, Tim Klinger, Michele Franceschini, and Djallel Boueaffouf. Scalable recollections for continual lifelong learning. arXiv preprint arXiv:1711.06761, 2018.  
Anthony Robins. Catastrophic forgetting, rehearsal and pseudorehearsal. _Connection Science_, 7(2): 123-146, 1995.  
Jonathan Schwarz, Jelena Luketina, Wojciech M Czarnecki, Agnieszka Grabska-Barwinska, Yee Whye Teh, Razvan Pascanu, and Raia Hadsell. Progress & compress: A scalable framework for continual learning. arXiv preprint arXiv:1805.06370, 2018.  
Joan Serrà, Dídac Surís, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. arXiv preprint arXiv:1801.01423, 2018.  
Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep generative replay. In Advances in Neural Information Processing Systems, pp. 2994-3003, 2017.  
Ragav Venkatesan, Hemanth Venkateswara, Sethuraman Panchanathan, and Baoxin Li. A strategy for an uncompromising incremental learner. arXiv preprint arXiv:1705.00744, 2017.  
Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, Zhengyou Zhang, and Yun Fu. Incremental classifier learning with generative adversarial networks. arXiv preprint arXiv:1802.00853, 2018.  
Friedemann Zenke, Ben Poole, and Surya Ganguli. Improved multitask learning through synaptic intelligence. arXiv preprint arXiv:1703.04200, 2017.
