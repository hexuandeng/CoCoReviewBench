# ON DISTRIBUTED ADAPTIVE OPTIMIZATION WITH GRADIENT COMPRESSION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study COMP-AMS, a distributed optimization framework based on gradient averaging and adaptive AMSGrad algorithm. Gradient compression is applied to reduce the communication in the gradient transmission process, whose bias is corrected by the tool of error feedback. Our convergence analysis of COMP-AMS shows that such gradient averaging strategy yields same convergence rate as standard AMSGrad, and also exhibits linear speedup effect w.r.t. the number of local workers. Compared with recently proposed protocols on distributed adaptive methods, COMP-AMSis simple and convenient. Numerical experiments are conducted to justify the theoretical findings, and demonstrate that the proposed method can achieve same test accuracy as full-gradient AMSGrad with substantial communication savings. With its simplicity and efficiency, COMP-AMScan serve as a useful distributed training framework for adaptive methods.

# 1 INTRODUCTION

Deep neural network has achieved the state-of-the-art learning performance on numerous AI applications, e.g., computer vision (Goodfellow et al., 2014; He et al., 2016; Voulodimos et al., 2018), Natural Language Processing (Graves et al., 2013; Young et al., 2018; Zhang et al., 2018), Reinforcement Learning (Mnih et al., 2013; Silver et al., 2017) and recommendation systems (Covington et al., 2016; Wei et al., 2017). With the increasing sheet size of data observations and growing complexity of deep neural networks, standard single-machine training procedures encounter at least two major challenges:

- Due to the limited computing power of a single-machine, processing the massive number of data samples takes a long time — training is too slow. Many real-world applications can not even afford spending that much time on training.  
- In many scenarios, data are stored in multiple servers, possibly at different locations, due to the storage constraints (massive user behavior data, Internet images, etc.) or privacy reasons (Chang et al., 2018). Hence, transmitting data among servers might be costly.

Distributed learning framework (Dean et al., 2012) has been a common training strategy to tackle the above two issues. For example, in centralized distributed stochastic gradient descent (SGD) protocol, data are located at  $n$  local nodes, at which the gradients of the model are computed in parallel. In each iteration, a central server aggregates the local gradients, updates the global model, and transmits back the updated model to the local nodes for subsequent gradient computation. As we can see, this setting naturally solves aforementioned issues: 1) We use  $n$  computing nodes to train the model, so the time per training epoch can be largely reduced; 2) There is no need to transmit the local data to central server. Besides, distributed training also provides stronger error tolerance since the training process could continue even one local machine breaks down. As a result of these advantages, there has been a surge of study and applications on distributed systems (Boyd et al., 2011; Nedic & Ozdaglar, 2009; Duchi et al., 2011; Goyal et al., 2017; Hong et al., 2017; Lu et al., 2019; Koloskova et al., 2019).

Gradient compression. Among many optimization strategies, SGD is still the most popular prototype in distributed training for its simplicity and effectiveness (Chilimbi et al., 2014; Agarwal et al., 2018; Mikami et al., 2018). Yet, when the deep learning model is very large, the communication between local nodes and central server could be expensive, and the burdensome gradient transmission

would slow down the whole training system. Thus, reducing the communication cost in distributed SGD has become an active topic, and an important ingredient of large-scale distributed systems (e.g., Seide et al. (2014)). Solutions based on quantization, sparsification and other compression techniques of the local gradients are proposed, e.g., Alistarh et al. (2017); Wen et al. (2017); Wangni et al. (2018); Stich et al. (2018); Aji & Heafield (2017); Bernstein et al. (2018); De Sa et al. (2017); Yang et al. (2019); Ivkin et al. (2019). However, it has been observed both theoretically and empirically (Stich et al., 2018; Ajalloeian & Stich, 2020), that directly updating with the compressed gradients usually brings non-negligible performance downgrade in terms of convergence speed and accuracy. To tackle this problem, studies (e.g., Stich et al. (2018); Karimireddy et al. (2019)) show that the technique of error feedback can to a large extent remedy the issue of such gradient compression, achieving same convergence rate as full-gradient SGD.

Adaptive optimization. In recent years, adaptive optimization algorithms (e.g. AdaGrad (Duchi et al., 2010), Adam (Kingma & Ba, 2014) and AMSGrad (Reddi et al., 2018)) have become popular because of their superior empirical performance. These methods use different implicit learning rates for different coordinates that keep changing adaptively throughout the training process, based on the learning trajectory. In many cases, adaptive methods have been shown to converge faster than SGD, sometimes with better generalization as well. However, the body of literature that extends adaptive methods to distributed training is still fairly limited. In particular, even the simple gradient averaging approach, though appearing standard, has not been analyzed for adaptive optimization algorithms. Given that distributed SGD with compressed gradient averaging can match the performance of standard SGD, one natural question is: is it also true for adaptive methods? In this work, we try to fill this gap formally, by analyzing COMP-AMS, a distributed adaptive optimization framework using the gradient averaging protocol, with communication-efficient gradient compression.

# 1.1 OUR CONTRIBUTIONS

Specifically, we focus on a simple algorithm design leveraging the adaptivity of AMSGrad and the computational virtue of local gradient compression. Our contributions summarize as follows:

- We consider COMP-AMS, a synchronous distributed adaptive optimization framework based on global averaging with gradient compression, which is efficient in both communication and memory as no local moment estimation is needed. Our scheme is coupled with an error-feedback technique to reduce the bias implied by the compression step.  
- We provide the convergence analysis of distributed COMP-AMS (with  $n$  workers) in smooth non-convex optimization, where data heterogeneity is allowed. In the special case of  $n = 1$  (single machine), similar to SGD, gradient compression with error feedback in adaptive method achieves same convergence rate  $\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$  as the standard full-gradient counterpart. Also, we show that with a properly chosen learning rate, COMP-AMS achieves  $\mathcal{O}\left(\frac{1}{\sqrt{nT}}\right)$  convergence, implying a linear speedup in terms of the number of local workers to attain  $\mathcal{O}(\delta)$ -stationary point.  
- We present numerical experiments on three tasks to validate our theoretical findings, and show that COMP-AMS has comparable performance with other distributed adaptive methods, and matches the accuracy of full-precision AMSGrad with substantially reduced communication cost. Thus, it can serve as a convenient distributed training strategy in practice.

# 2 RELATED WORK

# 2.1 DISTRIBUTED SGD WITH COMPRESSED GRADIENTS

Quantization. As we mentioned before, SGD is the most commonly adopted optimization method in distributed training of deep neural nets. To reduce the expensive communication in large-scale distributed systems, extensive works have considered various compression techniques applied to the gradient transaction procedure. The first strategy is quantization. Dettmers (2016) condenses 32-bit floating numbers into 8-bits when representing the gradients. Seide et al. (2014); Bernstein et al. (2018); Karimireddy et al. (2019); Bernstein et al. (2019) use the extreme 1-bit information (sign) of the gradients, combined with tricks like momentum, majority vote and memory. Other quantization-based methods include QSGD (Alistarh et al., 2017; Wu et al., 2018; Zhang et al.,

2017) and LPC-SVRG (Yu et al., 2019b), leveraging unbiased stochastic quantization. The saving in communication of quantization methods is moderate: for example, 8-bit quantization reduces the cost to  $25\%$  (compared with 32-bit full-precision). Even in the extreme 1-bit case, the largest compression ratio is around  $1/32 \approx 3.1\%$ .

Sparsification. Gradient sparsification is another popular solution which may provide higher compression rate. Instead of commuting the full gradient, each local worker only passes a few coordinates to the central server and zeros out the others. Thus, we can more freely choose higher compression ratio (e.g.,  $1\%$ ,  $0.1\%$ ), still achieving impressive performance in many applications (Lin et al., 2018). Stochastic sparsification methods, including uniform and magnitude based sampling (Wangni et al., 2018), select coordinates based on some sampling probability, yielding unbiased gradient compressors with proper scaling. Deterministic methods are simpler, e.g., Random- $k$ , Top- $k$  (Stich et al., 2018; Shi et al., 2019) (selecting  $k$  elements with largest magnitude), Deep Gradient Compression (Lin et al., 2018), but usually lead to biased gradient estimation. In Ivkin et al. (2019), the central server identifies heavy-hitters from the count-sketch (Charikar et al., 2002) of the local gradients, which can be regarded as a noisy variant of Top- $k$  strategy. More applications and analysis of compressed distributed SGD can be found in Jiang & Agrawal (2018); Shen et al. (2018); Alistarh et al. (2018); Basu et al. (2019); Jiang et al. (2018), among others.

Error Feedback (EF). Biased gradient estimation, which is a consequence of many aforementioned methods (e.g., signSGD, Top- $k$ ), undermines the model training, both theoretically and empirically, with slower convergence and worse generalization (Ajalloeian & Stich, 2020; Beznosikov et al., 2020). The technique of error feedback is able to "correct for the bias" and fix the convergence issues. In this procedure, the difference between the true stochastic gradient and the compressed one is accumulated locally, which is then added back to the local gradients in later iterations. Stich et al. (2018); Karimireddy et al. (2019) prove the  $\mathcal{O}\left(\frac{1}{T}\right)$  and  $\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$  convergence rate of EF-SGD in strongly convex and non-convex setting respectively, matching the rates of vanilla SGD (Nemirovski et al., 2009; Ghadimi & Lan, 2013). More works on the convergence rate of SGD with error feedback include (Zheng et al., 2019; Stich & Karimireddy, 2019), among other related papers.

# 2.2 ADAPTIVE OPTIMIZATION

In each SGD update, all the gradient coordinates share the same learning rate. This latter is either constant or decreasing through the iterations. Adaptive optimization methods cast different learning rate on each dimension. For instance, AdaGrad, developed in Duchi et al. (2010), divides the gradient element-wisely by  $\sqrt{\sum_{t=1}^{T} g_t^2} \in \mathbb{R}^d$ , where  $g_t \in \mathbb{R}^d$  is the gradient vector at time  $t$  and  $d$  is the model dimensionality. Thus, it intrinsically assigns different learning rates to different coordinates throughout the training—elements with smaller previous gradient magnitude tend to move a larger step via larger learning rate. Other adaptive methods include AdaDelta (Zeiler, 2012) and Adam (Kingma & Ba, 2014), which introduce momentum and moving average of second moment estimation into AdaGrad hence leading to better performances. AMSGrad (Reddi et al., 2018), and potential improvements as in Wang et al. (2019), fix the potential convergence issue of Adam. In this paper, we will use AMSGRAD as the prototype, which is summarized in Algorithm 1.

In general, adaptive optimization methods in many cases exhibit faster convergence than SGD. Thus, they have been widely used in training deep learning models in language and computer vision appli

# Algorithm 1 AMSGRAD

1: Input: parameters  $\beta_{1},\beta_{2},\epsilon$  , learning rate  $\eta_t$  
2: Initialize:  $\theta_{1} \in \Theta$ ,  $m_{0} = \mathbf{0} \in \mathbb{R}^{d}$ ,  $v_{0} = \mathbf{0} \in \mathbb{R}^{d}$  
3: for  $t = 1,\dots ,T$  do  
4: Compute stochastic gradient  $g_{t}$  at  $\theta_{t}$  
5:  $m_{t} = \beta_{1}m_{t - 1} + (1 - \beta_{1})g_{t}$  
6:  $v_{t} = \beta_{2}v_{t - 1} + (1 - \beta_{2})g_{t}^{2}$  
7:  $\hat{v}_t = \max (\hat{v}_{t - 1}, v_t)$  
8:  $\theta_{t + 1} = \theta_t - \eta_t\frac{\theta_t}{\sqrt{\hat{v}_t + \epsilon}}$  
9: end for

cations, e.g., Choi et al. (2019); You et al. (2020); Zhang et al. (2020). In distributed setting, Nazari et al. (2019) proposes a decentralized system in online optimization, but communication efficiency is not considered. Mostly relevant to our work, Chen et al. (2020) proposed a distributed training algorithm based on Adam, which requires every local node to store a local estimation of the moments of the gradient. Thus, one has to keep extra two more tensors of the model size on each local worker, which may be less feasible in terms of memory particularly with large models. More recently, Tang et al. (2021) proposes an Adam pre-conditioned momentum SGD method. We will present more detailed comparisons to these works in Section 3.

# 3 COMMUNICATION-EFFICIENT ADAPTIVE OPTIMIZATION

Consider the distributed optimization task where  $n$  workers jointly solve a large finite-sum optimization problem in the form of

$$
\min  _ {\theta \in \Theta} \frac {1}{n} \sum_ {i = 1} ^ {n} f _ {i} (\theta) := \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {x \sim \chi_ {i}} \left[ F _ {i} (\theta ; x) \right], \tag {1}
$$

where the non-convex function  $f_{i}$  represents the average loss (over the local data samples) for worker  $i\in [n]$  and  $\theta$  the global model parameter taking value in  $\Theta$ , a subset of  $\mathbb{R}^d$ .  $\mathcal{X}_i$  is the data distribution on each local node, which might be different.

# 3.1 GRADIENT COMPRESSORS

In this paper, we mainly consider deterministic  $q$ -deviate compressors defined as below.

Assumption 1. The gradient compressor  $\mathcal{C}:\mathbb{R}^d\mapsto \mathbb{R}^d$  is  $q$ -deviate: for  $\forall x\in \mathbb{R}^d$ ,  $\exists 0\leq q < 1$  such that  $\| \mathcal{C}(x) - x\| \leq q\| x\|$ .

Larger  $q$  indicates heavier compression, while smaller  $q$  implies better approximation of the true gradient.  $q = 0$  implies  $\mathcal{C}(x) = x$ , i.e., no compression. In the following, we give two popular and efficient  $q$ -deviate compressors that will be adopted in this paper.

Definition 1 (Top- $k$ ). For  $x \in \mathbb{R}^d$ , denote  $\mathcal{S}$  as the size- $k$  set of  $i \in [d]$  with largest  $k$  magnitude  $|x_i|$ . The Top- $k$  compressor is defined as  $\mathcal{C}(x)_i = x_i$ , if  $i \in \mathcal{S}$ ;  $\mathcal{C}(x)_i = 0$  otherwise.

Definition 2 (Block-Sign). For  $x \in \mathbb{R}^d$ , define  $M$  blocks indexed by  $\mathcal{B}_i$ ,  $i = 1, \dots, M$ , with  $d_i := |\mathcal{B}_i|$ . The Block-Sign compressor is defined as  $\mathcal{C}(x) = [sign(x_{\mathcal{B}_1}) \frac{\|x_{\mathcal{B}_1}\|_1}{d_1}, \dots, sign(x_{\mathcal{B}_M}) \frac{\|x_{\mathcal{B}_M}\|_1}{d_M}]$ .

Remark 1. It is well-known (Stich et al., 2018) that for Top- $k$ ,  $q^2 = 1 - \frac{k}{d}$ . For Block-Sign, by Cauchy-Schwartz inequality we have  $q^2 = 1 - \min_{i \in [M]} \frac{1}{d_i}$  where  $M$  and  $d_i$  are defined in Definition 2 (Zheng et al., 2019).

The intuition behind Top- $k$  is that, it has been observed empirically that when training many deep models, most gradients are typically very small, and gradients with large magnitude contain most information. The Block-Sign compressor is a simple extension of the 1-bit SIGN compressor (Seide et al., 2014; Bernstein et al., 2018), adapted to different gradient magnitude in different blocks, which, for neural nets, are usually set as the distinct network layers. The scaling factor in Definition 2 is to preserve the (possibly very different) gradient magnitude in each layer. In principle, Top- $k$  would perform the best when the gradient is effectively sparse, while Block-Sign compressor is favorable by nature when most gradients have similar magnitude within each layer.

# 3.2 COMP-AMS: DISTRIBUTED ADAPTIVE TRAINING BY GRADIENT AGGREGATION

We present in Algorithm 2 the proposed communication-efficient distributed adaptive method in this paper, COMP-AMS. This framework can be regarded as an analogue to the standard synchronous distributed SGD: in each iteration, each local worker transmits to the central server the compressed stochastic gradient computed using local data. Then the central server takes the average of local gradients, and performs an AMSGrad update. In Algorithm 2, line 7-8 depict the error feedback operation at local nodes.  $e_{t,i}$  is the accumulated error from gradient compression on the  $i$ -th worker up to time  $t - 1$ . This residual is added back to  $g_{t,i}$  to get the "corrected" gradient. In Section 4

Algorithm 2 Distributed COMP-AMS with error-feedback  
1: Input: parameters  $\beta_{1},\beta_{2},\epsilon$  , learning rate  $\eta_t$    
2: Initialize: central server parameter  $\theta_{1}\in \Theta \subseteq \mathbb{R}^{d};e_{1,i} = 0$  the error accumulator for each worker;  $m_0 = 0,v_0 = 0,\hat{v}_0 = 0$    
3: for  $t = 1,\dots ,T$  do   
4: parallel for worker  $i\in [n]$  do:   
5: Receive model parameter  $\theta_{t}$  from central server   
6: Compute stochastic gradient  $g_{t,i}$  at  $\theta_{t}$    
7: Compute  $\tilde{g}_{t,i} = \mathcal{C}(g_{t,i} + e_{t,i},k)$    
8: Update the error  $e_{t + 1,i} = e_{t,i} + g_{t,i} - \tilde{g}_{t,i}$    
9: Send  $\tilde{g}_{t,i}$  back to central server   
10: end parallel   
11: Central server do:   
12:  $\bar{g}_t = \frac{1}{n}\sum_{i = 1}^n\tilde{g}_{t,i}$    
13:  $m_t = \beta_1m_{t - 1} + (1 - \beta_1)\bar{g}_t$    
14:  $v_{t} = \beta_{2}v_{t - 1} + (1 - \beta_{2})\bar{g}_{t}^{2}$    
15:  $\hat{v}_t = \max (v_t,\hat{v}_{t - 1})$    
16: Update the global model  $\theta_{t + 1} = \theta_t - \eta_t\frac{m_t}{\sqrt{\hat{v}_t + \epsilon}}$    
17: end for

& 5, we will show that error feedback, similar to the case of SGD, also brings good convergence behavior under gradient compression in distributed AMSGrad.

Comparison with related methods. Next, we discuss the differences between COMP-AMS and two recently proposed methods also trying to solve the distributed adaptive optimization problem. Note that, the first important difference is that these two methods both are based on the Adam optimizer, while we use AMSGrad as the prototype.

- Comparison with Chen et al. (2020). Chen et al. (2020) develops a quantized variant of Adam (Kingma & Ba, 2014), called "QAdam". In this method, each worker keeps a local copy of the moment estimates, commonly noted  $m$  and  $v$ , and compresses and transmits the ratio  $\frac{m}{v}$  as a whole to the server. Thus, that method is very much like the compressed distributed SGD, with the exception that the ratio  $\frac{m}{v}$  plays the role of the gradient vector  $g$  communication-wise. Thus, two local moment estimators are additionally required, which have same size as the deep learning model. In our COMP-AMS, the moment estimates  $m$  and  $v$  are kept and updated only at the central server, thus not introducing any extra variable (tensor) on local nodes during training (except for the error accumulator). Hence, COMP-AMS is not only effective in communication reduction, but also efficient in terms of memory (space), which is feasible when training large-scale learners like BERT and CTR prediction models, e.g. Devlin et al. (2019); Zhao et al. (2020), to lower the hardware consumption happening in practice.  
- Comparison with Tang et al. (2021) The recent work (Tang et al., 2021) proposes "1BitAdam". They first run some warm-up training steps using standard Adam, and then store the second moment moving average  $v$ . Then, distributed Adam training starts with  $v$  frozen. Thus, 1BitAdam is actually more like a distributed momentum SGD with some pre-conditioned coordinate-wise learning rates. The number of warm-up steps also needs to be carefully tuned, otherwise bad pre-conditioning may hurt the learning performance. Our COMP-AMS is simpler, as no pre-training is needed. Also, 1BitAdam requires extra tensors for  $m$  locally, while COMP-AMS does not need additional local memory.

# 4 CONVERGENCE ANALYSIS

For the convergence analysis of COMP-AMS we will make following additional assumptions.

Assumption 2. (Smoothness) For  $\forall i\in [n]$ ,  $f_{i}$  is  $L$ -smooth:  $\| \nabla f_i(\theta) - \nabla f_i(\vartheta)\| \leq L\| \theta -\vartheta \|$ .

Assumption 3. (Unbiased and bounded stochastic gradient) For  $\forall t > 0$ ,  $\forall i \in [n]$ , the stochastic gradient is unbiased and uniformly bounded:  $\mathbb{E}[g_{t,i}] = \nabla f_i(\theta_t)$  and  $\| g_{t,i} \| \leq G$ .

Assumption 4. (Bounded variance) For  $\forall t > 0$ ,  $\forall i \in [n]$ : (i) the local variance of the stochastic gradient is bounded:  $\mathbb{E}[\| g_{t,i} - \nabla f_i(\theta_t)\|^2] < \sigma^2$ ; (ii) the global variance is bounded by  $\frac{1}{n}\sum_{i=1}^{n}\|\nabla f_i(\theta_t) - \nabla f(\theta_t)\|^2 \leq \sigma_g^2$ .

In Assumption 3, the uniform bound on the stochastic gradient is common in the convergence analysis of adaptive methods, e.g., Reddi et al. (2018); Zhou et al. (2018); Chen et al. (2019). The global variance bound  $\sigma_g^2$  in Assumption 4 characterizes the difference among local objective functions, which, is mainly caused by different local data distribution  $\mathcal{X}_i$  in (1). In classical distributed setting where all the workers can access the same dataset and local data are assigned randomly,  $\sigma_g^2 \equiv 0$ . The scenario where  $\mathcal{X}_i$ 's are different gives rise to the recently proposed Federated Learning (FL) (McMahan et al., 2017) framework where local data can be non-i.i.d. While typical FL method with periodical model averaging is not the focus of this present paper, we consider the global variance in our analysis to shed some light on the impact of non-i.i.d. data distribution in the federated setting for broader interest and future investigation.

We derive the following general convergence rate of COMP-AMS in the distributed setting.

Theorem 1. Denote  $C_0 = \sqrt{\frac{4(1 + q^2)^3}{(1 - q^2)^2}G^2 + \epsilon}$ ,  $C_1 = \frac{\beta_1}{1 - \beta_1} +\frac{2q}{1 - q^2}$ ,  $\theta^{*} = \arg \min f(\theta)$ . Under Assumption 1 to Assumption 4, with  $\eta_t = \eta \leq \frac{\epsilon}{3C_0\sqrt{2L\max\{2L,C_2\}}}$ , Algorithm 2 satisfies

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} [ \| \nabla f (\theta_ {t}) \| ^ {2} ] \leq 2 C _ {0} \Big (\frac {\mathbb {E} [ f (\theta_ {1}) - f (\theta^ {*}) ]}{T \eta} + \frac {\eta L \sigma^ {2}}{n \epsilon} + \frac {3 \eta^ {2} L C _ {0} C _ {1} \sigma^ {2}}{n \epsilon^ {2}} \\ + \frac {1 2 \eta^ {2} q ^ {2} L C _ {0} \sigma_ {g} ^ {2}}{(1 - q ^ {2}) ^ {2} \epsilon^ {2}} + \frac {(1 + C _ {1}) G ^ {2} d}{T \sqrt {\epsilon}} + \frac {\eta (1 + 2 C _ {1}) C _ {1} L G ^ {2} d}{T \epsilon}). \\ \end{array}
$$

The LHS of Theorem 1 is the expected squared norm of the gradient from a uniformly chosen iterate  $t \in [T]$ , which is a common convergence measure. From Theorem 1, we see that the more compression we apply to the gradient vectors (i.e., larger  $q$ ), the larger the gradient magnitude is, i.e., the slower the algorithm converges. This is intuitive as heavier compression loses more gradient information which would slower down the learner to find a good solution.

Note that, COMP-AMS with  $n = 1$  naturally reduces to the single-machine (sequential) AMSGrad (Algorithm 1) with compressed gradients instead of full-precision ones. The paper (Karimireddy et al., 2019) specifically analyzed this case for SGD, showing that single-machine SGD with error feedback has the same convergence rate as vanilla SGD using full gradients. In alignment with the conclusion in Karimireddy et al. (2019), for adaptive AMSGrad, we have a similar result.

Corollary 1. When  $n = 1$ , under Assumption 1 to Assumption 4, setting the stepsize as  $\eta = \min \left\{\frac{\epsilon}{3C_0\sqrt{2L\max\{2L,C_2\}}},\frac{1}{\sqrt{T}}\right\}$ , Algorithm 2 satisfies:

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} [ \| \nabla f (\theta_ {t}) \| ^ {2} ] \leq \mathcal {O} (\frac {1}{\sqrt {T}} + \frac {\sigma^ {2}}{\sqrt {T}} + \frac {d}{T}).
$$

Corollary 1 states that with error feedback, single machine AMSGrad with biased compressed gradients can also match the convergence rate  $\mathcal{O}\left(\frac{1}{\sqrt{T}} + \frac{d}{T}\right)$  of standard AMSGrad (Zhou et al., 2018) in non-convex optimization. It also achieves the same rate  $\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$  of vanilla SGD (Karimireddy et al., 2019) when  $T$  is sufficiently large. In other words, EF also fixes the convergence issue of using compressed gradients in AMSGrad.

Linear Speedup. In Theorem 1, the convergence rate is derived assuming constant learning rate. By carefully choosing a decreasing learning rate dependent on the number of workers, we have the following simplified statement.

Corollary 2. Under the same setting as Theorem 1, set  $\eta = \min \left\{\frac{\epsilon}{3C_0\sqrt{2L\max\{2L,C_2\}}},\frac{\sqrt{n}}{\sqrt{T}}\right\}$ . The COMP-AMS iterates admit

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} [ \| \nabla f (\theta_ {t}) \| ^ {2} ] \leq \mathcal {O} \left(\frac {1}{\sqrt {n T}} + \frac {\sigma^ {2}}{\sqrt {n T}} + \frac {n \left(\sigma^ {2} + \sigma_ {g} ^ {2}\right)}{T}\right). \tag {2}
$$

In Corollary 2, we see that the global variance  $\sigma_g^2$  appears in the  $\mathcal{O}\left(\frac{1}{T}\right)$  term, which says that it asymptotically has no impact on the convergence. This matches the result of momentum SGD (Yu et al., 2019a). When  $T \geq \mathcal{O}(n^3)$  is sufficiently large, the third term in (2) vanishes, and the convergence rate becomes  $\mathcal{O}\left(\frac{1}{\sqrt{nT}}\right)$ . Therefore, to reach a  $\mathcal{O}(\delta)$  stationary point, one worker ( $n = 1$ ) needs  $T = \mathcal{O}\left(\frac{1}{\delta^2}\right)$  iterations, while distributed training with  $n$  workers requires only  $T = \mathcal{O}\left(\frac{1}{N\delta^2}\right)$  iterations, which is  $n$  times faster than single machine training. That is, COMP-AMS has a linear speedup in terms of the number of the local workers. Such acceleration effect has also been reported for compressed SGD (Zheng et al., 2019; Jiang & Agrawal, 2018) and momentum SGD (Yu et al., 2019a) with error feedback.

# 5 EXPERIMENTS

In this section, we provide numerical results on several real-world datasets. Our main objective is to validate the theoretical results, and demonstrate that the proposed COMP-AMS can approach the learning performance of full-precision AMSGrad with significantly reduced communication costs.

# 5.1 DATASETS AND MODELS

Our experiments are conducted on various image and text datasets. The MNIST (LeCun et al., 1998) contains 60000 training samples of  $28 \times 18$  gray-scale hand-written digits from 10 classes, and 10000 test samples. We train MNIST with a Convolutional Neural Network (CNN), which has two convolutional layers followed by two fully connected layers with ReLu activation. Dropout is applied after the max-pooled convolutional layer with rate 0.5. The CIFAR-10 dataset (Krizhevsky et al., 2009) consists of  $50000 \times 32$  RGB natural images from 10 classes for training and 10000 images for testing, which is trained by LeNet-5 (LeCun et al., 1998). Moreover, we also implement ResNet-18 (He et al., 2016) on this dataset. The IMDB movie review (Maas et al., 2011) is a popular binary classification dataset for sentiment analysis. Each movie review is tokenized by top-2000 most frequently appeared words and transformed into integer vectors, which is of maximal length 500. We train an Long-Short Term Memory (LSTM) network with a 32-dimensional embedding layer and 64 LSTM cells, followed by two fully connected layers before output. Cross-entropy loss is used for all the tasks. In each training iteration, data sample are randomly assigned to the workers.

# 5.2 GENERAL EVALUATION AND COMMUNICATION EFFICIENCY

We compare COMP-AMS with full-precision distributed AMSGrad, QAdam (Chen et al., 2020) and 1BitAdam (Tang et al., 2021). For COMP-AMS, Top- $k$  picks top  $1\%$  gradient coordinates (i.e., sparsity 0.01). QAdam and 1BitAdam both use 1-bit quantization to achieve high compression. For MNIST and CIFAR-10, the local batch size on each worker is set to be 32. For IMDB, the local batch size is 16. The hyper-parameters in COMP-AMS are set as default  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.999$  and  $\epsilon = 10^{-8}$ , which are also used for QAdam and 1BitAdam. For 1BitAdam, the epochs for warm-up training is set to be 1/20 of the total epochs. For all methods, we tune the initial learning rate over a fine grid and report the best results averaged over three independent runs. Our experiments are performed on a GPU cluster with NVIDIA Tesla V100 cards.

The training loss and test accuracy on MNIST + CNN, CIFAR-10 + LeNet and IMDB + LSTM are reported in Figure 1. On CIFAR-10, we deploy a decreasing learning rate schedule which is divided by 10 at the 40-th and 80-th epoch, respectively. We observe:

- On MNIST, all the methods can approach the training loss and test accuracy of full-precision AMSGrad. The 1BitAdam seems slightly better, but the gap is very small. On CIFAR-10, COMP-AMS with Block-Sign performs the best and matches AMSGrad in terms of test accuracy.  
- On IMDB, COMP-AMS with Top- $k$  has both the fastest convergence and best generalization. This is because the IMDB text data is sparse (with many padded zeros), where Top- $k$  is expected to work better than sign. The 1BitAdam converges slowly. We believe one possible reason is that 1BitAdam is quite sensitive to the quality of the warm-up training. For sparse text data, the estimation of second moment  $v$  is more unstable, making the strategy of freezing  $v$  by warm-up less effective.

![](images/0572a63cd468d869a586369238d4afce7e9a4d0287e124f26d45d9974655557d.jpg)

![](images/e89f06b32de77df1236f92d544c400baa10e13f6c26b4cf06ac46eccbe335dda.jpg)

![](images/ede4a64906098378101b3924d41aa46af3e5ee13af49cbca30967f5577674f99.jpg)

![](images/a23e73b395b4ade31454389f81e70d54083e65fcaecbd4a602d7527fe514c98a.jpg)  
Figure 1: Training loss and test accuracy vs. epochs, on MNIST + CNN, CIFAR-10 + LeNet and IMDB + LSTM with  $n = 16$  local workers.

![](images/feeb791cea726ce4e6f93ef46051b0a4a99375e379d26af1a9542d24c3ed7db4.jpg)

![](images/c0aeb1ec457d3dad52b3a704c019185ed71299eb9dd1af7fb908ae48758e00df.jpg)

Communication Efficiency. In Figure 2, we plot the training loss and test accuracy against the number of bits transmitted to the central server during the distributed training process, where we assume the full-precision gradient is represented using 32 bits. As we can see, COMP-AMS-Top $k$  achieves around  $100\mathrm{x}$  communication reduction, to attain similar accuracy as the full-precision AMSGrad. The saving of Block-Sign is around  $30\mathrm{x}$ , but it gives slightly higher accuracy than Top $k$  on MNIST and CIFAR-10. In all cases, COMP-AMS can substantially reduce the communication cost compared with full-precision distributed AMSGrad, without losing accuracy. Due to the space limitation, we provide more results on ResNet-18 in Appendix A.

![](images/6c631f9d98a165d00c797e1b330f6f0680c55eb52554f50866c269dfdb518b0b.jpg)

![](images/0fce7c2b6cdca816f3a8befc189c0bc63f5bfe00cf0be2431a568b4403565fd2.jpg)

![](images/1cf1a9466b296d571a7fe720cd98a9c66942ddc8ee1135c63693008d51a9c85e.jpg)

![](images/da5e902cd4712dd84b1bdabf001c4afb3b2e766d27793b457caa879b58ba8d92.jpg)  
Figure 2: Train loss and Test accuracy vs. No. bits transmitted, on MNIST + CNN, CIFAR-10 + LeNet and IMDB + LSTM with  $n = 16$  local workers.

![](images/b2f25ee8e0885ef7afc34f1b30a9896ccf230aaf4ffe201ffda140f58b884fef.jpg)

![](images/7ad927453b9560ac313ecc360690df8ac1fd863449e0538b55a46f4108e9c74d.jpg)

# 5.3 LINEAR SPEEDUP OF COMP-AMS

![](images/cf91d4d8ffc41acfb7bf72e96d8dab1c08dd4fb1908549ea668028d722a6cae3.jpg)  
Figure 3: The linear speedup of COMP-AMS with varying  $n$ . Left: MNIST with Block-Sign compressor on CNN. Right: CIFAR-10 with Top- $k$ -0.01 compression on LeNet.

![](images/58002a20eb35934a60913a9f56414511d76572f00372afe4bfd467dc4a802f03.jpg)

Corollary 2 reveals the linear speedup of COMP-AMS in distributed training. In Figure 3, we present the training loss on MNIST and CIFAR-10 against the number of iterations, with varying number of workers  $n$ . We use COMP-AMS with Block-Sign on MNIST, and Top- $k$ -0.01 on CIFAR. As suggested by the theory, we use  $5 \times 10^{-4} \sqrt{n}$  as the learning rate. From Figure 3, we see the number of iterations to achieve a certain loss exhibits a strong linear relationship with  $n$ —it decreases by half whenever we double  $n$ , which justifies the linear speedup of COMP-AMS.

# 5.4 DISCUSSION

We provide a brief summary of our empirical observations. The proposed COMP-AMS is able to match the learning performance of full-gradient AMSGrad in all the experiments. In particular, for data/model involving some sparsity structure, COMP-AMS with the Top- $k$  compressor could be more effective. Also, our results reveal that 1BitAdam with second moment pre-conditioning might have unstable performance on some tasks in practice.

We would like to emphasize that, the primary goal of the experiments is to show that COMP-AMS is able to match the performance of full-precision AMSGrad, but not to show that COMP-AMS is "better than" other algorithms. This is because, different methods use different underlying optimization algorithms (e.g., AMSGrad, Adam, momentum SGD). Comparing COMP-AMS with other distributed training methods would be largely affected by the comparison among these optimization protocols, which typically depends on the specific data and model in practice. Our results say that: whenever one wants to use AMSGrad to train a deep neural network, he/she can simply employ the distributed COMP-AMS scheme to gain a linear speedup in training time with learning performance as good as the full-precision training, taking little communication cost and memory consumption.

# 6 CONCLUSION

In this paper, we study the simple, convenient, yet unexplored gradient averaging strategy for distributed adaptive optimization called COMP-AMS. Top- $k$  and Block-Sign compressor are incorporated for communication efficiency, whose biases are compensated by the error feedback strategy. We develop the convergence rate of COMP-AMS, and show that same as the case of SGD, for AMSGrad, compressed gradient averaging with error feedback matches the convergence of full-gradient AMSGrad, and linear speedup can be obtained in the distributed training. Numerical experiments are conducted to justify the theoretical findings, and demonstrate that COMP-AMS provides comparable performance with other distributed adaptive methods, and achieves similar accuracy as full-precision AMSGrad with significantly reduced communication overhead. Given the simple architecture and hardware (memory) efficiency, we expect COMP-AMS to serve as a useful distributed adaptive optimization framework in practice.

# REFERENCES

Naman Agarwal, Ananda Theertha Suresh, Felix X. Yu, Sanjiv Kumar, and Brendan McMahan. cpsgd: Communication-efficient and differentially-private distributed SGD. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montreal, Canada, pp. 7575-7586, 2018.  
Ahmad Ajalloeian and Sebastian U Stich. Analysis of sgd with biased gradient estimators. arXiv preprint arXiv:2008.00051, 2020.  
Alham Fikri Aji and Kenneth Heafield. Sparse communication for distributed gradient descent. arXiv preprint arXiv:1704.05021, 2017.  
Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communication-efficient sgd via gradient quantization and encoding. In Advances in Neural Information Processing Systems, pp. 1709-1720, 2017.  
Dan Alistarh, Torsten Hoefler, Mikael Johansson, Sarit Khirirat, Nikola Konstantinov, and Cédric Renggli. The convergence of sparsified gradient methods. arXiv preprint arXiv:1809.10505, 2018.  
Debraj Basu, Deepesh Data, Can Karakus, and Suhas N. Diggavi. Qsparse-local-sgd: Distributed SGD with quantization, sparsification and local computations. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 14668-14679, 2019.  
Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. *signsgd: Compressed optimisation for non-convex problems*. In International Conference on Machine Learning, pp. 560–569. PMLR, 2018.  
Jeremy Bernstein, Jiawei Zhao, Kamyar Azizzadenesheli, and Anima Anandkumar. signsgd with majority vote is communication efficient and fault tolerant. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.  
Aleksandr Beznosikov, Samuel Horváth, Peter Richtárik, and Mher Safaryan. On biased compression for distributed learning. CoRR, abs/2002.12410, 2020.  
Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, Jonathan Eckstein, et al. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning, 3(1):1-122, 2011.  
Ken Chang, Niranjan Balachandar, Carson K. Lam, Darwin Yi, James M. Brown, Andrew Beers, Bruce R. Rosen, Daniel L. Rubin, and Jayashree Kalpathy-Cramer. Distributed deep learning networks among institutions for medical imaging. J. Am. Medical Informatics Assoc., 25(8): 945-954, 2018.  
Moses Charikar, Kevin C. Chen, and Martin Farach-Colton. Finding frequent items in data streams. In Automata, Languages and Programming, 29th International Colloquium, ICALP 2002, Malaga, Spain, July 8-13, 2002, Proceedings, volume 2380 of Lecture Notes in Computer Science, pp. 693-703. Springer, 2002.  
Congliang Chen, Li Shen, Haozhi Huang, Qi Wu, and Wei Liu. Quantized adam with error feedback. arXiv preprint arXiv:2004.14180, 2020.  
Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of A class of adam-type algorithms for non-convex optimization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.  
Trishul Chilimbi, Yutaka Suzue, Johnson Apacible, and Karthik Kalyanaraman. Project adam: Building an efficient and scalable deep learning training system. In Symposium on Operating Systems Design and Implementation, pp. 571-582, 2014.

Dami Choi, Christopher J. Shallue, Zachary Nado, Jaehoon Lee, Chris J. Maddison, and George E. Dahl. On empirical comparisons of optimizers for deep learning. CoRR, abs/1910.05446, 2019.  
Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems, Boston, MA, USA, September 15-19, 2016, pp. 191-198. ACM, 2016.  
Christopher De Sa, Matthew Feldman, Christopher Ré, and Kunle Olukotun. Understanding and optimizing asynchronous low-precision stochastic gradient descent. In Proceedings of the 44th Annual International Symposium on Computer Architecture, pp. 561-574, 2017.  
Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V. Le, Mark Z. Mao, Marc'Aurelio Ranzato, Andrew W. Senior, Paul A. Tucker, Ke Yang, and Andrew Y. Ng. Large scale distributed deep networks. In Advances in Neural Information Processing Systems 25: 26th Annual Conference on Neural Information Processing Systems 2012. Proceedings of a meeting held December 3-6, 2012, Lake Tahoe, Nevada, United States, pp. 1232-1240, 2012.  
Tim Dettmers. 8-bit approximations for parallelism in deep learning. In *Yoshua Bengio and Yann LeCun (eds.), 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016.*  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pp. 4171-4186. Association for Computational Linguistics, 2019.  
John C. Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. In *COLT 2010 - The 23rd Conference on Learning Theory*, Haifa, Israel, June 27-29, 2010, pp. 257-269, 2010.  
John C Duchi, Alekh Agarwal, and Martin J Wainwright. Dual averaging for distributed optimization: Convergence analysis and network scaling. IEEE Transactions on Automatic control, 57(3): 592-606, 2011.  
Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341-2368, 2013.  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada, pp. 2672-2680, 2014.  
Priya Goyal, Piotr Dollár, Ross B. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: training image in 1 hour. CoRR, abs/1706.02677, 2017.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey E. Hinton. Speech recognition with deep recurrent neural networks. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2013, Vancouver, BC, Canada, May 26-31, 2013, pp. 6645-6649. IEEE, 2013.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pp. 770-778. IEEE Computer Society, 2016.  
Mingyi Hong, Davood Hajinezhad, and Ming-Min Zhao. Prox-pda: The proximal primal-dual algorithm for fast distributed nonconvex optimization and learning over networks. In International Conference on Machine Learning, pp. 1529-1538, 2017.  
Nikita Ivkin, Daniel Rothchild, Enayat Ullah, Vladimir Braverman, Ion Stoica, and Raman Arora. Communication-efficient distributed SGD with sketching. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 13144-13154, 2019.

Jiawei Jiang, Fangcheng Fu, Tong Yang, and Bin Cui. Sketchml: Accelerating distributed machine learning with data sketches. In Proceedings of the 2018 International Conference on Management of Data, SIGMOD Conference 2018, Houston, TX, USA, June 10-15, 2018, pp. 1269-1284. ACM, 2018.  
Peng Jiang and Gagan Agrawal. A linear speedup analysis of distributed deep learning with sparse and quantized communication. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 2530-2541, 2018.  
Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian U Stich, and Martin Jaggi. Error feedback fixes signsgd and other gradient compression schemes. arXiv preprint arXiv:1901.09847, 2019.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Anastasia Koloskova, Sebastian U Stich, and Martin Jaggi. Decentralized stochastic optimization and gossip algorithms with compressed communication. In International Conference on Machine Learning, pp. 3478-3487, 2019.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Yujun Lin, Song Han, Huizi Mao, Yu Wang, and Bill Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018.  
Songtao Lu, Xinwei Zhang, Haoran Sun, and Mingyi Hong. Gnsd: A gradient-tracking based nonconvex stochastic algorithm for decentralized optimization. In 2019 IEEE Data Science Workshop (DSW), pp. 315-321, 2019.  
Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies, pp. 142-150, 2011.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics, pp. 1273-1282. PMLR, 2017.  
Hiroaki Mikami, Hisahiro Suganuma, Yoshiki Tanaka, Yuichi Kageyama, et al. Massively distributed sgd: Imagenet/resnet-50 training in a flash. arXiv preprint arXiv:1811.05233, 2018.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller. Playing atari with deep reinforcement learning. CoRR, abs/1312.5602, 2013.  
Parvin Nazari, Davoud Ataee Tarzanagh, and George Michailidis. Dadam: A consensus-based distributed adaptive gradient method for online optimization. arXiv preprint arXiv:1901.09109, 2019.  
Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. IEEE Transactions on Automatic Control, 54(1):48, 2009.  
Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization, 19(4):1574-1609, 2009.  
Sashank J Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In International Conference on Learning Representations, 2018.

Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns. In INTERSPEECH 2014, 15th Annual Conference of the International Speech Communication Association, Singapore, September 14-18, 2014, pp. 1058-1062. ISCA, 2014.  
Zebang Shen, Aryan Mokhtari, Tengfei Zhou, Peilin Zhao, and Hui Qian. Towards more efficient stochastic decentralized learning: Faster convergence and sparse communication. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 4631-4640. PMLR, 2018.  
Shaohuai Shi, Kaiyong Zhao, Qiang Wang, Zhenheng Tang, and Xiaowen Chu. A convergence analysis of distributed sgd with communication-efficient gradient sparsification. In *IJCAI*, pp. 3411-3417, 2019.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy P. Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nat., 550(7676):354-359, 2017.  
Sebastian U. Stich and Sai Praneeth Karimireddy. The error-feedback framework: Better rates for SGD with delayed gradients and compressed communication. CoRR, abs/1909.05350, 2019. URL http://arxiv.org/abs/1909.05350.  
Sebastian U Stich, Jean-Baptiste Cordonnier, and Martin Jaggi. Sparsified sgd with memory. In Advances in Neural Information Processing Systems, pp. 4447-4458, 2018.  
Hanlin Tang, Shaoduo Gan, Ammar Ahmad Awan, Samyam Rajbhandari, Conglong Li, Xiangru Lian, Ji Liu, Ce Zhang, and Yuxiong He. 1-bit adam: Communication efficient large-scale training with adam's convergence speed. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 10118-10129. PMLR, 2021.  
Athanasios Voulodimos, Nikolaos Doulamis, Anastasios D. Doulamis, and Eftychios Protopapadakis. Deep learning for computer vision: A brief review. Comput. Intell. Neurosci., 2018: 7068349:1-7068349:13, 2018.  
Jun-Kun Wang, Xiaoyun Li, Belhal Karimi, and Ping Li. An optimistic acceleration of amsgrad for nonconvex optimization. arXiv preprint arXiv:1903.01435, 2019.  
Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsification for communication-efficient distributed optimization. In Advances in Neural Information Processing Systems, pp. 1299-1309, 2018.  
Jian Wei, Jianhua He, Kai Chen, Yi Zhou, and Zuoyin Tang. Collaborative filtering and deep learning based recommendation system for cold start items. Expert Systems with Applications, 69:29-39, 2017.  
Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. arXiv preprint arXiv:1705.07878, 2017.  
Jiaxiang Wu, Weidong Huang, Junzhou Huang, and Tong Zhang. Error compensated quantized SGD and its applications to large-scale distributed optimization. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 5321-5329. PMLR, 2018.  
Guandao Yang, Tianyi Zhang, Polina Kirichenko, Junwen Bai, Andrew Gordon Wilson, and Chris De Sa. Swalp: Stochastic weight averaging in low precision training. In International Conference on Machine Learning, pp. 7015-7024. PMLR, 2019.

Yang You, Jing Li, Sashank J. Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training BERT in 76 minutes. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.  
Tom Young, Devamanyu Hazarika, Soujanya Poria, and Erik Cambria. Recent trends in deep learning based natural language processing [review article]. IEEE Comput. Intell. Mag., 13(3):55-75, 2018.  
Hao Yu, Rong Jin, and Sen Yang. On the linear speedup analysis of communication efficient momentum SGD for distributed non-convex optimization. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 7184-7193. PMLR, 2019a.  
Yue Yu, Jiaxiang Wu, and Junzhou Huang. Exploring fast and communication-efficient algorithms in large-scale distributed networks. In The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan, volume 89 of Proceedings of Machine Learning Research, pp. 674-683. PMLR, 2019b.  
Matthew D. Zeiler. ADADELTA: an adaptive learning rate method. CoRR, abs/1212.5701, 2012.  
Hantian Zhang, Jerry Li, Kaan Kara, Dan Alistarh, Ji Liu, and Ce Zhang. Zipml: Training linear models with end-to-end low precision, and a little bit of deep learning. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pp. 4035-4043. PMLR, 2017.  
Lei Zhang, Shuai Wang, and Bing Liu. Deep learning for sentiment analysis: A survey. Wiley Interdiscip. Rev. Data Min. Knowl. Discov., 8(4), 2018.  
Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q. Weinberger, and Yoav Artzi. Revisiting few-sample BERT fine-tuning. CoRR, abs/2006.05987, 2020.  
Weijie Zhao, Deping Xie, Ronglai Jia, Yulei Qian, Ruiquan Ding, Mingming Sun, and Ping Li. Distributed hierarchical GPU parameter server for massive scale deep learning ads systems. In Proceedings of Machine Learning and Systems 2020, MLSys 2020, Austin, TX, USA, March 2-4, 2020. mlsys.org, 2020.  
Shuai Zheng, Ziyue Huang, and James T. Kwok. Communication-efficient distributed blockwise momentum SGD with error-feedback. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 11446-11456, 2019.  
Dongruo Zhou, Yiqi Tang, Ziyan Yang, Yuan Cao, and Quanquan Gu. On the convergence of adaptive gradient methods for nonconvex optimization. CoRR, abs/1808.05671, 2018.
