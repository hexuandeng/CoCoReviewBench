# SCALABLE ONE-PASS OPTIMISATION OF HIGH-DIMENSIONAL WEIGHT-UPDATE HYPERPARAMETERS BY IMPLICIT DIFFERENTIATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Machine learning training methods depend plenitfully and intricately on hyperparameters, motivating automated strategies for their optimisation. Many existing algorithms restart training for each new hyperparameter choice, at considerable computational cost. Some hypergradient-based one-pass methods exist, but these either cannot be applied to arbitrary optimiser hyperparameters (such as learning rates and momenta) or take several times longer to train than their base models. We extend these existing methods to develop an approximate hypergradient-based hyperparameter optimiser which is applicable to any continuous hyperparameter appearing in a differentiable model weight update, yet requires only one training episode, with no restarts. We also provide a motivating argument for convergence to the true hypergradient, and perform tractable gradient-based optimisation of independent learning rates for each model parameter. Our method performs competitively from varied random hyperparameter initialisations on several UCI datasets and Fashion-MNIST (using a one-layer MLP), Penn Treebank (using an LSTM) and CIFAR-10 (using a ResNet-18), in time only  $2 - 3\mathrm{x}$  greater than vanilla training.

# 1 INTRODUCTION

Many machine learning methods are governed by hyperparameters: quantities which, unlike model parameters or weights, control the training process itself (e.g. optimiser settings, dropout probabilities and dataset configurations). As suitable hyperparameter selection is crucial to system performance (e.g. Kohavi & John (1995)), it is a pillar of efforts to automate machine learning (Hutter et al., 2018, Chapter 1), spawning several hyperparameter optimisation (HPO) algorithms (e.g. Bergstra & Bengio (2012); Snoek et al. (2012; 2015); Falkner et al. (2018)). However, HPO is computationally intensive and random search is an unexpectedly strong baseline (but beatable; Turner et al. (2021)); beyond random or grid searches, HPO is relatively underused in research (Bouthillier & Varoquaux, 2020).

Recently, Lorraine et al. (2019) used gradient-based updates to adjust hyperparameters during training, displaying impressive optimisation performance and scalability to high-dimensional hyperparameters. Despite their computational efficiency (since updates occur before final training performance is known), Lorraine et al.'s algorithm only applies to hyperparameters on which the loss function depends explicitly (such as  $\ell_2$  regularisation), notably excluding optimiser hyperparameters.

Our work extends Lorraine et al.'s algorithm to support arbitrary continuous inputs to a differentiable weight update formula, including learning rates and momentum factors. We demonstrate our algorithm handles a range of hyperparameter initialisations and datasets, improving test loss after a single training episode ('one pass'). Relaxing differentiation-through-optimisation (Domke, 2012) and hypergradient descent's (Baydin et al., 2018) exactness allows us to improve computational and memory efficiency. Our scalable one-pass method improves performance from arbitrary hyperparameter initialisations, and could be augmented with a further search over those initialisations if desired.

# 2 WEIGHT-UPDATE HYPERPARAMETER TUNING

In this section, we develop our method. Expanded derivations and a summary of differences from Lorraine et al. (2019) are given in Appendix C.

![](images/3eafc311e86e68e190c7ac6295f39d4c84567008db196370cb498b6862e9a778.jpg)  
(a) Training Space

![](images/1c31de39740f44711a1e9df95cf331e5da0c4156fde6b8af8e3bf42d456ac700.jpg)  
(b) Validation Space

![](images/4909f0ac56577be35ff9eea17bb727c9020f8efe1b87b0810d842b8d829a4b80.jpg)  
Figure 1: Summary of our derivation (Section 2): sample online hypergradient descent using exact hypergradients from the implicit function theorem (IFT)  $(\longrightarrow)$ , with our method's approximate hypergradients  $(\longrightarrow)$  superimposed. We target optimal validation loss  $(\bullet)$ , adjusting weights w based on the training loss. Classical weight updates (for fixed hyperparameters  $\lambda$ ) converge  $(\longrightarrow, \longrightarrow)$  to the best-response line  $\mathbf{w}^{*}(\lambda)$ $(\longrightarrow)$ ; the IFT gives hyperparameter updates  $(\longrightarrow)$  leading to a minimum of validation loss along  $\mathbf{w}^{*}(\lambda)$ . Our approximate hyperparameter updates  $(\longrightarrow)$  differ in magnitude from these exact updates, but still give useful guidance.

# 2.1 IMPLICIT FUNCTION THEOREM IN BILEVEL OPTIMISATION

Consider some model with learnable parameters  $\mathbf{w}$ , training loss  $\mathcal{L}_T$ , optimisation hyperparameters  $\lambda$  and validation loss  $\mathcal{L}_V$ . We use  $\mathbf{w}^*$ ,  $\lambda^*$  to represent the optimal values of these quantities, found by solving the following bilevel optimisation problem:

$$
\mathbf {\Lambda} ^ {(a)} \lambda^ {*} = \underset {\boldsymbol {\lambda}} {\arg \min } \mathcal {L} _ {V} (\boldsymbol {\lambda}, \mathbf {w} ^ {*} (\boldsymbol {\lambda})) \text {, s u c h t h a t} \quad \mathbf {\Lambda} ^ {(b)} \mathbf {w} ^ {*} (\boldsymbol {\lambda}) = \underset {\mathbf {w}} {\arg \min } \mathcal {L} _ {T} (\boldsymbol {\lambda}, \mathbf {w}). \tag {1}
$$

The optimal model parameters  $\mathbf{w}^*$  may vary with  $\lambda$ , making  $\mathcal{L}_V$  an implicit function of  $\lambda$  alone.

We approach the outer optimisation (1a) similarly to Lorraine et al. (2019) and Majumder et al. (2019), using the hypergradient: the total derivative of  $\mathcal{L}_V$  with respect to the hyperparameters  $\lambda$ . One strategy for solving (1) is therefore to alternate between updating  $\mathbf{w}$  for several steps using  $\frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$  and updating  $\lambda$  using the hypergradient, as shown in Figure 1 (by  $\longrightarrow$  and  $\longrightarrow$ ).

Carefully distinguishing the total differential  $\mathrm{d}\lambda$  and the partial differential  $\partial \lambda$ , we have

$$
\frac {\mathrm {d} \mathcal {L} _ {V}}{\mathrm {d} \boldsymbol {\lambda}} = \frac {\partial \mathcal {L} _ {V}}{\partial \boldsymbol {\lambda}} + \frac {\partial \mathcal {L} _ {V}}{\partial \mathbf {w} ^ {*}} \frac {\partial \mathbf {w} ^ {*}}{\partial \boldsymbol {\lambda}}. \tag {2}
$$

While derivatives of  $\mathcal{L}_V$  are easily computed for typical loss functions, the final derivative of the optimal model parameters  $(\frac{\partial\mathbf{w}^*}{\partial\lambda})$  presents some difficulty. Letting square brackets indicate the evaluation of the interior at the subscripted values, we may rewrite  $\frac{\partial\mathbf{w}^*}{\partial\lambda}$  as follows:

Theorem 1 (Cauchy's Implicit Function Theorem (IFT)) Suppose for some  $\lambda'$  and  $\mathbf{w}'$  that  $\left[\frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}\right]_{\lambda',\mathbf{w}'} = \mathbf{0}$ . If  $\frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$  is a continuously differentiable function with invertible Jacobian, then there exists a function  $\mathbf{w}^*(\lambda)$  over an open subset of hyperparameter space, such that  $\lambda'$  lies in the open subset,  $\left[\frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}\right]_{\lambda,\mathbf{w}^*(\lambda)} = \mathbf{0}$  and

$$
\frac {\partial \mathbf {w} ^ {*}}{\partial \boldsymbol {\lambda}} = - \left(\frac {\partial^ {2} \mathcal {L} _ {T}}{\partial \mathbf {w} \partial \mathbf {w} ^ {\intercal}}\right) ^ {- 1} \frac {\partial^ {2} \mathcal {L} _ {T}}{\partial \mathbf {w} \partial \boldsymbol {\lambda} ^ {\intercal}}. \tag {3}
$$

$\mathbf{w}^{*}(\lambda)$  is called the best response of  $\mathbf{w}$  to  $\lambda$  (Figure 1). While (3) suggests a route to computing  $\frac{\partial \mathbf{w}^{*}}{\partial \lambda}$ , inverting a potentially high-dimensional Hessian in  $\mathbf{w}$  is not computationally tractable.

# 2.2 APPROXIMATE BEST-RESPONSE DERIVATIVE

To develop and justify a computationally tractable approximation to (3), we mirror the strategy of Lorraine et al. (2019). Consider the broad class of weight optimisers with updates of the form

$$
\mathbf {w} _ {i} (\boldsymbol {\lambda}) = \mathbf {w} _ {i - 1} (\boldsymbol {\lambda}) - \mathbf {u} (\boldsymbol {\lambda}, \mathbf {w} _ {i - 1} (\boldsymbol {\lambda})) \tag {4}
$$

for some arbitrary differentiable function  $\mathbf{u}$ , with  $i$  indexing each update iteration. We deviate here from the approach of Lorraine et al. (2019) by admitting general functions  $\mathbf{u}(\boldsymbol{\lambda},\mathbf{w})$ , rather than assuming the particular choice  $\mathbf{u}_{\mathrm{SGD}} = \eta \frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$ . In particular, this allows  $\boldsymbol{\lambda}$  to include optimiser hyperparameters. Differentiating (4) and unrolling the recursion gives

$$
\left[ \frac {\partial \mathbf {w} _ {i}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda} ^ {\prime}} = - \sum_ {0 \leq j <   i} \left(\prod_ {k \leq j} \left[ \mathbf {I} - \frac {\partial \mathbf {u}}{\partial \mathbf {w}} \right] _ {\boldsymbol {\lambda} ^ {\prime}, \mathbf {w} _ {i - 1 - k}}\right) \left[ \frac {\partial \mathbf {u}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda} ^ {\prime}, \mathbf {w} _ {i - 1 - j}}, \tag {5}
$$

where  $j$  indexes our steps back through time from  $\mathbf{w}_{i-1}$  to  $\mathbf{w}_0$ , and all  $\mathbf{w}$  depend on the current hyperparameters  $\lambda'$  — see Appendix C.2 for the full derivation. Now, we follow Lorraine et al. and assume  $\mathbf{w}_0, \ldots, \mathbf{w}_{i-1}$  to be equal to  $\mathbf{w}_i$ . With this, we simplify the product in (5) to a  $j$ th power by evaluating all derivatives at  $\mathbf{w}_i$ . This result is then used to approximate  $\frac{\partial \mathbf{w}^*}{\partial \lambda}$  by further assuming that  $\mathbf{w}_i \approx \mathbf{w}^*$ . These two approximations lead to

$$
\left[ \frac {\partial \mathbf {w} _ {i}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda} ^ {\prime}} \approx - \left[ \sum_ {0 \leq j <   i} \left(\mathbf {I} - \frac {\partial \mathbf {u}}{\partial \mathbf {w}}\right) ^ {j} \frac {\partial \mathbf {u}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda} ^ {\prime}, \mathbf {w} _ {i} \left(\boldsymbol {\lambda} ^ {\prime}\right)} \approx \left[ \frac {\partial \mathbf {w} ^ {*}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda} ^ {\prime}}. \tag {6}
$$

We reinterpret  $i$  as a predefined look-back distance, trading off accuracy and computational efficiency.

These approximations actually combine to give  $\mathbf{w}_i = \mathbf{w}^*$  for all  $i$ , which is initially inaccurate, but as training proceeds we would expect it to become more correct. In mitigation, we perform several weight updates prior to each hyperparameter update. This means derivatives in earlier terms of the series of (6) (which are likely the largest, dominant terms) are evaluated at weights closer to  $\mathbf{w}^*$ , therefore making the summation more accurate. In Section 4, we show that the approximations described here result in an algorithm that is both practical and effective.

Our approximate result (6) combines the general weight update of Majumder et al. (2019) with the overall approach and constant-weight assumption of Lorraine et al. (2019). The latter empirically show that an approximation similar to (6) leads to a directionally-accurate approximate hypergradient; we illustrate the approximate updates from our derivations in Figure 1 (by  $\longrightarrow$ ).

# 2.3 CONVERGENCE TO BEST-RESPONSE DERIVATIVE

To justify the approximations in (6), note that the central part of that equation is a truncated Neumann series. Taking the limit  $i\to \infty$  , when such a limit exists, results in the closed form

$$
\left[ \frac {\partial \mathbf {w} ^ {*}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda} ^ {\prime}} \approx - \left[ \left(\frac {\partial \mathbf {u}}{\partial \mathbf {w}}\right) ^ {- 1} \frac {\partial \mathbf {u}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda} ^ {\prime}, \mathbf {w} ^ {*} (\boldsymbol {\lambda} ^ {\prime})}. \tag {7}
$$

This is precisely the result of the IFT (Theorem 1) applied to  $\mathbf{u}$  instead of  $\frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$ ; that is, substituting the simple SGD update  $\mathbf{u}_{\mathrm{SGD}}(\boldsymbol {\lambda},\mathbf{w}) = \eta \frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$  into (7) recovers (3) exactly. Thus, under certain conditions, our approximation (6) converges to the true best-response Jacobian in the limit of infinitely long look-back windows.

# 2.4 HYPERPARAMETER UPDATES

Substituting (6) into (2) yields a tractable approximation for the hypergradient  $\frac{\mathrm{d}\mathcal{L}_V}{\mathrm{d}\lambda}$ , with which we can update hyperparameters by gradient descent. Our implementation in Algorithm 1, Figure 4, closely parallels Lorraine et al.'s algorithm, invoking Jacobian-vector products (Pearlmutter, 1994) during gradient computation for memory efficiency via the grad_outputs argument, which also provides the repeated multiplication for the  $j$ th power in (6). Thus, we retain the  $\mathcal{O}(|\mathbf{w}| + |\lambda|)$  time and memory cost of Lorraine et al. (2019), where  $|\cdot|$  denotes cardinality. The core loop to compute the sum in (6) comes from an algorithm of Liao et al. (2018). Note that Algorithm 1 approximates  $\frac{\mathrm{d}\mathcal{L}_V}{\mathrm{d}\lambda}$  retrospectively by considering only the last weight update rather than any future ones.

Unlike differentiation-through-optimisation (Domke, 2012), Algorithm 1 crucially estimates hypergradients without reference to old model parameters, thanks to the approximate-hypergradient construction of Lorraine et al. (2019) and (6). We thus do not store network weights at multiple time

steps, so gradient-based HPO becomes possible on previously-intractable large-scale problems. In essence, we develop an approximation to online hypergradient descent (Baydin et al., 2018).

Optimiser hyperparameters generally do not affect the optimal weights, suggesting their hypergradients should be zero. However, in practice,  $\mathbf{w}^*$  is better reinterpreted as the approximately optimal weights obtained after a finite training episode. These certainly depend on the optimiser hyperparameters, which govern the convergence of  $\mathbf{w}$ , thus justifying our use of the bilevel framework.

We emphasise training is not reset after each hyperparameter update — we simply continue training from where we left off, using the new hyperparameters. Consequently, Algorithm 1 avoids the time cost of multiple training restarts. While our locally greedy hyperparameter updates threaten a short-horizon bias (Wu et al., 2018), we still realise practical improvements in our experiments.

# 2.5 REINTERPRETATION OF ITERATIVE OPTIMISATION

Originally, we stated the IFT (3) in terms of minima of  $\mathcal{L}_T$  (zeros of  $\frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$ ), and substituting  $\mathbf{u}_{\mathrm{SGD}} = \eta \frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$  into (7) recovers this form of (3). However, in general, (7) recovers the Theorem for zeros of  $\mathbf{u}$ , which are not necessarily minima of the training loss. Despite this, our development can be compared to Lorraine et al. (2019) by expressing  $\mathbf{u}$  as the derivative of an augmented 'pseudo-loss' function. Consider again the simple SGD update  $\mathbf{u}_{\mathrm{SGD}}$ , which provides the weight update rule  $\mathbf{w}_i = \mathbf{w}_{i-1} - \eta \frac{\partial\mathcal{L}_T}{\partial\mathbf{w}}$ . By trivially defining a pseudo-loss  $\overline{\mathcal{L}} = \eta \mathcal{L}_T$ , we may absorb  $\eta$  into a loss-like derivative, yielding  $\mathbf{w}_i = \mathbf{w}_{i-1} - \frac{\partial\overline{\mathcal{L}}}{\partial\mathbf{w}}$ . More generally, we may write  $\overline{\mathcal{L}} = \int \mathbf{u}(\boldsymbol{\lambda},\mathbf{w}) \, \mathrm{d}\mathbf{w}$ .

Expressing the update in this form suggests a reinterpretation of the role of optimiser hyperparameters. Conventionally, our visualisation of gradient descent has the learning rate control the size of steps over some undulating landscape. Instead, we propose fixing a unit step size, with the 'learning rate' scaling the landscape underneath. Similarly, we suppose a 'momentum' could, at every point, locally squash the loss surface in the negative-gradient direction and stretch it in the positive-gradient direction. In aggregate, these transformations straighten out optimisation trajectories and bring local optima closer to the current point. While more complex hyperparameters lack a clear visualisation in this framework, it nevertheless allows a broader class of hyperparameters to 'directly alter the loss function' instead of remaining completely independent, circumventing the problem with optimiser hyperparameters noted by Lorraine et al. (2019). Figure 2 illustrates this argument.

# 3 RELATED WORK

Kohavi & John (1995) first noted different problems respond optimally to different hyperparameters; Hutter et al. (2018, Chapter 1) summarise the resulting hyperparameter optimisation (HPO) field.

Black-box HPO treats the training process as atomic, selecting trial configurations by grid search (coarse or intractable at scale), random search (Bergstra & Bengio (2012); often more efficient) or population-based searches (mutating promising trials). Pure Bayesian Optimisation (Močkus et al., 1978; Snoek et al., 2012) guides the search with a predictive model; many works seek to exploit its sample efficiency (e.g. Swersky et al. (2014a); Snoek et al. (2015); Wang et al. (2016); Kandasamy et al. (2017); Lévesque et al. (2017); Perrone et al. (2019)). However, these methods require each proposed configuration to be fully trained, incurring considerable computational expense. Other techniques infer information during a training run — from learning curves (Provost et al., 1999; Swersky et al., 2014b; Domhan et al., 2015; Chandrashekaran & Lane, 2017; Klein et al., 2017), smaller surrogate problems (Petrak, 2000; van den Bosch, A. et al., 2004; Krueger et al., 2015; Sparks et al., 2015; Thornton et al., 2013; Sabharwal et al., 2016) or intelligent resource allocation (Jamieson & Talwalkar, 2016; Li et al., 2018; Falkner et al., 2018; Bertrand et al., 2017; Wang et al., 2018). Such techniques could be applied on top of our algorithm to improve performance.

In HPO with nested bilevel optimisation, hyperparameters are optimised conditioned on optimal weights, enabling updates during training, with a separate validation set mitigating overfitting risk; this relates to meta-learning (Franceschi et al., 2018). Innovations include differentiable unrolled stochastic gradient descent (SGD) updates (Domke, 2012; Maclaurin et al., 2015; Baydin et al., 2018; Shaban et al., 2019; Majumder et al., 2019), conjugate gradients or hypernetworks (Lorraine & Duvenaud, 2018; Lorraine et al., 2019; MacKay et al., 2019; Fu et al., 2017), solving one level while penalising suboptimality of the other (Mehra & Hamm, 2020), and deploying Cauchy's implicit

![](images/5f6c8b6ba457cc4aa5b15c128d315d8461d821e6122fa25292c9c3cbf0e72c96.jpg)

![](images/a7625381132ba7cb27c13be433670e284cee109bdb27fdd5682ec34101a9b6fe.jpg)  
Figure 2: Reinterpreting the role of learning rate  $\eta$  over a loss function  $\mathcal{L}$ . (a) In the classical setting,  $\eta$  scales the gradient of some fixed  $\mathcal{L}$ . (b) In our setting,  $\eta$  scales  $\mathcal{L}$  to form a 'pseudo-loss'  $\overline{\mathcal{L}}$ , whose gradient is used as-is: our loss function has become dependent on  $\eta$ . The same weight updates are obtained in both (a) and (b).

![](images/f9c3158c0fb75bfb798c8e495f66f98c9cd6c62e65e7d5cdf56b96c9323f0116.jpg)  
Figure 3: Sample hyperparameter trajectories from training UCI Energy under  $\text{Our}^{WD+LR}$  configuration. Background shading gives a non-HPO baseline with hyperparameters fixed at the corresponding initial point; these results are interpolated by a Gaussian process. Note the trajectories are generally attracted to the valley of high performance at learning rates around  $10^{-1}$  and weight decays below  $10^{-2}$ .

function theorem (Larsen et al., 1996; Bengio, 2000; Luketina et al., 2016; Lorraine et al., 2019). Donini et al. (2020) extrapolate training to optimise arbitrary learning rate schedules, extending earlier work on gradient computation (Franceschi et al., 2017), but do not immediately accommodate other hyperparameters. While non-smooth procedures exist (Lopez-Ramos & Beferull-Lozano, 2020), most methods focus on smooth hyperparameters which augment the loss function (e.g. weight regularisation) and work with the augmented loss directly. Consequently, they cannot handle optimiser hyperparameters not represented in the loss function (e.g. learning rates and momentum factors; Lorraine et al. (2019)). Many of these methods compute hyperparameter updates locally (not over the entire training process), which may induce short-horizon bias (Wu et al., 2018), causing myopic convergence to local optima.

Modern developments include theoretical reformulations of bilevel optimisation to improve performance (Liu et al., 2020; Li et al., 2020), optimising distinct hyperparameters for each model parameter (Lorraine et al., 2019; Jie et al., 2020), and computing forward-mode hypergradient averages using more exact techniques than we do (Micaelli & Storkey, 2020). Although these approaches increase computational efficiency and the range of tunable parameters, achieving both benefits at once remains challenging. Our algorithm accommodates a diverse range of differentiable hyperparameters, but retains the efficiency of existing approaches (specifically Lorraine et al. (2019)).

# 4 EXPERIMENTS

We now empirically evaluate our approach, using the hardware and software detailed in Appendix A.1. Our code is available at https://github.com/anonymised.

Throughout, we train models using SGD with weight decay and momentum. We uniformly sample initial learning rates, weight decays and momenta, using logarithmic and sigmoidal transforms (see Appendix B.3), applying each initialisation in the following eight settings:

# Algorithm 1

Scalable One-Pass Optimisation of High-Dimensional Weight-Update Hyperparameters by Implicit Differentiation

while training continues do

for  $t\gets 1$  to  $T$  do  $\triangleright T$  steps of weight updates

$$
\mathbf {w} \leftarrow \mathbf {w} - \mathbf {u} (\boldsymbol {\lambda}, \mathbf {w})
$$

end for

$\mathbf{p} = \mathbf{v} = \left[\frac{\partial\mathcal{L}_V}{\partial\mathbf{w}}\right]_{\mathbf{A},\mathbf{w}}$  ▷Initialise accumulators

for  $j\gets 1$  to  $i$  do  $\triangleright$  Accumulate first summand in (6)

$\mathbf{v}\gets \mathbf{v} - \mathrm{grad}(\mathbf{u}(\pmb {\lambda},\mathbf{w}),\mathbf{w},\mathrm{grad\_outpu s} = \mathbf{v})$

$$
\mathbf {p} \leftarrow \mathbf {p} + \mathbf {v}
$$

end for

$\mathbf{g}_{\mathrm{indirect}} = -\mathrm{grad}(\mathbf{u}(\pmb {\lambda},\mathbf{w}),\pmb {\lambda},\mathrm{grad\_outputs} = \mathbf{p})$

$$
\triangleright \operatorname {N o w} \mathbf {g} _ {\text {i n d i r e c t}} \approx - \left[ \frac {\partial \mathcal {L} _ {V}}{\partial \mathbf {w}} \left(\frac {\partial \mathbf {u}}{\partial \mathbf {w}}\right) ^ {- 1} \frac {\partial \mathbf {u}}{\partial \lambda} \right] _ {\lambda , \mathbf {w}}
$$

$$
\boldsymbol {\lambda} \leftarrow \boldsymbol {\lambda} - \kappa \left(\left[ \frac {\partial \mathcal {L} _ {V}}{\partial \boldsymbol {\lambda}} \right] _ {\boldsymbol {\lambda}, \mathbf {w}} + \mathbf {g} _ {\text {i n d i r e c t}}\right)
$$

For some meta-learning rate  $\kappa$

end while

10812

(a) Baseline: Random  $(\times 1000)$

![](images/9b798dfdc5a4ad9dc43dcb52b22848738691c8879b43801b52b7bfdc6f1fd5f9.jpg)  
Figure 4: Left: Pseudocode for our method. Right: Median final test loss on UCI Energy after 400 hyperparameter updates from random initialisations, for various update intervals  $T$  and look-back distances  $i$  of (a) no hyper-parameter tuning, Random, and (b) our proposed method,  $Ours^{WD + LR + M}$ .  
(b) Our approach:  $Ours^{WD + LR + M}$  (×1000)

Random No HPO; hyperparameters held constant at their initial values, as in random search.

Random  $(\times \mathbf{LR})$  Random with an extra hyperparameter increasing or decreasing the learning rate by multiplication after each update step.

Random (3-batched) Random reprocessed to retain only the best result of every three runs. Allows Random to exceed our methods' computational budgets; imitates population training.

Lorraine Lorraine et al. (2019)'s method optimising weight decay; other hyperparameters constant.

Baydin Baydin et al. (2018)'s method optimising learning rates only; other hyperparameters constant.

Ours $^{\text{WD+LR}}$  Algorithm 1 updating weight decay and learning rate; other hyperparameters constant.

Ours $^{\text{WD+LR+M}}$  Optimising all hyperparameters (adding momentum to  $Ours^{WD+LR}$ )

OursWD+HDLR+M Ours  $^{WD + LR + M}$  with independent learning rates for each model parameter

Diff-through-Opt Optimising all hyperparameters using exact hypergradients (Domke, 2012)

Any unoptimised hyperparameters are fixed at their random initial values. Ideally, we seek resilience to poor initialisations, which realistically arise in unguided hyperparameter selection. Hyperparameters are tuned on the validation set; this is combined with the training set for our Random settings, so each algorithm observes the same data. We use the UCI dataset split sizes of Gal & Ghahramani (2016) and standard  $60\% / 20\% / 20\%$  splits for training/validation/test datasets elsewhere, updating hyperparameters every  $T = 10$  batches with look-back distance  $i = 5$  steps.

Numerical data is normalised to zero-mean, unit-variance. For efficiency, the computational graph is detached after each hyperparameter update, so we never differentiate through hyperparameter updates — essentially, back-propagation and momentum histories are truncated. Our approximate hypergradient is passed to Adam (Kingma & Ba, 2015) with meta-learning rate  $\kappa = 0.05$  and default  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.999$ . While these meta-hyperparameters are not tuned, previous work indicates performance is progressively less sensitive to higher-order hyperparameters (Franceschi et al., 2017; Majumder et al., 2019). As some settings habitually chose unstable high learning rates, we clip these to  $[10^{-10}, 1]$  throughout.

UCI Energy: Proof of Concept First, we broadly illustrate Algorithm 1 on UCI Energy, using a one-layer multi-layer perceptron (MLP) with 50 hidden units and ReLU (Glorot et al., 2011) activation functions, trained under  $Ours^{WD+LR}$  for 4000 full-batch epochs without learning rate clipping (see Appendix B.1 for details). Figure 3 shows the evolution of learning rate and weight decay from a variety of initialisations, overlaid on the performance obtained when all hyperparameters are kept fixed during training. Notice the trajectories are attracted towards the region of lowest test loss, indicating our algorithm is capable of useful learning rate and weight decay adjustment.

Table 1: Final test MSEs after training UCI datasets for 4000 full-batch epochs from each of 200 random hyperparameter initialisations, showing best and bootstrapped average runs. Uncertainties are standard errors; bold values lie in the error bars of the best algorithm.  

<table><tr><td rowspan="2">Method</td><td colspan="4">UCI Energy</td><td colspan="4">UCI Kin8nm (×1000)</td><td colspan="4">UCI Power</td></tr><tr><td colspan="2">Mean</td><td>Median</td><td>Best</td><td colspan="2">Mean</td><td>Median</td><td>Best</td><td colspan="2">Mean</td><td>Median</td><td>Best</td></tr><tr><td>Random</td><td>21</td><td>±2</td><td>7.8</td><td>±0.4</td><td>0.120</td><td>38</td><td>±2</td><td>33</td><td>±3</td><td>6.44</td><td>67</td><td>±6</td></tr><tr><td>Random (×LR)</td><td>31</td><td>±2</td><td>13</td><td>±3</td><td>0.133</td><td>47</td><td>±2</td><td>45</td><td>±5</td><td>6.43</td><td>105</td><td>±8</td></tr><tr><td>Random (3-batched)</td><td>4.5</td><td>±0.6</td><td>4</td><td>±2</td><td>0.141</td><td>18</td><td>±2</td><td>13</td><td>±1</td><td>6.72</td><td>19</td><td>±2</td></tr><tr><td>Lorraine</td><td>22</td><td>±2</td><td>8.2</td><td>±0.5</td><td>0.161</td><td>38</td><td>±2</td><td>34</td><td>±3</td><td>6.39</td><td>67</td><td>±6</td></tr><tr><td>Baydin</td><td>5.5</td><td>±0.2</td><td>6.70</td><td>±0.06</td><td>0.103</td><td>24.5</td><td>±0.8</td><td>27</td><td>±1</td><td>6.52</td><td>17.82</td><td>±0.07</td></tr><tr><td>OursWD+LR</td><td>2.5</td><td>±0.1</td><td>2.2</td><td>±0.2</td><td>0.147</td><td>10</td><td>±1</td><td>7.6</td><td>±0.1</td><td>6.44</td><td>17.33</td><td>±0.01</td></tr><tr><td>OursWD+LR+M</td><td>1.28</td><td>±0.08</td><td>0.9</td><td>±0.2</td><td>0.139</td><td>10</td><td>±1</td><td>6.95</td><td>±0.08</td><td>5.92</td><td>17.4</td><td>±0.2</td></tr><tr><td>OursWD+HDLR+M</td><td>1.9</td><td>±0.2</td><td>1.2</td><td>±0.3</td><td>0.165</td><td>16</td><td>±4</td><td>8.1</td><td>±0.1</td><td>6.29</td><td>18</td><td>±1</td></tr><tr><td>Diff-through-Opt</td><td>1.10</td><td>±0.09</td><td>0.48</td><td>±0.09</td><td>0.123</td><td>8.0</td><td>±0.2</td><td>7.17</td><td>±0.06</td><td>6.09</td><td>17.17</td><td>±0.02</td></tr></table>

# 4.1 UCI DATASETS: ESTABLISHING ROBUSTNESS

Next, we consider the same 50-hidden-unit MLP applied to eight standard UCI datasets (Boston Housing, Concrete, Energy, Kin8nm, Naval, Power, Wine, Yacht) in a fashion analogous to Gal & Ghahramani (2016) (we do not consider dropout or Bayesian formulations). We train 200 hyperparameter initialisations for 4000 full-batch epochs. Table 1 shows results for three datasets, with complete loss evolution and distribution plots in Appendix B.4. Some extreme hyperparameters caused numerical instability and NaN final losses, which our averages ignore. NaN results are not problematic: they indicate extremely poor initialisations, which should be easier for the user to rectify than merely mediocre hyperparameters. Error bars are based on 1000 sets of bootstrap samples.

Given the sparse distribution of strong hyperparameter initializations, random sampling unsurprisingly achieves generally poor averages. Random  $(\times LR)$  applies a learning rate multiplier, uniformly chosen in [0.995, 1.001]; these limits allow extreme initial learning rates to revert to more typical values after 4000 multiplications. This setting's poor performance shows naive learning rate schedules cannot match our algorithms' average improvements. A stronger baseline is Random (3-batched), which harshly simulates the greater computational cost of our methods by retaining only the best of every three Random trials (according to validation loss). This setting comes close to, but cannot robustly beat, our methods — a claim reinforced by final test loss distributions (Figure 9, Appendix B.4).

Lorraine et al. (2019)'s algorithm is surprisingly indistinguishable from Random in our trials, though it must account for three random hyperparameters while varying only one (the weight decay). We surmise that learning rates and momenta are more important hyperparameters to select, and poor choices cannot be overcome by intelligent use of weight decay. Baydin et al. (2018)'s algorithm, however, broadly matches the performance of Random (3-batched), indicating successful intervention in its sole optimisable hyperparameter (learning rate). Variance is also much lower than the preceding algorithms, suggesting greater stability. That said, despite concentrating on a more important hyperparameter, Baydin still suffers from being unable to control every hyperparameter.

Our scalar algorithms  $(Ours^{WD+LR}$  and  $Ours^{WD+LR+M})$  appear generally more robust to these initialisations, with average losses beating Random, Lorraine and Baydin. Figures 8 and 9 (Appendix B.4) clearly distinguish these algorithms from the preceding: we achieve performant results over a wider space of initial hyperparameters. Given it considers more hyperparameters,  $Ours^{WD+LR+M}$  predictably outperforms  $Ours^{WD+LR}$ , although sometimes a slightly higher variance in the former obscures this difference. Unlike pure HPO, the non-Random algorithms in Table 1 vary hyperparameters during training, combining aspects of HPO and schedule learning. They are thus more flexible than conventional, static-hyperparameter techniques, even in high dimensions (Lorraine et al., 2019).

Diff-through-Opt exactly differentiates the current loss with respect to the hyperparameters, over the same  $i = 5$  look-back window and  $T = 10$  update interval. As an exact version of  $Ours^{WD + LR + M}$  (though subject to the same short-horizon bias), it unsurprisingly outperforms the other algorithms (Figures 8 and 9, Appendix B.4). However, our scalar methods' proximity to this exact baseline is reassuring given our much-reduced memory requirements and generally comparable error bars. In these experiments, lengthening Diff-through-Opt's look-back horizon to all 4000 training steps, and repeating those steps for 30 hyperparameter updates, did not improve its performance (Appendix B.7).

Ours $^{WD + HDLR + M}$  theoretically mitigates short-horizon bias (Wu et al., 2018) by adapting appropriately to high- and low-curvature directions. Surprisingly, however, performance is often uncompetitive with scalar methods — training often diverges, causing large and wildly-varying final losses, which

obscure very promising clusters of low-loss runs. In many of these runs, a subset of learning rates become large — presumably those for low-curvature directions. Large scalar learning rates are widely understood to yield high-variance results; it appears a similar phenomenon affects our high-dimensional algorithm. This punctuates a trend in our results: optimising more hyperparameters can lead to better solutions, but also provides more scope for divergent behaviour — possibly because we adapt to optimisation dynamics which suddenly change. We speculate learning rate regularisation would be beneficial, but leave investigation of high-dimensional dynamics to future work.

# 4.2 LARGE-SCALE DATASETS: PRACTICAL SCALABILITY

Fashion-MNIST: HPO in Multi-Layer Perceptrons We train the same single 50-unit hidden layer MLP on 10 epochs of Fashion-MNIST (Xiao et al., 2017), using 50-sample batches. Table 2 and Figure 12a (Appendix B.6) show average test set cross-entropies over 100 initialisations. Clearly-outlying final losses (above  $10^{3}$ ) are set to NaN to stop them dominating our error bars.

Echoing Section 4.1, for arbitrary hyperparameter initialisations, our methods generally converge more robustly to lower losses, even when (as in Figure 12a, Appendix B.6) NaN solutions are included in our statistics. Importantly, we see mini-batches provide sufficient gradient information for our HPO task. Diff-through-Opt's failure to beat our methods is surprising; we suppose noisy approximate gradients may regularise our algorithms, preventing them seeking the short-horizon optimum so directly, thus mitigating short-horizon bias (see our sensitivity study in Section 4.3). Diff-through-Opt with long look-back horizons does not improve performance for equal computation (Appendix B.7). Finally, median and best loss evolution plots are shown in Figures 5b and 5c, the latter including results for a Bayesian Optimisation baseline. For more details, see Appendices B.5 and B.6.

Penn Treebank: HPO in Recurrent Networks Now, we draw inspiration from Lorraine et al. (2019)'s large-scale trials: a 2-layer, 650-unit LSTM (Hochreiter & Schmidhuber, 1997) with learnable embedding, trained on the standard Penn Treebank-3-subset benchmark dataset (Marcus et al., 1999) for 72 epochs. Lorraine et al.'s algorithm performed better without dropout, which we omit. To focus our study, we also omit activation regularisation and predefined learning rate schedules, though we retain training gradient clipping to a Euclidean norm of 0.25. Training considers length-70 subsequences of 40 parallel sequences, using 50 random hyperparameter initialisations. Again, clearly outlying final perplexities (above  $10^{5}$ ) are set to NaN.

Table 2 and Figure 12b (Appendix B.6) show final test perplexities. They reflect our intuition that adjusting progressively more hyperparameters reduces average test losses, continuing the trend we have seen thus far. Highly bimodal final loss distributions for some algorithms cause wide bootstrap-sampled error bars. Learning rate adjustments show particular gains:  $Ours^{WD+LR}$  and  $Ours^{WD+LR+M}$  perform well in less-optimal configurations. NaN runs — previously argued to be unproblematic — cause the largest performance differences between our methods (Figure 12b, Appendix B.6).

CIFAR-10: HPO in Convolutional Networks Finally, to demonstrate scalability, we train a ResNet-18 (He et al., 2016) on CIFAR-10 (Krizhevsky, 2009) for 72 epochs. We use the vanilla normalised dataset (since unbiased data augmentation over both training and validation datasets exhausts our GPU memory) and optimise hyperparameters as before, using 100-image batches.

Table 2 and Figure 12c (Appendix B.6) show our results. Our gains are now more marginal, with this setting apparently robust to its initialisation, though the general ranking of algorithms remains similar,

Table 2: Final test *cross-entropy (†perplexity) on larger datasets. Bold values are the lowest in class.  

<table><tr><td rowspan="2">Method</td><td colspan="4">Fashion-MNIST*</td><td colspan="3">Penn Treebank†</td><td colspan="3">CIFAR-10*</td></tr><tr><td>Mean</td><td>Median</td><td>Best</td><td>Mean</td><td>Median</td><td>Best</td><td>Mean</td><td>Median</td><td>Best</td><td></td></tr><tr><td>Random (× LR)</td><td>0.85 ± 0.07</td><td>0.51 ± 0.05</td><td>0.334</td><td>4700 ± 600</td><td>3000 ± 3000</td><td>170</td><td>1.90 ± 0.05</td><td>1.9 ± 0.1</td><td>0.823</td><td></td></tr><tr><td>Random (3-batched)</td><td>1.62 ± 0.08</td><td>2.0 ± 0.2</td><td>0.446</td><td>7000 ± 600</td><td>9400 ± 700</td><td>138</td><td>20 ± 20</td><td>1.7 ± 0.1</td><td>0.798</td><td></td></tr><tr><td>Lorraine</td><td>0.8 ± 0.1</td><td>0.5 ± 0.1</td><td>0.351</td><td>470 ± 40</td><td>470 ± 80</td><td>170</td><td>1.62 ± 0.07</td><td>1.55 ± 0.08</td><td>0.834</td><td></td></tr><tr><td>Baydin</td><td>0.87 ± 0.07</td><td>0.52 ± 0.05</td><td>0.343</td><td>4700 ± 600</td><td>3000 ± 3000</td><td>170</td><td>1.70 ± 0.06</td><td>1.6 ± 0.1</td><td>0.727</td><td></td></tr><tr><td>OursWD+LR</td><td>0.49 ± 0.07</td><td>0.410 ± 0.002</td><td>0.340</td><td>4200 ± 600</td><td>1000 ± 2000</td><td>147</td><td>1.55 ± 0.04</td><td>1.5 ± 0.1</td><td>0.762</td><td></td></tr><tr><td>OursWD+LR+M</td><td>0.375 ± 0.002</td><td>0.375 ± 0.003</td><td>0.336</td><td>350 ± 10</td><td>360 ± 30</td><td>126</td><td>1.33 ± 0.05</td><td>1.19 ± 0.02</td><td>0.681</td><td></td></tr><tr><td>OursWD+HDLR+M</td><td>0.374 ± 0.002</td><td>0.373 ± 0.003</td><td>0.336</td><td>290 ± 50</td><td>220 ± 20</td><td>103</td><td>1.23 ± 0.05</td><td>1.17 ± 0.02</td><td>0.631</td><td></td></tr><tr><td rowspan="2">Diff-through-Opt</td><td>0.402 ± 0.004</td><td>0.389 ± 0.003</td><td>0.358</td><td>1300 ± 900</td><td>300 ± 600</td><td>190</td><td>1.59 ± 0.07</td><td>1.6 ± 0.1</td><td>0.735</td><td></td></tr><tr><td>0.388 ± 0.001</td><td>0.387 ± 0.002</td><td>0.359</td><td>300 ± 10</td><td>280 ± 20</td><td>114</td><td>1.196 ± 0.008</td><td>1.202 ± 0.006</td><td>0.941</td><td></td></tr></table>

![](images/cef7a8dc4ee4cd8306b5ccd96421106b530f26b46bc0e800562a98ce9061f84f.jpg)

![](images/5362ebb31eb597c1037271675471cabc307abccae412edb813dfcbda7109b9fa.jpg)

![](images/b094f151314d9f401a8aad304a8c263b6c11b7473b35ef194b245ebc9d3e7463.jpg)

![](images/621cbb79dfd1558e0c962bb62b8fa4d0fcd46409791e07862eeb99a2d446ff12.jpg)  
(a) Single-Pass Runtimes  
Figure 5: Illustrations of training a one-layer 50-unit MLP on Fashion-MNIST over 100 random hyperparameter initialisations. We include a Bayesian Optimisation baseline from Appendix B.5 and loss evolution plots from Appendix B.6.  
(b) Median Loss Evolution  
(c) Best Loss Evolution

and we retain useful improvements over our baselines. However, our best final accuracies fall short of state-of-the-art, suggesting a more intricate and clever setting may yield further performance gains.

# 4.3 MISCELLANEOUS STUDIES

Experiment Runtimes For our larger-scale experiments, we illustrate comparative runtimes in Figure 14 (Appendix B.6.1, where they are discussed in detail). Note from our Fashion-MNIST sample (Figure 5a) that all HPO algorithms except  $Ours^{WD + HDLR + M}$  have computational cost similar to naively training two fixed hyperparameter initialisations, despite achieving substantial HPO effect. This compares extremely favourably to HPO methods relying on repeated retraining.

UCI Energy: Sensitivity Study Finally, we consider a range of update intervals  $T$  and look-back distances  $i$  on UCI Energy, performing 400 hyperparameter updates from 100 random initialisations on each, using  $Ours^{WD + LR + M}$ . We plot the median final test losses for each choice of  $T$  and  $i$  to the right of Figure 4, and also the median performance with no HPO (Random), which we outperform in every case. Performance generally improves with larger  $T$  — likely because the total number of weight updates increases with  $T$ , since the number of hyperparameter updates is fixed at 400. Performance also improves with smaller  $i$ . While a larger  $i$  gives more terms in our approximating series (6), these extra terms become more and more biased: they are evaluated at the current weight values instead of at the progressively more different past weight values. We theorise a smaller  $i$  avoids some of this bias, improving performance. More details are given in Appendix B.2.

# 5 CONCLUSION AND FUTURE WORK

We have presented an algorithm for optimising continuous hyperparameters, specifically those appearing in a differentiable weight update, including optimiser hyperparameters. Our method requires only a single training pass, has a motivating true-hypergradient convergence argument, and demonstrates practical benefits at a range of experimental scales without greatly sacrificing training time. We also tackle traditionally-intractable per-parameter learning rate optimisation on non-trivial datasets. However, this setting surprisingly underperformed its scalar counterpart; further work is necessary to understand this result.

As future work, our myopic approach could be extended to longer horizons by incorporating the principles of recent work by Micaelli & Storkey (2020), which presents a promising research direction. We also depend inconveniently on meta-hyperparameters, which are not substantially tuned. Ultimately, we desire systems which are completely independent of human configuration, thus motivating investigations into the removal of these settings.

# SOCIETAL IMPACT / ETHICS STATEMENT

Our work fits into the broad subfield of automatic machine learning (AutoML), which aims to use automation to strip away the tedious work that is necessary to implement practical ML systems. Our method focuses on automating the oft-labelled 'black art' of optimising hyperparameters. This contributes towards the democratisation of ML techniques, which we hope will improve accessibility to non-experts.

However, our method also has some associated risks. For one, developers of unethical machine learning applications (e.g. for mass surveillance, identity theft or automated weaponry) may use our techniques to improve their systems' performance. The heavily metric-dominated nature of our field creates additional concerns — for instance, end-users of our method may not appreciate that optimising naively for training and validation loss alone may result in dangerously poorly-trained or unethical models if the chosen metric, developer's intentions and moral good do not align.

More broadly, users may rely excessively on HPO techniques to optimise their models' performance, which can lead to poor results and inaccurate comparisons if the HPO strategy is imperfect. Further, reducing the need to consider hyperparameter tuning abstracts away an important component of how machine learning methods work in practice. Knowledge of this component may then become less accessible, inhibiting understanding and future research insights from the wider community.

Our datasets are drawn from standard benchmarks, and thus should not introduce new societal risks. Similarly, our work aims to decrease the computational burden of HPO, which should mitigate the environmental impact of ML training — an ever more important goal in our increasingly environmentally-conscious society.

# REPRODUCIBILITY STATEMENT

All datasets we use are publicly available; for the Penn Treebank dataset, we provide a link in Table 4 (Appendix A.2). What little data processing we perform is fully explained in the corresponding subsection of Section 4.  
Our mathematical arguments are presented fully and with disclosure of all assumptions in Section C.  
Source code for all our experiments is provided to reviewers, and will be made available on GitHub after deanonymisation. This source code contains a complete description of our experimental environment, configuration files and instructions on the reproduction of our experiments.

# REFERENCES

Baydin, A. G., Cornish, R., Rubio, D. M., Schmidt, M., and Wood, F. Online Learning Rate Adaptation with Hypergradient Descent. arXiv:1703.04782 [cs, stat], February 2018.  
Bengio, Y. Gradient-Based Optimization of Hyperparameters. Neural Computation, 12(8):1889-1900, August 2000.  
Bergstra, J. and Bengio, Y. Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13(Feb):281-305, 2012.  
Bertrand, H., Ardon, R., Perrot, M., and Bloch, I. Hyperparameter Optimization of Deep Neural Networks: Combining Hyperband with Bayesian Model Selection. In *Conference Sur l'Apprentissage Automatique*, pp. 5, 2017.  
Bouthillier, X. and Varoquaux, G. Survey of machine-learning experimental methods at NeurIPS2019 and ICLR2020. Research Report, Inria Saclay Ile de France, January 2020.  
Chandrashekaran, A. and Lane, I. R. Speeding up Hyper-parameter Optimization by Extrapolation of Learning Curves Using Previous Builds. In Ceci, M., Hollmén, J., Todorovski, L., Vens, C., and Džeroski, S. (eds.), Machine Learning and Knowledge Discovery in Databases, Lecture Notes in Computer Science, pp. 477-492, Cham, 2017. Springer International Publishing.

Domhan, T., Springenberg, J. T., and Hutter, F. Speeding Up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves. In Twenty-Fourth International Joint Conference on Artificial Intelligence, June 2015.  
Domke, J. Generic Methods for Optimization-Based Modeling. In Artificial Intelligence and Statistics, pp. 318-326, March 2012.  
Donini, M., Franceschi, L., Frasconi, P., Pontil, M., and Majumder, O. MARTHE: Scheduling the Learning Rate Via Online Hypergradients. In Twenty-Ninth International Joint Conference on Artificial Intelligence, volume 3, pp. 2119-2125, July 2020.  
Falkner, S., Klein, A., and Hutter, F. BOHB: Robust and Efficient Hyperparameter Optimization at Scale. In International Conference on Machine Learning, pp. 1437-1446. PMLR, July 2018.  
Franceschi, L., Donini, M., Frasconi, P., and Pontil, M. Forward and Reverse Gradient-Based Hyperparameter Optimization. In International Conference on Machine Learning, pp. 1165-1173, July 2017.  
Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., and Pontil, M. Bilevel Programming for Hyperparameter Optimization and Meta-Learning. In International Conference on Machine Learning, pp. 1568-1577. PMLR, July 2018.  
Fu, J., Ng, R., Chen, D., Ilievski, I., Pal, C., and Chua, T.-S. Neural Optimizers with Hypergradients for Tuning Parameter-Wise Learning Rates. In Automatic Machine Learning Workshop at ICML 2017, pp. 8, Sydney, August 2017.  
Gal, Y. and Ghahramani, Z. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. In International Conference on Machine Learning, pp. 1050-1059, June 2016.  
Glorot, X., Bordes, A., and Bengio, Y. Deep Sparse Rectifier Neural Networks. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 315-323, June 2011.  
Grefenstette, E., Amos, B., Yarats, D., Htut, P. M., Molchanov, A., Meier, F., Kiela, D., Cho, K., and Chintala, S. Generalized Inner Loop Meta-Learning. arXiv:1910.01727 [cs, stat], October 2019.  
He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, June 2016.  
Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 9(8):1735-1780, November 1997.  
Hutter, F., Kotthoff, L., and Vanschoren, J. (eds.). Automated Machine Learning: Methods, Systems, Challenges. Springer, 2018.  
Jamieson, K. and Talwalkar, A. Non-stochastic Best Arm Identification and Hyperparameter Optimization. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, pp. 240-248, May 2016.  
Jie, R., Gao, J., Vasnev, A., and Tran, M.-N. Adaptive Multi-level Hyper-gradient Descent. arXiv:2008.07277 [cs], August 2020.  
Kandasamy, K., Dasarathy, G., Schneider, J., and Poczos, B. Multi-fidelity Bayesian Optimisation with Continuous Approximations. arXiv:1703.06240 [stat], March 2017.  
Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization. In arXiv:1412.6980 [Cs], San Diego, CA, USA, May 2015.  
Klein, A., Falkner, S., Springenberg, J. T., and Hutter, F. Learning Curve Prediction with Bayesian Neural Networks. In 5th International Conference on Learning Representations, 2017.  
Kohavi, R. and John, G. H. Automatic Parameter Selection by Minimizing Estimated Error. In Prieditis, A. and Russell, S. (eds.), Machine Learning Proceedings 1995, pp. 304-312. Morgan Kaufmann, San Francisco (CA), January 1995.

Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. MSc Thesis, University of Toronto, April 2009.  
Krueger, T., Panknin, D., and Braun, M. Fast Cross-Validation via Sequential Testing. Journal of Machine Learning Research, 16(33):1103-1155, 2015.  
Larsen, J., Hansen, L., Svarer, C., and Ohlsson, M. Design and regularization of neural networks: The optimal use of a validation set. In Neural Networks for Signal Processing VI. Proceedings of the 1996 IEEE Signal Processing Society Workshop, pp. 62-71, September 1996.  
Levesque, J., Durand, A., Gagné, C., and Sabourin, R. Bayesian optimization for conditional hyperparameter spaces. In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 286-293, May 2017.  
Li, J., Gu, B., and Huang, H. Improved Bilevel Model: Fast and Optimal Algorithm with Theoretical Guarantee. arXiv:2009.00690 [cs, stat], September 2020.  
Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. arXiv:1603.06560 [cs, stat], June 2018.  
Liao, R., Xiong, Y., Fetaya, E., Zhang, L., Yoon, K., Pitkow, X., Urtasun, R., and Zemel, R. Reviving and Improving Recurrent Back-Propagation. In International Conference on Machine Learning, pp. 3082-3091. PMLR, July 2018.  
Liu, R., Mu, P., Yuan, X., Zeng, S., and Zhang, J. A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton. In Proceedings of the International Conference on Machine Learning. PMLR, 2020.  
Lopez-Ramos, L. M. and Beferull-Lozano, B. Online Hyperparameter Search Interleaved with Proximal Parameter Updates. arXiv:2004.02769 [cs, eess, stat], April 2020.  
Lorraine, J. and Duvenaud, D. Stochastic Hyperparameter Optimization through Hypernetworks. arXiv:1802.09419 [cs], March 2018.  
Lorraine, J., Vicol, P., and Duvenaud, D. Optimizing Millions of Hyperparameters by Implicit Differentiation. arXiv:1911.02590 [cs, stat], November 2019.  
Luketina, J., Berglund, M., Greff, K., and Raiko, T. Scalable Gradient-Based Tuning of Continuous Regularization Hyperparameters. In International Conference on Machine Learning, pp. 2952-2960. PMLR, June 2016.  
MacKay, M., Vicol, P., Lorraine, J., Duvenaud, D., and Grosse, R. Self-Tuning Networks: Bilevel Optimization of Hyperparameters using Structured Best-Response Functions. arXiv:1903.03088 [cs, stat], March 2019.  
Maclaurin, D., Duvenaud, D., and Adams, R. Gradient-based Hyperparameter Optimization through Reversible Learning. In International Conference on Machine Learning, pp. 2113-2122. PMLR, June 2015.  
Majumder, O., Donini, M., and Chaudhari, P. Learning the Learning Rate for Gradient Descent by Gradient Descent. In 6th AutoML Workshop at ICML 2019, pp. 8, 2019.  
Marcus, M. P., Santorini, B., Marcinkiewicz, M. A., and Taylor, A. Treebank-3, 1999.  
Mehra, A. and Hamm, J. Penalty Method for Inversion-Free Deep Bilevel Optimization. arXiv:1911.03432 [cs, math, stat], June 2020.  
Micaelli, P. and Storkey, A. Non-greedy Gradient-based Hyperparameter Optimization Over Long Horizons. arXiv:2007.07869 [cs, stat], July 2020.  
Mockus, J., Tiešis, V., and Žilinskas, A. The Application of Bayesian Methods for Seeking the Extremum. Towards Global Optimisation, 2:117-129, 1978.  
Nogueira, F. Bayesian Optimization: Open source constrained global optimization tool for Python, 2014.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 12, Vancouver, Canada, 2019.  
Pearlmutter, B. A. Fast exact multiplication by the Hessian. Neural Computation, 6(1):147-160, January 1994.  
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournaepau, D., Brucher, M., Perrot, M., and Duchesnay, É. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12(85):2825-2830, 2011.  
Perrone, V., Shen, H., Seeger, M., Archambeau, C., and Jenatton, R. Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning. arXiv:1909.12552 [cs, stat], September 2019.  
Petrak, J. Fast Subsampling Performance Estimates for Classification Algorithm Selection. Technical Report TR-2000-07, Austrian Research Institute for Artificial Intelligence, 2000.  
Provost, F., Jensen, D., and Oates, T. Efficient progressive sampling. In Proceedings of the Fifth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '99, pp. 23-32, San Diego, California, USA, August 1999. Association for Computing Machinery.  
Sabharwal, A., Samulowitz, H., and Tesauro, G. Selecting near-optimal learners via incremental data allocation. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI'16, pp. 2007-2015, Phoenix, Arizona, February 2016. AAAI Press.  
Shaban, A., Cheng, C.-A., Hatch, N., and Boots, B. Truncated Back-propagation for Bilevel Optimization. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1723-1732, April 2019.  
Snoek, J., Larochelle, H., and Adams, R. P. Practical Bayesian Optimization of Machine Learning Algorithms. In Pereira, F., Burges, C. J. C., Bottou, L., and Weinberger, K. Q. (eds.), Advances in Neural Information Processing Systems 25, pp. 2951-2959. Curran Associates, Inc., 2012.  
Snoek, J., Rippel, O., Swersky, K., Kiros, R., Satish, N., Sundaram, N., Patwary, M., Prabhat, M., and Adams, R. Scalable Bayesian Optimization Using Deep Neural Networks. In International Conference on Machine Learning, pp. 2171-2180. PMLR, June 2015.  
Sparks, E. R., Talwalkar, A., Haas, D., Franklin, M. J., Jordan, M. I., and Kraska, T. Automating model search for large scale machine learning. In Proceedings of the Sixth ACM Symposium on Cloud Computing, SoCC '15, pp. 368-380, Kohala Coast, Hawaii, August 2015. Association for Computing Machinery.  
Swersky, K., Duvenaud, D., Snoek, J., Hutter, F., and Osborne, M. A. Raiders of the Lost Architecture: Kernels for Bayesian Optimization in Conditional Parameter Spaces. arXiv:1409.4011 [stat], September 2014a.  
Swersky, K., Snoek, J., and Adams, R. P. Freeze-Thaw Bayesian Optimization. arXiv:1406.3896 [cs, stat], June 2014b.  
Thornton, C., Hutter, F., Hoos, H. H., and Leyton-Brown, K. Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '13, pp. 847-855, Chicago, Illinois, USA, August 2013. Association for Computing Machinery.  
Turner, R., Eriksson, D., McCourt, M., Kiili, J., Laaksonen, E., Xu, Z., and Guyon, I. Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020. arXiv:2104.10201 [cs, stat], August 2021.

van den Bosch, A., Verbrugge, R., Taatgen, N., and Schomaker, L. Wrapped progressive sampling search for optimizing learning algorithm parameters. In Proceedings of the Belgium-Netherlands Conference on Artificial Intelligence, BNAIC'04, pp. 219-226, October 2004.  
Wang, J., Xu, J., and Wang, X. Combination of Hyperband and Bayesian Optimization for Hyperparameter Optimization in Deep Learning. arXiv:1801.01596 [cs], January 2018.  
Wang, Z., Hutter, F., Zoghi, M., Matheson, D., and de Feitas, N. Bayesian Optimization in a Billion Dimensions via Random Embeddings. Journal of Artificial Intelligence Research, 55:361-387, February 2016.  
Williams, C. K. I. and Rasmussen, C. E. Gaussian Processes for Regression. In Touretzky, D. S., Mozer, M. C., and Hasselmo, M. E. (eds.), Advances in Neural Information Processing Systems 8, pp. 514-520. MIT Press, 1996.  
Wu, Y., Ren, M., Liao, R., and Grosse, R. B. Understanding Short-Horizon Bias in Stochastic Meta-Optimization. In 6th International Conference on Learning Representations, Vancouver, BC, Canada, 2018. OpenReview.net.  
Xiao, H., Rasul, K., and Vollgraf, R. Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747 [cs, stat], September 2017.
