# WIN: WEIGHT-DECAY-INTEGRATED NESTEROV AC-CELERATION FOR ADAPTIVE GRADIENT ALGORITHMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Training deep networks on increasingly large-scale datasets is computationally challenging. In this work, we explore the problem of "how to accelerate the convergence of adaptive gradient algorithms in a general manner", and aim at providing practical insights to boost the training efficiency. To this end, we propose an effective and general Weight-decay-Integrated Nesterov acceleration (Win) for adaptive algorithms to enhance their convergence speed. Taking AdamW and Adam as examples, we minimize a dynamical loss per iteration which combines the vanilla training loss and a dynamic regularizer inspired by proximal point method (PPM) to improve the convexity of the problem. To introduce Nesterov-alike-acceleration into AdamW and Adam, we respectively use the first- and second-order Taylor approximations of vanilla loss to update the variable twice while fixing the above dynamic regularization brought by PPM. In this way, we arrive at our Win acceleration (like Nesterov acceleration) for AdamW and Adam that uses a conservative step and a reckless step to update twice and then linearly combines these two updates for acceleration. Next, we extend this Win acceleration to LAMB and SGD. Our transparent acceleration derivation could provide insights for other accelerated methods and their integration into adaptive algorithms. Besides, we prove the convergence of Win-accelerated adaptive algorithms and justify their convergence superiority over their non-accelerated counterparts by taking AdamW and Adam as examples. Experimental results testify the faster convergence speed and superior performance of our Win-accelerated AdamW, Adam, LAMB and SGD over their non-accelerated counterparts on vision classification tasks and language modeling tasks with both CNN and Transformer backbones. We hope Win acceleration shall be a default acceleration option for all popular optimizers in deep learning community to improve the training efficiency.

# 1 INTRODUCTION

Deep neural networks (DNNs) are effective to model realistic data and have been successfully applied to many applications, e.g. image classification (He et al., 2016) and speech recognition (Sainath et al., 2013). Typically, their training models can be formulated as a nonconvex problem:

$$
\min  _ {\boldsymbol {z} \in \mathbb {R} ^ {d}} F (\boldsymbol {z}) := \mathbb {E} _ {\boldsymbol {\zeta} \sim \mathcal {D}} [ f (\boldsymbol {z}, \boldsymbol {\zeta}) ] + \frac {\lambda}{2} \| \boldsymbol {z} \| _ {2} ^ {2}, \tag {1}
$$

where  $z \in \mathbb{R}^d$  is the model parameters; sample  $\zeta$  is drawn from a data distribution  $\mathcal{D}$ ; the loss  $f$  is differentiable;  $\lambda$  is a constant. Though many algorithms, e.g. gradient descent (Cauchy et al., 1847) and variance-reduced algorithms (Johnson & Zhang, 2013), can solve problem (1), SGD (Robbins & Monro, 1951) uses the compositional structure in (1) to efficiently estimate gradient via minibatch data, and has become a dominant algorithm to train DNNs in practice because of its higher efficiency and effectiveness. However, on sparse data or ill-conditioned problems, SGD suffers from slow convergence speed (Kingma & Ba, 2014), as it scales the gradient uniformly in all parameter coordinate and ignores the data or problem properties on each coordinate. To resolve this issue, recent work has proposed a variety of adaptive methods, e.g. Adam (Kingma & Ba, 2014) and AdamW (Loshchilov & Hutter, 2018), that scale each gradient coordinate according to the current geometry curvature of the loss  $F(z)$ . This coordinate-wise scaling greatly accelerates the optimization convergence and helps them, e.g. Adam and AdamW, become default optimizers to train DNNs.

Unfortunately, along with the increasing scale of both datasets and models, efficient DNN training even with SGD or adaptive algorithms has become very challenging. In this work, we are particularly

interested in the problem of "how to accelerate the convergence of adaptive algorithms in a general manner" because of their dominant popularity across many DNNs. Heavy ball acceleration (Polyak, 1964) and Nesterov acceleration (Nesterov, 2003) are widely used in SGD but are rarely studied in adaptive algorithms. Among the very few, NAdam (Dozat, 2016) simplifies Nesterov acceleration to estimate the first moment of gradient in Adam while totally ignoring the second-order moments, which is not exact Nesterov acceleration and may not inherit its full acceleration merit.

Contributions: In this work, based on a recent Nesterov-type acceleration formulation (Nesterov et al., 2018) and proximal point method (PPM) (Moreau, 1965), we propose a new Weight-decay-Integrated Nesterov acceleration (Win for short) to accelerate adaptive algorithms, and also further analyze the convergence of Win-accelerated adaptive algorithms to justify their convergence superiority by taking AdamW and Adam as examples. Our main contributions are highlighted below.

Firstly, we use PPM to rigorously derive our Win acceleration for accelerating adaptive algorithms. By taking AdamW and Adam as examples, at the  $k$ -iteration, we follow PPM spirit and minimize a dynamically regularized loss  $F(z) + \frac{1}{2\eta_k}\|z - x_k\|_{\sqrt{v_k + \nu}}^2$  with the second-order gradient moment  $v_k$  and the stabilizing constant  $\nu$  in AdamW and Adam. Then to introduce Nesterov-alike acceleration and also make the problem solvable iteratively, we respectively approximate  $F(z)$  by its first- and second-order Taylor expansions to update the variable  $z$  twice while always fixing the above dynamic regularization and also an extra regularizer  $\frac{1}{2\eta_k}\|z\|_{\sqrt{v_k + \nu}}^2$  induced by the weight decay in AdamW. As a result, we achieve at our Win acceleration, a Nesterov-alike acceleration, for AdamW and Adam that uses a conservative step and a reckless step to update twice and then linearly combines these two updates for acceleration. Then we extend this Win acceleration to LAMB (You et al., 2019) and SGD. The above acceleration derivation is transparent and general which could motivate other accelerations and provide examples to introduce other accelerations into adaptive algorithms.

Secondly, we prove the convergence of our Win-accelerated AdamW and Adam. For both, to find an  $\epsilon$ -approximate first-order stationary point, their stochastic gradient complexity is  $\mathcal{O}\left(\frac{c_{\infty}^{2.5}}{\nu^{0.5}\epsilon^4}\right)$  and matches the lower bound  $\Omega\left(\frac{1}{\epsilon^4}\right)$  in (Arjevani et al., 2019; 2020) (up to constant factors) under the same conditions, where  $c_{\infty}$  upper bounds the  $\ell_{\infty}$  norm of stochastic gradient. Moreover, this complexity improves a factor  $\mathcal{O}\left(\frac{1}{\nu^{0.75}}\right)$  over the complexity  $\mathcal{O}\left(\frac{c_{\infty}^{2.5}}{\nu^{1.25}\epsilon^4}\right)$  of Adam-type optimizers in (Zhou et al., 2018; Guo et al., 2021), e.g. Adam, AdaGrad (Duchi et al., 2011), AdaBound (Luo et al., 2018), since  $\nu$  is often small, e.g.  $\mathcal{O}(\nu) = 10^{-4}$  in practice. Indeed, Win-accelerated Adam and AdamW also enjoy superior complexity than other Adam variants, e.g. Adabelief (Zhuang et al., 2020) with complexity  $\mathcal{O}\left(\frac{c_2^6}{\nu^2\epsilon^4}\right)$  and RMSProp (Zhou et al., 2018) with complexity  $\mathcal{O}\left(\frac{c_{\infty}^{0.5}d^{0.5}}{\nu\epsilon^4}\right)$ , especially on over-parameterized networks, where  $c_2$  is the maximum  $\ell_{2}$ -norm of stochastic gradient.

Finally, experimental results on both vision classification tasks and language modeling tasks show that our Win-accelerated algorithms, i.e. accelerated AdamW, Adam, LAMB and SGD, can accelerate the convergence speed and also improve the performance of their corresponding non-accelerated counterparts by a remarkable margin on both CNN and transformer architectures. All these results show the strong compatibility, generalization and superiority of our acceleration technique.

# 2 RELATED WORK

In the context of deep learning, when considering efficiency and generalization, one often prefers to adopt SGD and adaptive gradient algorithms, e.g. Adam, instead of other algorithms, e.g. variance-reduced algorithms (Johnson & Zhang, 2013), to solve problem (1). But, in practice, adaptive algorithms often suffer from inferior generalization performance than SGD. To solve this issue, AdamW (Loshchilov & Hutter, 2018) proposes a decoupled weight decay which introduces an  $\ell_2$ -like regularization into Adam to decay network weight iteratively, and its effectiveness is widely validated on ViTs (Touvron et al., 2021) and CNNs (Touvron et al., 2021). Later, LAMB (You et al., 2019) scales the update in AdamW to the weight magnitude for getting rid of too large or small update, but suffers from unsatisfactory performance on small batch. In this work, we aim to design a general acceleration approach to accelerate these adaptive algorithms.

Heavy-ball acceleration (Polyak, 1964) and Nesterov acceleration (Nesterov, 2003) are two classical acceleration techniques, and their effectiveness in SGD is well testified. Later, NAdam (Dozat, 2016) integrates Nesterov acceleration into the first-order gradient moment estimation but ignores the second-order gradient moments which harms the acceleration effect. Recently, for full gra

dient decent algorithm, a new general Nesterov-type acceleration (Nesterov et al., 2018) directly interpolates two variables to look ahead for correction, and is more flexible than vanilla Nesterov acceleration (Nesterov, 2003) which interpolates the variable and gradient. See more discussion in Sec. 3.2. In this work, we use proximal point method to introduce this new acceleration into adaptive algorithms by a rigorous and transparent derivation and also necessary adaptations (or tailors).

# 3 WEIGHT-DECAY-INTEGRATED NESTEROV ACCELERATION

To accelerate full gradient descent algorithm, given a full gradient  $\nabla F(\pmb{z}_k)$  of problem (1) at the  $k$ -th iteration, Nesterov-type acceleration (Nesterov et al., 2018) generally uses a conservative step  $\eta_k$  and a reckless step  $\bar{\eta}_k$  to update two sequences  $\pmb{x}_{k+1}$  and  $\pmb{y}_{k+1}$  respectively, and then linearly combines them to update the variable  $\pmb{z}_{k+1}$  of the problem. Similar formulations are also observed and proved in many recent works, e.g. (Allen-Zhu & Orecchia, 2014; Bansal & Gupta, 2019; Ahn & Sra, 2022). In general, their acceleration formulation can be formally formulated as

$$
\boldsymbol {x} _ {k + 1} = \boldsymbol {x} _ {k} - \eta_ {k} \nabla F (\boldsymbol {z} _ {k}), \quad \boldsymbol {y} _ {k + 1} = \boldsymbol {z} _ {k} - \bar {\eta} _ {k} \nabla F (\boldsymbol {z} _ {k}), \quad \boldsymbol {z} _ {k + 1} = \rho_ {k} \boldsymbol {x} _ {k + 1} + (1 - \rho_ {k}) \boldsymbol {y} _ {k + 1}. \tag {2}
$$

This acceleration enjoys provably faster convergence rate for full gradient descent method on convex problems (Nesterov et al., 2018), and is then empirically validated in many convex and nonconvex cases, e.g. (Wilson et al., 2017; Nado et al., 2021). Despite its effectiveness, such acceleration is rarely explored in adaptive gradient algorithms, especially for network training. In deterministic optimization setting, another widely used optimization-stabilizing and acceleration approach is proximal point method (PPM) (Moreau, 1965; Rockafellar, 1976). At the  $k$ -th iteration, PPM optimizes an  $\ell_2$ -regularized loss  $F(z) + \frac{1}{2\eta_k}\|z - z_{k-1}\|_2^2$  instead of the vanilla loss  $F(z)$ . This small change enhances the convexity of the problem, accelerating and also stabilizing optimization process (Kim et al., 2022; Zhou et al., 2021). To make the  $\ell_2$ -regularized problem solvable iteratively, PPM approximates the loss  $F(z)$  by its first- or second-order Taylor expansion so that each iteration has a close-form solution (see below). At below, we borrow the idea in PPM to induce a Weight-decay-Integrated Nesterov acceleration (Win) for adaptive algorithms by using AdamW and Adam as examples in Sec. 3.1, and then extend this acceleration technique to LAMB and SGD in Sec. 3.2.

# 3.1 WIN-ACCELERATED ADAMW AND ADAM

To begin with, following most adaptive gradient algorithms, e.g. Adam and AdamW, we estimate the first- and second-order moments  $\pmb{m}_k$  and  $\pmb{v}_k$  of gradient as follows:

$$
\boldsymbol {g} _ {k} = \frac {1}{b} \sum_ {i = 1} ^ {b} \nabla f \left(\boldsymbol {z} _ {k}; \boldsymbol {\zeta} _ {i}\right), \quad \boldsymbol {m} _ {k} = \left(1 - \beta_ {1}\right) \boldsymbol {m} _ {k - 1} + \beta_ {1} \boldsymbol {g} _ {k}, \quad \boldsymbol {v} _ {k} = \left(1 - \beta_ {2}\right) \boldsymbol {v} _ {k - 1} + \beta_ {2} \boldsymbol {g} _ {k} ^ {2}, \tag {3}
$$

where  $\pmb{g}_k$  is the average gradient on a minibatch data of size  $b$ ,  $\beta_1 \in [0,1]$  and  $\beta_2 \in [0,1]$ . For the initialization, we set  $\pmb{m}_0 = \pmb{g}_0$ ,  $\pmb{v}_0 = \pmb{g}_0^2$ . For brevity, with a small scalar  $\nu > 0$ , we define

$$
\boldsymbol {s} _ {k} = \sqrt {\boldsymbol {v} _ {k} + \nu}, \quad \boldsymbol {u} _ {k} = \boldsymbol {m} _ {k} / \sqrt {\boldsymbol {v} _ {k} + \nu}. \tag {4}
$$

Then following the spirit of PPM, at the  $k$ -th iteration, we minimize a regularized loss  $F(\pmb{x}) + \frac{1}{2\eta_k} \| \pmb{x} - \pmb{x}_k \|_{\pmb{s}_k}^2$ . Here we use the regularizer  $\| \pmb{x} - \pmb{x}_k \|_{\pmb{s}_k}^2$  instead of the  $\ell_2$ -regularization  $\| \pmb{x} - \pmb{x}_k \|_2^2$ , since 1) this new regularization allows us to handle adaptive algorithms as shown below Eqn. (5), and 2) it also helps increase the convexity of the problem and actually further considers the different sharpness properties of each coordinates of the loss because of different elements in  $\pmb{s}_k$ , thus also speeding up the convergence. To make the problem solvable iteratively, we approximate the vanilla loss  $F(z)$  by its first-order Taylor expansion at the point  $\pmb{z}_k$  and update  $\pmb{x}_{k+1}$  as

$$
\boldsymbol {x} _ {k + 1} = \operatorname {a r g m i n} _ {\boldsymbol {x}} F (\boldsymbol {z} _ {k}) + \left\langle \boldsymbol {m} _ {k}, \boldsymbol {x} - \boldsymbol {z} _ {k} \right\rangle + \frac {1}{2 \eta_ {k}} \| \boldsymbol {x} - \boldsymbol {x} _ {k} \| _ {\boldsymbol {s} _ {k}} ^ {2} + \frac {\lambda}{2} \| \boldsymbol {x} \| _ {\boldsymbol {s} _ {k}} ^ {2} = \frac {1}{1 + \lambda \eta_ {k}} \left(\boldsymbol {x} _ {k} - \eta_ {k} \boldsymbol {u} _ {k}\right), \tag {5}
$$

where  $\| \pmb{x}\|_{\pmb{s}_k} = \sqrt{\langle\pmb{x},\pmb{s}_k*\pmb{x}\rangle}$  with an element-wise product operation  $*$ ,  $m_k$  is used to approximate the full gradient  $\nabla F(z_k)$  for Taylor expansion. We add a small regularization  $\frac{\lambda}{2}\|\pmb{x}\|_{\pmb{s}_k}^2$ , since 1) it can largely improve the generalization performance in practice (Loshchilov & Hutter, 2018; Touvron et al., 2021); 2) it allows us to derive Adam ( $\lambda = 0$ ) and AdamW ( $\lambda > 0$ ). Specifically, when  $\lambda = 0$ , the updating (5) becomes the exact Adam algorithm. If  $\lambda > 0$ , the updating (5) is an approximation to the updating rule  $\pmb{x}_{k + 1} = (1 - \lambda \eta_k)\pmb{x}_k - \eta_k\pmb{u}_k$  of AdamW. This is because consider  $\lambda \eta_k$  is small in practice, we can approximate  $(1 + \lambda \eta_k)^{-1} = 1 - \lambda \eta_k + \mathcal{O}(\lambda^2\eta_k^2)$  and thus  $\frac{1}{1 + \lambda\eta_k} (\pmb{x}_k - \eta_k\pmb{u}_k) = [1 - \lambda \eta_k + \mathcal{O}(\lambda^2\eta_k^2)]\pmb{x}_k - [\eta_k - \mathcal{O}(\lambda \eta_k^2) + \mathcal{O}(\lambda^3\eta_k^3)]\pmb{u}_k$  which becomes AdamW by ignoring the ignorable terms related to  $\mathcal{O}(\eta_k^2)$  or  $\mathcal{O}(\eta_k^3)$ . This is also one reason that we adopt the regularizer  $\| \pmb{x} - \pmb{x}_k\|_{\pmb{s}_k}^2$  in (5) instead of the  $\ell_2$ -regularization in PPM, since we can flexibly derive Adam and AdamW.

Algorithm 1: Win-Accelerated AdamW, Adam and LAMB  
Input: initialization  $\pmb{x}_0 = \pmb{z}_0 = \mathbf{0}$  step size  $\{(\eta_k,\bar{\eta}_k)\}_{k = 0}^T$  , moment parameters  $\{\beta_{1},\beta_{2}\}$  Output:  $(\bar{x},\bar{z})$  uniformly selected from  $\{(x_{k},z_{k})\}_{k = 0}^{T}$  while  $k <   T$  do  $\begin{array}{rlr}{\pmb{g}_k = \frac{1}{b}\sum_{i = 1}^b\nabla f(\pmb {z}_k;\pmb {\zeta}_i)} & {} & \\ {\pmb {m}_k = (1 - \beta_1)\pmb {m}_{k - 1} + \beta_1\pmb {g}_k} & {} & {/\star \pmb {m}_0 = \pmb {g}_0\star /}\\ {\pmb {v}_k = (1 - \beta_2)\pmb {v}_{k - 1} + \beta_2\pmb {g}_k^2} & {} & {/\star \pmb {v}_0 = \pmb {g}_0^2\star /}\\ {\pmb {u}_k = \frac{\pmb{m}_k}{\sqrt{\pmb{v}_k + \nu}}\mathrm{~for~AdamW~and~Adam,~}\pmb {u}_k = \frac{\|\pmb{x}_k\|_2}{\|\pmb{m}_k / \sqrt{\pmb{v}_k + \nu}\|_2}\frac{\pmb{m}_k}{\sqrt{\pmb{v}_k + \nu}}\mathrm{~for~LAMB}}\\ {\pmb{x}_{k + 1} = \frac{1}{1 + \lambda\eta_k} (\pmb {x}_k - \eta_k\pmb {u}_k)} & {} & \\ {\pmb{z}_{k + 1} = \bar{\eta}_k\tau_k\pmb{x}_{k + 1} + \eta_k\tau_k(\pmb {z}_k - \bar{\eta}_k\pmb {u}_k)\mathrm{~with~}\tau_k = \frac{1}{\eta_k + \bar{\eta}_k + \lambda\eta_k\bar{\eta}_k}} & {} & \end{array}$

Similarly, we minimize a regularized loss objective  $F(\pmb{z}) + \frac{1}{2\eta_k} \| \pmb{z} - \pmb{x}_{t+1} \|_{\pmb{s}_k}^2$  again, and further approximate  $F(\pmb{z})$  by its second-order approximation  $F(\pmb{z}_k) + \langle \pmb{m}_k, \pmb{z} - \pmb{z}_k \rangle + \frac{1}{2\bar{\eta}_k} \| \pmb{z} - \pmb{z}_k \|_{\pmb{s}_k}^2$ :

$$
\begin{array}{l} \boldsymbol {z} _ {k + 1} = \operatorname {a r g m i n} _ {\boldsymbol {z}} F (\boldsymbol {z} _ {k}) + \left\langle \boldsymbol {m} _ {k}, \boldsymbol {z} - \boldsymbol {z} _ {k} \right\rangle + \frac {1}{2 \bar {\eta} _ {k}} \| \boldsymbol {z} - \boldsymbol {z} _ {k} \| _ {\boldsymbol {s} _ {k}} ^ {2} + \frac {1}{2 \eta_ {k}} \| \boldsymbol {z} - \boldsymbol {x} _ {k + 1} \| _ {\boldsymbol {s} _ {k}} ^ {2} + \frac {\lambda}{2} \| \boldsymbol {z} \| _ {\boldsymbol {s} _ {k}} ^ {2} \tag {6} \\ = \bar {\eta} _ {k} \tau_ {k} \boldsymbol {x} _ {k + 1} + \eta_ {k} \tau_ {k} \left(\boldsymbol {z} _ {k} - \bar {\eta} _ {k} \boldsymbol {u} _ {k}\right), \\ \end{array}
$$

where  $\tau_{k} = \frac{1}{\eta_{k} + \bar{\eta}_{k} + \lambda\eta_{k}\bar{\eta}_{k}}$ ,  $m_{k}$  is used to approximate  $\nabla F(\pmb{x}_k)$  as guaranteed by Theorem 1 in Sec. 4,  $\bar{\eta}_k$  approximates the inverse of the local smoothness parameter of  $F(z)$  around  $\pmb{z}_k$ . Here we use a regularizer  $\| z - x_{k + 1}\|_{s_k}^2$  with the latest update  $\pmb{x}_{k + 1}$  instead of  $\pmb{x}_k$  as an anchor point, since the latest update  $\pmb{x}_{k + 1}$  could often provide better regularization for the concurrent optimization.

Now we have used PPM to rigorously derive our Win-accelerated AdamW and Adam in Eqns. (3), (5) and (6). For more clarity, we summarize their algorithmic steps in Algorithm 1 in which we omit the bias-correlation term for simplicity. When  $\lambda = 0$ , it is Win-accelerated Adam; if  $\lambda > 0$ , it gives Win-accelerated AdamW. Generally, AdamW can greatly improve the generalization performance of Adam by simply adding a weight decay (i.e. the regularizer  $\frac{\lambda}{2} \|\cdot\|_{\mathbf{s}_k}^2$ ) into Adam as observed in many works, e.g. (Loshchilov & Hutter, 2018; Touvron et al., 2021). Our Win-acceleration is quite simple and efficient, since our accelerated AdamW/Adam only adds one extra simple algorithmic step, i.e. the seventh step in Algorithm 1, on vanilla AdamW/Adam. Moreover, for the only extra hyper-parameter, the reckless step  $\bar{\eta}_k$ , in Algorithm 1 over AdamW/Adam, we always set it  $2 \times$  larger than the conservative step  $\eta_k$  for all iterations, i.e.  $\bar{\eta}_k = 2\eta_k$ , working well in our all experiments.

Now we discuss the relations between Nesterov-type acceleration (2) and our Win acceleration (6). For comparison, we introduce a virtual sequence  $\pmb{y}_{k + 1} = \pmb{z}_k - \bar{\eta}_k\pmb{u}_k$  in Win, and rewrite (6) as

$$
\boldsymbol {x} _ {k + 1} = \left(1 + \lambda \eta_ {k}\right) ^ {- 1} \left(\boldsymbol {x} _ {k} - \eta_ {k} \boldsymbol {u} _ {k}\right), \quad \boldsymbol {y} _ {k + 1} = \boldsymbol {z} _ {k} - \bar {\eta} _ {k} \boldsymbol {u} _ {k}, \quad \boldsymbol {z} _ {k + 1} = \bar {\eta} _ {k} \tau_ {k} \boldsymbol {x} _ {k + 1} + \eta_ {k} \tau_ {k} \boldsymbol {y} _ {k + 1}, \tag {7}
$$

where  $\pmb{u}_k$  is defined in (4). By comparing Nesterov-type acceleration (2) with our Win acceleration (7), one can observe some similarity and also differences as well. For similarity, both acceleration use a conservative step  $\eta_k$  and a reckless step  $\bar{\eta}_k$  to update  $\pmb{x}_{k+1}$  and  $\pmb{y}_{k+1}$  respectively, and then linearly combine  $\pmb{x}_{k+1}$  and  $\pmb{y}_{k+1}$  to obtain  $\pmb{z}_{k+1}$ . For the differences, the first one is that Win has a weight-decay-alike factor  $\frac{1}{1 + \lambda\eta_k}$  in (7) which slightly decays the variable  $\pmb{x}_k$  like AdamW and also the update  $\pmb{u}_k$ , while Nesterov acceleration has no such important factor. Note, this weight decay can greatly benefit generalization in practice as shown in many works, e.g. (Loshchilov & Hutter, 2018; Touvron et al., 2021; Liu et al., 2021). Another difference is that for almost all acceleration techniques, including Nesterov-type acceleration in (2), the sum of the linear combination factors (e.g.  $\rho_k$  and  $1 - \rho_k$  in (2)) is always one. In contrast, in Eqn. (7), Win uses  $\bar{\eta}_k\tau_k + \eta_k\tau_k = 1 - \frac{\lambda\eta_k\bar{\eta}_k}{\eta_k + \bar{\eta}_k + \lambda\eta_k\bar{\eta}_k} < 1$  when  $\lambda > 0$ , which indeed gives a second weigh decay. Since these two differences are caused by the weight decay, we call our acceleration "weight-decay-integrated Nesterov acceleration" (Win).

# 3.2 EXTENSION TO LAMB AND SGD

Here we generalize Win acceleration to LAMB (You et al., 2019) and SGD (Robbins & Monro, 1951). For LAMB, it scales the update  $\pmb{u}_k$  of AdamW in Eqn. (4) so that  $\pmb{u}_k$  is at the same magnitude of the network weight  $\pmb{x}_k$ . That is, it changes the update rule  $\pmb{x}_{k + 1} = (1 - \lambda \eta_k)\pmb{x}_k - \eta_k\pmb{m}_k / \pmb{s}_k$  in AdamW to  $\pmb{x}_{k + 1} = \pmb{x}_k - \eta_k\frac{\|\pmb{x}_k\|_2}{\|\pmb{r}_k + \lambda\pmb{x}_k\|_2} (\pmb{r}_k + \lambda \pmb{x}_k)$  where  $\pmb{r}_k = \pmb{m}_k / s_k$ . This modification is to

avoid too large or small update, improving optimization efficiency. To extend Win acceleration to LAMB, we inherit this scaling spirit, and scale the update  $\pmb{u}_k$  in (4) to the following one:

$$
\boldsymbol {u} _ {k} = \left(\| \boldsymbol {x} _ {k} \| _ {2} / \| \boldsymbol {m} _ {k} / \boldsymbol {s} _ {k} \| _ {2}\right) \cdot \left(\boldsymbol {m} _ {k} / \boldsymbol {s} _ {k}\right). \tag {8}
$$

We scale  $\pmb{m}_k / s_k$  instead of  $(\pmb{m}_k / s_k + \lambda \pmb{x}_k)$  in LAMB, as our scaling can be repeatedly used to update our two sequences  $\pmb{x}_k$  and  $\pmb{z}_k$ . Next, we can respectively follow Eqn. (5) and (6) to update the two sequences  $\pmb{x}_k$  and  $\pmb{z}_k$ . See detailed steps of Win-accelerated LAMB in Algorithm 1.

For SGD, applying Win acceleration to it is quite direct. Specifically, the only algorithmic difference between SGD and AdamW on the  $\ell_2$ -regularized problems is that SGD has no second-order moment  $v_k$  while AdamW has. So we can borrow the acceleration framework of AdamW in Sec. 3.1 to accelerate SGD by setting  $s_k = 1 \in \mathbb{R}^d$  in Eqn. (4), (5) and (6), and obtain WIN-accelerated SGD:

$$
\boldsymbol {m} _ {k} = \beta_ {1} \boldsymbol {m} _ {k - 1} + \beta_ {1} ^ {\prime} \boldsymbol {g} _ {k}, \boldsymbol {x} _ {k + 1} = \frac {1}{1 + \lambda \eta_ {k}} (\boldsymbol {x} _ {k} - \eta_ {k} \boldsymbol {m} _ {k}), \boldsymbol {z} _ {k + 1} = \bar {\eta} _ {k} \tau_ {k} \boldsymbol {x} _ {k + 1} + \eta_ {k} \tau_ {k} (\boldsymbol {z} _ {k} - \bar {\eta} _ {k} \boldsymbol {m} _ {k}), \tag {9}
$$

where  $\beta_1' \in [0, 1]$  is dampening parameter. Here we slightly modify the moment  $m_k$  to accord with the one used in Nesterov-accelerated SGD (e.g. SGD-M in Pytorch) whose updating steps are

$$
\boldsymbol {m} _ {k} = \beta_ {1} \boldsymbol {m} _ {k - 1} + \beta_ {1} ^ {\prime} (\boldsymbol {g} _ {k} + \lambda \boldsymbol {x} _ {k}), \quad \boldsymbol {x} _ {k + 1} = (1 - \lambda \eta_ {k}) \boldsymbol {x} _ {k} - \eta_ {k} (\boldsymbol {g} _ {k} + \beta_ {2} \boldsymbol {m} _ {k}). \tag {10}
$$

By comparing Win-accelerated SGD and SGD-M in (10), one can find their big differences mainly caused by their different acceleration strategies and ways to handle weight decay. Win-accelerated SGD is derived from PPM and a recently proposed acceleration (2), while SGD-M modifies another previous Nesterov-type acceleration (Nesterov, 2003) (with formulation  $\pmb{m}_k = \beta_1\pmb{m}_{k - 1} - \frac{\eta_k}{b}\sum_{i = 1}^b\nabla f(\pmb{x}_k + \eta_k\pmb{m}_{k - 1};\pmb{\zeta}_i)$  and  $\pmb{x}_{k + 1} = \pmb{x}_k + \pmb{m}_k$  ) to better train networks. See more mechanisms of previous Nesterov acceleration and (10) in (Sutskever et al., 2013; Bengio et al., 2013).

# 4 CONVERGENCE ANALYSIS

Here we investigate the convergence performance of Win-accelerated algorithms by taking accelerated AdamW, Adam and SGD as examples, as these algorithms are more preferably used in deep learning field. Moreover, since we aim to accelerate deep network training which is highly nonconvex problems, we focus on analyzing nonconvex problems to accord with the practical setting.

For analysis, we follow previous works on optimizer, e.g. (Kingma & Ba, 2014; Reddi et al., 2019; Luo et al., 2019; Duchi et al., 2011; Xie et al., 2022), and introduce necessary assumptions.

Assumption 1 (L-smoothness). We say a function  $f(z, \cdot)$  to be L-smooth w.r.t.  $z$ , if for  $\forall z_1, z_2$  and  $\forall \zeta \sim \mathcal{D}$ , we have  $\| \nabla f(z_1, \zeta) - \nabla f(z_2, \zeta) \|_2 \leq L \| z_1 - z_2 \|_2$  with a universal constant  $L$ .

Assumption 2 (Unbiased and bounded gradient estimation). The gradient estimation  $\pmb{g}_k$  is unbiased, i.e. for  $\forall k$ ,  $\mathbb{E}[\pmb{g}_k] = \nabla F(\pmb{z}_k)$ , and its magnitude and variance are bounded, namely, for  $\forall k$ ,  $\| \pmb{g}_k \|_{\infty} \leq c_{\infty}$  and  $\mathbb{E}[\| \nabla F(\pmb{z}_k) - \pmb{g}_k \|_2] \leq \sigma$  with two universal constants  $c_{\infty}$  and  $\sigma$ .

Next, we first define a dynamic function  $F_{k}(z)$  at the  $k$ -th iteration which is real loss minimized by our algorithms. It combines the vanilla loss  $F(z)$  in (1) and a dynamic regularization  $\frac{\lambda}{2} \| z \|_{s_k}^2$ :

$$
F _ {k} (z) = F (z) + \frac {\lambda}{2} \| z \| _ {\mathbf {s} _ {k}} ^ {2} = \mathbb {E} _ {\boldsymbol {\zeta}} [ f (\boldsymbol {z}; \boldsymbol {\zeta}) ] + \frac {\lambda}{2} \| z \| _ {\mathbf {s} _ {k}} ^ {2}, \tag {11}
$$

where  $s_k$  is given in (4). To obtain (11), following PPM spirit and Eqn. (5), one can approximate  $F(z)$  by its first-order Taylor expansion, and obtain Eqn. (5) with  $x$  replaced by  $z$  to update  $z_{k+1} = \frac{1}{1 + \lambda\eta_k}(z_k - \eta_k m_k / s_k)$ . Since  $\lambda \eta_k$  is very small, one can follow the discussion below Eqn. (5) and approximate  $z_{k+1}$  as  $z_{k+1} = (1 - \lambda \eta_k) z_k - \eta_k m_k / s_k$  which becomes the update rule of AdamW. This is the reason that our analysis on Win-accelerated AdamW involves a dynamic loss  $F_k(z)$  in (11). Note, for Win-accelerated Adam  $(\lambda = 0)$ ,  $F_k(z)$  degenerates to the vanilla loss  $F(z)$ .

With these assumptions, we analyze the convergence behaviors of our accelerated algorithms on general nonconvex problems, and summarize our main results in Theorem 1 with proof in Appendix D.

Theorem 1. Suppose Assumptions 1 and 2 hold, and  $\pmb{x}_{\star} \in \operatorname{argmin}_{\pmb{x}} F(\pmb{x})$ . By setting  $\bar{\eta}_k = \gamma \eta_k$ ,  $\gamma > 1$ ,  $\eta_k = \eta \leq \frac{\nu^{0.5} b \epsilon^2}{36 c^{1.5} (\gamma + 1)^2 L \sigma^2}$ ,  $\beta_1 \leq \frac{\nu^{0.5} b \epsilon^2}{6 c \sigma^2}$ ,  $\beta_2 \in (0, 1)$ ,  $c = (c_{\infty}^2 + \nu)^{0.5}$  after  $T = \mathcal{O}\left(\frac{c_{\infty}^{2.5} \Delta \sigma^2 L}{\nu^{0.5} b \epsilon^4}\right)$  iterations with minibatch size  $b$  and  $\Delta = F(\pmb{x}_0) - F(\pmb{x}_{\star})$ , the sequence  $\{(x_k, z_k)\}_{k=0}^T$  generated by Win-accelerated AdamW and Adam in Algorithm 1 satisfies the following four properties.

a) The gradient  $\nabla F_{k}(\pmb{x}_{k})$  of the sequence  $\{\pmb{x}_k\}_{k = 0}^T$  can be upper bounded by

$$
\frac {1}{T} \sum_ {k = 0} ^ {T - 1} \mathbb {E} \left[ \| \nabla F _ {k} (\boldsymbol {x} _ {k}) \| _ {2} ^ {2} + \frac {1}{4} \| \boldsymbol {m} _ {k} + \lambda \boldsymbol {x} _ {k} * \boldsymbol {s} _ {k} \| _ {2} ^ {2} \right] \leq \epsilon^ {2}.
$$

b) The gradient moment  $\pmb{m}_k$  can well estimate the full gradient  $\nabla F(\pmb{x}_k)$  and  $\nabla F(\pmb{z}_k)$ :

$$
\frac {1}{T} \sum_ {k = 0} ^ {T - 1} \max  \left\{\mathbb {E} \| \boldsymbol {m} _ {k} - \nabla F (\boldsymbol {x} _ {k}) \| _ {2} ^ {2}, \mathbb {E} \| \boldsymbol {m} _ {k} - \nabla F (\boldsymbol {z} _ {k}) \| _ {2} ^ {2} \right\} \leq \left(1 6 + \frac {1}{2 c} \nu^ {0. 5} L\right) \epsilon^ {2}.
$$

c) The sequence  $\{\pmb{x}_k, \pmb{z}_k\}$  satisfies

$$
\frac {1}{T} \sum_ {k = 0} ^ {T - 1} \Bigl \{\mathbb {E} \| \boldsymbol {x} _ {k} - \boldsymbol {x} _ {k + 1} \| _ {\boldsymbol {s} _ {k}} ^ {2}, \mathbb {E} \| \boldsymbol {z} _ {k + 1} - \boldsymbol {z} _ {k} \| _ {2} ^ {2}, \mathbb {E} \| \boldsymbol {z} _ {k} - \boldsymbol {x} _ {k} \| _ {2} ^ {2} \Bigr \} \leq \left\{\frac {\eta^ {2} \epsilon^ {2}}{4 (1 + \lambda \eta) ^ {2}}, \frac {\nu^ {1 . 5} \beta_ {1} ^ {2} \epsilon^ {2}}{4 c (1 - \beta_ {1}) ^ {3} L ^ {2}}, \frac {\nu^ {0 . 5} \epsilon^ {2}}{4 c L} \right\}.
$$

d) The total stochastic gradient complexity to achieve the above three properties is  $\mathcal{O}\left(\frac{c_{\infty}^{2.5}\Delta\sigma^2L}{\nu^{0.5}\epsilon^4}\right)$ .

Theorem 1 guarantees the convergence of Win-accelerated AdamW and Adam in Algorithm 1 on nonconvex problems. When  $\lambda >0$ , Algorithm 1 corresponds to Win-accelerated AdamW; and if  $\lambda = 0$ , it becomes Win-accelerated Adam. For both cases, Theorem 1 holds. Theorem 1 a) shows that by running at most  $T = \mathcal{O}\left(\frac{c_{\infty}^{2.5}\Delta\sigma^2L}{\nu^{0.5}b\epsilon^4}\right)$  iterations, the average gradient  $\frac{1}{T}\sum_{k = 0}^{T - 1}\mathbb{E}\big[\| \nabla F_k(\pmb{x}_k)\| _2^2\big]$  is upper bounded by  $\epsilon^2$ , guaranteeing the algorithmic convergence. Theorem 1 b) indicates the gradient moment  $\pmb{m}_{k}$  can well estimate the full gradient  $\nabla F(\pmb {z}_k)$  and also  $\nabla F(\pmb {x}_k)$  because of their small distances, guaranteeing the good Taylor approximation used in Eqn. (5) and (6). Moreover, in Theorem 1 c), one can find that although Algorithm 1 uses a conservative step  $\eta_{k}$  and a reckless step  $\bar{\eta}_k = \gamma \eta_k$  ( $\forall \gamma >1$ ) to update, the two sequences  $\pmb{x}_{k + 1}$  and  $\pmb{z}_{k + 1}$  can converge to each other, which could be the key for the good convergence behavior of both Win-accelerated AdamW and Adam.

Now we discuss the stochastic gradient complexity of Win-accelerated Adam and AdamW. Theorem 1 d) shows that to find an  $\epsilon$ -approximate first-order stationary point, both Win-accelerated Adam and AdamW have the complexity  $\mathcal{O}\left(\frac{c_{\infty}^{2.5}\sigma^2L}{\nu^{0.5}\epsilon^4}\right)$  which matches the lower bound  $\Omega\left(\frac{1}{\epsilon^4}\right)$  in (Arjevani et al., 2019; 2020) (up to constant factors) under the same Assumptions 1 and 2. Our accelerated Adam and AdamW enjoy superior complexity over Adam-type optimizers, e.g. Adam, AdaGrad (Duchi et al., 2011), AdaBound (Luo et al., 2018), whose previously best known complexity under the same assumptions is  $\mathcal{O}\left(\frac{c_{\infty}^{2.5}\sigma^2L}{\nu^{1.25}\epsilon^4}\right)$  in (Zhou et al., 2018; Chen et al., 2021; Guo et al., 2021). By comparison, both accelerated Adam and AdamW improve their complexity by a factor  $\mathcal{O}\left(\frac{1}{\nu^{0.75}}\right)$ , where  $\nu$  is often very small in practice, e.g.  $\mathcal{O}(\nu) = 10^{-4}$ . Since the convergence of AdamW has not been proved yet in the literatures, here we cannot directly compare with it. Moreover, the complexity of Win-accelerated Adam and AdamW is also lower than  $\mathcal{O}\left(\frac{c_2^6\sigma^2L}{\nu^2\epsilon^4}\right)$  of Adabelief (Zhuang et al., 2020) and  $\mathcal{O}\left(\frac{c_{\infty}^{0.5}d^{0.5}\sigma^{2}L}{\nu\epsilon^4}\right)$  of RMSProp (Tijmen & Geoffrey, 2012; Zhou et al., 2018), especially on over-parameterized networks, since for a  $d$ -dimensional gradient, its  $\ell_2$ -norm upper bound  $c_2$  is often much larger than the  $\ell_{\infty}$ -norm  $c_{\infty}$  and can be  $\sqrt{d} \times$  larger for worse case.

Now we discuss the convergence performance of Win-accelerated SGD in Theorem 2.

Theorem 2. Suppose Assumptions 1 and 2 hold, and  $\pmb{x}_{\star} \in \operatorname{argmin}_{\pmb{x}} F(\pmb{x})$ . By setting  $\bar{\eta}_k = \gamma \eta_k$ ,  $\gamma > 1$ ,  $\eta_k = \eta \leq \frac{b\epsilon^2}{36(\gamma + 1)^2L\sigma^2}$ ,  $\beta_1 \leq \frac{b\epsilon^2}{6\sigma^2}$  and  $\beta_1' = 1 - \beta_1$ , after  $T = \mathcal{O}\left(\frac{\Delta\sigma^2L}{\nu^{0.5}b\epsilon^4}\right)$  iterations with minibatch size  $b$  and  $\Delta = F(\pmb{x}_0) - F(\pmb{x}_{\star})$ , the sequence  $\{(x_{k},z_{k})\}_{k = 0}^{T}$  generated by Win-accelerated SGD in (9) satisfies the four properties in Theorem 1 with  $\nu = c_{\infty} = c = 1$  and  $s_k = 1 \in \mathbb{R}^d$ .

See its proof in Appendix E. Theorem 2 also guarantees the convergence of Win-accelerated SGD. By using the hyper-parameter settings in Theorem 2, the sequence  $\{(x_k,z_k)\}_{k = 0}^T$  generated by Win-accelerated SGD satisfies the four properties in Theorem 1 with  $\nu = c_{\infty} = c = 1$  and  $s_k = 1$ . It shows the complexity  $\mathcal{O}\left(\frac{L\sigma^2}{\epsilon^4}\right)$  of Win-accelerated SGD which also matches the lower bound  $\Omega \left(\frac{1}{\epsilon^4}\right)$  in (Arjevani et al., 2019; 2020) (up to constant factors) under the same Assumptions 1 and 2.

# 5 EXPERIMENTS

Here we investigate the performance of our accelerated algorithms on two representative tasks, including vision classification tasks and also natural language modeling tasks. For vision tasks, we test accelerated algorithms on both CNNs, e.g. ResNet (He et al., 2016), and vision transformers (ViTs), such as ViT (Dosovitskiy et al., 2020) and PoolFormer (Yu et al., 2021). For natural lan

Table 2: ImageNet top-1 accuracy (\%) of ResNet50&101 whose official optimizer is LAMB due to the stronger data augmentation for better performance. * is reported in (Wightman et al., 2021).  

<table><tr><td rowspan="2">Epoch</td><td colspan="4">ResNet50</td><td rowspan="2">Avg.</td><td colspan="4">ResNet101</td></tr><tr><td>100</td><td>200</td><td>300</td><td></td><td>100</td><td>200</td><td>300</td><td>avg.</td></tr><tr><td>SAM</td><td>77.3</td><td>78.7</td><td>79.4</td><td>78.5</td><td>79.5</td><td>81.1</td><td>81.6</td><td>80.7</td><td></td></tr><tr><td>SGD-H</td><td>75.3</td><td>76.9</td><td>77.2</td><td>76.5</td><td>77.7</td><td>78.6</td><td>78.8</td><td>78.4</td><td></td></tr><tr><td>SGD-M</td><td>77.0</td><td>78.6</td><td>79.3</td><td>78.3</td><td>79.3</td><td>81.0</td><td>81.4</td><td>80.6</td><td></td></tr><tr><td>SGD-Win</td><td>78.0</td><td>79.2</td><td>79.7</td><td>79.0+0.7</td><td>80.1</td><td>81.2</td><td>81.6</td><td>81.0+0.4</td><td></td></tr><tr><td>Adam</td><td>76.9</td><td>78.4</td><td>78.8</td><td>78.1</td><td>78.4</td><td>80.2</td><td>80.6</td><td>79.7</td><td></td></tr><tr><td>Adam-Win</td><td>77.8</td><td>78.8</td><td>79.3</td><td>78.7+0.6</td><td>79.2</td><td>80.6</td><td>81.0</td><td>80.3+0.6</td><td></td></tr><tr><td>AdamW</td><td>77.0</td><td>78.9</td><td>79.3</td><td>78.4</td><td>78.9</td><td>79.9</td><td>80.4</td><td>79.7</td><td></td></tr><tr><td>AdamW-Win</td><td>78.0</td><td>79.3</td><td>79.9</td><td>79.1+0.7</td><td>80.2</td><td>81.1</td><td>81.3</td><td>80.9+1.2</td><td></td></tr><tr><td>LAMB</td><td>77.0</td><td>79.2</td><td>79.8*</td><td>78.7</td><td>79.4</td><td>81.1</td><td>81.3*</td><td>80.6</td><td></td></tr><tr><td>LAMB-Win</td><td>78.4</td><td>79.7</td><td>80.1</td><td>79.4+0.7</td><td>80.6</td><td>81.5</td><td>81.7</td><td>81.3+0.7</td><td></td></tr></table>

guage modeling tasks, we use LSTM (Schmidhuber et al., 1997) and Transformer-XL (Dai et al., 2019) to evaluate the proposed accelerated algorithms for sequence modeling.

For clarity, we call our accelerated algorithm "X-Win", where "X" denotes vanilla optimizers, e.g. Adam. In all experiments, we do not change model architectures and data augmentations, and only replace the default optimizer with ours. Moreover, for all experiments, our accelerated algorithms, e.g. AdamW-Win, always use the default optimizer-inherent hyper-parameters of the vanilla optimizers, e.g. first- and second-order moment parameters  $\beta_{1}$  and  $\beta_{2}$  in AdamW; and their reckless step  $\bar{\eta}_{k}$  always satisfies  $\bar{\eta}_{k} = 2\eta_{k}$ . These settings well reduce the parameter-tuning cost of our algorithms. In the experiments, same with other optimizers, we only slightly tune other widely tuned hyper-parameters around the vanilla ones, e.g. step size and warm-up epochs, etc, which is reasonable, as our accelerated algorithms have two step sizes and the vanilla ones are not very suitable.

# 5.1 RESULTS ON VISION CLASSIFICATION TASKS

Results on ResNet18. Here we follow the conventional supervised training setting used in ResNets (He et al., 2016) and evaluate our accelerated algorithms on ImageNet (Deng et al., 2009). Due to limited space, we defer the hyper-parameter settings of the four accelerated algorithms in Table 1 into Appendix A.

Table 1 shows that our accelerated algorithms can improve the corresponding non-accelerated versions by a remarkable margin. For instance, AdamW-Win, Adam-Win and LAMB-Win respectively make  $3.1\%$ ,  $2.2\%$  and  $2.6\%$  improvement over their corresponding non-accelerated coun

Table 1: ImageNet top-1 accuracy  $(\%)$  of ResNet18. *,† and ‡ are respectively reported in (Chen et al., 2021), (Zhuang et al., 2020) and (Liu et al., 2019).  

<table><tr><td>AdaBound</td><td>68.1*</td><td>Radam</td><td>67.7*</td></tr><tr><td>Nadam</td><td>68.8</td><td>Padam</td><td>70.1*</td></tr><tr><td>SGD-H</td><td>67.3</td><td>AdaBelief</td><td>70.1†</td></tr><tr><td>SGD-M</td><td>70.2*</td><td>Adam</td><td>66.5‡</td></tr><tr><td>SGD-Win</td><td>70.7+0.5</td><td>Adam-Win</td><td>68.7+2.2</td></tr><tr><td>AdamW</td><td>67.9*</td><td>LAMB</td><td>68.5</td></tr><tr><td>AdamW-Win</td><td>71.0+3.1</td><td>LAMB-Win</td><td>71.1+2.6</td></tr></table>

terparts, AdamW, Adam and LAMB. Moreover, SGD-Win improves SGD-H (i.e. SGD + heavy ball) by  $3.4\%$ , and also surpasses SGD-M by  $0.5\%$ , where SGD-M is the Nesterov-accelerated SGD introduced in Sec. 3.2, also validating the superiority of our Win acceleration. Besides, our accelerated algorithms, i.e. SGD-Win, AdamW-Win and LAMB-Win, beat several other optimizers, e.g. AdaBound, Radam (Liu et al., 2019), Nadam, Padam (Chen et al., 2021) and AdaBelief, in which Nadam uses Nesterov acceleration to estimate its first-order gradient moment. Actually, LAMB-Win sets a new SoTA top-1 accuracy on ResNet18. All these results show the strong compatibility and superiority of our Win-acceleration in adaptive algorithms.

Results on ResNet50&101. Here we adopt the training setting in (Wightman et al., 2021) to train ResNet50&101, as this setting uses stronger data augmentation and largely improves CNNs' performance. See augmentation details and our algorithmic hyper-parameter settings in Appendix A. Here LAMB is the default optimizer because of its higher performance than other optimizers caused by the stronger augmentations (Wightman et al., 2021). All optimizers in Table 2 are under this setting.

Table 2 reports the top-1 accuracy of the compared optimizers on ImageNet. By comparison, one can observe that our accelerated algorithms consistently outperform their corresponding non-accelerated version. For example, across the three training epoch settings on ResNet50 and ResNet101, LAMB-Win always achieves remarkable improvement over LAMB which is the official optimizer for this training recipe. Specifically, LAMB-Win makes  $0.7\%$  average improvement over LAMB on both ResNet50 and ResNet101. For AdamW-Win and Adam-Win, they also respectively improve their

Table 3: ImageNet top-1 accuracy (%) of ViT and PoolFormer whose default optimizers are both AdamW. * and ◆ are respectively reported in (Touvron et al., 2021) and (Yu et al., 2021).  

<table><tr><td rowspan="2">Epoch</td><td colspan="3">ViT-S</td><td colspan="3">ViT-B</td><td colspan="3">PoolFormer-S12</td></tr><tr><td>150</td><td>300</td><td>avg.</td><td>150</td><td>300</td><td>avg.</td><td>150</td><td>300</td><td>avg.</td></tr><tr><td>SGD-M</td><td>77.4</td><td>79.4</td><td>78.4</td><td>79.6</td><td>80.0</td><td>79.8</td><td>69.7</td><td>74.3</td><td>72.0</td></tr><tr><td>SGD-Win</td><td>78.1</td><td>80.1</td><td>79.1+0.7</td><td>80.4</td><td>80.8</td><td>80.6+0.8</td><td>71.1</td><td>74.5</td><td>72.8+0.8</td></tr><tr><td>Adam</td><td>77.3</td><td>79.3</td><td>78.3</td><td>79.0</td><td>79.7</td><td>79.4</td><td>74.3</td><td>76.3</td><td>75.3</td></tr><tr><td>Adam-Win</td><td>78.6</td><td>80.2</td><td>79.4+1.1</td><td>80</td><td>80.5</td><td>80.3+0.9</td><td>75.6</td><td>77.1</td><td>76.4+1.1</td></tr><tr><td>AdamW</td><td>78.3</td><td>79.8*</td><td>79.1</td><td>79.5</td><td>81.8*</td><td>80.7</td><td>75.2</td><td>77.1*</td><td>76.2</td></tr><tr><td>AdamW-Win</td><td>79.3</td><td>80.8</td><td>80.1+1.0</td><td>81.0</td><td>82.2</td><td>81.6+0.9</td><td>76.7</td><td>77.6</td><td>77.2+1.0</td></tr><tr><td>LAMB</td><td>78.0</td><td>79.6</td><td>78.8</td><td>80.3</td><td>80.8</td><td>80.6</td><td>75.4</td><td>77.4</td><td>76.4</td></tr><tr><td>LAMB-Win</td><td>79.3</td><td>80.6</td><td>80.0+1.2</td><td>81.0</td><td>81.4</td><td>81.2+0.6</td><td>76.7</td><td>78.0</td><td>77.4+1.0</td></tr></table>

![](images/6d1df64a603e14496633725f061ff21f204414cb745c0a5e3db65013a5046bf6.jpg)

![](images/3e3efddcdfff77c499b9d6c2767e20fb35ff78ad3cf3d8174abb66bd55a8c1d5.jpg)

![](images/1dfc79ce25abdff8dcf9f318b1dd29fac7daaaf46f8a9790a14114cef3517f99.jpg)

![](images/b2dddf3c76b6c40a3ba4dec49c06adca777cd79bf88e10bef2894899c2873621.jpg)

![](images/035026cc961b83a223640143eb88c37d7baa51f609d01c3a3d4cf6340699f23f.jpg)  
Figure 1: Visualization of training and test losses on ImageNet. In all figures, training loss is larger than test one, as training data use random augmentations, e.g. random crop and clip, while test data only adopt the centralization crop which eases the recognition difficulty and thus has small loss.

![](images/a676bbbd833efd4bc96ea10986bf7b6f715d2dfc7fc715a4d891ac02d3db6d1e.jpg)

![](images/ebccba61f9c41087421382004ee7427f5c0f7c7790133f342df168a9cad48045.jpg)

![](images/8d1b2f250fb3ae4d3cb3bf069feda76b0417cc0371a4c5a51613806747669311.jpg)

counterparts by  $0.7\%$  and  $0.6\%$  on ResNet50,  $1.2\%$  and  $0.6\%$  on ResNet101. SGD-Win also makes  $2.5\%$  and  $0.8\%$  overall improvement over heavy-ball accelerated SGD (SGD-H) and Nesterov accelerated SGD (SGD-M) on ResNet50, and also has similar advantage on ResNet101. These improvements are not trivial because of the following two reasons. 1) Since the performance is already high and may approach the model limit, it is already very hard to make very large improvement. This can be testified by the fact that in (Wightman et al., 2021), using LAMB to train ResNet50 for even 600 epochs only gives  $80.4\%$  top-1 accuracy. In contrast, our accelerated LAMB-Win uses 300 epochs (only half computational cost) to achieve  $80.2\%$ . 2) By comparing the previous optimizers, including SAM, SGD-M, Adam, AdamW and LAMB, one can observe smaller accuracy gap  $(\leq 0.2\%)$  between the best optimizer and the runner-up. For example, on ResNet101, the SoTA optimizer, i.e. SAM, only makes  $0.1\%$  average improvement over the runner-up, namely LAMB. All these comparisons show the non-travail improvement of our accelerated algorithms over their counterparts, and also the superiority of our acceleration technique.

Results on ViTs. We follow the widely used official training setting of ViTs (Touvron et al., 2021; Yu et al., 2021). To evaluate the performance of our accelerated algorithms, we select two popular and representative ViT architectures, including ViT (Dosovitskiy et al., 2020) and PoolFormer (Yu et al., 2021). See the training setting and our hyper-parameter settings in Appendix A.

We test our accelerated algorithms under different model sizes and different training epochs, and report the results in Table 3. One can find that since AdamW and LAMB use the decoupled weight decay, they enjoy better performance than SGD and Adam, which is also observed in other works, e.g. (Xiao et al., 2021; Nado et al., 2021). Moreover, under different training settings, our accelerated algorithms consistently outperform the corresponding non-accelerated counterparts. Specifically, compared the default AdamW optimizer on both ViT and PoolFormer, our accelerated AdamW-Win respectively makes about  $1.0\%$ ,  $0.9\%$ ,  $1.0\%$  average improvement under the two training epoch settings on ViT-S, ViT-B and PoolFormer-S12. For Adam-Win and LAMB-Win, one can also observe their remarkable improvements on the three ViT backbones. Moreover, our accelerated SGD-Win also outperforms the Nesterov-accelerated SGD denoted as "SGD-M" by non-trivial margins under all settings. All these results are consistent with the observations on ResNets, and they together demonstrate the advantage of our accelerated optimizers for deep network training.

Results Analysis. Here we investigate the convergence behaviors of our accelerated algorithms, and aim to explain their better test performance over their non-accelerated counterparts. In Fig. 1, we plot the curves of training and test losses along with the training epochs on ResNet18 and ViT-B. One can find that our accelerated algorithms, e.g. AdamW-Win, show much faster convergence behaviors than their non-accelerated counterparts, e.g. AdamW. Moreover, SGD-Win also converges faster than Nesterove-accelerated SGD, i.e. SGD-M. We also plot the curves of test accuracy in Fig. 2, showing the superior convergence speed of AdamW-Win and LAMB-Win over their non-accelerated versions. Fig. 3 in Appendix A also reveals SGD-Win and

Adam-Win enjoy faster convergence than their non-accelerated counterparts in terms of test accuracy. So these faster convergence behaviors could contribute to our accelerated algorithms for their higher performance over non-accelerated counterparts under the same computational cost.

Robust Analysis. For the only extra hyperparameter  $\bar{\eta}_k$  in our accelerated algorithms over their non-accelerated counterparts, in experiments, we always set  $\bar{\eta}_k = \gamma \eta_k$ , where  $\gamma = 2$  determines the relation between the reckless step  $\bar{\eta}_k$  and the con

Table 4: Effects of  $\gamma$  to top-1 accuracy (\%) of AdamW-Win and LAMB-Win on ResNet50.

<table><tr><td>γ</td><td>1.5</td><td>2</td><td>3</td><td>4</td><td>6</td><td>8</td></tr><tr><td>AdamW-Win</td><td>77.9</td><td>78.0</td><td>78.0</td><td>77.9</td><td>78.1</td><td>78.0</td></tr><tr><td>LAMB-Win</td><td>78.3</td><td>78.4</td><td>78.4</td><td>78.4</td><td>78.5</td><td>78.3</td></tr></table>

servative step  $\eta_{k}$ . Here we investigate the effects of  $\gamma$  to the accelerated algorithms on ResNet50 by taking AdamW-Win and LAMB-Win as examples because of their superior performance. Table 4 shows the stable performance of AdamW-Win and LAMB-Win when tuning  $\gamma$  in a relatively large range, thus testifying the robustness of AdamW-Win and LAMB-Win to the hyper-parameter  $\gamma$ .

# 5.2 RESULTS ON NATURAL LANGUAGE MODELING TASKS

Results on LSTM. We follow AdaBelief to test our accelerated algorithms via training three-layered LSTM (Schmidhuber et al., 1997) on the Penn TreeBank dataset (Marcinkiewicz, 1994) for 200 epochs. See optimization and training details in Appendix A.

From Table 5, one can observe that our Win-accelerated algorithms consistently surpass the corresponding non-accelerated counterparts, and actually bring 1.2 overall average perplexity improvement over the four non-accelerated counterparts.

Table 5: Test perplexity of LSTM on Penn Treebank. * is reported by AdaBelief (Zhuang et al., 2020).

<table><tr><td>AdaBound</td><td>63.6*</td><td>Radam</td><td>70.0*</td></tr><tr><td>Yogi</td><td>67.5*</td><td>AdaBelief</td><td>61.2*</td></tr><tr><td>SGD-H</td><td>67.4</td><td>Padam</td><td>63.2*</td></tr><tr><td>SGD-M</td><td>63.8*</td><td>Adam</td><td>64.3*</td></tr><tr><td>SGD-Win</td><td>61.6+2.2</td><td>Adam-Win</td><td>62.7+1.6</td></tr><tr><td>AdamW</td><td>67.0*</td><td>LAMB</td><td>66.8</td></tr><tr><td>AdamW-Win</td><td>66.5+0.5</td><td>LAMB-Win</td><td>66.2+0.6</td></tr></table>

Results on Transformer-XL. We adopt a widely used language sequence model, i.e. Transformer-XL (Dai et al., 2019), to further evaluate the performance of our accelerated algorithms. Since 1) Adam is the most popular and used optimizer in NLP models, including Transformer-XL, and 2) our limited resource cannot well tune the hyperparameters of other optimizers in Sec. 5.1, we take Adam as an example to show the superiority of our accelerated

algorithms. Follow the official setting of Transformer-XL-base, we use Adam-Win with the default hyper-parameters of Adam on the WikiText-103 dataset. See more details in Appendix A.

Table 6: Test PPL of Transformer-XLbase on WikiText-103 where Adam is the official optimizer. * is reported in the official implementation.

<table><tr><td rowspan="2">Transformer-XL</td><td colspan="4">Training Steps</td></tr><tr><td>50k</td><td>100k</td><td>200k</td><td>avg.</td></tr><tr><td>Adam</td><td>28.5</td><td>25.5</td><td>24.2*</td><td>26.7</td></tr><tr><td>Adam-Win</td><td>26.7</td><td>25.0</td><td>24.0</td><td>25.2+1.5</td></tr></table>

Table 6 shows that under different training steps, our accelerated Adam-Win always achieves lower test PPL than the official Adam optimizer. Specifically, it improves 1.5 average test PPL over Adam on the three test cases. All these results are consistent with observations on vision tasks, and they together demonstrate the advantages of our accelerated algorithms.

# 6 CONCLUSION

In this work, we adopt proximal point method to derive a weight-decay-integrated Nesterov acceleration for AdamW and Adam, and extend it to LAMB and SGD. Moreover, we prove the convergence of our accelerated algorithms, i.e. accelerated AdamW, Adam and SGD, and observe the superiority of the accelerated Adam-type algorithm over the vanilla ones in terms of stochastic gradient complexity. Finally, experimental results validate the advantages of our accelerated algorithms.

![](images/b8e4e4e288d366f972021c64aa8b7dcfc70af85a05d55aae4e64df03b67171d5.jpg)  
Figure 2: Test accuracy curves of AdamW-Win and LAMB-Win on ResNet18.

# REFERENCES

Kwangjun Ahn and Suvrit Sra. Understanding nesterov's acceleration via proximal point method. In Symposium on Simplicity in Algorithms (SOSA), pp. 117-130. SIAM, 2022.  
Zeyuan Allen-Zhu and Lorenzo Orecchia. Linear coupling: An ultimate unification of gradient and mirror descent. arXiv preprint arXiv:1407.1537, 2014.  
Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. arXiv preprint arXiv:1912.02365, 2019.  
Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Ayush Sekhari, and Karthik Sridharan. Second-order information in non-convex stochastic optimization: Power and limitations. In Conference on Learning Theory, pp. 242–299. PMLR, 2020.  
Nikhil Bansal and Anupam Gupta. Potential-function proofs for gradient methods. Theory of Computing, 15(1):1-32, 2019.  
Yoshua Bengio, Nicolas Boulanger-Lewandowski, and Razvan Pascanu. Advances in optimizing recurrent networks. In 2013 IEEE international conference on acoustics, speech and signal processing, pp. 8624-8628. IEEE, 2013.  
Augustin Cauchy et al. Méthode générale pour la résolution des systèmes d'équations simultanées. Comp. Rend. Sci. Paris, 25(1847):536-538, 1847.  
Jinghui Chen, Dongruo Zhou, Yiqi Tang, Ziyan Yang, Yuan Cao, and Quanquan Gu. Closing the generalization gap of adaptive gradient methods in training deep neural networks. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, pp. 3267-3275, 2021.  
Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 702-703, 2020.  
Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 2978-2988, 2019.  
J. Deng, W. Dong, R. Socher, L. Li, K. Li, and F. Li. Imagenet: A large-scale hierarchical image database. In Proc. IEEE Conf. Computer Vision and Pattern Recognition, pp. 248-255, 2009.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.  
Timothy Dozat. Incorporating nesterov momentum into adam. 2016.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(7), 2011.  
Zhishuai Guo, Yi Xu, Wotao Yin, Rong Jin, and Tianbao Yang. A novel convergence analysis for algorithms of the adam family and beyond. arXiv e-prints, pp. arXiv-2104, 2021.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
R. Johnson and T. Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Proc. Conf. Neural Information Processing Systems, pp. 315-323, 2013.  
Junhyung Lyle Kim, Panos Toulis, and Anastasios Kyrillidis. Convergence and stability of the stochastic proximal point algorithm with momentum. In Learning for Dynamics and Control Conference, pp. 1034-1047. PMLR, 2022.

D. Kingma and J. Ba. Adam: A method for stochastic optimization. Int'l Conf. Learning Representations, 2014.  
Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. In International Conference on Learning Representations, 2019.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10012-10022, 2021.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018.  
Liangchen Luo, Yuanhao Xiong, Yan Liu, and Xu Sun. Adaptive gradient methods with dynamic bound of learning rate. In International Conference on Learning Representations, 2018.  
Liangchen Luo, Yuanhao Xiong, Yan Liu, and Xu Sun. Adaptive gradient methods with dynamic bound of learning rate. arXiv preprint arXiv:1902.09843, 2019.  
Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. Using Large Corpora, 273, 1994.  
Jean-Jacques Moreau. Proximate et dualité dans un espace hilbertien. Bulletin de la Société mathématique de France, 93:273-299, 1965.  
Zachary Nado, Justin M Gilmer, Christopher J Shallue, Rohan Anil, and George E Dahl. A large batch optimizer reality check: Traditional, generic optimizers suffice across batch sizes. arXiv preprint arXiv:2102.06356, 2021.  
Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2003.  
Yuri Nesterov et al. Lectures on convex optimization, volume 137. Springer, 2018.  
Boris T Polyak. Some methods of speeding up the convergence of iteration methods. Ussr computational mathematics and mathematical physics, 4(5):1-17, 1964.  
Sashank J Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. arXiv preprint arXiv:1904.09237, 2019.  
H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 22(3):400-407, 1951.  
R Tyrrell Rockafellar. Monotone operators and the proximal point algorithm. SIAM journal on control and optimization, 14(5):877-898, 1976.  
T. Sainath, A. Mohamed, B. Kingsbury, and B. Ramabhadran. Deep convolutional neural networks for LVCSR. In ICASSP, pp. 8614-8618. IEEE, 2013.  
Jürgen Schmidhuber, Sepp Hochreiter, et al. Long short-term memory. Neural Comput, 9(8):1735-1780, 1997.  
Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In International conference on machine learning, pp. 1139-1147. PMLR, 2013.  
Tieleman Tijmen and Hinton Geoffrey. Lecture 6.5-rmsprop: Divide the gradient by a run-ning average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4, 2012.  
Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, pp. 10347-10357. PMLR, 2021.

Ross Wightman, Hugo Touvron, and Hervé Jégou. Resnet strikes back: An improved training procedure in timm. arXiv preprint arXiv:2110.00476, 2021.  
Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nati Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. Advances in neural information processing systems, 30, 2017.  
Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Dólar, and Ross Girshick. Early convolutions help transformers see better. Advances in Neural Information Processing Systems, 34:30392-30400, 2021.  
Xingyu Xie, Pan Zhou, Huan Li, Zhouchen Lin, and Shuicheng Yan. Adan: Adaptive nesterov momentum algorithm for faster optimizing both cnns and vits. A x r i v, 2022.  
Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. In International Conference on Learning Representations, 2019.  
Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. arXiv preprint arXiv:2111.11418, 2021.  
Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE International Conference on Computer Vision, pp. 6023-6032, 2019.  
Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018.  
Dongruo Zhou, Jinghui Chen, Yuan Cao, Yiqi Tang, Ziyan Yang, and Quanquan Gu. On the convergence of adaptive gradient methods for nonconvex optimization. arXiv preprint arXiv:1808.05671, 2018.  
Pan Zhou, Hanshu Yan, Xiaotong Yuan, Jiashi Feng, and Shuicheng Yan. Towards understanding why lookahead generalizes better than sgd and beyond. Advances in Neural Information Processing Systems, 34:27290-27304, 2021.  
Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar C Tatikonda, Nicha Dvornek, Xenophon Papademetris, and James Duncan. Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. Advances in Neural Information Processing Systems, 33:18795-18806, 2020.
