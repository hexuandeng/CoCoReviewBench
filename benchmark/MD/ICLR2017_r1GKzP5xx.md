# RECURRENT NORMALIZATION PROPAGATION

César Laurent, Nicolas Ballas & Pascal Vincent*

Montreal Institute for Learning Algorithms (MILA)

Département d'Informatique et de Recherche Opérationnelle

Université de Montréal

Montréal, Québec, Canada

{firstname_lastname}@umontreal.ca

# ABSTRACT

We propose a LSTM parametrization that preserves the means and variances of the hidden states and memory cells across time. While having training benefits similar to Recurrent Batch Normalization and Layer Normalization, it does not need to estimate statistics at each time step, therefore, requiring fewer computations overall. We also investigate the parametrization impact on the gradient flows and present a way of initializing the weights accordingly.

We evaluate our proposal on language modelling and image generative modelling tasks. We empirically show that it performs similarly or better than other recurrent normalization approaches, while being faster to execute.

# 1 INTRODUCTION

Recurrent neural network have shown remarkably good performances for sequential modelling tasks including machine translation (Bahdanau et al., 2015), visual captioning (Xu et al., 2015; Yao et al., 2015) or question answering (Hermann et al., 2015). However, such models remain notoriously hard to train with gradient backpropagation. As the number of time steps in the input sequence increases, the contractive or expanding effects associated with the state-to-state transformation at each time step can shrink or grow exponentially, leading respectively to vanishing or exploding gradients (Hochreiter, 1991; Bengio et al., 1994; Pascanu et al., 2012). In particular, with gradient vanishing, states at a given time are not influenced by changes happening much earlier in the sequence, preventing the model from learning long-term dependencies.

While the long-term dependencies problem is unsolvable in absolute (Hochreiter, 1991; Bengio et al., 1994), different RNN parameterizations, such as LSTM or GRU (Hochreiter & Schmidhuber, 1997; Cho et al., 2014) can help mitigate it. Furthermore, the LSTM parametrization has been recently extended to include layer-wise normalization (Cooijmans et al., 2016; Ba et al., 2016), building upon Batch Normalization (BN) (Ioffe & Szegedy, 2015). By normalizing the hidden state distributions to a fix scale and shift through the different time steps, normalized LSTMs have been shown to ease training, resulting in a parametrization that converges faster than a standard LSTM.

However, normalized LSTM introduces extra-computations as it involves standardizing the hidden states, enforcing their means and variances at each time step. By contrast, we propose a LSTM reparametrization that allows by construction to cheaply preserve the normalization of the hidden states through time. Our approach can be seen as the recurrent counterpart to the recent normalization propagation applied in feedforward network (Arpit et al., 2016). It results in faster training convergence similar to Layer Normalization (LN) and Recurrent Batch Normalization while requiring fewer operations per timestep and generalizing naturally to variable length sequences.

In addition, we investigate the impact of our parametrization, and more generally of normalized LSTM, on the vanishing and exploding gradient problems. We observe that layerwise normalization provides a direct way to orient LSTM behavior toward either gradient explosion or vanishing, and therefore biases the LSTM either towards reliably storing bits of information throughout time or allowing it to be more sensitive to new input changes.

We empirically validate our proposal on character-level language modelling on the Penn Treebank corpus (Marcus et al., 1993) and on image generative modelling, applying our normalisation to the DRAW architecture (Gregor et al., 2015).

The paper is structured as follows: section 2 provides a brief overview of the Batch-Normalized LSTM, in section 3 we derive our Normalized LSTM, section 4 investigates the impact of such normalization on the gradient flow, section 5 presents some experimental results, and we conclude in section 5.

# 2 PRE-REQUISITS

# 2.1 BN-LSTM

Batch-Normalized Long Short-Term Memory (BN-LSTM) (Cooijmans et al., 2016) is a reparametrization of LSTM that takes advantage of Batch Normalization (BN) to address the Covariate Shift (Shimodaira, 2000) occurring between time steps. Changes in the LSTM output at one time-step are likely to cause correlated changes in the summed inputs of the sequence next timesteps. This Temporal Covariate Shift can slow down the training process as the parameters of the model must not only be updated to minimize the cost of the task at hand but also adapt to the changing distribution of the inputs. In other words, the latter time steps in a LSTM need to account for the shifting distribution of the previous hidden states.

BN-LSTM proposes to reduce this temporal covariate shift by fixing the mean and the variance at each time step, relying on the BN transform

$$
\operatorname {B N} (\mathbf {x}; \gamma , \beta) = \gamma \odot \frac {\mathbf {x} - \widehat {\mathbb {E}} [ \mathbf {x} ]}{\sqrt {\operatorname {V a r} [ \mathbf {x} ] + \epsilon}} + \beta \tag {1}
$$

where  $\widehat{\mathbb{E}}[\mathbf{x}],\widehat{\mathrm{Var}}[\mathbf{x}]$  are the activation mean and variance estimated from the mini-batch samples. Given an input sequence  $\mathbf{X} = (\mathbf{x}_1,\mathbf{x}_2,\dots ,\mathbf{x}_T)$ , the BN-LSTM defines a sequence of hidden states  $\mathbf{h}_t$  and memory cell states  $\mathbf{c}_t$  according to

$$
\left( \begin{array}{l} \tilde {\mathbf {i}} _ {t} \\ \tilde {\mathbf {f}} _ {t} \\ \tilde {\mathbf {o}} _ {t} \\ \tilde {\mathbf {g}} _ {t} \end{array} \right) = \mathrm {B N} \left(\mathbf {W} _ {x} \mathbf {x} _ {t}; \gamma_ {x}, \beta_ {x}\right) + \mathrm {B N} \left(\mathbf {W} _ {h} \mathbf {h} _ {t - 1}; \gamma_ {h}, \beta_ {h}\right) + \mathbf {b} \tag {2}
$$

$$
\mathbf {c} _ {t} = \sigma (\tilde {\mathbf {i}} _ {t}) \odot \tanh  \left(\tilde {\mathbf {g}} _ {t}\right) + \sigma (\tilde {\mathbf {f}} _ {t}) \odot \mathbf {c} _ {t - 1} \tag {3}
$$

$$
\mathbf {h} _ {t} = \sigma \left(\tilde {\mathbf {o}} _ {t}\right) \odot \tanh  \left(\mathrm {B N} \left(\mathbf {c} _ {t}; \gamma_ {c}, \beta_ {c}\right)\right), \tag {4}
$$

where  $\mathbf{W}_h\in \mathbb{R}^{d_h\times 4d_h},\mathbf{W}_x\in \mathbb{R}^{d_x\times 4d_h},\mathbf{b}\in \mathbb{R}^{4d_h}$  and the initial states  $\mathbf{h}_0\in \mathbb{R}^{d_h},\mathbf{c}_0\in \mathbb{R}^{d_h}$  are model parameters.  $\sigma$  is the logistic sigmoid function, and  $\odot$  denotes the Hadamard product. Ba et al. (2016) latter extended this parametrization by estimating the normalizing statistics  $(\widehat{\mathbb{E}} [\mathbf{x}],\widehat{\mathrm{Var}} [\mathbf{x}])$  using the different feature channels rather than mini-batch samples in order to naturally generalize to variable length sequences.

# 2.2 NORMALIZATION PROPAGATION

While increasing the training convergence speed relatively to a standard LSTM (Cooijmans et al., 2016), BN-LSTM needs to perform more computations per sample as it requires to compute  $3x$  the BN transform at each time step.

On the other hand, Normalization Propagation (Norm Prop) (Arpit et al., 2016) aims at preserve the normalization of the input throughout the network. Unlike BN, the normalization doesn't rely on the statistics of the mini-batch. Instead, it is the structure of the network itself that maintains the normalization. We therefore propose a LSTM reparametrization that preserves the normalization through the different time steps in order to avoid those extra computation.

# 3 NORMALIZED LSTM

While Norm Prop properties are appealing for recurrent models, its application to LSTM is not straightforward due to the memory cell structure. In this section we show how to derive a LSTM reparametrization that preserves normalization of the state  $\mathbf{h}_t$  through time.

# 3.1 CONSTRUCTION THE NORMALIZED LSTM

Following (Arpit et al., 2016; Salimans & Kingma, 2016), we first compensate for the distribution changes induced by the weight matrices in the gates and cell candidate  $\mathbf{g}_t$  computations

$$
\left( \begin{array}{l} \tilde {\mathbf {i}} _ {t} \\ \tilde {\mathbf {f}} _ {t} \\ \tilde {\mathbf {o}} _ {t} \\ \tilde {\mathbf {g}} _ {t} \end{array} \right) = \gamma_ {x} \frac {\mathbf {W} _ {x}}{| | \mathbf {W} _ {x , i} | | _ {2}} \mathbf {x} _ {t} + \gamma_ {h} \frac {\mathbf {W} _ {h}}{| | \mathbf {W} _ {h , i} | | _ {2}} \mathbf {h} _ {t - 1} + \mathbf {b}. \tag {5}
$$

where  $||\mathbf{W}_{..,i}||_2$  is the vector of L2-norm of each line of the matrix and  $\gamma_{x}$  and  $\gamma_{h}$  are the trainable rescaling factors that restore the representation power lost in the rescaling of the weight matrices.

To preserve the constant error carousel mechanism mechanism of the LSTM, we use the usual cell update,

$$
\mathbf {c} _ {t} = \sigma (\tilde {\mathbf {i}} _ {t}) \odot \tanh  (\tilde {\mathbf {g}} _ {t}) + \sigma (\tilde {\mathbf {f}} _ {t}) \odot \mathbf {c} _ {t - 1} \tag {6}
$$

The evolution of  $\mathbf{c}_t$  through time can be seen as a geometric series, with  $\sigma(\tilde{\mathbf{f}}_t)$  as constant ratio. Since  $\sigma(\cdot)$  is upper-bounded by (and in practice smaller than) 1,  $\mathbf{c}_t$  will converge in expectation to a fixed value. This is the reason why in BN-LSTM the mini-batch statistics converge to a fixed value after a few time steps (Cooijmans et al., 2016). Moreover, if we consider that  $\tilde{\mathbf{1}}_t$ ,  $\tilde{\mathbf{f}}_t$ ,  $\tilde{\mathbf{g}}_t$  and  $\mathbf{c}_{t-1}$  are independants, we can use the variance product rule of two independent random variables  $X$  and  $Y$

$$
\operatorname {V a r} [ X Y ] = \operatorname {V a r} [ X ] \operatorname {V a r} [ Y ] + \operatorname {V a r} [ X ] \mathbb {E} [ Y ] ^ {2} + \operatorname {V a r} [ Y ] \mathbb {E} [ X ] ^ {2} \tag {7}
$$

to compute  $\operatorname{Var}[\mathbf{c}_t]$ . Considering that  $\mathbb{E}[\tanh (\tilde{\mathbf{g}}_t)] = 0$  and assuming that the cell has converged i.e.  $\operatorname{Var}[\mathbf{c}_t] = \operatorname{Var}[\mathbf{c}_{t - 1}]$ , we have

$$
\operatorname {V a r} \left[ \mathbf {c} _ {t} \right] = \operatorname {V a r} \left[ \tanh  \left(\tilde {\mathbf {g}} _ {t}\right) \right] \frac {\operatorname {V a r} \left[ \sigma \left(\tilde {\mathbf {i}} _ {t}\right) \right] + \mathbb {E} \left[ \sigma \left(\tilde {\mathbf {i}} _ {t}\right) \right] ^ {2}}{1 - \operatorname {V a r} \left[ \sigma \left(\tilde {\mathbf {f}} _ {t}\right) \right] - \mathbb {E} \left[ \sigma \left(\tilde {\mathbf {f}} _ {t}\right) \right] ^ {2}} \tag {8}
$$

We can therefore analytically or numerically compute the mean and variance of each of those elements, assuming that both input  $\mathbf{x}_t$  and hidden state  $\mathbf{h}_{t-1}$  are independent drawn from  $\mathcal{N}(0,1)$

$$
\mathbb {E} \left[ \mathbf {y} _ {t} \right] = \mathbb {E} \left[ f \left(\left(\gamma_ {x} + \gamma_ {h}\right) z\right) \right], \quad z \sim \mathcal {N} (0, 1) \tag {9}
$$

$$
\operatorname {V a r} \left[ \mathbf {y} _ {t} \right] = \operatorname {V a r} \left[ f \left(\left(\gamma_ {x} + \gamma_ {h}\right) z\right) \right], \quad z \sim \mathcal {N} (0, 1) \tag {10}
$$

where  $f(\cdot)$  is  $\tanh (\cdot)$  for  $\mathbf{g}_t$  and  $\sigma (\cdot)$  for the gates  $\mathbf{i}_t, \mathbf{o}_t, \mathbf{f}_t$ . We can then compute the value to which  $\mathrm{Var}[\mathbf{c}_t]$  converges. Using this variance estimate, we compensate  $\mathbf{c}_t$  in order to compute the next hidden state  $\mathbf{h}_t$ .

$$
\mathbf {h} _ {t} = \sigma (\tilde {\mathbf {o}} _ {t}) \odot \tanh  \left(\frac {\gamma_ {c} \mathbf {c} _ {t}}{\sqrt {\operatorname {V a r} [ \mathbf {c} _ {t} ]}}\right) \tag {11}
$$

Since we assumed that  $\mathrm{Var}[\mathbf{h}_{t - 1}] = 1$ , we need to correct for the variance induced by the product of the tanh with the output gate. Using again the variance product rule (equation 7) we obtain

$$
\operatorname {V a r} \left[ \mathbf {h} _ {t} \right] = \operatorname {V a r} \left[ \tanh  \left(\frac {\gamma_ {c} \mathbf {c} _ {t}}{\sqrt {\operatorname {V a r} \left[ \mathbf {c} _ {t} \right]}}\right) \right] (\operatorname {V a r} [ \sigma (\tilde {\mathbf {o}} _ {t}) ] + \mathbb {E} [ \sigma (\tilde {\mathbf {o}} _ {t}) ] ^ {2}) \tag {12}
$$

We can estimate this variance through similar computation than equation 10. Scaling  $\mathbf{h}_t$  with  $1 / \sqrt{\mathrm{Var}[\mathbf{h}_t]}$  ensure that its variance is 1 and so the propagation is maintained throughout the recurrence.

# 3.2 PROPOSED REPARAMETRIZATION

Using equations 5, 6 and 11, we propose the following reparametrization of the LSTM, simply called the Normalized LSTM

$$
\left( \begin{array}{l} \tilde {\mathbf {i}} _ {t} \\ \tilde {\mathbf {f}} _ {t} \\ \tilde {\mathbf {o}} _ {t} \\ \tilde {\mathbf {g}} _ {t} \end{array} \right) = \gamma_ {x} \frac {\mathbf {W} _ {x}}{| | \mathbf {W} _ {x , i} | | _ {2}} \mathbf {x} _ {t} + \gamma_ {h} \frac {\mathbf {W} _ {h}}{| | \mathbf {W} _ {h , i} | | _ {2}} \mathbf {h} _ {t - 1} + \mathbf {b} \tag {13}
$$

$$
\mathbf {c} _ {t} = \sigma (\tilde {\mathbf {i}} _ {t}) \odot \tanh  (\tilde {\mathbf {g}} _ {t}) + \sigma (\tilde {\mathbf {f}} _ {t}) \odot \mathbf {c} _ {t - 1} \tag {14}
$$

$$
\mathbf {h} _ {t} = \frac {1}{\sqrt {\operatorname {V a r} [ \mathbf {h} _ {t} ]}} \left[ \sigma (\tilde {\mathbf {o}} _ {t}) \odot \tanh  \left(\frac {\gamma_ {c} \mathbf {c} _ {t}}{\sqrt {\operatorname {V a r} [ \mathbf {c} _ {t} ]}}\right) \right] \tag {15}
$$

where  $\mathrm{Var}[\mathbf{c}_t]$  and  $\mathrm{Var}[\mathbf{h}_t]$  are computed using equations 8 and 12, respectively. Those two variances are estimated at the initialization of the neural network, and are then kept fixed during the training.

Note that the reparametrization of equation 13 is identical to Weight Normalization (Weigth Norm) (Salimans & Kingma, 2016). The main difference comes from equation 15, where we compensate for the variance of  $\mathbf{c}_t$ , the tanh and  $\sigma (\tilde{\mathbf{o}}_t)$ , which ensures a normalized propagation. Overall, this reparamertization is equivalent in spirit to the BN-LSTM, but it benefits from the same advantages that Norm Prop has over BN: There is no dependence on the mini-batch size and the computation is the same for training and inference. Also, the rescaling of the matrices can be done before the recurrence, leading to computation time closer to a vanilla LSTM.

# 3.3 WEIGHTS INITIALIZATION

With such reparametrization of the weight matrices, one can think that the scale of the initialization of the weights doesn't matter in the learning process anymore. It is actually true for the forward and backward computation of the layer

$$
y _ {i} = \frac {a \mathbf {W} _ {i}}{| | a \mathbf {W} _ {i} | | _ {2}} x = \frac {\mathbf {W} _ {i}}{| | \mathbf {W} _ {i} | | _ {2}} x \tag {16}
$$

$$
\frac {\partial y _ {i}}{\partial x} = \frac {a \mathbf {W} _ {i}}{| | a \mathbf {W} _ {i} | | _ {2}} = \frac {\mathbf {W} _ {i}}{| | \mathbf {W} _ {i} | | _ {2}} \tag {17}
$$

and since the variance of both foward and backward passes is fixed, using an initialization scheme such as Glorot (Glorot & Bengio, 2010) doesn't make sense with Norm Prop. However, the update of the parameters is affected by their scale:

$$
\frac {\partial y _ {i}}{\partial \mathbf {W} _ {i j}} \frac {\partial \mathcal {L}}{\partial y _ {i}} = \frac {1}{\left\| \mathbf {W} _ {i} \right\| _ {2}} \left[ x _ {j} - y _ {i} \frac {\mathbf {W} _ {i j}}{\left\| \mathbf {W} _ {i} \right\| _ {2}} \right] \frac {\partial \mathcal {L}}{\partial y _ {i}} \tag {18}
$$

The scale of the parameters affect the learning rate of the layer: the bigger the weights, the smaller the update. This induces a regularization effect in Norm Prop that is also present in BN (Ioffe & Szegedy, 2015). However, this could possibly be an issue for such parametrization: different initializations lead to different learning rates, and it is true even with adaptive step rules, such as Adam (Kingma & Ba, 2014). Moreover, the parameters that are not normalized (such as  $\gamma$  and b) aren't affected by this effect, and so they are not regularized. This is the reason why forcing the weight matrices to have a unit L2 norm of the lines, as proposed in Arpit et al. (2016), helps the training procedure.

To still benefit from the reduction of the learning rate, which is known to ease the optimization (Vogl et al., 1988), we propose to simply force the unit L2 norm of the lines of the matrices and combine it with a global learning rate decay schedule.

# 4 GRADIENT PROPAGATION IN NORMALIZED LSTM

In this section we study the gradient flow in the Normalized LSTM. Since this reparametrization is similar to the BN-LSTM, the analysis we do here can be transposed to the BN-LSTM case.

# 4.1 THE EXPLODING AND VANISHING GRADIENTS PROBLEM

Given an input sequence  $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$ , we consider a recurrent network, parametrized by  $\theta$ , that defines a sequence of hidden states  $\mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, \mathbf{x}_t)$  and cost function  $\mathcal{L}$  which evaluates the model performance on a given task. Such network is usually trained using backpropagation through time, where the backpropagation is applied on the time-unrolled model. The chain rule can be applied in order to compute the derivative of the loss  $\mathcal{L}$  with respect to parameters  $\theta$ .

$$
\frac {\partial \mathcal {L}}{\partial \theta} = \sum_ {1 \leq t \leq T} \frac {\partial \mathcal {L} _ {t}}{\partial \theta} = \sum_ {1 \leq t \leq T} \sum_ {1 \leq k \leq t} \frac {\partial \mathcal {L} _ {t}}{\partial \mathbf {h} _ {k}} \frac {\partial \mathbf {h} _ {k}}{\partial \mathbf {h} _ {t}} \frac {\partial \mathbf {h} _ {t}}{\partial \theta}. \tag {19}
$$

The factors  $\frac{\partial\mathbf{h}_k}{\partial\mathbf{h}_t} = \prod_{k\leq l\leq t}\frac{\partial\mathbf{h}_l}{\partial\mathbf{h}_{l - 1}}$  transports the error "in time" from step  $t$  back to step  $k$  and are also the cause of vanishing or exploding gradient in RNN (Pascanu et al., 2012). Indeed, if the Jacobian  $\frac{\partial\mathbf{h}_l}{\partial\mathbf{h}_{l - 1}}$  has singular value different from 1, the factor  $\frac{\partial\mathbf{h}_k}{\partial\mathbf{h}_t}$ , which is a product of  $t - k$  Jacobian matrices will either explode or vanish.

# 4.2 GRADIENT OF THE NORMALIZED LSTM

To study the gradient propagation of the Normalized LSTM, we first need to derive it. Using equation 13-15, we can write the gradient of  $\mathbf{h}_t$  with respect to  $\mathbf{h}_{t - 1}$

$$
\mathbf {a} _ {t} = \frac {1}{\sqrt {\operatorname {V a r} [ \mathbf {h} _ {t} ]}} \tanh  \left(\frac {\gamma_ {c} \mathbf {c} _ {t}}{\sqrt {\operatorname {V a r} [ \mathbf {c} _ {t} ]}}\right) \tag {20}
$$

$$
\frac {\partial \mathbf {h} _ {t}}{\partial \mathbf {h} _ {t - 1}} = \frac {\partial \mathbf {o} _ {t}}{\partial \mathbf {h} _ {t - 1}} \odot \mathbf {a} _ {t} + \mathbf {o} _ {t} \odot \frac {\partial \mathbf {a} _ {t}}{\partial \mathbf {h} _ {t - 1}} \odot \left[ \frac {\partial \mathbf {i} _ {t}}{\partial \mathbf {h} _ {t - 1}} \odot \mathbf {g} _ {t} + \mathbf {i} _ {t} \odot \frac {\partial \mathbf {g} _ {t}}{\partial \mathbf {h} _ {t - 1}} + \frac {\partial \mathbf {f} _ {t}}{\partial \mathbf {h} _ {t - 1}} \odot \mathbf {c} _ {t - 1} \right] \tag {21}
$$

As we can see in equation 21 with the normalization, the gradient depends not only on the derivative of the cell candidate, the gates and the output tanh, but also on the variance of  $\mathbf{h}_t$  and  $\mathbf{c}_t$ .

If we assume that  $\mathbf{h}_{t-1}$  and  $\mathbf{x}_t$  are independant, we can compute the variance of  $\mathbf{c}_t$ . Neglecting the weight matrices and the effect of the gates, we can write from equations 8 and 12

$$
\operatorname {V a r} \left[ \mathbf {c} _ {t} \right] \approx \operatorname {V a r} \left[ \mathbf {g} _ {t} \right] = \operatorname {V a r} \left[ \tanh  (z) \right], \quad z \sim \mathcal {N} \left(0, \gamma_ {x} ^ {2} + \gamma_ {h} ^ {2}\right) \tag {22}
$$

$$
\operatorname {V a r} \left[ \mathbf {h} _ {t} \right] = \operatorname {V a r} \left[ \tanh  (z) \right], \quad z \sim \mathcal {N} \left(0, \gamma_ {c} ^ {2} \left(\gamma_ {x} ^ {2} + \gamma_ {h} ^ {2}\right)\right) \tag {23}
$$

In both cases, the variance depends explicitly on the value of the different  $\gamma$ : The bigger the  $\gamma$ , the higher the variance. Neglecting again the weight matrices, we can now write the equations of the cell candidates  $\mathbf{g}_t$  and the gates  $\mathbf{i}_t, \mathbf{o}_t$  and  $\mathbf{f}_t$  with respect to  $\mathbf{h}_{t-1}$

$$
\frac {\partial \mathbf {g} _ {t}}{\partial \mathbf {h} _ {t - 1}} = \frac {\partial \tanh  (\tilde {\mathbf {g}} _ {t})}{\partial \tilde {\mathbf {g}} _ {t}} \frac {\partial \tilde {\mathbf {g}} _ {t}}{\partial \mathbf {h} _ {t - 1}} = \left(1 - \tanh  \left(\gamma_ {x} \mathbf {x} _ {t} + \gamma_ {h} \mathbf {h} _ {t - 1}\right) ^ {2}\right) \gamma_ {h} \tag {24}
$$

$$
\frac {\partial \mathbf {i} _ {t}}{\partial \mathbf {h} _ {t - 1}} = \frac {\partial \sigma (\tilde {\mathbf {i}} _ {t})}{\partial \tilde {\mathbf {i}} _ {t}} \frac {\partial \tilde {\mathbf {i}} _ {t}}{\partial \mathbf {h} _ {t - 1}} = \sigma \left(\gamma_ {x} \mathbf {x} _ {t} + \gamma_ {h} \mathbf {h} _ {t - 1}\right) \left(1 - \sigma \left(\gamma_ {x} \mathbf {x} _ {t} + \gamma_ {h} \mathbf {h} _ {t - 1}\right)\right) \gamma_ {h} \tag {25}
$$

The gradients of  $\mathbf{o}_t$  and  $\mathbf{f}_t$  are equivalent to equation 25. The effect of the  $\gamma$  here is double: They appear both in the activation function, where they control the saturation regime, and  $\gamma_h$  also appears as a multiplicative term in the gradient. They should therefore be small enough to prevent the activation from saturating too much, but at the same time  $\gamma_h$  can't be too small, because it can also make the gradients vanish. Putting it all together, we have

$$
\frac {\partial \mathbf {h} _ {t}}{\partial \mathbf {h} _ {t - 1}} = \frac {\partial \mathbf {o} _ {t}}{\partial \tilde {\mathbf {o}} _ {t}} \gamma_ {h} \odot \mathbf {a} _ {t} + \mathbf {o} _ {t} \odot \frac {\partial \mathbf {a} _ {t}}{\partial \tilde {\mathbf {a}} _ {t}} \frac {\gamma_ {c}}{\sqrt {\operatorname {V a r} [ \mathbf {c} _ {t} ]}} \odot \gamma_ {h} \left[ \frac {\partial \mathbf {i} _ {t}}{\partial \tilde {\mathbf {i}} _ {t}} \odot \mathbf {g} _ {t} + \mathbf {i} _ {t} \odot \frac {\partial \mathbf {g} _ {t}}{\partial \tilde {\mathbf {g}} _ {t}} + \frac {\partial \mathbf {f} _ {t}}{\partial \tilde {\mathbf {f}} _ {t}} \odot \mathbf {c} _ {t - 1} \right] \tag {26}
$$

In this equations we can see that the diffrent  $\gamma$  directly scale the gradient, and they also control the saturation of the activation functions. Bad initialization of  $\gamma$  could thus lead to saturation or explosion regimes. Figure 1 shows the norm of the gradient with respect to  $\gamma_{x}$  and  $\gamma_{h}$  in a simulated LSTM. As we can see, one important parameter is the ratio between  $\gamma_{h}$  and  $\gamma_{x}$ : They control most of the propagation of the gradients. If  $\gamma_{x} > \gamma_{h}$ , the network will focus more on the input and so the gradients will tend to vanish more. On the other hand, if  $\gamma_{x} > \gamma_{h}$ , the network will tend have less vanishing gradients, but will focus less on its inputs.

![](images/21c63c9e6f8021df50d1c34d197c669971c60624620eb8cd81ed2abe61ddc205.jpg)  
Figure 1: Norm of the gradients for one time step in an LSTM with respect to  $\gamma_{x}$  and  $\gamma_{h}$  (simulation). Left:  $\gamma_{c} = 0.1$ . Right:  $\gamma_{c} = 1.0$ .

![](images/1109fe9aa4354252c9cc2128360c7cd0775fb48639e3d7eb0b5f8f4281487d4d.jpg)

# 5 EXPERIMENTS

# 5.1 CHARACTER-LEVEL LANGUAGE MODELLING

The first task we explore is character-level language modelling on the Penn Treebank corpus (Marcus et al., 1993). The goal is to predict the next character of the sequence given the previous ones. We use the same splits as Mikolov et al. (2012) and the same training procedure as Cooijmans et al. (2016), ie we train on sequences of length 100, with random stating point. The model is a 1000 units LSTM followed by a softmax classifier. We use orthogonal initialization for the weight matrices. Because Norm Prop requires normalized inputs, we multiply the one-hot inputs vector with an untrained but fixed orthogonal matrix. This triks does not only help the optimization of Norm Prop, but also all other variants.

To compare the convergence properties of Norm Prop against LN and BN, we first ran experiments using Adam (Kingma & Ba, 2014) with learning rate 2e-3, exponential decay of 1e-3 and gradient clipping at 1.0. As explained in section 3.3, we rescale the matrices such that they have a unit norm on the lines. For Norm Prop, we use  $\gamma_{x} = \gamma_{h} = 2$  and  $\gamma_{c} = 1$ , for LN all the  $\gamma = 1.0$  and for BN all the  $\gamma = 0.1$ . The results are presented in Table 1 and in Figure 2.

<table><tr><td>Model</td><td>Validation</td><td>Time</td></tr><tr><td>Baseline</td><td>1.455</td><td>386</td></tr><tr><td>Weight Norm</td><td>1.438</td><td>402</td></tr><tr><td>Batch Norm</td><td>1.433</td><td>545</td></tr><tr><td>Layer Norm</td><td>1.439</td><td>530</td></tr><tr><td>Norm Prop</td><td>1.422</td><td>413</td></tr></table>

Table 1: Perplexity (bits-per-character) on sequences of length 100 from the Penn Treebank validation set, and training time (seconds) per epoch.

To show the potential of Norm Prop against other state-of-the-art system, we followed Ha et al. (2016) and apply dropout on both the input and output layer  $(p = 0.1)$  and recurrent dropout inside the LSTM  $(p = 0.1)$ . We also used the Batch Data Normalization scheme presented by Arpit et al. (2016), so we standardize each input example using the mini-batch statistics and use population statistics at inference time. Finally, we also reduce the learning rate decay to 1e-4, to compensate for the fact that a network with dropout needs more time to train. The results are presented in Table 2.

As we can see in Figure 2 and in Table 1, Norm Prop compares really well against the other reparametrization. Also Norm Prop is roughly  $30\%$  computationally faster than BN and LN. LN shows better optimization performances, but also overfits more. We also see that both optimization and generalization are better than the ones from Weight Norm, which shows the importance of compensating for the variance of  $\mathbf{c}_t$  and  $\mathbf{h}_t$ . Moreover, although Norm Prop doesn't combine well with

![](images/317f7a39832687d2479c377b920e36a7074fa3c8758794481b1651e64ae303c4.jpg)  
Figure 2: Perplexity (bits-per-character) on sequences of length 100 from the Penn Treebank corpus. The dashed lines are the training curves, and the solid ones are the validation curves.

<table><tr><td>Model</td><td>Test</td></tr><tr><td>Recurrent Dropout LSTM (Semeniuta et al., 2016)</td><td>1.301</td></tr><tr><td>Zoneout LSTM (Krueger et al., 2016)</td><td>1.27</td></tr><tr><td>Layer Norm LSTM (Ha et al., 2016)</td><td>1.267</td></tr><tr><td>HyperLSTM (Ha et al., 2016)</td><td>1.265</td></tr><tr><td>Norm Prop LSTM (ours)</td><td>1.262</td></tr><tr><td>Layer Norm HyperLSTM (Ha et al., 2016)</td><td>1.250</td></tr></table>

Table 2: Perplexity (bits-per-character) of the full Penn Treebank test sequence.

dropout in feed-forward networks (Arpit et al., 2016), it works well with recurrent dropout, as we can see in Table 2. We believe it is because recurrent dropout is less affecting its output distribution than dropout, because we copy the variable at the previous time step instead of setting it to 0. With such regularization, Norm Prop compares well with other state-of-the-art models.

# 5.2 DRAW

The second task we explore is a generative modelling task on binarized MNIST (Larochelle & Murray, 2011) using the Deep Recurrent Attentive Writer (DRAW) (Gregor et al., 2015) architecture. DRAW is a variational auto-encoder, where both encoder and decoder are LSTMs, and has two attention mechanisms to select where to read and where to write.

We use Jörg Bornschein's implementation<sup>2</sup>, with the same hyper-parameters as Gregor et al. (2015), ie the read and write size are 2x2 and 5x5 respectively, the number of glimpses is 64, the LSTMs have 256 units and the dimention of  $z$  is 100. We use Adam with learning rate of 1e-2, exponential decay of 1e-3 and mini-batch size of 128. We use orthogonal initialization and force the norm of the lines of the matrices to be 1. For Norm Prop, we use  $\gamma_{x} = \gamma_{h} = \gamma_{c} = 0.5$ . The test variational bound for the first 100 epochs is presented in Figure 3.

As we can see in Figure 3, both Weight Norm and Norm Prop outperform the baseline network by a significant margin. Also, as expected, Norm Prop performs better than Weight Norm, showing one again the importance of the compensation of the variance of  $\mathbf{c}_t$  and  $\mathbf{h}_t$ . Table 3 shows the test variational bound after 200 epochs of training. Norm Prop also compares favorably against LN.

![](images/074f0167c0c6bd05585746b320824134fa0a4c0ac6699164b643dd853956f0f6.jpg)

Figure 3: Test negative log-likelihood on binarized MNIST.  

<table><tr><td>Model</td><td>DRAW</td></tr><tr><td>Baseline (ours)</td><td>84.30</td></tr><tr><td>Layer Norm (Ba et al., 2016)</td><td>82.09</td></tr><tr><td>Weight Norm (ours)</td><td>81.98</td></tr><tr><td>Norm Prop (ours)</td><td>81.17</td></tr></table>

Table 3: Test variational log likelihood (nats) after 200 epochs of training.

# 6 CONCLUSION

Based on the BN-LSTM, we have shown how to build a Normalized LSTM that is able to preserve the variance its output at each time step, by compensating for the variance of the cell and the hidden state. Such LSTM can be seen as the Norm Prop version of the BN-LSTM, and thus benefits from the same advantages that Norm Prop has over BN, while being way faster to compute. Also, we propose a scheme to initialize the weight matrices that takes into account the reparametrization. Moreover, we have derived the gradients of this LSTM and pointed out the importance of the initialization of the rescaling parameters. We have validated the performances of the Normalized LSTM on two different tasks, showing similar performances than BN-LSTM and LN-LSTM, while being significantly faster in computation time. Also, unlike the feedforward case, this architecture works well with recurrent dropout, leading to close to state-of-the-art performance on the character-level language modelling task.

Future work includes trying this architecture on more challenging tasks and also studying the impact of not keeping the variance estimates of the cell and the hidden states fixed during the learning process.

# ACKNOWLEDGMENTS

Part of this work was funded by Samsung. We used Theano (Theano Development Team, 2016), Blocks and Fuel (van Merrienboer et al., 2015) for our experiments. We also want to thanks Caglar Culcehre and Tim Coolijmans for the talks and Jörg Bornschein for his DRAW implementation.

# REFERENCES

D. Arpit, Y. Zhou, B. U Kota, and V. Govindaraju. Normalization propagation: A parametric technique for removing internal covariate shift in deep networks. arXiv preprint, 2016.  
J. L. Ba, J. R. Kiros, and G. E Hinton. Layer normalization. arXiv preprint, 2016.  
D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. *ICLR*, 2015.

Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. Neural Networks, IEEE Transactions on, 1994.  
K. Cho, B. Van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using rnnc encoder-decoder for statistical machine translation. arXiv preprint, 2014.  
T. Coolijmans, N. Ballas, C. Laurent, and A. Courville. Recurrent batch normalization. arXiv preprint, 2016.  
X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In Aistats, volume 9, pp. 249-256, 2010.  
K. Gregor, I. Danihelka, A. Graves, D. J. Rezende, and D. Wierstra. Draw: A recurrent neural network for image generation. arXiv preprint, 2015.  
D. Ha, A. Dai, and Q. V Le. Hypernetworks. arXiv preprint, 2016.  
K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman, and P. Blunsom. Teaching machines to read and comprehend. In NIPS, 2015.  
S. Hochreiter. Untersuchungen zu dynamischen neuronalen netzen. Master's thesis, 1991.  
S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8), 1997.  
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. CoRR, abs/1502.03167, 2015.  
D. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint, 2014.  
D. Krueger, T. Maharaj, J. Kramár, M. Pezeshki, N. Ballas, N. R. Ke, A. Goyal, Y. Bengio, H. Larochelle, A. Courville, et al. Zoneout: Regularizing rnns by randomly preserving hidden activations. arXiv preprint, 2016.  
H. Larochelle and I. Murray. The neural autoregressive distribution estimator. AISTATS, 2011.  
M. P. Marcus, M. Marcinkiewicz, and B. Santorini. Building a large annotated corpus of english: The penn treebank. Comput. Linguist., 1993.  
T. Mikolov, I. Sutskever, A. Deoras, H. Le, S. Kombrink, and J. Cernocky. Subword language modeling with neural networks. preprint, 2012.  
R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. arXiv preprint, 2012.  
T. Salimans and D. P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. arXiv preprint, 2016.  
S. Semeniuta, A. Severyn, and E. Barth. Recurrent dropout without memory loss. CoRR, abs/1603.05118, 2016.  
H. Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of statistical planning and inference, 2000.  
Theano Development Team. Theano: A Python framework for fast computation of mathematical expressions. arXiv preprint, 2016.  
B. van Merrienboer, D. Bahdanau, V. Dumoulin, D. Serdyuk, D. Warde-Farley, J. Chorowski, and Y. Bengio. Blocks and fuel: Frameworks for deep learning. CoRR, abs/1506.00619, 2015.  
T. P. Vogl, J. K. Mangis, A. K. Rigler, W. T. Zink, and D. L. Alkon. Accelerating the convergence of the back-propagation method. Biological Cybernetics, 59(4):257-263, 1988.  
K. Xu, J. Ba, R. Kiros, A. Courville, R. Salakhutdinov, R. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. arXiv preprint, 2015.  
L. Yao, A. Torabi, K. Cho, N. Ballas, C. Pal, H. Larochelle, and A. Courville. Describing videos by exploiting temporal structure. In ICCV, 2015.