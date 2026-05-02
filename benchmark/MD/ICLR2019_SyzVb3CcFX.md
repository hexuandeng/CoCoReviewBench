# TIME-AGNOSTIC PREDICTION:  
PREDICTING PREDICTABLE VIDEO FRAMES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Prediction is arguably one of the most basic functions of an intelligent system. In general, the problem of predicting events in the future or between two waypoints is exceedingly difficult. However, most phenomena naturally pass through relatively predictable bottlenecks—while we cannot predict the precise trajectory of a robot arm between being at rest and holding an object up, we can be certain that it must have picked the object up. To exploit this, we decouple visual prediction from a rigid notion of time. While conventional approaches predict frames at regularly spaced temporal intervals, our time-agnostic predictors (TAP) are not tied to specific times so that they may instead discover predictable "bottleneck" frames no matter when they occur. We evaluate our approach for future and intermediate frame prediction across three robotic manipulation tasks. Our predictions are not only of higher visual quality, but also correspond to coherent semantic subgoals in temporally extended tasks.

# 1 INTRODUCTION

Imagine taking a bottle of water and laying it on its side. Consider what happens to the surface of the water as you do this: which times can you confidently make predictions about? The surface is initially flat, then becomes turbulent, until it is flat again, as shown in Fig 1. Predicting the exact shape of the turbulent liquid is extremely hard, but its easy to say that it will eventually settle down.

Prediction is thought to be fundamental to intelligence (Bar, 2009; Clark, 2013; Hohwy, 2013). If an agent can learn to predict the future, it can take anticipatory actions, plan through its predictions, and use prediction as a proxy for representation learning. The key difficulty in prediction is uncertainty. Visual prediction approaches attempt to mitigate uncertainty by predicting iteratively in heuristically chosen small timesteps, such as, say,  $0.1s$ . In the bottle-tilting case, such approaches generate blurry images of the chaotic states at  $t = 0.1s$ ,  $0.2s$ , ..., and this blurriness compounds to make predictions unusable within a few steps. Sophisticated probabilistic approaches have been proposed to better handle this uncertainty (Babaeizadeh et al., 2018; Lee et al., 2018; Denton & Fergus, 2018; Xue et al., 2016).

What if we instead change the goal of our prediction models? Fixed time intervals in prediction are in many ways an artifact of the fact that cameras and monitors record and display video at fixed frequencies. Rather than requiring predictions at regularly spaced future frames, we ask: if a frame prediction is treated as a bet on that frame occurring at some future point, what should we predict? Such time-agnostic prediction (TAP) has two immediate effects: (i) the predictor may skip more uncertain states in favor of less uncertain ones, and (ii) while in the standard approach, a prediction is wrong if it occurs at  $t \pm \epsilon$  rather than at  $t$ , our formulation considers such predictions equally correct.

Recall the bottle-tilting uncertainty profile. Fig 1 depicts uncertainty profiles for several other prediction settings, including both forward/future prediction (given a start frame) and intermediate prediction (given start and end frames). Our time-agnostic reframing of the prediction problem targets the minima of these profiles, where prediction is intuitively easiest. We refer to these minima states as "bottlenecks."

At this point, one might ask: are these "easy" bottlenecks actually useful to predict? Intuitively, bottlenecks naturally correspond to reliable subgoals—an agent hoping to solve the maze in Fig 1 (e) would do well to target its bottlenecks as subgoals. In our experiments, we evaluate the usefulness of our predictions as subgoals in simulated robotic manipulation tasks.

![](images/f54ac7499aa2def29432a4827d342b53adf67251724e192acab3c6fc08f3e9a2.jpg)  
(a)

![](images/36d27c9596e78c9ac08ca7710047fd503243e23bf3a0e82016a004e852c52a6a.jpg)  
(b)  
Figure 1: (a) Over time as the bottle is tilted, the uncertainty first rises and then falls as the bottle is held steady after tilting. (b)-(e) Similar uncertainty profiles corresponding to various scenarios—a ball rolling down the side of a bowl, a car driving on a highway with an exit  $100\mathrm{m}$  away, an iron pellet tossed in the direction of a magnet, and intermediate frame prediction in a maze traversal given start and end states. The red asterisks along the x-axis correspond to the asterisks in the maze—these "bottleneck" states must occur in any successful traversal.

![](images/c330f6dd5171a7a4b7ddf129ee95ecbda90eb86e49ab3aa240b4d8420289df0b.jpg)  
(c)

![](images/5286581606b9dc8b4e96df2a41390d5d394b2dd156ad3e5681a994281fcd4e3f.jpg)  
(d)

![](images/3a00debdd6cedb29b2922e88a532807d8fc064542c97a187493a145523a186ec.jpg)  
(e)

Our main contributions are: (i) we reframe the video prediction problem to be time-agnostic, (ii) we propose a novel technical approach to solve this problem, (iii) we show that our approach effectively identifies "bottleneck states" across several tasks, and (iv) we show that these bottlenecks correspond to subgoals that aid in planning towards complex end goals.

# 2 RELATED WORK

Visual prediction approaches. Prior visual prediction approaches regress directly to future video frames in the pixel space (Ranzato et al., 2014; Oh et al., 2015) or in a learned feature space (Hadsell et al., 2006; Mobahi et al., 2009; Jayaraman & Grauman, 2015; Wang et al., 2016; Vondrick et al., 2016b; Kitani et al., 2012). The success of generative adversarial networks (GANs) (Goodfellow et al., 2014; Mirza & Osindero, 2014; Radford et al., 2015; Isola et al., 2017) has inspired many video prediction approaches (Mathieu et al., 2015; Vondrick et al., 2016a; Finn & Levine, 2017; Xue et al., 2016; Oh et al., 2015; Ebert et al., 2017; Finn et al., 2016; Larsen et al., 2016; Lee et al., 2018). While adversarial losses aid in producing photorealistic image patches, prediction has to contend with a more fundamental problem: uncertainty. Several approaches (Walker et al., 2016; Xue et al., 2016; Denton & Fergus, 2018; Lee et al., 2018; Larsen et al., 2016; Babaeizadeh et al., 2018) exploit conditional variational autoencoders (VAE) (Kingma & Welling, 2013) to train latent variable models for video prediction. Pixel-autoregression (Oord et al., 2016; van den Oord et al., 2016; Kalchbrenner et al., 2016) explicitly factorizes the joint distribution over all pixels to model uncertainty, at a high computational cost.

Like these prior approaches, we too address the uncertainty problem in video prediction. We propose a general time-agnostic prediction (TAP) framework for prediction tasks. While all prior work predicts at fixed time intervals, we aim to identify inherently low-uncertainty bottleneck frames with no associated timestamp. We show how TAP may be combined with conditional GANs as well as VAEs, to handle the residual uncertainty in its predictions.

Bottlenecks. In hierarchical reinforcement learning, bottlenecks are proposed for discovery of options (Sutton et al., 1999) in low-dimensional state spaces in (McGovern & Barto, 2001; Şimsek & Barto, 2009; Bacon, 2013; Metzen, 2013). Most approaches (Şimsek & Barto, 2009; Bacon, 2013; Metzen, 2013) construct full transition graphs and apply notions of graph centrality to locate bottlenecks. A multi-instance learning approach is applied in (McGovern & Barto, 2001) to mine states that occur in successful trajectories but not in others. We consider the use of our bottleneck predictions as subgoals for a hierarchical planner, which is loosely related to options in that both aim to break down temporally extended trajectories into more manageable chunks. Unlike these prior works, we use predictability to identify bottlenecks, and apply this to unlabeled high-dimensional visual state trajectories.

# 3 TIME-AGNOSTIC PREDICTION OF BOTTLENECK FRAMES

In visual prediction, the goal is to predict a set of unobserved target video frames given some observed context frames. In forward prediction, the context is the first frame, and the target is all future frames. In the bidirectionally conditioned prediction case, the context is the first and the last frame, and the frames in between are the target. In Fig 1, we may wish to predict future images of the tilting bottle, or intermediate images of an agent who traverses the maze successfully.

# 3.1 MINIMUM-OVER-TIME LOSS

In standard fixed-time video prediction models (Ranzato et al., 2014; Oh et al., 2015; Mathieu et al., 2015; Vondrick et al., 2016a; Walker et al., 2016; Finn & Levine, 2017; Xue et al., 2016; Oh et al., 2015; Ebert et al., 2017; Finn et al., 2016; Lee et al., 2018; Denton & Fergus, 2018), a frame  $x_{\tau}$  (video frame at time  $\tau$ ) is selected in advance to be the training target for some given input frames  $c$ . For instance, in a

![](images/d95a3cf1a0f8cdd6dccc2b6985adb35d3851d5b98dca0fdfacc3ea964803c77b.jpg)  
Figure 2: The TAP minimum-over time loss.

typical forward prediction setup, the input may be  $c = x_0$ , and the target frame may be set to  $x_{\tau} = x_{1}$ . A predictor  $G$  takes context frames  $c$  as input and produces a single frame  $G(c)$ .  $G$  is trained as:

$$
G ^ {*} = \underset {G} {\arg \min } \mathcal {L} _ {0} (G) = \underset {G} {\arg \min } \mathcal {E} (G (c), x _ {\tau}), \tag {1}
$$

where  $\mathcal{E}$  is a measure of prediction error, such as  $\| G(c) - x_{\tau}\| _1$  . This predictor may be applied recursively at test time to generate more predictions as  $G(G(c))$ $G(G(G(c)))$  , and so on.

We propose to depart from this fixed-time paradigm by decoupling prediction from a rigid notion of time. Instead of predicting the video frame at a specified time  $\tau$ , we propose to predict predictable bottleneck video frames through a time-agnostic predictor (TAP), as motivated in Sec 1. To train this predictor, we minimize the following "minimum-over-time" loss:

$$
G ^ {*} = \underset {G} {\arg \min } \mathcal {L} (G) = \underset {G} {\arg \min } \underset {t \in \mathrm {T}} {\min } \mathcal {E} (G (c), x _ {t}), \tag {2}
$$

where the key difference from Eq 1 is that the loss is now a minimum over target times of a time-indexed error  $\mathcal{E}_t = \mathcal{E}(\cdot, x_t)$ . The target times are defined by a set of time indices  $T$ . For forward prediction starting from input  $c = x_0$ , we may set targets to  $T = \{1, 2, \ldots\}$ . Fig 2 depicts this idea schematically. Intuitively, the penalty for a prediction is determined based on the closest matching ground truth target frame. This loss incentivizes the model to latch on to "bottleneck" states in the video, i.e., those with low uncertainty. In the bottle-tilting example, this would mean producing an image of the bottle after the water has come to rest.

One immediate concern with this minimum-over-time TAP loss might be that it could produce degenerate predictions very close to the input conditioning frames  $c$ . However, as in the tilting bottle and other cases in Fig 1, uncertainty is not always lowest closest to the observed frames. Moreover, target frame indices  $\mathrm{T}$  are always disjoint from the input context frames, so the model's prediction must be different from input frames by at least one step, which is no worse than the one-step-forward prediction of Eq 1. In our experiments, we show cases where the minimum-over-time loss above captures natural bottlenecks successfully. Further, Sec 3.2 shows how it is also possible to explicitly penalize predictions near input frames  $c$ .

This minimum loss may be viewed as adaptively learning the time offset  $\tau$ , but in fact, the predictor's task is even simpler since it is not required to provide a timestamp accompanying its prediction. For example, in Fig 1(e), it need only specify which points in the maze the agent will go through; it need not specify when. Lifting the requirement of a timestamped prediction relieves TAP approaches of a significant implicit burden.

Recursive TAP. TAP models may also be trained for recursive prediction, by minimizing the following loss:

$$
G ^ {*} = \underset {G} {\arg \min } \mathcal {L} _ {\operatorname {r e c}} (G) = \underset {G} {\arg \min } \sum_ {r} \underset {t \in \mathrm {T} (r)} {\min } \mathcal {E} (G (c (r)), x _ {t}), \tag {3}
$$

where  $c(r)$  and  $\mathrm{T}(r)$  are the input and target set at recursion level  $r$ , both dynamically adapted based on the previous prediction. The input  $c(r)$  may be set to the previous prediction  $G(c(r - 1))$ , so that the sequence of predictions is  $(G(c(0)), G(G(c(0))), \ldots)$ .  $\mathrm{T}(r)$  is set to target all times after the last prediction. In other words, if the prediction at  $r = 0$  was closest to frame  $x_5$ , the targets for  $r = 1$  are set to  $\mathrm{T}(1) = \{6, 7, \ldots\}$ . While we also test recursive TAP in Sec 4, in the rest of this section, we discuss the non-recursive formulation, building on Eq 2, for simplicity.

Bidirectional TAP. Finally, while the above description of TAP has focused on forward prediction, the TAP loss of Eq 2 easily generalizes to bidirectional prediction. Given input frames  $c = (x_0, x_{\mathrm{last}})$ , fixed-time bidirectional predictors might target, say, the middle frame  $x_{\tau} = x_{\mathrm{last}/2}$ . Instead, bidirectional TAP models target all intermediate frames, i.e.,  $\mathrm{T} = \{1, 2, \dots, \mathrm{last} - 1\}$  in Eq 2. As in forward prediction, the model has incentive to predict predictable frames. In the maze example from Fig 1, this would mean producing an image of the agent at one of the asterisks.

# 3.2 FROM MINIMUM TO GENERALIZED MINIMUM TAP LOSS

Within the time-agnostic prediction paradigm, we may still want to specify preferences for some times over others, or for some visual properties of the predictions. Consider the minimum-over-time loss  $\mathcal{L}$  in Eq 2. Taking the minimum inside, this may be rewritten as:

$$
\mathcal {L} (G) = \min  _ {t \in T} \mathcal {E} _ {t} = \mathcal {E} _ {\arg \min  _ {t \in T} \mathcal {E} _ {t}}, \tag {4}
$$

where we use the time-indexed error  $\mathcal{E}_t$  as shorthand for  $\mathcal{E}(.x_t)$ . We may now extend this to the following "generalized minimum" loss, where the outer and inner errors are decoupled:

$$
\mathcal {L} ^ {\prime} (G) = \mathcal {E} _ {\arg \min  _ {t \in T} \varepsilon_ {t} ^ {\prime}}. \tag {5}
$$

Now,  $\mathcal{E}_t^\prime$ , over which the minimum is computed, could be designed to express preferences about which frames to predict. In the simplest case,  $\mathcal{E}_t^\prime = \mathcal{E}_t$ , and the loss reduces to Eq 2. Instead, suppose that predictions at some times are preferred over others. Let  $w(t)$  express the preference value for all target times  $t\in T$ , so that higher  $w(t)$  indicates higher preference. Then we may set  $\mathcal{E}_t^\prime = \mathcal{E}_t / w(t)$  so that times  $t$  with higher  $w(t)$  are preferred in the arg min. In our experiments, we set  $w(t)$  to linearly increase with time during forward prediction and to a truncated discrete Gaussian centered at the midpoint in bidirectional prediction.

At this point, one might ask: could we not directly incorporate preferences into the outer error? For instance, why not simply optimize  $\min_t\mathcal{E}_t / w(t)$ ? Unfortunately, that would have the side-effect of downweighting the errors computed against frames with higher preferences  $w(t)$ , which is counterproductive. Decoupling the outer and inner errors instead, as in Eq 5, allows applying preferences  $w(t)$  only to select the target frame to compute the outer loss against; the outer loss itself penalizes prediction errors equally regardless of which frame was selected.

The generalized minimum formulation may be used to express other kinds of preferences too. For instance, when using predictions as subgoals in a planner, perhaps some states are more expensive to reach than others. We also use the generalized minimum to select frames using different criteria than the prediction penalty itself, as we will discuss in Sec 3.5.

# 3.3 TIME-AGNOSTIC CONDITIONAL GANS

TAP is not limited to simple losses such as  $\ell_1$  or  $\ell_2$  errors; it can be extended to handle expressive GAN losses to improve perceptual quality. A standard conditional GAN (CGAN) in fixed-time video prediction targeting time  $\tau$  works as follows: given a "discriminator"  $D$  that outputs 0 for input-prediction tuples and 1 for input-ground truth tuples, the generator  $G$  is trained to fool the discriminator. The discriminator in turn is trained adversarially using a binary cross-entropy loss. The CGAN objective is written as:

$$
G^{*} = \operatorname *{arg  min}_{G}\max_{D}\mathcal{L}_{\text{\rm{cga n}}} (G,D),
$$

$$
\mathcal {L} _ {\mathrm {c g a n}} (G, D) = \log (D (c, x _ {\tau})) + \log (1 - D (c, G (c)) \tag {6}
$$

To make this compatible with TAP, we train  $|\mathrm{T}|$  discriminators  $\{D_t\}$ , one per timestep. Then, analogous to Eq 2, we may define a time-agnostic CGAN loss:

$$
G^{*} = \operatorname *{arg  min}_{G}\min_{t\in T}\max_{D_{t}}\mathcal{L}^{t}_{\mathrm{cga n}}(G,D_{t}),
$$

$$
\mathcal {L} _ {\mathrm {c g a n}} ^ {t} (G, D _ {t}) = \log D _ {t} (c, x _ {t}) + \log \left(1 - D _ {t} (c, G (c))\right) + \sum_ {t ^ {\prime} \neq t} \log \left(1 - D _ {t} \left(c, x _ {t ^ {\prime}}\right)\right), \tag {7}
$$

Like Eq 6, Eq 7 defines a cross-entropy loss. The first two terms are analogous to Eq 6 — for the  $t$ -th discriminator, the  $t$ -th frame provides a positive, and the generated frame provides a negative

instance. The third term treats ground truth video frames occurring at other times  $x_{t'} \neq t$  as negatives. In practice, we train a single discriminator network with  $|T|$  outputs that serve as  $\{D_t\}$ . Further, for computational efficiency, we approximate the summation over  $t' \neq t$  by sampling a single frame at random for each training video at each iteration. Appendix A provides additional details.

# 3.4 TIME-AGNOSTIC CONDITIONAL VAES

While TAP targets low-uncertainty bottleneck states, it may be integrated with a conditional variational autoencoder (CVAE) to handle residual uncertainty at these bottlenecks. In typical fixed-time CVAE predictors targeting time  $\tau$ , variations in a latent code  $z$  input to a generator  $G(c,z)$  must capture stochasticity in  $x_{\tau}$ . At training time,  $z$  is sampled from a posterior distribution  $q_{\phi}(z|x_{\tau})$  with parameters  $\phi$ , represented by a neural network. At test time,  $z$  is sampled from a prior  $p(z)$ . The training loss combines a log-likelihood term with a KL-divergence from the prior:

$$
\mathcal {L} _ {\mathrm {c v a e}} (G, \phi) = D _ {K L} \left(q _ {\phi} \left(z \mid x _ {\tau}\right), p (z)\right) - \mathbb {E} _ {z \sim q _ {\phi} \left(z \mid x _ {\tau}\right)} \ln p _ {G} \left(x _ {\tau} \mid c, z\right), \tag {8}
$$

where we might set  $p_G$  to a Laplacian distribution such that the second term reduces to a  $l1$ -reconstruction loss  $-\ln p_G(x_\tau | c, z) = \| G(c, z) - x_\tau \|_1$ . In a time-agnostic CVAE, rather than capturing stochasticity at a fixed time  $\tau$ ,  $z$  must now capture stochasticity at bottlenecks: e.g., when the agent crosses one of the asterisks in the maze of Fig 1, which pose is it in? The bottleneck's time index varies and is not known in advance. For computational reasons (see Appendix B), we pass the entire video  $X$  into the inference network  $q_\phi$ , similar to Babaeizadeh et al. (2018). The negative log-likelihood term is adapted to be a minimum-over-time:

$$
\mathcal {L} _ {\mathrm {c v a e}} (G, \phi) = D _ {K L} \left(q _ {\phi} (z | X), p (z)\right) + \min  _ {t \in \mathrm {T}} \mathbb {E} _ {z \sim q _ {\phi} (z | X)} \left[ - \ln p _ {G} \left(x _ {t} | c, z\right) \right]. \tag {9}
$$

# 3.5 COMBINED LOSS, NETWORK ARCHITECTURE, AND TRAINING

We train time-agnostic CVAE-GANs with the following combination of a generalized minimum loss (Sec 3.2) and the CVAE KL divergence loss (Sec 3.4):

$$
G ^ {*} = \underset {G} {\arg \min} \underset {\phi} {\min} \left[ D _ {K L} (q _ {\phi} (z | X), p (z)) + \mathcal {E} _ {\arg \min _ {t \in \mathrm {T}} \varepsilon_ {t} ^ {\prime}} \right],
$$

$$
\mathcal {E} _ {t} = \max _ {D, D ^ {\prime}} \mathcal {L} _ {\mathrm {c g a n}} ^ {t} (G, D _ {t}) + \mathcal {L} _ {\mathrm {c v a e - g a n}} ^ {t} (G, D _ {t} ^ {\prime}) + \| G (c, z) - x _ {t} \| _ {1},
$$

$$
\mathcal {E} _ {t} ^ {\prime} = \| G (c, z) - x _ {t} \| _ {1} / w (t). \tag {10}
$$

The outer error  $\mathcal{E}_t$  absorbs the CGAN discriminator errors (Sec 3.3), while the inner error  $\mathcal{E}_t^\prime$  is a simple  $\ell_1$  error, weighted by user-specified time preferences  $w(t)$  (Sec 3.2). Omitting GAN error terms in  $\mathcal{E}_t^\prime$  helps stabilize training, since learned errors may not always be meaningful especially early on in training. As in VAE-GANs (Larsen et al., 2016; Lee et al., 2018), the training objective includes a new term  $\mathcal{L}_{\mathrm{cvae-gan}}^t$ , analogous to  $\mathcal{L}_{\mathrm{cga}}^t$  (Eq 7). We set up  $\mathcal{L}_{\mathrm{cga}}^t$  to use samples  $z$  from the prior  $p(z)$ , while  $\mathcal{L}_{\mathrm{cvae-gan}}^t$  instead samples  $z$  from the posterior  $q_{\phi}(z|X)$ , and employs separate discriminators  $\{D_t^\prime\}$ . The  $\ell_1$  loss also samples  $z$  from the posterior. We omit expectations over the VAE latent  $z$  to keep notation simple.

Frame generation in the predictor involves first generating appearance flow-transformed input frames (Zhou et al., 2016) and a frame with new uncopied pixels. These frames are masked and averaged to produce the output. Full architecture and training details are in Appendix C.

# 4 EXPERIMENTS

We have proposed a time-agnostic prediction (TAP) paradigm that is different from the fixed-time paradigm followed in prior prediction work. In our experiments, we focus on comparing TAP against a representative fixed-time prediction model, keeping network architectures fixed. We use three simulated robot manipulation settings: object grasping (50k episodes), pick-and-place (75k episodes), and multi-object pushing (55k episodes). Example episodes from each task are shown in Fig 3 (videos in Supp).  $5\%$  of the data is set aside for testing. We use  $64 \times 64$  images.

For grasping (15 frames per episode), the arm moves to a single object on a table, selects a grasp, and lifts it vertically. For pick-and-place (20 frames), the arm additionally places the lifted object at a new position before performing a random walk. For pushing (40 frames), two objects are initialized at random locations and pushed to random final positions. Object shapes and colors in all three settings are randomly generated. Fig 3 shows example episodes.

![](images/6bde642cc53ba903e8fbdc3cdd573f69b835a5b6b26a17ea7f3e81d3baf44ba6.jpg)  
Grasping episode (length  $T = 15$ )

![](images/3f47afe216643681d0770b5f9219bc0d693fd5b6d42cead2fa49ee19c106586f.jpg)  
Pick-and-place episode (length  $T = 20$ )

![](images/890fa2ca501ff4c8630de77f4e8064dfc3e0d59d64a97bbbf4c0c285e6ffc904.jpg)  
Two-object pushing episode (length  $T = 40$ )  
Figure 3: (Best seen in pdf) One sample episode each for grasping, pick-and-place, and pushing. Time overlaid on each frame.  
Figure 4: Forward prediction  $\ell_1$  error. TAP methods (red) perform better than fixed-time predictors over all time steps.

![](images/3d1688a72c6bcd50300bb19e65424c5f76043ac2f4e6ea5e8bf4e2835daec3c2.jpg)

<table><tr><td rowspan="2">Setting → 
Method ↓</td><td colspan="2">Grasping (15 steps)</td><td colspan="2">Pick-and-place (20 steps)</td><td colspan="2">Pushing (30 steps)</td></tr><tr><td colspan="2">min ℓ1 err match-step</td><td colspan="2">min ℓ1 err match-step</td><td colspan="2">min ℓ1 err match-step</td></tr><tr><td>fix</td><td>0.0153</td><td>7.61±2.52</td><td>0.0366</td><td>10.58±4.82</td><td>0.07223</td><td>10.76±5.82</td></tr><tr><td>MIN (ours)</td><td>0.0104</td><td>7.31±2.68</td><td>0.0256</td><td>8.23±6.07</td><td>0.0365</td><td>10.59±6.61</td></tr><tr><td>GENMIN (ours)</td><td>0.0121</td><td>6.77±2.44</td><td>0.0269</td><td>8.49±4.51</td><td>0.0338</td><td>10.82±5.79</td></tr><tr><td>GENMIN W/O GAN (ours)</td><td>0.0117</td><td>6.74±2.41</td><td>0.0235</td><td>9.21±4.90</td><td>0.0411</td><td>11.07±5.80</td></tr><tr><td>GENMIN + VAE (ours)</td><td>0.0156</td><td>7.07±2.75</td><td>0.0432</td><td>6.12±4.81</td><td>0.0447</td><td>10.99±6.15</td></tr><tr><td>GENMIN + VAE BEST-OF-100 (ours)</td><td>0.0121</td><td>-</td><td>0.0196</td><td>-</td><td>0.0236</td><td>-</td></tr></table>

Table 1: Bidirectional frame prediction performance on: grasping, pick-and-place, and two-object pushing. Lower min  $\ell_1$  err is better. match-step denotes which times are being predicted. It is clear that TAP methods make better predictions than fixed-time prediction at the same time offsets.

Forward prediction. First, we evaluate our approach for forward prediction on grasping. The first frame ("start") is provided as input. We train fixed-time baselines (architecture same as ours, using  $\ell_1$  and GAN losses same as MIN and GENMIN) that target predictions at exactly 0.25, 0.50, 0.75, 1.0 fraction of the episode length (FIX0.25,..., FIX1.00). MIN and GENMIN are TAP with/without the generalized minimum of Sec 3.2. For GENMIN, we evaluate different choices of the time preference vector  $w(t)$  (Sec 3.2). We set  $w(t) = \beta + t / 15$ , so that our preference increases linearly from  $\beta$  to  $\beta + 1$ . Since  $w(t)$  applies multiplicatively, low  $\beta$  corresponds to high disparity in preferences ( $\beta = \infty$  reduces to MIN, i.e., no time preference). GENMIN2 is our approach with  $\beta = 2$  and so on.

Fig 5 shows example predictions from all methods for the grasping task. In terms of visual quality of predictions and finding a semantically coherent bottleneck, GENMIN2, GENMIN4, and GENMIN7 perform best—they reliably produce a grasp on the object while it is still on the table. With little or no time preferences, MIN and GENMIN10 produce images very close to the start, while GENMIN0.5 places too high a value on predictions farther away, and produces blurry images of the object after lifting.

Quantitatively, for each method, we report the min and arg min index of the  $\ell_1$  distance to all frames in the video, as "min  $\ell_1$  err" and "match-step" ("which future ground truth frame is the prediction closest to?"). Fig 4 shows a scatter plot, where each dot or square is one model. TAP (various models with varying  $\beta$ ) produces an even larger variation in stepsizes than fixed-time methods explicitly targeting the entire video (FIX0.75 and FIX1.0 fall short of producing predictions at 0.75 and 1.0 fraction of the episode length). TAP also produces higher quality predictions (lower error) over that entire range. From these quantitative and qualitative results, we see that TAP not only successfully encourages semantically coherent bottleneck predictions, it also produces higher quality predictions than fixed-time prediction over a range of time offsets.

Intermediate frame prediction. Next, we evaluate our approaches for bidirectionally conditioned prediction in all three settings. Initial and final frames are provided as input, and the method is trained to generate an intermediate frame. The FIX baseline now targets the middle frame. As before, MIN and GENMIN are our TAP models. The GENMIN time preference  $w(t)$  is bell-shaped and varies from 2/3 at the ends to 1 at the middle frame.

Figs 6 and 9 show examples from the three settings. TAP approaches successfully discover interesting bottlenecks in each setting. For grasping (Fig 6 (left)), both MIN and GENMIN consistently produce clear images of the arm at the point at which it picks up the object. Pick-and-place (Fig 6, right) is harder because it is more temporally extended, and the goal image does not specify how to grasp the object. FIX struggles to produce any coherent predictions, but GENMIN once again identifies

![](images/bd5d8f466b4488f2fb18c89ec8055641dd11004a7e8d2e89e7263eea8e100a11.jpg)

![](images/4a3e59614995824bec80c1467ee2d70d709bb279617a399d859cf32c2c505725.jpg)  
Figure 5: (Best seen in pdf) Forward prediction results on grasping comparing fixed-time predictors and our approach. Each row is a separate example. First column is the input. Thereafter, each column corresponds to the output of a different model per the column title. More in Appendix Fig 17.

![](images/241b19543bec3dc656710dec1672628cefc5808e622e3ac7683167c92cfdc685.jpg)  
Figure 6: (Best seen in pdf) Bidirectional prediction results comparing fixed-time prediction and our approach. (Left) Grasping results. First two columns are inputs (start and goal). Thereafter, each column corresponds to the output of a different model per the column title. "match" is the ground truth image closest to the GENMIN prediction. More in Appendix Fig 13. (Right) Similar results for pick-and-place. More in Appendix Fig 14.

bottlenecks reliably—in examples #3 and #1, it predicts the “pick” and the “place” respectively. For the pushing setting (Fig 9 (left)), GENMIN frequently produces images with one object moved and the other object fixed in place, which again is a semantically coherent bottleneck for this task. In row #1, it moves the correct object first to generate the subgoal, so that objects do not collide.

Table 1 shows quantitative results over the full test set. As in forward prediction, we report min  $\ell_1$  error and the best-matching frame index ("match-step") for all methods. MIN and GENMIN consistently yield higher quality predictions (lower error) than FIX at similar times on average. As an example, GENMIN reduces FIX errors by  $21\%$ ,  $26.5\%$ , and  $53.2\%$  on the three tasks—these are consistent and large gains that increase with increasing task complexity/duration. Additionally, while all foregoing results were reported without the CVAE approach of Sec 3.4, Table 1 shows results for GENMIN+VAE, and Fig 9 shows example predictions for pick-and-place. In our models, individual stochastic predictions from GENMIN+VAE produce higher  $\ell_1$  errors than GENMIN. However, the CVAE helps capture meaningful sources of stochasticity at the bottlenecks—in Fig 9 (right), it produces different grasp configurations to pick up the object in each case. To measure this, we evaluate the best of 100 stochastic predictions from GENMIN+VAE in Table 1 (GENMIN+VAE BEST-OF-100). On pick-and-place and pushing, the best VAE results are significantly better than any of the deterministic methods. Table 1 also shows results for our method without the GAN (GENMIN w/o GAN)—while its  $\ell_1$  errors are comparable, we observed a drop in visual quality.

As indicated in Sec 3.1 and Eq 3, TAP may also be applied recursively. Fig 7 compares consecutive subgoals for the pick-and-place task produced by recursive TAP versus a recursive fixed-time model. Recursion level  $r = 1$  refers to the first subgoal, and  $r = 2$  refers to the subgoal generated when the first subgoal is provided as the goal input (start input is unchanged). In example #1, FIX struggles while "ours" identifies the "place" bottleneck at  $r = 1$ , and subsequently the "pick" botleneck at  $r = 2$ .

![](images/b313b7378daeb3bbcfcbee0a9caa07e526765d8eb3290a4bb36f576ebcf03b51.jpg)  
Figure 7: Recursive bidirectional prediction on pick-and-place.  $r = 1,2$  denote generated subgoals at different levels of recursion. Higher  $r \Rightarrow$  earlier in time.

![](images/e2a1f48c5843eacbde51668d339a7536a3db4addc6ace4c431cc43b7b9cfda94.jpg)  
Figure 9: (Best seen in pdf) (Left) Bidirectional prediction results on two-object pushing. More in Appendix Fig 15. (Right) When used with a VAE (Sec 3.4), our approach captures residual stochasticity at the bottleneck. In these results from the pick-and-place task, GENMIN+VAE produces images that are all of the arm in contact with the object on the table, but at different points on the object, and with different arm/gripper poses.

Finally, we test on "BAIR pushing" (Ebert et al., 2017), a real-world dataset that is commonly used in visual prediction tasks. The data consists of 30-frame clips of random motions of a Sawyer arm tabletop. While this dataset does not have natural bottlenecks like in grasping, TAP (min  $\ell_{1}$  error 0.046 at match-step 15.42) still performs better than FIX (0.059 at 15.29). Qualitatively, as Fig 8 shows, even though BAIR pushing contains incoherent random arm motions, TAP consistently produces predictions that plausibly lie on the path from start to goal image. In example #1, given a start and goal image with one object displaced, "ours" correctly moves the arm to the object before displacement, whereas FIX struggles.

![](images/af1d7fa2e085c0aaa15b131142ea1c15b3690dfc4a9c71df069bd3e304d9011c.jpg)  
Figure 8: Bidirectional prediction results on BAIR pushing data. The first two columns are the inputs, and the next two correspond to FIX and GENMIN.

Bottleneck discovery frequency. We have thus far relied on qualitative results to assess how often our approach finds coherent bottlenecks. For pushing, we test bottleneck discovery frequency more quantitatively. We make the reasonable assumption that bottlenecks in 2-object pushing correspond to states where one object is pushed and the other is in place. Our metric exploits knowledge of true object positions at start and goal states. First, for this evaluation, we restrict both GENMIN and FIX to synthesize predictions purely by warping and masking inputs. Thus, we can track where the pixels at ground truth object locations in start and goal images end up in the prediction, i.e., where did each object move? We then compute a score that may be thresholded to detect when only

one object moves (details in Appendix E). As Fig 10 shows, GENMIN predicts bottlenecks much more frequently ( $\sim 60\%$  of the time) than FIX. As hypothesized, our time-agnostic approach does indeed identify and latch on to low-uncertainty states to improve its predictions.

![](images/94ac3bff0bef97feca191ed08c954bf495083ed78ab58c481e6e25cc0a624c10.jpg)  
Figure 10: Bottleneck frequency vs. score threshold

Hierarchical planning evaluation. Finally, we discuss experiments directly evaluating our intermediate predictions as visual subgoals for hierarchical planning for pushing tasks. A forward Visual MPC planner (Ebert et al., 2017) accepts the subgoal object positions (computed as above for evaluating bottlenecks). Start and goal object positions are also known. Internally, Visual MPC makes action-conditioned fixed-time forward predictions of future object positions to find an action sequence that reaches the subgoal object positions, with a planning horizon of 15. Additional implementation

<table><tr><td></td><td>2-object</td><td>3-object</td></tr><tr><td>direct</td><td>12.9±0.6</td><td>15.8±0.6</td></tr><tr><td>FIX</td><td>12.5±0.5</td><td>17.6±0.6</td></tr><tr><td>GENMIN</td><td>11.9±0.6</td><td>12.9±0.7</td></tr></table>

Table 2: Multi-object pushing errors (in cm).

Given start and goal images, our model produces a subgoal. Visual MPC plans towards this subgoal for half the episode length, then switches to the final goal. We compare this scheme against (i) "direct": planning directly towards the final goal for the entire episode length, and (ii) FIX: subgoals from a center-frame predictor. The error measure is the mean of object distances to goal states (lower is better). As an upper bound, single-object pushing with the planner yields  $\sim 5\mathrm{cm}$  error. Results for two-object and three-object pushing are shown in Table 2. GENMIN does best on both, but especially

on the more complex three-object task. Since Visual MPC has thus far been demonstrated to work only on pushing tasks, our hierarchical planning evaluation is also limited to this task. Going forward, we plan to adapt Visual MPC to allow testing TAP on more complex temporally extended tasks like block-stacking, where direct planning breaks down and subgoals offer greater value.

# 5 CONCLUSIONS

The standard paradigm for prediction tasks demands that a predictor not only make good predictions, but that it make them on a set schedule. We have argued for redefining the prediction task so that the predictor need only care that its prediction occur at some time, rather than that it occur at a specific scheduled time. We define this time-agnostic prediction task and propose novel technical approaches to solve it, that require relatively small changes to standard prediction methods. Our results show that reframing prediction objectives in this way yields higher quality predictions that are also semantically coherent—unattached to a rigid schedule of regularly specified timestamps, model predictions instead naturally attach to specific semantic "bottleneck" events, like a grasp. In our preliminary experiments with a hierarchical visual planner, our results suggest that such predictions could serve as useful subgoals for complex tasks. We hope to build further on these results.

# REFERENCES

Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. Stochastic variational video prediction. *ICLR*, 2018.  
Pierre-Luc Bacon. On the bottleneck concept for options discovery. M.S. Thesis, McGill University, 2013.  
Moshe Bar. The proactive brain: memory for predictions. Philosophical Transactions of the Royal Society B: Biological Sciences, 364(1521):1235-1243, 2009.  
Andy Clark. Whatever next? predictive brains, situated agents, and the future of cognitive science. Behavioral and brain sciences, 36(3):181-204, 2013.  
Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. arXiv preprint arXiv:1802.07687, 2018.  
Frederik Ebert, Chelsea Finn, Alex X. Lee, and Sergey Levine. Self-supervised visual planning with temporal skip connections. CORL, 2017.  
Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion. In International Conference on Robotics and Automation (ICRA), 2017.  
Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. In Neural Information Processing Systems (NIPS), 2016.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In Computer vision and pattern recognition, 2006 IEEE computer society conference on, volume 2, pp. 1735-1742. IEEE, 2006.  
Jakob Hohwy. The predictive mind. Oxford University Press, 2013.  
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. CVPR, 2017.  
Dinesh Jayaraman and Kristen Grauman. Learning image representations tied to ego-motion. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1413-1421, 2015.  
Nal Kalchbrenner, Aaron van den Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Video pixel networks. arXiv preprint arXiv:1610.00527, 2016.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Kris M Kitani, Brian D Ziebart, James Andrew Bagnell, and Martial Hebert. Activity forecasting. In European Conference on Computer Vision, pp. 201-214. Springer, 2012.  
Dirk P Kroese, Reuven Y Rubinstein, Izack Cohen, Sergey Porotsky, and Thomas Taimre. Cross-entropy method. In Encyclopedia of Operations Research and Management Science, pp. 326-333. Springer, 2013.  
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, and Ole Winther. Autoencoding beyond pixels using a learned similarity metric. ICML, 2016.  
Alex X Lee, Richard Zhang, Frederik Ebert, Pieter Abbeel, Chelsea Finn, and Sergey Levine. Stochastic adversarial video prediction. arXiv preprint arXiv:1804.01523, 2018.  
Michael Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. arXiv preprint arXiv:1511.05440, 2015.  
Amy McGovern and Andrew G Barto. Automatic discovery of subgoals in reinforcement learning using diverse density. In ICML, 2001.  
Jan Hendrik Metzen. Online skill discovery using graph-based clustering. In European Workshop on Reinforcement Learning, pp. 77-88, 2013.  
Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.  
Hossein Mobahi, Ronan Collobert, and Jason Weston. Deep learning from temporal coherence in video. In Proceedings of the 26th Annual International Conference on Machine Learning, pp. 737-744. ACM, 2009.  
Augustus Odena, Vincent Dumoulin, and Chris Olah. Deconvolution and checkerboard artifacts. Distill, 2016. doi: 10.23915/distill.00003. URL http://distill.pub/2016/deconv-checkerboard.  
Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games. In Advances in Neural Information Processing Systems, 2015.  
Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759, 2016.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
MarcAurelio Ranzato, Arthur Szlam, Joan Bruna, Michael Mathieu, Ronan Collobert, and Sumit Chopra. Video (language) modeling: a baseline for generative models of natural videos. arXiv preprint arXiv:1412.6604, 2014.  
Özgür Şimsek and Andrew G Barto. Skill characterization based on betweenness. In Advances in neural information processing systems, pp. 1497-1504, 2009.  
Richard S Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 112(1-2):181-211, 1999.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026-5033. IEEE, 2012.  
Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelCNN decoders. In Advances in Neural Information Processing Systems, pp. 4790-4798, 2016.

Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. In Advances In Neural Information Processing Systems, pp. 613-621, 2016a.  
Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Anticipating visual representations from unlabeled video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 98-106, 2016b.  
Jacob Walker, Carl Doersch, Abhinav Gupta, and Martial Hebert. An uncertain future: Forecasting from static images using variational autoencoders. In European Conference on Computer Vision, pp. 835-851. Springer, 2016.  
Xiaolong Wang, Ali Farhadi, and Abhinav Gupta. Actions~ transformations. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2658-2667, 2016.  
Tianfan Xue, Jiajun Wu, Katherine Bouman, and Bill Freeman. Visual dynamics: Probabilistic future frame synthesis via cross convolutional networks. In Advances in Neural Information Processing Systems, pp. 91-99, 2016.  
Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. View synthesis by appearance flow. In European conference on computer vision, pp. 286-301. Springer, 2016.

![](images/f29914747647e6acb5dfdfbbd87a092e8c0a8ac6375ed7053f9a417f074502fd.jpg)  
Figure 11: Training time network schematic. At test time, only the predictor  $G$  is used, and  $z \sim \mathcal{N}(0, \mathcal{I})$ . Loss terms (as used in Eq 10) are in red.

In these appendices, we provide details omitted in the main text for space. Note that more supplementary material, such as video examples, is hosted at: https://sites.google.com/view/ta-pred
