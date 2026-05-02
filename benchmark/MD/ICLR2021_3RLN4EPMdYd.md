# REVISITING HIERARCHICAL APPROACH FOR PERSISTENT LONG-TERM VIDEO PREDICTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Learning to predict the long-term future of video frames is notoriously challenging due to the inherent ambiguities in the distant future and the dramatic amplification of the prediction error through time. Despite the recent advances in the literature, existing approaches are limited to moderately short-term prediction (less than a few seconds), while extrapolating it to a longer future quickly leads to destruction in structure and content. In this work, we revisit hierarchical models in video prediction. Our method generates future frames by first estimating a sequence of dense semantic structures and subsequently translating the estimated structures to pixels by video-to-video translation. Despite its simplicity, we show that modeling structures and their dynamics in a categorical structure space with a stochastic sequential estimator leads to surprisingly successful long-term prediction. We evaluate our method on three challenging video prediction datasets involving car driving and human dancing, and demonstrate that it can generate complicated scene structures and motions over a very long time horizon (i.e., thousands frames), setting a new standard of video prediction with orders of magnitude longer prediction time than existing approaches. Full videos are available at https://bit.ly/2EyDSem.

# 1 INTRODUCTION

Video prediction aims to generate future frames conditioned on a short video clip. It has received much attention in recent years as forecasting the future of visual sequence is critical in improving the planning for model-based reinforcement learning (Finn et al., 2016; Hafner et al., 2019; Ha & Schmidhuber, 2018), forecasting future event (Hoai & Torre, 2013), action (Lan et al., 2014), and activity (Lan et al., 2014; Ryoo, 2011). To make it truly beneficial for these applications, video prediction should be capable of forecasting long-term future. Many previous approaches have formulated video prediction as a conditional generation task by recursively synthesizing future frames conditioned on the previous frames (Vondrick et al., 2016; Tulyakov et al., 2018; Denton & Fergus, 2018; Babaeizadeh et al., 2018; Castrejon et al., 2019; Villegas et al., 2019). Despite their success in short-term forecasting, however, none of these approaches have been successful in synthesizing convincing long-term future, due to the challenges in modeling complex dynamics and extrapolating from short sequences to much longer future. As the prediction errors easily accumulates and amplifies through time, the quality of the predicted frames quickly degrades over time.

One way to reduce the error propagation is to extrapolate in a low dimensional structure space instead of directly estimating pixel-level dynamics in a video. Therefore, many hierarchical modeling approaches are proposed (Villegas et al., 2017b; Wichers et al., 2018; Liang et al., 2017; Yan et al., 2018; Walker et al., 2017; Kim et al., 2019). These approaches first generate a sequence using a low-dimensional structure representation, and subsequently generate appearance conditioned on the predicted structures. Hierarchical approaches are potentially promising for long-term prediction since learning structure-aware dynamics allows the model to generate semantically accurate motion and content in the future. However, previous approaches often employed too specific and incomprehensive structures such as human body joints (Villegas et al., 2017b; Yan et al., 2018; Yang et al., 2018; Walker et al., 2017; Kim et al., 2019) or face landmarks (Yan et al., 2018; Yang et al., 2018). Moreover, they made oversimplified assumptions of the future by using a deterministic loss or assuming homogeneous content. We therefore argue that the benefit of hierarchical model has been underestimated and their impact on long-term video prediction has not been properly demonstrated.

In this paper, we propose a hierarchical model with a general structure representation (i.e., dense semantic label map) for long-term video prediction with complex scenes. We abstract the scene as categorical labels for each pixel, and predict the motion and content change in this label space

![](images/a1784086083eb8570276600737224132ca7c62e1a266f0e6d7d74e57cf96e74d.jpg)  
Figure 1: The overall framework of the proposed hierarchical approach. Given the context frames and its label maps extracted by the segmentation network, our model predicts the future frames by estimating the semantic label maps using a stochastic sequence estimator (Section 2.1) and converting the predicted labels to RGB frames by using a conditional image sequence generator (Section 2.2).

using variational sequence models. Given the context frames and the predicted label maps, we then generate the textures by translating the sequence of label maps to the RGB frames. As dense label maps are generic and universal, we can learn comprehensive scene dynamics from object motion to even dramatic scene change. We can also capture the multi-modal dynamics in the dense label space with the stochastic prediction of the variational sequence model. Our experiments demonstrate that we can generate a surprisingly long-term future of videos, from driving scenes to human dancing, including the complex motion of multiple objects and even an evolution of the content in a distant future. We also show that the predicted frame quality is preserved through time, which enables persistent future prediction virtually near-infinite time horizon. For scalable evaluation of long-term prediction at this scale, we also propose a novel metric called shot-wise FVD, which enables the evaluation of spatio-temporal prediction quality without ground-truth frames and is consistent with human perception.

# 2 METHOD

Given the context frames  $\mathbf{x}_{1:C} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_C\}$ , our goal is to synthesize the future frames  $\mathbf{x}_{C+1:T} = \{\mathbf{x}_{C+1}, \mathbf{x}_{C+2}, \dots, \mathbf{x}_T\}$  up to an arbitrary long-term future  $T$ . Let  $\mathbf{s}_t \in \mathbb{R}^{W \times H \times N}$  denote a dense label map of the frame  $\mathbf{x}_t$  defined over  $N$  categories, which is inferred by the pre-trained semantic segmentation model<sup>1</sup>. Then given the context frames  $\mathbf{x}_{1:C}$  and the label maps  $\mathbf{s}_{1:C}$ , our hierarchical framework synthesizes the future frames  $\hat{\mathbf{x}}_{C+1:T}$  by the following steps.

- The structure generator takes the context label maps as input and produces a sequence of the future label maps by  $\hat{\mathbf{s}}_{C+1:T} \sim G_{\mathrm{struct}}(\mathbf{s}_{1:C}, \mathbf{z}_{1:T})$ , where  $\mathbf{z}_t$  denotes the latent variable encoding the stochasticity of the structure.  
- Given the context frames and the predicted structures, the image generator produces the RGB frames by  $\hat{x}_{C + 1:T} \sim G_{\mathrm{image}}(\mathbf{x}_{1:C}, \{\mathbf{s}_{1:C}, \hat{\mathbf{s}}_{C + 1:T}\})$ .

Figure 1 illustrates the overall pipeline. Note that there are various factors beyond motion that make spatio-temporal variations in the label maps, such as the emergence of new objects, partial observability, or even the dramatic scene changes by the global camera motion (e.g., panning). By learning to model these dynamics in the semantic labels with the stochastic sequence estimator and conditioning the video generation with the estimated labels, the proposed hierarchical model can synthesize convincing frames into the very long-term future. Below, we describe each component.

# 2.1 SEQUENTIAL DENSE STRUCTURE GENERATOR

Our structure generator models the dynamics in label maps by  $p(\mathbf{s}_{\leq T}|\mathbf{z}_{\leq T}) = \prod_{t=1}^{T}p(\mathbf{s}_t|\mathbf{s}_{< t},\mathbf{z}_{\leq t})$ . We employ the sequence model based on VAE (Denton & Fergus, 2018) since (1) it provides a probabilistic framework to handle stochastic variations of the structure and (2) it can easily incorporate discrete sequences (i.e., label maps). Specifically, we optimize the following variational

lower-bound of the sequence:

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {\leq T} | \mathbf {s} _ {\leq T})} [ \log p _ {\theta} (\mathbf {s} _ {t} | \mathbf {z} _ {\leq t}, \mathbf {s} _ {<   t}) ] - \beta D _ {K L} (q _ {\phi} (\mathbf {z} _ {t} | \mathbf {s} _ {\leq t}) | | p _ {\psi} (\mathbf {z} _ {t} | \mathbf {s} _ {<   t})) \tag {1}
$$

where  $q_{\phi}(\mathbf{z}_{\leq T}|\mathbf{s}_{\leq T})$  is the approximated posterior distribution and  $p_{\psi}(\mathbf{z}_t|\mathbf{s}_{< t})$  is the prior distribution. At each step, it learns to represent and reconstruct each label map  $\mathbf{s}_t$  through CNN-based encoder  $f^{\mathrm{enc}}$  and decoder  $f^{\mathrm{dec}}$ , respectively, while the temporal dependency of the label maps is modeled stochastically by the two LSTMs as follows:

$$
\mu_ {\phi} (t), \sigma_ {\phi} (t) = \operatorname {L S T M} _ {\phi} (\mathbf {h} _ {t}) \quad \text {w h e r e} \quad \mathbf {h} _ {t} = f ^ {\mathrm {e n c}} (\mathbf {s} _ {t}),
$$

$$
\mathbf {z} _ {t} \sim \mathcal {N} \left(\mu_ {\phi} (t), \sigma_ {\phi} (t)\right), \\ \mathbf {x} _ {t} = \operatorname {L S T M} _ {C} \left(\mathbf {h} _ {t - 1}, \mathbf {z} _ {t}\right) \quad \text {w h e r e} \quad \mathbf {h} _ {t} = f ^ {\mathrm {e n c}} (s _ {t - 1}) \tag {2}
$$

$$
\mathbf {g} _ {t} = \operatorname {L S T M} _ {\theta} \left(\mathbf {h} _ {t - 1}, \mathbf {z} _ {t}\right) \quad \text {w h e r e} \quad \mathbf {h} _ {t - 1} = f ^ {\mathrm {e n c}} \left(\mathbf {s} _ {t - 1}\right), \tag {2}
$$

$$
\mathbf {s} _ {t} = f ^ {\mathrm {d e c}} \left(\mathbf {g} _ {t}\right),
$$

where  $\mathrm{LSTM}_{\theta}$  and  $\mathrm{LSTM}_{\phi}$  respectively approximate the generative and the posterior distributions recurrently up to the time step  $t$ . Unlike Denton & Fergus (2018); Villegas et al. (2019) that exploit skip-connection from the last observed context frame during both training and testing time, we skip hidden representations of the encoder to the decoder at every time step during testing to handle long-term dynamics in structure. We also apply aggressive dilated convolutions to encode global motion context. Please see Appendix C.1 for more detail.

During training, we apply teacher-forcing by feeding ground-truth label maps and sampling the latent  $\mathbf{z}_t$  from the posterior distribution  $\mathcal{N}(\mu_{\phi}(t),\sigma_{\phi}(t)) = \mathrm{LSTM}_{\phi}(f^{\mathrm{enc}}(\mathbf{s}_t))$ . During inference, we recursively generate label maps by (1) sampling  $\mathbf{z}_t$  from the prior distribution  $\mathcal{N}(\mu_{\psi}(t),\sigma_{\psi}(t)) = \mathrm{LSTM}_{\psi}(f^{\mathrm{enc}}(\hat{\mathbf{s}}_{t - 1}))$ , (2) producing the frame  $\hat{\mathbf{s}}_t$  through the decoder, and (3) discretizing the predicted label map by taking pixel-wise maximum. Such discretization provides additional merits for being more robust against error propagation than continuous structures such as 2D keypoints (Villegas et al., 2017b; Yan et al., 2018; Yang et al., 2018) or optical flow (Walker et al., 2017).

Extension to object boundary prediction When the pre-trained instance-wise segmentation model is available, we can optionally extend the structure generator to jointly predict the object boundary maps. Such structure can add a notion of object instance, and is useful to improve the image generator in sequences with many occluding objects. Let  $\mathbf{e}_t \in \{0,1\}^{W \times H}$  denotes the object boundary map at frame  $\mathbf{x}_t$ . Then we train the denoising autoencoder  $\hat{\mathbf{e}}_t = G_{\mathrm{edge}}(\mathbf{s}_t)$  that produces a boundary map given a label map for each frame<sup>1</sup>. We train  $G_{\mathrm{edge}}$  using the conditional GAN objective to maximize the joint probability between the label and boundary maps  $p(\mathbf{s}_t, \mathbf{e}_t)$ . The boundary and label maps are then combined as the output of the structure generator  $\bar{\mathbf{s}}_t = [\hat{\mathbf{s}}_t, \hat{\mathbf{e}}_t]$ , and is used as an input to the image generator described below. See Appendix C.2 for implementation details.

# 2.2 STRUCTURE-CONDITIONAL Pixel SEQUENCE GENERATOR

Given a sequence of the structures and the context frames, the image generator learns to model the conditional distribution of the RGB frames by  $p(\mathbf{x}_{\leq T}|\mathbf{s}_{\leq T}) = \prod_{t=1}^{T}p(\mathbf{x}_t|\mathbf{x}_{<t},\mathbf{s}_{\leq t})$ . We formulate this task as a video-to-video translation problem, and employ a state-of-the-art conditional video generation model (Wang et al., 2018). Specifically, the video synthesis network  $F$  in Wang et al. (2018) consists of three main components; the generator  $H$ , occlusion mask predictor  $M$ , and optical flow estimator  $W$ , which are combined to generate each frame  $\hat{\mathbf{x}}_t$  by the following operation:

$$
\hat {\mathbf {x}} _ {t} = F \left(\hat {\mathbf {x}} _ {t - \tau : t - 1}, \mathbf {s} _ {t - \tau : t}\right) = \left(1 - \mathbf {m} _ {t}\right) \odot \hat {\mathbf {w}} _ {t - 1} + \mathbf {m} _ {t} \odot \mathbf {h} _ {t} \tag {3}
$$

where  $\hat{\mathbf{w}}_{t-1} = W(\hat{\mathbf{x}}_{t-\tau:t-1}, \mathbf{s}_{t-\tau:t})$  is the warped previous frame  $\hat{\mathbf{x}}_{t-1}$  using the estimated optical flow,  $\mathbf{h}_t = H(\hat{\mathbf{x}}_{t-\tau:t-1}, \mathbf{s}_{t-\tau:t})$  is the hallucinated frame at  $t$ , and  $\mathbf{m}_t = M(\hat{\mathbf{x}}_{t-\tau:t-1}, \mathbf{s}_{t-\tau:t})$  is the soft occlusion mask blending  $\hat{\mathbf{w}}_{t-1}$  and  $\mathbf{h}_t$ . Unlike models synthesizing future frames by transforming the context frames  $\mathbf{x}_C$  (Villegas et al., 2019; 2017b; Yan et al., 2018; Tulyakov et al., 2018), this model is appropriate to synthesize the long-term future since it can handle both transformation of the existing objects (via  $\hat{\mathbf{w}}_{t-1}$ ) and synthesis of the emerging objects (via  $\mathbf{h}_t$ ).

To ensure both frame-level and video-level generation quality, the video synthesis network  $F$  is trained against a conditional image discriminator  $D_{I}$  and a conditional video discriminator  $D_{V}$  through adversarial learning by

$$
\mathcal {L} _ {I} (F, D _ {I}) = \mathbb {E} _ {\phi_ {I} (\mathbf {x} _ {\leq T}, \mathbf {s} _ {\leq T})} [ \log D _ {I} (\mathbf {x} _ {i}, \mathbf {s} _ {i}) ] + \mathbb {E} _ {\phi_ {I} (\hat {\mathbf {x}} _ {\leq T}, \mathbf {s} _ {\leq T})} [ \log (1 - D _ {I} (\hat {\mathbf {x}} _ {i}, \mathbf {s} _ {i})) ], \tag {4}
$$

$$
\mathcal {L} _ {V} (F, D _ {V}) = \mathbb {E} _ {\phi_ {V} (\mathbf {w} _ {<   T}, \mathbf {x} _ {\leq T}, \mathbf {s} _ {\leq T})} [ \log D _ {V} (\mathbf {x} _ {i - 1: i - \tau^ {\prime}}, \mathbf {w} _ {i - 2: i - \tau^ {\prime}}) ] +
$$

$$
\mathbb {E} _ {\phi_ {V} \left(\mathbf {w} _ {<   T}, \hat {\mathbf {x}} _ {\leq T}, \mathbf {s} _ {\leq T}\right)} \left[ \log \left(1 - D _ {V} \left(\hat {\mathbf {x}} _ {i - 1: i - \tau^ {\prime}}, \mathbf {w} _ {i - 2: i - \tau^ {\prime}}\right)\right) \right], \tag {5}
$$

where  $\mathcal{L}_I(F, D_I)$  and  $\mathcal{L}_V(F, D_V)$  are frame-level and video-level adversarial losses, respectively. For efficient training, we follow (Wang et al., 2018) to adopt sampling operators  $\phi_I(\mathbf{x}_{\leq T}, \mathbf{s}_{\leq T}) = (\mathbf{x}_i, \mathbf{s}_i)$  and  $\phi_V(\mathbf{w}_{<T}, \mathbf{x}_{\leq T}, \mathbf{s}_{\leq T}) = (\mathbf{w}_{i-2:i-\tau'}, \mathbf{x}_{i-1:i-\tau'}, \mathbf{s}_{i-1:i-\tau'})$  for frame and video-level adversarial learning objectives, respectively, where  $i$  is an integer sampled from  $\mathrm{U}(1,T)$  and  $\mathrm{U}(\tau' + 1, T + 1)$  for frame sampling operator  $\phi_I$  and video sampling operator  $\phi_V$ , respectively. Then the final learning objective is formulated by

$$
\min  _ {F} \max  _ {D _ {I}, D _ {V}} \mathcal {L} _ {I} (F, D _ {I}) + \mathcal {L} _ {V} (F, D _ {V}). \tag {6}
$$

During training, the model is trained to estimate a frame sequence using the ground-truth structures, while we use the predicted structures by the structure generator during testing. See Appendix C.3 for implementation details.

# 3 RELATED WORK

The task of predicting future video frames has been extensively studied over the past few years (Villegas et al., 2019; Denton & Fergus, 2018; Lee et al., 2018; Babaeizadeh et al., 2018; Vondrick et al., 2016; Finn et al., 2016; Clark et al., 2019; Castrejon et al., 2019). Early approaches focus on modeling simple and deterministic dynamics using regression loss (Srivastava et al., 2015; Ranzato et al., 2014) or predictive coding (Lotter et al., 2017). However, such deterministic models may not be appropriate for modeling stochastic variations in real-world videos. Recently, deep generative models have been employed to model dynamics in complex videos. Babaeizadeh et al. (2018) proposed a variational approach for modeling stochasticity in a sequence. Denton & Fergus (2018) incorporated more flexible frame-wise inference models. The prediction quality of the variational models has been further improved by employing rich structure in latent variables (Castrejon et al., 2019), maximally increasing the network parameters (Villegas et al., 2019), or incorporating adversarial loss (Vondrick et al., 2016; Clark et al., 2019; Tulyakov et al., 2018; Villegas et al., 2017a). Despite their success, these approaches still fail to generate long-term videos and the common artifacts were losing object structures, switching object categories, etc. Our approach addresses these issues by explicitly predicting dynamics in a low-dimensional label map using a hierarchical model.

Hierarchical models studied in the past for video prediction were usually in specific video domains (Villegas et al., 2017b; Wichers et al., 2018; Liang et al., 2017; Yan et al., 2018; Walker et al., 2017; Minderer et al., 2019). Villegas et al. (2017b) employed LSTMs to predict human body joints and visual analogy making to create textures, which is extended by Wichers et al. (2018) and Minderer et al. (2019) to unsupervised approaches. Other approaches employed sequential VAE (Yan et al., 2018) or GAN (Yang et al., 2018) to predict the human body posture. However, these approaches are designed specifically for certain objects, and evaluated under simple videos containing only a single moving object. Also, they mostly utilized deterministic model to learn structure-level dynamics. These simplified assumptions on video content and dynamics limit their application to real-world videos. We propose to resolve these limitations by using dense semantic label maps as universal representation and stochastic sequential estimator for modeling video dynamics.

# 4 EXPERIMENTS

# 4.1 EVALUATION METRICS

Below we describe the evaluation metrics used in the experiment. Unless otherwise specified, we use  $64 \times 64$  images for all quantitative evaluation for fair comparison to the existing works. Full videos are available on the anonymous website: https://bit.ly/2EyDSem.

Short-term prediction We employ three conventional metrics in literature to evaluate the prediction performance on short-term videos (i.e., less than 50 frames): VGG cosine similarity (CSIM) measuring the frame-wise perceptual similarity using VGG features, mean Intersection-over-Union (mIoU) measuring the structure-level similarity using the label maps extracted by the pre-trained

segmentation network (Zhu et al., 2019) and Fréchet Video Distance (FVD) (Unterthiner et al., 2018) measuring a Fréchet distance between the ground-truth videos and the generated ones in a video representation space. Detailed evaluation protocols are described in Appendix A.

Long-term prediction Compared to short-term prediction, evaluating prediction quality of arbitrary long-term video is challenging for a number of reasons. First, the ground-truth videos are seldom available at this scale, making it impossible to adopt metrics based on frame-wise comparison (e.g., CSIM, mIoU). Second, the uncertainty of future prediction increases exponentially with time, making it intractable to employ density-based metrics (e.g., FVD) as it requires exponentially many samples. Since our goal is to demonstrate video prediction at the scale of hundreds to thousands of frames, we introduce a novel metric based on FVD, which enables evaluation of temporal and frame-level synthesis quality without ground-truth sequences and allows tractable evaluation through time. Specifically, we introduce a shot-level video-quality evaluation metric as

$$
\operatorname {S h o t F V D} (t) = \sum_ {t = 1} ^ {T} \operatorname {F V D} \left(\hat {X} _ {t: t + \omega - 1}, X _ {\omega}\right) \tag {7}
$$

which computes a FVD between the ground-truth shots  $X_{\omega} = \{X_{\omega}^{1}, \dots, X_{\omega}^{L}\}$  and the predictions  $\hat{X}_{t:t + \omega - 1} = \{\hat{X}_{t:t + \omega - 1}^{1}, \dots, \hat{X}_{t:t + \omega - 1}^{M}\}$  in a sliding window manner, where  $L$  denotes the total number of overlapping shots in the training video and  $M$  denotes the number of predicted shots. The shot-wise FVD evaluates the synthesis quality in a short interval defined by  $\omega$  through time. We show in the experiment that it is indeed aligned well with human perception. We also compute Inception Score (Salimans et al., 2016) for frame-level quality evaluation, which does not require ground-truths thus appropriate for long-term evaluation. Please find Appendix A for more details.

Human evaluation We conduct a user study on Amazon Mechanical Turk (AMT). To evaluate the quality at different time scales, we presented 5 second (50 frames) videos extracted at 1, 250, 400th predicted frames for all methods and counted their chosen ratio as the best. More details are in Appendix A.

# 4.2 RESULTS ON HUMAN DANCING SEQUENCES

We first present our results on human dancing videos collected from the Web (Wang et al., 2018).

Baselines We compare our method with two state-of-the-art video prediction models. SVG-extend (Villegas et al., 2019) is directly operating on RGB frames via a stochastic estimator and serves as our baseline for a non-hierarchical model. Following the paper, we employ the largest model with maximum parameters for fair comparison. We employ Villegas et al. (2017b) as a hierarchical model designed specifically for long-term prediction of human motion using pose (Newell et al., 2016). All models are trained to predict 40 future frames given 5 context frames.

Short-term prediction results We evaluate the short-term prediction performance by synthesizing 50 future frames given the 5 context frames and comparing them to groundtruths. Table 1 summarizes the quantitative evaluation results (see Figure A for qualitative results). Even in a short prediction interval, our method generates substantially higher quality samples than the baselines in terms of modeling appearance (CSIM), structure (mIoU), and motion (FVD). This is

because the dancing sequences contain rapid and complex variations in dynamics, making prediction task particularly challenging. We discuss more detailed analysis across methods in a long-term prediction task below.

Table 1: Quantitative comparisons of short-term prediction results.  

<table><tr><td>Model</td><td>CSIM(↑)</td><td>mIoU(↑)</td><td>FVD(↓)</td></tr><tr><td>SVG-extend</td><td>0.6654</td><td>0.0519</td><td>2125.29</td></tr><tr><td>Villegas et al.</td><td>0.7637</td><td>0.1755</td><td>1987.55</td></tr><tr><td>Ours</td><td>0.8164</td><td>0.3454</td><td>1398.98</td></tr></table>

Long-term prediction results We evaluate 500 frame prediction results based on shot-wise FVD, frame-wise Inception score, and human evaluation. Figure 2 and 3 summarize the quantitative and qualitative comparisons, respectively. We notice that the SVG-extend fails to model complex dynamics in dancing even early in prediction. This is because the dancing motions involve fast transition and frequent self-occlusions, resulting in catastrophic error propagation through time. On the other hand, as shown in the Inception score, Villegas et al. (2017b) produces much reasonable future frames as it exploits human body joints for generation. However, the deterministic LSTM module is not strong enough to capture complex dancing motions, and ends up generating static postures in the long term that leads to very high FVD scores. In contrast, we observe that our method generates both realistic frames and convincing motions, leading to stable Inception and FVD scores

![](images/cd06c15182a2b5c0f07604f4a5d7efcfe732fa5eb3f71f85727fd6836d1409a5.jpg)

![](images/67b66bdb9d40a043d8c13ddef04b2e6f576a3bf749eeac0d97f197190f2ca248.jpg)

(c) Human evaluation (most-preferred ratio)  

<table><tr><td>Model</td><td>t=1</td><td>t=250</td><td>t=400</td></tr><tr><td>SVG-Extend</td><td>3.9</td><td>2.3</td><td>3.1</td></tr><tr><td>Villegas et al.</td><td>6.6</td><td>9.9</td><td>9.1</td></tr><tr><td>Ours</td><td>89.5</td><td>87.8</td><td>87.8</td></tr></table>

![](images/b4632793804ff68c37d74e10cd1aa060f46862cac29ee591d5fa7269583fdd8c.jpg)  
Figure 2: Quantitative comparisons of the long-term prediction on human dancing sequences.  
Figure 3: Qualitative comparisons of long-term video generation results across models on human dancing sequences. All models are conditioned on the same context frames. Click the image to play the video in a browser. More results are available at https://bit.ly/2EyDSem.

in the long-term future. The human evaluation (Figure 2(c)) also shows the consistent results that our method outperforms all methods from the short- to long-term prediction. When our method is applied to predict up to 2040 frames in  $128 \times 128$  resolution (see Figure C in the appendix), we observe that it generates future frames that are convincing and involve diverse dance motions without noticeable quality degradation through time. In addition to the synthesis quality, Figure B illustrates that our method can generate diverse and interesting motions with the stochastic structure estimator.

# 4.3 RESULTS ON KITTI BENCHMARK

Baselines In addition to SVG-extend, we employ the future segmentation model (Bhattacharyya et al., 2019) (Bayes-WD-SL) that predicts future dense label maps as a strong baseline for hierarchical model. Since it generates only the label maps, we employ the same image generator with ours to produce RGB frames from the predicted label maps. All models are trained to predict 15 future frames given 5 context frames. Please refer to the appendix for the detailed architecture settings.

Short-term prediction results Similar to the previous section, we evaluate the short-term prediction quality by synthesizing 50 frames given the 5 frames. Table 2 summarizes the quantitative results (see Figure D for qualitative analysis). We observe that SVG-extend performs worse in terms of structural accuracy (mIoU) and motion (FVD), as it loses the object structure and generates arbitrary pixel dynamics.

The Bayes-WD-SL performs better in terms of mIoU, as it conditions the frame prediction with the structure estimation. However, as shown high FVD score, it fails to model temporal variations in the structure, largely due to its feedforward architecture. This unstructured motion leads to disastrous failure in a long-term prediction, as we will discuss later. In contrary, our method generates semantically accurate motions and structures, and substantially outperforms the others in all metrics.

Long-term prediction results Figure 4 and 5 summarize the quantitative and qualitative comparisons, respectively. In all metrics, our method outperforms the other baselines with substantial margins, showing that our method synthesizes both high-quality frames and motion. We notice that

Table 2: Quantitative comparisons of short-term prediction results.  

<table><tr><td>Model</td><td>CSIM(↑)</td><td>mIoU(↑)</td><td>FVD(↓)</td></tr><tr><td>SVG-extend</td><td>0.6664</td><td>0.3529</td><td>1448.84</td></tr><tr><td>Bayes-WD-SL</td><td>0.6533</td><td>0.4225</td><td>956.05</td></tr><tr><td>Ours</td><td>0.6789</td><td>0.5137</td><td>762.73</td></tr></table>

![](images/a1c1d08f5c9d989231d59af53f2a01be9c9655d75e28c429dc9cff568eb77469.jpg)  
(a) Inception score

![](images/81885424667c29b81aa7485c55a6f56b8be2484248553decba7b31a85baf4629.jpg)  
(b) Shot-wise FVD

(c) Human evaluation (most-preferred ratio)  

<table><tr><td>Model</td><td>t=1</td><td>t=250</td><td>t=400</td></tr><tr><td>SVG-Extend</td><td>13.1</td><td>22.9</td><td>27.2</td></tr><tr><td>Bayes-WD-SL</td><td>23.0</td><td>6.6</td><td>6.8</td></tr><tr><td>Ours</td><td>63.9</td><td>70.5</td><td>66.0</td></tr></table>

![](images/3ddf8e6b84dd2551fadc32e2481aaeeede60349588d7b283f28e517a2ef931e7.jpg)  
Figure 4: Quantitative comparisons of the long-term generation quality on KITTI Benchmark.

![](images/d8daae5bb60719b5d5ee55695d29130a938f2e0604354b61759c9b05040ca602.jpg)  
Figure 5: Qualitative comparisons of long-term video generation results across models. Although all models succeed in generating plausible frames in a short-term ( $t = 1 \sim 40$ ), only our approach can generate persistent and convincing futures even in the end ( $t = 251 \sim 500$ ). Click the image to play the video in a browser. More results are available at https://bit.ly/2EyDSem.  
Figure 6: Long-term prediction results on a high-resolution KITTI sequence. Both the frames and the label maps are predicted by our method. Our method can generate high-resolution frames  $(256 \times 256$  pixels) into the long-term future without particular quality degradation. Click the image to play the video in a browser. More results are available at https://bit.ly/2EyDSem.

SVG-extend simulates arbitrary pixel motions, resulting in relatively constant Inception and shotwise FVD scores through time. However, these unstructured motions lead to rapid destruction in recognizable concepts in the synthesized frames (Figure 5). The hierarchical model (Bayes-WD-SL) generates more convincing frames in a short-term as shown in lower shot-wise FVD, but fails to extrapolate in long-term. Importantly, such trends in shot-wise FVD (Figure 4(b)) aligns well with human evaluation results (Figure 4(c)), showing that it is appropriate metric for evaluation of long-term prediction. Interestingly, we observe that the Inception Score of our method increases through time. It is because the long-term prediction by our method often leads to scenes with simple and typical structures, such as a highway, where the image generator can produce more high-quality frames than complicated scenes appearing early in the test videos. Note that such transition of the scene is still reasonable as it is frequently observed in the training data. Nonetheless, we observe that our method generates reasonable sequences through time while maintaining its quality in a reasonable range even in a distant future. Finally, Figure 6 illustrates the prediction results over 2500 future frames. We observe that the generated sequences are reasonable in both structure and motion, and capture interesting translations of the scenes through time (e.g., from suburban to rural areas).

Table 3: Comparison to future segmentation methods on Cityscapes dataset.  

<table><tr><td>Model</td><td>person</td><td>rider</td><td>car</td><td>truck</td><td>bus</td><td>train</td><td>motorcycle</td><td>bicycle</td><td>mIoU GT</td></tr><tr><td>S2S (Luc et al., 2017)</td><td>0.37</td><td>0.18</td><td>0.70</td><td>0.43</td><td>0.55</td><td>0.26</td><td>0.27</td><td>0.38</td><td>0.39</td></tr><tr><td>F2F (Luc et al., 2018)</td><td>0.33</td><td>0.20</td><td>0.72</td><td>0.53</td><td>0.58</td><td>0.38</td><td>0.30</td><td>0.25</td><td>0.41</td></tr><tr><td>Ours</td><td>0.41</td><td>0.28</td><td>0.77</td><td>0.75</td><td>0.73</td><td>0.49</td><td>0.40</td><td>0.45</td><td>0.54</td></tr></table>

![](images/3d93cad8ccbca1444038bd77be48ce1879198443e6f9ba0e914565843a4f4dfc.jpg)  
Figure 7: Long-term prediction results on  $256 \times 512$  Cityscapes dataset. Click the image to play the video in a browser. For more qualitative results and details, please refer to Figure G and Figure H in the appendix and the project website: https://bit.ly/2EyDSem.

# 4.4 RESULTS ON CITYSCAPES DATASET

To further evaluate the quality of structure prediction, we compare our structure generator with existing future segmentation methods that directly predict the segmentation map of the future frames.

Baselines We compare our method with S2S (Luc et al., 2017) and F2F (Luc et al., 2018), which are the state-of-the-arts in the future segmentation literature. S2S is a deterministic model based on fully-convolutional network that predicts future semantic label maps in a multi-scale and autoregressive manner. F2F is an extension of S2S to future instance segmentation task by predicting the high-level feature maps of Mask R-CNN (He et al., 2017). Similar to KITTI, we use the label maps extracted by the pre-trained semantic segmentation network (Zhu et al., 2019) for the training.

Evaluation We follow the standard evaluation protocols in the literature (Luc et al., 2017; 2018) for quantitative and qualitative comparisons. For each validation sequence, each model is provided with 4 contexts, and produces up to 29th frame. Then, we compute mIoUs between ground-truths and predictions at 20th time-step. For fair comparisons with the future instance segmentation model (i.e., F2F), we follow Luc et al. (2018) and measure mIoUs only for moving objects. For S2S and F2F, we use publicly available pre-trained models provided by the authors.

Results Table 3 presents the quantitative evaluation results (See Figure G for qualitative results). The two baselines produce blurry predictions, resulting in inaccurate structures and mislabelings. These problems have been widely observed in the literature and are attributed to the inability to handle stochasticity (Denton & Fergus, 2018). Our method outperforms the two baselines in all classes as it can handle highly stochastic nature of complex driving scenes. Figure 7 and Figure H illustrates the long-term prediction results of our full model including the boundary map and image generator on  $256 \times 512$  resolution. It shows that our method can generate convincing future even in extremely complex and high-resolution videos. See Appendix B.3 for more detailed discussion.

# 5 CONCLUSION

We proposed a hierarchical approach for persistent video prediction. We revisit the hierarchical model with stochastic estimation of dense label maps and integration of image generator robust against temporal misprediction in pixels and structures. Our experimental results show that a carefully designed hierarchical model can learn to synthesize a very long-term future with convincing structures and dynamics even in complex videos. By scaling up the video prediction to order-of-magnitudes longer, we believe that our work can help the future research to focus on more challenging problems of learning a long-term dependency and eliminating explicit structure estimation.

# REFERENCES

Riza Alp Güler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In CVPR, 2018.  
Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H. Campbell, and Sergey Levine. Stochastic variational video prediction. In ICLR, 2018.  
Apratim Bhattacharyya, Mario Fritz, and Bernt Schiele. Bayesian prediction of future street scenes using synthetic likelihoods. In *ICLR*, 2019.  
J. Carreira and A. Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.  
Lluis Castrejon, Nicolas Ballas, and Aaron Courville. Improved conditional vrnns for video prediction. In ICCV, 2019.  
Aidan Clark, Jeff Donahue, and Karen Simonyan. Efficient video generation on complex datasets. arXiv preprint arXiv:1907.06571, 2019.  
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In CVPR, 2016.  
Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In ICML, 2018.  
Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. In NeurIPS. 2016.  
Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. *IJRR*, 2013.  
David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In NeurIPS. 2018.  
Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, 2019.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.  
K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask r-cnn. In ICCV, 2017.  
Minh Hoai and Fernando Torre. Max-margin early event detectors. IJCV, 2013.  
Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016.  
Yunj Kim, Seonghyeon Nam, In Cho, and Seon Joo Kim. Unsupervised keypoint learning for guiding class-conditional video prediction. In NeurIPS. 2019.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS. 2012.  
Tian Lan, Tsung-Chuan Chen, and Silvio Savarese. A hierarchical representation for future action prediction. In ECCV, 2014.  
Alex X. Lee, Richard Zhang, Frederik Ebert, Pieter Abbeel, Chelsea Finn, and Sergey Levine. Stochastic adversarial video prediction. arXiv preprint arXiv:1804.01523, 2018.  
Xiaodan Liang, Lisa Lee, Wei Dai, and Eric P. Xing. Dual motion gan for future-flow embedded video prediction. In ICCV. 2017.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólar, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, 2014.  
William Lotter, Gabriel Kreiman, and David Cox. Deep predictive coding networks for video prediction and unsupervised learning. In ICLR, 2017.  
Pauline Luc, Natalia Neverova, Camille Couprie, Jacob Verbeek, and Yann LeCun. Predicting deeper into the future of semantic segmentation. In ICCV, 2017.  
Pauline Luc, Camille Couprie, Yann LeCun, and Jakob Verbeek. Predicting future instance segmentation by forecasting convolutional features. In ECCV. 2018.  
Matthias Minderer, Chen Sun, Ruben Villegas, Forrester Cole, Kevin P Murphy, and Honglak Lee. Unsupervised learning of object structure and dynamics from videos. In NeurIPS. 2019.  
Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Qureshi, and Mehran Ebrahimi. Edgeconnect: Structure guided image inpainting using edge prediction. In ICCV, 2019.  
Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016.  
Marc'Aurelio Ranzato, Arthur Szlam, Joan Bruna, Michael Mathieu, Ronan Collobert, and Sumit Chopra. Video (language) modeling: a baseline for generative models of natural videos. arXiv preprint arXiv:1412.6604, 2014.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. IJCV, 2015.  
M. S. Ryoo. Human activity prediction: Early recognition of ongoing activities from streaming videos. In ICCV, 2011.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, 2016.  
Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, and Wang-chun Woo. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In NeurIPS, 2015.  
Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. Unsupervised learning of video representations using lstms. In ICML, 2015.  
C. Szegedy, Wei Liu, Yangqing Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.  
Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. MoCoGAN: Decomposing motion and content for video generation. In CVPR, 2018.  
Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.  
Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin, and Honglak Lee. Decomposing motion and content for natural video sequence prediction. In ICLR. 2017a.  
Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryull Sohn, Xunyu Lin, and Honglak Lee. Learning to Generate Long-term Future via Hierarchical Prediction. In ICML, 2017b.  
Ruben Villegas, Arkanath Pathak, Harini Kannan, Dumitru Erhan, Quoc V Le, and Honglak Lee. High fidelity video prediction with large stochastic recurrent neural networks. In NeurIPS. 2019.  
Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. In NeurIPS. 2016.

Jacob Walker, Kenneth Marino, Abhinav Gupta, and Martial Hebert. The pose knows: Video forecasting by generating pose futures. In ICCV, 2017.  
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Video-to-video synthesis. In NeurIPS, 2018.  
Ruben Wichers, Nevan Villegas, Dumitru Erhan, and Honglak Lee. Hierarchical long-term video prediction without supervision. In ICML. 2018.  
Xinchen Yan, Akash Rastogi, Ruben Villegas, Kalyan Sunkavalli, Eli Shechtman, Sunil Hadap, Ersin Yumer, and Honglak Lee. Mt-vae: Learning motion transformations to generate multimodal human dynamics. In ECCV, 2018.  
Ceyuan Yang, Zhe Wang, Xinge Zhu, Chen Huang, Jianping Shi, and Dahua Lin. Pose guided human video generation. In ECCV, 2018.  
Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. In *ICLR*, 2016.  
Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. TPAMI, 2017.  
Yi Zhu, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn D. Newsam, Andrew Tao, and Bryan Catanzaro. Improving semantic segmentation via video propagation and label relaxation. In CVPR, 2019.
