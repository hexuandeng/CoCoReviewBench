# NEURAL SYNTHESIS OF BINAURAL AUDIO

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a neural rendering approach for binaural sound synthesis that can produce realistic and spatially accurate binaural sound in realtime. The network takes, as input, a single-channel audio source and synthesizes, as output, two-channel binaural sound, conditioned on the relative position and orientation of the listener with respect to the source. We investigate deficiencies of the  $\ell_2$ -loss on raw waveforms in a theoretical analysis and introduce an improved loss that overcomes these limitations. In an empirical evaluation, we establish that our approach is the first to generate spatially accurate waveform outputs (as measured by real recordings) and outperforms existing approaches by a considerable margin, both quantitatively and in a perceptual study. We will release a first-of-its-kind binaural audio dataset as a benchmark for future research.

# 1 INTRODUCTION

The rise of artificial spaces, in augmented and virtual reality, necessitates efficient production of accurate spatialized audio. Spatial hearing (the capacity to interpret spatial clues from binaural signals), not only helps us to orient ourselves in 3D environments, it also establishes immersion in the space by providing the brain with congruous acoustic and visual input (Hendrix & Barfield, 1996). Binaural audio (left and right ear) even guides us in multi-person conversations: consider a scenario where multiple persons are speaking in a video call, making it difficult to follow the conversation. In the same situation in a real environment we are able to effortlessly focus on the speech from an individual (Hawley et al., 2004). Indeed, auditory sensation has primacy over even visual sensation as an input modality for scene understanding: (1) reaction times are faster for auditory stimulus compared to visual stimulus (Jose & Praveen, 2010) (2) auditory sensing provides a surround understanding of space as opposed to the directionality of visual sensation. For these reasons, the generation of accurate binarual signal is integral to full immersion in artificial spaces.

Most approaches to binaural audio generation rely on traditional digital signal processing (DSP) techniques, where each component – head related transfer function, room acoustics, ambient noise – is modeled as a linear time-invariant system (LTI) (Savioja et al., 1999; Zotkin et al., 2004; Sunder et al., 2015; Zhang et al., 2017). These linear systems are well-understood, relatively easy to model mathematically, and have been shown to produce perceptually plausible results – reasons why they are still widely used. Real acoustic propagation, however, has nonlinear wave effects that are not appropriately modeled by LTI systems. As a consequence, DSP approaches do not achieve perceptual authenticity in dynamic scenarios (Brinkmann et al., 2017), and fail to produce metrically accurate results, i.e., the generated waveform does not resemble recorded binaural audio well.

In this paper, we present an end-to-end neural synthesis approach that overcomes many of these limitations by efficiently synthesizing accurate and precise binaural audio. The end-to-end learning scheme naturally captures the linear and nonlinear effects of sound wave propagation and, being fully convolutional, is efficient to execute on commodity hardware. Our major contributions are (1) a novel binarization model that outperforms existing state of the art, (2) an analysis of the shortcomings of the  $\ell_2$ -loss on raw waveforms and a novel loss mitigating these shortcomings, (3) a real-world binaural dataset captured in a non-anechoic room. $^{1}$

Related Work. State of the art DSP techniques approach binaural sound spatialization as a stack of acoustic components, each of which is an LTI system. As accurate wave-based simulation of room impulse responses is computationally expensive and requires detailed geometry and material information, most real-time systems rely on simplified geometrical models (Välimäki et al., 2012; Savioja

![](images/6c3a3e15863be8213a977749ab712bce0217605754d03b33dd988c5d44329d22.jpg)  
Figure 1: System Overview. Given the source and listener position and orientation  $c_{1:T}$  at each time step, a single-channel input signal  $x_{1:T}$  is transformed into a binaural signal. The neural time warping module learns an accurate warp from the source position to the listeners left and right ear while respecting physical properties like monotonicity and causality. The Temporal ConvNet models nuanced effects like room reverberations or head- and ear-shape related modifications to the signal.

& Svensson, 2015). Head-related transfer functions are measured in an anechoic chamber (Li & Peissig, 2020) and high-quality spatialization requires binaural recordings at almost  $10\mathrm{k}$  different spatial positions (Armstrong et al., 2018). To generate binaural audio the DSP-based binaural renderers typically perform a series of convolutions with these component impulse responses (Savioja et al., 1999; Zotkin et al., 2004; Sunder et al., 2015; Zhang et al., 2017). For a more detailed discussion, see Appendix A.3.

Given their success in speech synthesis (Wang et al., 2017), neural networks gained increased attention for audio generation recently. While most approaches focus on models in frequency domain (Choi et al., 2018; Vasquez & Lewis, 2019), raw waveform models were long neglected due to the difficulty to model long-range dependencies on a high-frequency audio signal. With the success of WaveNet (Van Den Oord et al., 2016) however, direct wave-to-wave modeling is of increasing interest (Fu et al., 2017; Luo & Mesgarani, 2018; Donahue et al., 2019) and shows major improvements in speech enhancement (Defossez et al., 2020) and denoising (Rethage et al., 2018), speech synthesis (Kalchbrenner et al., 2018), and music style translation (Mor et al., 2019).

More recently, first steps towards neural sound spatialization have been undertaken with a focus on predicting spatial sound conditioned on visual information. One of the first works by Morgado et al. (2018) aims to spatialize sound conditioned on  $360^{\circ}$  video. Yet, their work is limited to first order ambisonics and can not model detailed binaural effects. More closely related is a line of papers originating from the 2.5D visual sound system by Gao & Grauman (2019b). In this work, binaural audio is generated conditioned on a video frame embedding such that object locations can contribute to where sound comes from. Yang et al. (2020); Lu et al. (2019); Zhou et al. (2020) build upon the same idea. Unfortunately, all these works have in common that they pose the spatialization task as an upmixing problem, i.e., their models are trained with a mixture of left and right ear binaural recording as pseudo mono input. By design, these methods fail to model time delays and reverberation effects caused by the difference between source and listener position.

# 2 A NEURAL NETWORK FOR BINAURAL SYNTHESIS

We consider the problem where a monoaural (single-channel) signal  $x_{1:T} = (x_1, \ldots, x_T)$  of length  $T$  is to be transformed into a binaural (stereophonic) signal  $y_{1:T}^{(l)}, y_{1:T}^{(r)}$  representing the listener's left ear and right ear, given a conditioning temporal signal  $c_{1:T}$ . This conditioning signal is the position and orientation of source and listener, respectively. Here  $x_t$ , and correspondingly  $y_t^{(l)}$  and  $y_t^{(r)}$ , are scalars representing an audio sample at time  $t$ . In other words, we aim to produce a function,

$$
\left(y _ {t} ^ {(l)}, y _ {t} ^ {(r)}\right) = f (x _ {t - \Delta : t} | \boldsymbol {c} _ {t - \Delta : t}),
$$

where  $\Delta$  is a temporal receptive field. Each  $c_{t} \in \mathbb{R}^{14}$  contains the 3D position of source and listener (three values each) and their orientations as quaternions (four values each). Note that in practice,  $c$  often is of lower frequency than the input and output signals  $x_{1:T}$  and  $y_{1:T}^{(l/r)}$  - source and listener positions would likely not be updated at  $48\mathrm{kHz}$  but rather at typical camera frame rates such as  $30-120\mathrm{Hz}$ . To simplify notation, we assume that  $c$  has already been upsampled to the same temporal resolution as the audio signals.

Our overall framework is shown in Figure 1. A neural time warping module first warps the single-channel input signal  $x_{1:T}$  into a two-channel signal  $x_{1:T}^{(l/r)}$ , where the channels represent left and right ear. The time warping compensates for coarse temporal effects and differences in time of sound arrival at left and right ear caused by the distance between source and listener. The second block in Figure 1 is a stack of  $N$  layers, each of which is a conditioned hyper-convolution (see Section 2.2) followed by a sine activation, which has been shown to be beneficial for modeling higher frequencies (Sitzmann et al., 2020). Following the design of WaveNet, we use kernel size 2 and double the dilation factor in each layer to increase the receptive field. This temporal ConvNet models nuanced effects caused by room reverberations, head and ear shape, or changing head orientations.

# 2.1 NEURAL TIME WARPING

Time warping is the task of mapping a source temporal sequence onto a target sequence and has a long tradition in temporal signal processing. Most prominently, dynamic time warping (DTW) finds application in tasks like speech recognition (Juang, 1984) or audio retrieval (Deng & Leung, 2015). DTW can be characterized as finding a warpfield  $\rho_{1:T}$  that warps a source signal  $x_{1:T}$  to a target signal  $\hat{x}_{1:T}$  such that the distance between the signals is minimized,

$$
\rho_ {1: T} = \underset {\tilde {\rho} _ {1: T}} {\arg \min } \sum_ {t} \| \hat {x} _ {t} - x _ {\tilde {\rho} _ {t}} \|, \quad \text {w h e r e} \rho_ {t} \in \{1, \dots , T \}, \tag {1}
$$

where the warpfield is typically constrained to respect physical properties such as monotonicity  $(\rho_{t} \geq \rho_{t-1})$  and causality  $(\rho_{t} \leq t)$ .

For binaural audio, there is a clear monotonous and causal relationship between source and target signal but the target signal is unknown at inference time. Instead, a warpfield can be estimated from the conditioning input  $c_{1:T}$ , i.e., from the spatial position and orientation of source and listener. A simple, parameter-free approach is geometric warping based on the speed of sound  $\nu_{\mathrm{sound}}$  and the distance between source and listener. Let  $p_t^{(\mathrm{src})}$  and  $p_t^{(\mathrm{lstn})}$  be the source and listener positions at time  $t$  (which are part of  $c_t$ ). Then,

$$
\rho_ {t} ^ {\left(\text {g e o m}\right)} = t - \left\| \boldsymbol {p} _ {t} ^ {\left(\text {s r c}\right)} - \boldsymbol {p} _ {t} ^ {\left(\text {l s t n}\right)} \right\| \cdot \frac {\text {a u d i o s a m p l e r a t e}}{\nu_ {\text {s o u n d}}}. \tag {2}
$$

This approach, however, fails to model important nuances such as the displacement between the left and right ear or diffraction delays as sound travels around the listener's head rather than straight through. In order to correct for those effects that geometric warping can not model properly, we estimate a neural warpfield  $\rho_{1:T}^{(\mathrm{neural})} = \mathrm{WarpNet}(c_{1:T})$  and add it to the geometric warpfield (cf. Figure 1),

$$
\rho_ {t} = \sigma^ {(\text {w a r p})} \left(\rho_ {t - 1}, \hat {\rho} _ {t}\right) \quad \text {w i t h} \quad \hat {\rho} _ {t} := \rho_ {t} ^ {(\text {n e u r a l})} + \rho_ {t} ^ {(\text {g e o m})}, \tag {3}
$$

where  $\sigma^{(\mathrm{warp})}(\rho_{t - 1},\hat{\rho}_t) = \max (\rho_{t - 1},\min (t,\hat{\rho}_t))$  is a recursive activation function that ensures monotonicity and causality. The WarpNet is a shallow temporal convolutional network with four layers and 64 channels each.

The warped signal can now be computed using the predicted warpfield. Since the warpfield elements  $\rho_{t}$  are typically not integers, we define the warped signal  $\hat{x}_{1:T}$  to be the linear interpolation of the original signal  $x_{1:T}$  at positions  $\lfloor \rho_t\rfloor$  and  $\lceil \rho_t\rceil$

$$
\hat {x} _ {t} = \left(\left[ \rho_ {t} \right] - \rho_ {t}\right) \cdot x _ {\left[ \rho_ {t} \right]} + \left(\rho_ {t} - \left[ \rho_ {t} \right]\right) \cdot x _ {\left[ \rho_ {t} \right]}. \tag {4}
$$

In practice two warpfields are generated, one for each ear. Note how we explicitly enforce physical constraints in the warping by  $\sigma^{(\mathrm{warp})}$ :  $\min(t, \hat{\rho}_t)$  ensures causality by enforcing that the  $t$ -th element of the warfield can not be larger than  $t$  itself. Monotonicity is enforced by  $\max(\rho_{t-1}, \cdot)$ : if an element has been warped from  $\rho_{t-1}$  to position  $t-1$ , the next element at position  $t$  must be warped from  $\rho_{t-1}$  or a succeeding position. In contrast to related approaches such as deformable convolutions (Dai et al., 2017) and spatial transformer networks (Jaderberg et al., 2015), our neural time warping therefore allows for constrained warping of input signals with arbitrary lengths and directly models a physical phenomenon of sound.

# 2.2 CONDITIONED HYPER-CONVOLUTIONS

Raw waveform models where the output depends on an input signal and an additional conditioning temporal signal have primarily been studied in speech synthesis (Van Den Oord et al., 2016). The

predominant approach towards such conditional temporal convolutions is an additive combination of the input signal  $\boldsymbol{x}_{1:T}$  and the conditioning signal  $c_{1:T}$ , i.e.,  $\boldsymbol{z}_{1:T} = \mathbf{W} * \boldsymbol{x}_{1:T} + \mathbf{V} * c_{1:T} + \boldsymbol{b}$ , such that the result of the convolution at time  $t$  is

$$
\boldsymbol {z} _ {t} = \sum_ {k = 1} ^ {K} \mathbf {W} _ {:,: k} \boldsymbol {x} _ {t - k + 1} + \sum_ {k = 1} ^ {K} \mathbf {V} _ {:,: k} \boldsymbol {c} _ {t - k + 1} + \boldsymbol {b}. \tag {5}
$$

Here,  $\mathbf{W} \in \mathbb{R}^{C_{\mathrm{out}} \times C_{\mathrm{in}} \times K}$  and  $\mathbf{V} \in \mathbb{R}^{C_{\mathrm{out}} \times C_{\mathrm{cond}} \times K}$  are tensors containing the weights for temporal convolutions of the  $C_{\mathrm{in}}$ -dimensional input signal  $x_{1:T}$  and the  $C_{\mathrm{cond}}$ -dimensional conditional signal  $c_{1:T}$  with a kernel size of  $K$ . Note that the convolutional weights  $\mathbf{W}$  and  $\mathbf{V}$  in this formulation are constant over time. Binaural filters in traditional digital signal processing, on the contrary, depend on the position of the sound source.

Inspired by the DSP formulation, we predict the convolutional weights for the input  $\pmb{x}_{1:T}$  of a layer and the bias as functions of the conditioning input  $\pmb{c}_{1:T}$ ,

$$
\boldsymbol {z} _ {t} = \sum_ {k = 1} ^ {K} \left[ \mathcal {H} ^ {(\mathbb {W})} \left(\boldsymbol {c} _ {1: t}\right) \right] _ {:,: k} \boldsymbol {x} _ {t - k + 1} + \mathcal {H} ^ {(\boldsymbol {b})} \left(\boldsymbol {c} _ {1: t}\right). \tag {6}
$$

This formulation is similar to the use of hyper-networks in Ha et al. (2017) but rather than generating them from intermediate feature maps, weights are generated from the conditioning input  $c_{1:t}$  that contains physical information about the relation between source and listener.  $\mathcal{H}^{(\mathbb{W})}$  and  $\mathcal{H}^{(b)}$  are small convolutional hyper-networks that receive  $c_{1:t}$  as input and predict the convolutional weights and the bias as their output, respectively. Therefore, not only is the input to the convolutional layer a temporal sequence but the weights and biases change over time as well. We show in Appendix A.2 that if  $\mathcal{H}^{(\mathbb{W})}$  and  $\mathcal{H}^{(b)}$  are linear networks, hyper-convolutions equal equation 5 plus a bilinear term.

# 2.3 DEFICIENCIES OF THE  $\ell_2$ -LOSS ON RAW WAVEFORMS

Training a generative audio model with an  $\ell_2$ -loss on the raw waveform is generally considered to result in poor audio quality and distorted signals particularly for speech. Therefore, a number of mostly spectrogram oriented alternative loss functions have been introduced over recent years (Kolbæk et al., 2020). Here, we provide an analytical explanation for a fundamental problem of phase estimation with the  $\ell_2$ -loss on the waveform and show that a simple additional loss term mitigates the problem. While correct phase estimation is not critical for single-channel audio, it is crucial for binaural audio as our ears are sensitive to interaural time differences as small as  $10\mu s$  (Brown & Duda, 1998). To start the analysis, let

$$
\mathcal {L} _ {2} \left(y _ {1: T}, \hat {y} _ {1: T}\right) = \sum_ {t} \left(y _ {t} - \hat {y} _ {t}\right) ^ {2} \tag {7}
$$

be the time-domain  $\ell_2$ -loss between the predicted audio signal  $y_{1:T}$  and the target  $\hat{y}_{1:T}$  and let  $Y_k, \hat{Y}_k \in \mathbb{C}$  denote the  $k$ -th frequency component of  $y_{1:t}$  and  $\hat{y}_{1:T}$  in the Fourier domain. We denote the amplitude error and angular phase error of the  $k$ -th frequency component as

$$
\mathcal {L} ^ {(\mathrm {a m p})} \left(Y _ {k}, \hat {Y} _ {k}\right) = \left| \left| Y _ {k} \right| - \left| \hat {Y} _ {k} \right| \right| \quad \text {a n d} \quad \mathcal {L} ^ {(\mathrm {p h a s e})} \left(Y _ {k}, \hat {Y} _ {k}\right) = \angle \left(Y _ {k}, \hat {Y} _ {k}\right), \tag {8}
$$

where  $|\cdot|$  is the modulus (or magnitude) of the complex number.

Lemma 1. Let  $\hat{Y} \in \mathbb{C}$  be a fixed complex number and  $Y \in \mathbb{B}_{\varepsilon, \hat{Y}} = \{Y \in \mathbb{C} : |Y - \hat{Y}| = \varepsilon\}$  be any complex number that has distance  $\varepsilon$  from  $\hat{Y}$ . Then, the expected amplitude error and the expected angular phase error with respect to  $\hat{Y}$  are

$$
\mathbb {E} _ {Y} \left(\mathcal {L} ^ {(\mathrm {a m p})} (Y, \hat {Y})\right) = \frac {1}{2 \pi} | \hat {Y} | \int_ {- \pi} ^ {\pi} \left| \left| \frac {\varepsilon}{| \hat {Y} |} + e ^ {i \varphi} \right| - 1 \right| d \varphi \quad \text {a n d} \tag {9}
$$

$$
\mathbb {E} _ {Y} \left(\mathcal {L} ^ {\text {(p h a s e)}} (Y, \hat {Y})\right) = \frac {1}{2 \pi} \int_ {- \pi} ^ {\pi} \operatorname {a r c c o s} \frac {\operatorname {R e} \left(\frac {\varepsilon}{| \hat {Y} |} e ^ {i \varphi} + 1\right)}{\left| \frac {\varepsilon}{| \hat {Y} |} + e ^ {i \varphi} \right|} d \varphi . \tag {10}
$$

![](images/8e17e0db0aec0b8a622a08e025bfbfe476323e5ca53b79234f8c676473c2c489.jpg)  
(a)  $\mathbb{E}_Y\big(\mathcal{L}^{(\mathrm{amp})}(Y,\hat{Y})\big)$

![](images/389aca751947e00c8b9e1386009508978bd2fd8e1acc5a68711a2bde005cd7ef.jpg)  
(b)  $\mathbb{E}_Y\big(\mathcal{L}^{(\mathrm{phase})}(Y,\hat{Y})\big)$

Proof. See Appendix A.1.

![](images/63aa371fcb6c905f06a6fdc98256aa04bdb6987b0a2564a7259fd0af50efe3a6.jpg)  
Figure 2: Expected amplitude and phase error from Lemma 1 as a function of  $\ell_2$ -value  $\varepsilon$  and target signal energy  $|\hat{Y}|$ .

Using Parseval's theorem, we write the time-domain  $\ell_2$ -loss as the  $\ell_2$ -loss on the complex spectrum,

$$
\mathcal {L} _ {2} \left(y _ {1: T}, \hat {y} _ {1: T}\right) = \sum_ {k} \left| Y _ {k} - \hat {Y} _ {k} \right| ^ {2}. \tag {11}
$$

Now, consider a single summand from equation 11 and denote the distance  $|Y_k - \hat{Y}_k|$  as  $\varepsilon$ . Lemma 1 allows us to analyze the expected amplitude and phase errors along this  $k$ -th frequency component. In Figure 2 we plot equation 9 and equation 10 as a function of the  $\ell_2$ -value  $\varepsilon$  and the target energy  $|\hat{Y}|$ . There are two key insights. First, the expected amplitude error is low even for large  $\ell_2$ -values – that is, in the early stage of training – as long as the target signal has high energy (top right part of Figure 2a). The phase, on the contrary, is barely optimized at all early in training when the  $\ell_2$ -loss is large, even for high energy components, see Figure 2b. Second, over the course of training, i.e., when the  $\ell_2$ -loss decreases over time, the expected amplitude error among all target energies decreases. The expected phase error, on the other hand, improves primarily for high energy components and mid- and low energy components tend to have poor phase accuracy even for small  $\ell_2$ -values.

The above analysis shows that optimizing raw waveforms with a time-domain  $\ell_2$ -loss leads to a strong focus on fitting the amplitudes but accurate phase reconstruction falls short. Since the models have limited capacity, the training data usually can only be fit up to an  $\ell_2$ -loss  $\varepsilon_{\mathrm{min}}$ . If this  $\varepsilon_{\mathrm{min}}$  is not sufficiently small, the signal's amplitude can be modeled well but phase errors will always be significant. This can be critical since small amplitude errors lead to a slight change in speech coloration but phase errors introduce perceivable distortions. To overcome the deficiencies of the time-domain  $\ell_2$ -loss in phase optimization, we add an explicit phase term to the loss function,

$$
\mathcal {L} \left(y _ {1: T}, \hat {y} _ {1: T}\right) = \mathcal {L} _ {2} \left(y _ {1: T}, \hat {y} _ {1: T}\right) + \lambda \mathcal {L} ^ {\text {(p h a s e)}} \left(\operatorname {S T F T} \left(y _ {1: T}\right), \operatorname {S T F T} \left(\hat {y} _ {1: T}\right)\right), \tag {12}
$$

where  $\mathrm{STFT}(y_{1:T})$  is the short-term Fourier transform of the audio signal  $y_{1:T}$ .

# 3 EVALUATION

Dataset. We recorded a total of 2 hours of binaural data from eight different speakers, four male and four female. The listener is a mannequin equipped with binaural microphones in its ears. Participants were asked to walk around the mannequin an a circle with  $1.5\mathrm{m}$  radius and have an unscripted conversation with it. We used an OptiTrack system to track position and orientation of source and listener throughout the captures. To the best of our knowledge, this is the only in-the-wild (i.e., not recorded in an anechoic chamber) binaural dataset of such size. The dataset will be made available to the public on acceptance of the paper. We use a validation sequence and the last two minutes from each participant as test data and train the models on the remaining data. See Appendix A.4 for a more detailed description.

Network Architecture. The WarpNet architecture is as described in Section 2.1. The temporal convolutional network consists of three sequential blocks. Each block is a stack of ten hyperconvolution layers with 128 channels, kernel size 2, and the dilation size is doubled after each layer. We train our models for 100 epochs using an Adam optimizer. Learning rates are decreased if between two epochs the loss on the training set did not improve. At inference, our model can produce binaural audio in real-time.

![](images/1c43fe4c7cd889e2cf715ad93f0bd2f17c917cf50509a04169cc7ec21d96cb63.jpg)  
Figure 3: Development of phase- and amplitude error as the  $\ell_2$ -loss decreases during training.

![](images/0ca862e84fc7f375b133bb097d696d7a84ca62c8ca11233519f4572256961ae2.jpg)

Table 1: Comparison of commonly used losses for audio modeling to our proposed  $\ell_2 +$  phase loss.  

<table><tr><td></td><td>raw waveform (l2 error ×103)</td><td>power spectrum (l2 error)</td><td>phase spectrum (angular error)</td></tr><tr><td>power spectrum + phase copy</td><td>1.276</td><td>0.048</td><td>1.563</td></tr><tr><td>multiscale STFT</td><td>2.279</td><td>0.043</td><td>1.996</td></tr><tr><td>Si-SDR</td><td>0.798</td><td>0.222</td><td>1.507</td></tr><tr><td>cross entropy on μ-law encoding</td><td>0.161</td><td>0.039</td><td>1.199</td></tr><tr><td>l2</td><td>0.141</td><td>0.037</td><td>0.886</td></tr><tr><td>l2 + phase loss (equation 12)</td><td>0.167</td><td>0.048</td><td>0.807</td></tr></table>

# 3.1 LOSS EVALUATION

In order to empirically validate our findings from Section 2.3, we train our proposed network with time-domain  $\ell_2$ -loss only and with the loss proposed in equation 12. Figure 3 shows how the phase error and amplitude error develop during training as the time-domain  $\ell_2$ -loss decreases. The model trained with  $\ell_2$ -loss only (Figure 3a) shows the behaviour that the analysis in Section 2.3 suggests: the amplitude is optimized aggressively, particularly in the beginning in training when the  $\ell_2$ -loss is still high. The phase, on the contrary, does hardly improve at all in the beginning and shows only moderate improvements as the  $\ell_2$ -loss becomes smaller. When training with time-domain  $\ell_2$ -loss and phase loss (Figure 3b), this effect is being compensated for. The amplitude is optimized less aggressively and phase improves from the beginning of training on.

Various audio losses have been proposed over time, ranging from optimizing the power spectrum only and copying the input's phase (Zhao et al., 2018; Gao & Grauman, 2019a) over a multiscale STFT loss for high frequency and high time resolution (Yamamoto et al., 2020) to optimization of the scale-invariant signal to distortion ratio (si-SDR, Le Roux et al. (2019); Heitkaemper et al. (2020); Luo & Mesgarani (2019)). With the introduction of WaveNet for speech synthesis (Van Den Oord et al., 2016), interpreting audio optimization as categorical optimization on a  $\mu$ -law encoded signal has become a prominent technique. As Table 1 shows, all these approaches fail to predict accurate phase and mostly result in meager power spectral and waveform optimization. Overall, our proposed loss retains accurate  $\ell_2$  and power spectral estimations while outperforming other criteria by a huge margin in phase error.

Perceptually, we observe a strong correlation between the phase error and noise and distortions in the generated binaural signal. In particular, our proposed loss was the only one that produced clean speech without perceivable distortions. This is consistent with our perceptual study in Table 4, where other approaches with different losses and architectures have been ranked less favorable.

# 3.2 MODEL EVALUATION

Ablation Study. In Table 2, we show the impact of our model's individual components compared to a vanilla temporal convolutional network baseline with a WaveNet-like architecture and ReLU activations. Number of layers, channels, and kernel sizes are the same as in our final system. Keeping amplitudes unchanged but compensating for interaural time differences, it is not surprising that neural time warping leads to a huge improvement in phase. Replacing regular convolutions with

Table 2: Ablation study. The components of the proposed binauralization network improve phase and amplitude and thereby the overall loss in time-domain.  

<table><tr><td></td><td></td><td>raw waveform (l2 error ×103)</td><td>power spectrum (l2 error)</td><td>phase spectrum (angular error)</td></tr><tr><td>(a)</td><td>vanilla temporal CNN</td><td>0.254</td><td>0.061</td><td>0.934</td></tr><tr><td>(b)</td><td>+ warping</td><td>0.206</td><td>0.061</td><td>0.849</td></tr><tr><td>(c)</td><td>+ hyper-conv</td><td>0.183</td><td>0.051</td><td>0.847</td></tr><tr><td>(d)</td><td>+ sine activation</td><td>0.167</td><td>0.048</td><td>0.807</td></tr></table>

![](images/0f9912bcee9c8bbe72f810606d68aaf25c918f1345070f994875fc920dcb1cf2.jpg)  
(a) Warping example. Top to bottom: source mono input; left ear binaural recording; geometric warping as in equation 2; neural time warping as in equation 4.

![](images/11e4b6c52b4f8eaa621521eaecc4df68db718baed7a0292a9dbd6e83d05ca2f3.jpg)  
(b) Amplitude and phase error for different warping schemes, warping plus bilinear amplitude scale, and the full system.  
Figure 4: Analysis of the warping module.

hyper-convolutions, on the contrary, is particularly beneficial to improve the power spectrum. Finally, replacing the ReLU activations by sine functions, which have been proven to retain high frequency details more reliably (Sitzmann et al., 2020), leads to an additional moderate improvement along waveform, phase, and amplitude error.

Neural Time Warping. The purpose of neural time warping is a strong initial alignment of the mono source signal to the left and right ear listener signal, respectively. Note the significant temporal shift between the mono signal and recorded left ear signal in Figure 4a. In the same figure, observe how geometric warping provides an approximate alignment to the reference signal, while the learned neural warping successfully corrects the inaccurate geometric warping and aligns the peaks and valleys more accurately. Although those adjustments seem small, the impact of neural warping on the phase error is significant, as shown in Figure 4b (red bars). Naturally, neural warping can not improve the amplitude (blue bars).

Temporal HyperConv Network. Neural warping provides an accurate alignment between input and target signal. This raises the question if a deep network is required on top of the warping module or if a linear amplitude adjustment can already yield convincing results. We therefore apply a learned bilinear term to the warped result,

$$
y _ {t} ^ {(l / r)} = x _ {t} ^ {(\text {w a r p e d})} \boldsymbol {a} ^ {T} \boldsymbol {c} _ {t} + b, \quad \boldsymbol {a} \in \mathbb {R} ^ {C _ {\text {c o n d}}}, b \in \mathbb {R} \tag {13}
$$

given the conditioning  $c_t$  and the warped signal  $x^{(\mathrm{warped})}$  for the left or right ear, respectively. Figure 4b shows that this leads to a slight improvement of the amplitude error but falls way behind the performance of the full system with a deep temporal network of hyper-convolutions after the warping module. Inspection of the mono and recorded signal in Figure 4a in fact reveals that the binaural recording undergoes nonlinear transformations beyond warping. Room reverberations, source speech directivity, and modifications caused by the shape of the listener's ear, for instance, are highly nonlinear effects and require complex transformations of the warped signal.

Many of these subtle effects depend on the position and orientation of source and listener in the room. It is therefore plausible that conditioned hyper-convolutions, which can model more complex dependencies between inputs and conditioning variables in a single layer, show better performance than standard convolutions, cf. Table 2 (b) versus (c). As Figure 5 reveals, hyper-convolutions also converge significantly faster than standard convolutions in the early stages of training.

![](images/8577fedf2e68e3701051b12f62bd5a851739759fd41cff181b60fe7229651370.jpg)  
Figure 5: Training loss of a model with hyper-convolutions and a model with standard convolutions. Hyper-convolutions lead to a significantly faster convergence.

Table 3: Comparison to state of the art approaches for binaural sound synthesis.  

<table><tr><td></td><td>raw waveform (l2 error ×103)</td><td>power spectrum (l2 error)</td><td>phase spectrum (angular error)</td></tr><tr><td>DSP</td><td>0.485</td><td>0.058</td><td>1.388</td></tr><tr><td>2.5D Sound</td><td>1.085</td><td>0.113</td><td>1.519</td></tr><tr><td>WaveNet</td><td>0.237</td><td>0.048</td><td>1.239</td></tr><tr><td>ours</td><td>0.167</td><td>0.048</td><td>0.807</td></tr></table>

Table 4: Mean opinion scores of different approaches. Participants were ask to rank cleanliness, spatialization, and overall realism on a Likert scale from 1 to 5.  

<table><tr><td></td><td>cleanliness</td><td colspan="2">spatialization</td><td>realism</td></tr><tr><td>DSP</td><td>3.48 ± 0.88</td><td colspan="2">3.75 ± 0.98</td><td>3.62 ± 0.90</td></tr><tr><td>2.5D Sound</td><td>2.70 ± 1.09</td><td colspan="2">3.18 ± 0.94</td><td>2.70 ± 1.03</td></tr><tr><td>WaveNet</td><td>1.20 ± 0.51</td><td colspan="2">2.92 ± 1.11</td><td>1.39 ± 0.71</td></tr><tr><td>ours</td><td>4.26 ± 0.89</td><td colspan="2">3.76 ± 0.91</td><td>3.88 ± 0.99</td></tr><tr><td>binaural recordings</td><td>3.69 ± 0.94</td><td colspan="2">3.88 ± 0.96</td><td>3.82 ± 0.88</td></tr></table>

# 3.3 STATE OF THE ART COMPARISON

The de-facto state of the art for binaural sound synthesis is still a traditional digital signal processing (DSP) model. The recently proposed 2.5D visual sound (Gao & Grauman, 2019b) network predicts a complex mask which the input is multiplied with to obtain left and right ear outputs. We compare to their approach and replace the visual features with our conditioning features  $c_{1:T}$ . We also provide a comparison to a WaveNet that proved to be generally strong in various generative audio problems (Rethage et al., 2018; Engel et al., 2017). Table 3 reveals that all of these approaches perform significantly worse than our proposed model.

For a perceptual evaluation, we asked 100 participants to rank a total of 2,000 audio snippets from 1 to 5 on a Likert scale according to three criteria: cleanliness of the signal, spatialization quality, and overall realism, see Table 4. All scores are below a 5 (indistinguishable from reality) because participants listened to results for a generic head-related transfer function rather to one that takes their explicit head and ear shape into account. Additionally, user's headphones are of different quality and not equalized. Our approach ranks favorably against other neural binauralization systems and is also preferred in terms of cleanliness and realism over the DSP baseline. Note that the binaural (ground truth) recordings score a bit lower on cleanliness because they contain ambient noise that is uncorrelated to the source input and therefore not modeled by our approach.

# 4 CONCLUSION

Our neural sound binauralization approach is the first purely data-driven end-to-end model that shows convincing performance compared to traditional state of the art binauralization methods. We were able to show effectiveness of our model both quantitatively and in a perceptual user study. Moreover, we unveiled and mitigated a fundamental issue with  $\ell_2$ -optimization on the raw waveform that affects not only this task but is relevant to other generative audio problems as well. The code and dataset will be released on acceptance of the paper.

# REFERENCES

Jont B. Allen and David A. Berkley. Image method for efficiently simulating small-room acoustics. The Journal of the Acoustical Society of America, 65(4):943-950, 1979.  
Cal Armstrong, Lewis Thresh, Damian Murphy, and Gavin Kearney. A perceptual evaluation of individual and non-individual hrfts: A case study of the sadie ii database. Applied Sciences, 8 (11):2029, 2018.  
Futoshi Asano, Yoiti Suzuki, and Toshio Sone. Role of spectral cues in median plane localization. The Journal of the Acoustical Society of America, 88(1):159-168, 1990.  
Durand R. Begault, Alexandra S. Lee, Elizabeth M. Wenzel, and Mark R. Anderson. Direct comparison of the impact of head tracking, reverberation, and individualized head-related transfer functions on the spatial perception of a virtual speech source. Journal of the Audio Engineering Society, 2000.  
Samuel Bellows and Timothy Leishman. High-resolution analysis of the directivity factor and directivity index functions of human speech. Journal of the Audio Engineering Society, 2019.  
Piotr Bilinski, Jens Ahrens, Mark R. P. Thomas, Ivan J. Tashev, and John C. Platt. Hrtf magnitude synthesis via sparse representation of anthropometric features. In IEEE Int. Conf. on Acoustics, Speech and Signal Processing, pp. 4468-4472, 2014.  
Fabian Brinkmann, Alexander Lindau, and Stefan Weinzierl. On the authenticity of individual dynamic binaural synthesis. The Journal of the Acoustical Society of America, 142(4):1784-1795, 2017.  
C Phillip Brown and Richard O Duda. A structural model for binaural sound synthesis. IEEE transactions on speech and audio processing, 6(5):476-488, 1998.  
Corey I. Cheng and Gregory H. Wakefield. Introduction to head-related transfer functions (hrtfs): Representations of hrfts in time, frequency, and space. Journal of the Audio Engineering Society, 49(4):231-249, 2001.  
Hyeong-Seok Choi, Jang-Hyun Kim, Jaesung Huh, Adrian Kim, Jung-Woo Ha, and Kyogu Lee. Phase-aware speech enhancement with deep complex u-net. In Int. Conf. on Learning Representations, 2018.  
Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Int. Conf. on Computer Vision, pp. 764-773, 2017.  
Alexandre Defossez, Gabriel Synnaeve, and Yossi Adi. Real time speech enhancement in the waveform domain. In Interspeech, 2020.  
James J Deng and Clement HC Leung. Dynamic time warping for music retrieval using time series modeling of musical emotions. IEEE Transactions on Affective Computing, 6(2):137-151, 2015.  
Chris Donahue, Julian McAuley, and Miller Puckette. Adversarial audio synthesis. In Int. Conf. on Learning Representations, 2019.  
Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Mohammad Norouzi, Douglas Eck, and Karen Simonyan. Neural audio synthesis of musical notes with wavenet autoencoders. In Int. Conf. on Machine Learning, pp. 1068-1077, 2017.  
Szu-Wei Fu, Yu Tsao, Xugang Lu, and Hisashi Kawai. Raw waveform-based speech enhancement by fully convolutional networks. In Asia-Pacific Signal and Information Processing Association Annual Summit and Conference, pp. 6-12, 2017.  
Ruohan Gao and Kristen Grauman. Co-separating sounds of visual objects. In Int. Conf. on Computer Vision, pp. 3879-3888, 2019a.  
Ruohan Gao and Kristen Grauman. 2.5d visual sound. In IEEE Conf. on Computer Vision and Pattern Recognition, 2019b.

Corentin Guezenoc and Renaud Seguier. Hrtf individualization: A survey. Journal of the Audio Engineering Society, october 2018.  
David Ha, Andrew Dai, and Quoc V Le. Hypernetworks. In Int. Conf. on Learning Representations, 2017.  
Monica L Hawley, Ruth Y Litovsky, and John F Culling. The benefit of binaural hearing in a cocktail party: Effect of location and type of interferer. *The Journal of the Acoustical Society of America*, 115(2):833-843, 2004.  
Jens Heitkaemper, Darius Jakobeit, Christoph Boeddeker, Lukas Drude, and Reinhold Haeb-Umbach. Demystifying tasnet: A dissecting approach. In IEEE Int. Conf. on Acoustics, Speech and Signal Processing, pp. 6359-6363, 2020.  
Claudia Hendrix and Woodrow Barfield. The sense of presence within auditory virtual environments. Presence: Teleoperators and Virtual Environments, 5(3):290-301, 1996.  
Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. In Advances in Neural Information Processing Systems, pp. 2017-2025, 2015.  
Shelton Jose and Kumar Gideon Praveen. Comparison between auditory and visual simple reaction times. *Neuroscience & Medicine*, 2010, 2010.  
B-H Juang. On the hidden markov model and dynamic time warping for speech recognition—a unified view. AT&T Bell Laboratories Technical Journal, 63(7):1213-1243, 1984.  
Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron van den Oord, Sander Dieleman, and Koray Kavukcuoglu. Efficient neural audio synthesis. arXiv preprint arXiv:1802.08435, 2018.  
Brian F. G. Katz. Boundary element method calculation of individual head-related transfer function. i. rigid model calculation. The Journal of the Acoustical Society of America, 110(5):2440-2448, 2001.  
Paulina Kocon and Brian B. Monson. Horizontal directivity patterns differ between vowels extracted from running speech. The Journal of the Acoustical Society of America, 144(1), 2018.  
Morten Kolbæk, Zheng-Hua Tan, Søren Holdt Jensen, and Jesper Jensen. On loss functions for supervised monaural time-domain speech enhancement. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:825-838, 2020.  
Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, and John R Hershey. Sdr-half-baked or well done? In IEEE Int. Conf. on Acoustics, Speech and Signal Processing, pp. 626-630, 2019.  
Song Li and Jürgen Peissig. Measurement of head-related transfer functions: A review. Applied Sciences, 10(14):5014, 2020.  
Yu-Ding Lu, Hsin-Ying Lee, Hung-Yu Tseng, and Ming-Hsuan Yang. Self-supervised audio spatialization with correspondence classifier. In IEEE Int. Conf. on Image Processing, pp. 3347-3351, 2019.  
Yi Luo and Nima Mesgarani. Tasnet: time-domain audio separation network for real-time, single-channel speech separation. In IEEE Int. Conf. on Acoustics, Speech and Signal Processing, pp. 696-700, 2018.  
Yi Luo and Nima Mesgarani. Conv-tasnet: Surpassing ideal time-frequency magnitude masking for speech separation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 27(8): 1256-1266, 2019.  
Noam Mor, Lior Wolf, Adam Polyak, and Yaniv Taigman. A universal music translation network. In Int. Conf. on Learning Representations, 2019.  
Pedro Morgado, Nuno Nvasconcelos, Timothy Langlois, and Oliver Wang. Self-supervised generation of spatial audio for 360 video. In Advances in Neural Information Processing Systems, pp. 362-372, 2018.

Sebastian Prepelita, Michele Geronazzo, Federico Avanzini, and Lauri Savioja. Influence of voxelization on finite difference time domain simulations of head-related transfer functions. The Journal of the Acoustical Society of America, 139(5):2489-2504, 2016.  
Dario Rethage, Jordi Pons, and Xavier Serra. A wavenet for speech denoising. In IEEE Int. Conf. on Acoustics, Speech and Signal Processing, pp. 5069-5073, 2018.  
Atul Rungta, Carl Schissler, Nicholas Rewkowski, Ravish Mehra, and Dinesh Manocha. Diffraction kernels for interactive sound propagation in dynamic environments. IEEE Transactions on Visualization and Computer Graphics, 24(4):1613-1622, 2018.  
Lauri Savioja and U. Peter Svensson. Overview of geometrical room acoustic modeling techniques. The Journal of the Acoustical Society of America, 138(2):708-730, 2015.  
Lauri Savioja, Jyri Huopaniemi, Tapio Lokki, and Ritta Väänänen. Creating interactive virtual acoustic environments. Journal of the Audio Engineering Society, 47(9):675-705, 1999.  
Barbara G. Shinn-Cunningham, Norbert Kopco, and Tara J. Martin. Localizing nearby sound sources in a classroom: Binaural room impulse responses. The Journal of the Acoustical Society of America, 117(5):3100-3115, 2005.  
Vincent Sitzmann, Julien NP Martel, Alexander W Bergman, David B Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. arXiv preprint arXiv:2006.09661, 2020.  
Kaushik Sunder, Jianjun He, Ee-Leng Tan, and Woon-Seng Gan. Natural sound rendering for headphones: Integration of signal processing techniques. IEEE Signal Processing Magazine, 32(2): 100-113, 2015.  
Vesa Valimäki, Julian D. Parker, Lauri Savioja, Julius O. Smith, and Jonathan S. Abel. Fifty years of artificial reverberation. IEEE Transactions on Audio, Speech, and Language Processing, 20 (5):1421-1448, 2012.  
Aäron Van Den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew W. Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. In ISCA Speech Synthesis Workshop, pp. 125, 2016.  
Sean Vasquez and Mike Lewis. Melnet: A generative model for audio in the frequency domain. arXiv preprint arXiv:1906.01083, 2019.  
Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, et al. Tacotron: Towards end-to-end speech synthesis. In Interspeech, 2017.  
Frederic L. Wightman and Doris J. Kistler. The dominant role of low-frequency interaural time differences in sound localization. The Journal of the Acoustical Society of America, 91(3):1648-1661, 1992.  
Kazuhiko Yamamoto and Takeo Igarashi. Fully perceptual-based 3d spatial sound individualization with an adaptive variational autoencoder. ACM Transaction on Graphics, 36(6), 2017.  
Ryuichi Yamamoto, Eunwoo Song, and Jae-Min Kim. Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In IEEE Int. Conf. on Acoustics, Speech and Signal Processing, pp. 6199-6203, 2020.  
Karren Yang, Bryan Russell, and Justin Salamon. Telling left from right: Learning spatial correspondence of sight and sound. In IEEE Conf. on Computer Vision and Pattern Recognition, pp. 9932-9941, 2020.  
Wen Zhang, Prasanga Samarasinghe, Hanchi Chen, and Thushara Abhayapala. Surround by sound: A review of spatial audio recording and reproduction. Applied Sciences, 7(5):532, 2017.  
Hang Zhao, Chuang Gan, Andrew Rouditchenko, Carl Vondrick, Josh McDermott, and Antonio Torralba. The sound of pixels. In European Conf. on Computer Vision, pp. 570-586, 2018.

Hang Zhou, Xudong Xu, Dahua Lin, Xiaogang Wang, and Ziwei Liu. Sep-stereo: Visually guided stereophonic audio generation by associating source separation. In European Conf. on Computer Vision, 2020.  
Dmitry N. Zotkin, Ramani Duraiswami, and Larry S. Davis. Rendering localized spatial audio in a virtual auditory space. IEEE Transactions on Multimedia, 6(4):553-564, 2004.
