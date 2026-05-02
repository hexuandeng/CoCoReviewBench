# MIDI-DDSP: DETAILED CONTROL OF MUSICAL PERFORMANCE VIA HIERARCHICAL MODELING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Musical expression requires control of both what notes are played, and how they are performed. Conventional audio synthesizers provide detailed expressive controls, but at the cost of realism. Black-box neural audio synthesis and concatenative samplers can produce realistic audio, but have few mechanisms for control. In this work, we introduce MIDI-DDSP a hierarchical model of musical instruments that enables both realistic neural audio synthesis and detailed user control. Starting from interpretable Differentiable Digital Signal Processing (DDSP) synthesis parameters, we infer musical notes and high-level properties of their expressive performance (such as timbre, vibrato, dynamics, and articulation). This creates a 3-level hierarchy (notes, performance, synthesis) that affords individuals the option to intervene at each level, or utilize trained priors (performance given notes, synthesis given performance) for creative assistance. Through quantitative experiments and listening tests, we demonstrate that this hierarchy can reconstruct high-fidelity audio, accurately predict performance attributes for a note sequence, independently manipulate the attributes of a given performance, and as a complete system, generate realistic audio from a novel note sequence. By utilizing an interpretable hierarchy, with multiple levels of granularity, MIDI-DDSP opens the door to assistive tools to empower individuals across a diverse range of musical experience.

# 1 INTRODUCTION

Generative models are most useful to creators if they can generate realistic outputs, afford many avenues for control, and easily fit into existing creative workflows (Huang et al., 2020). Deep generative models are expressive function approximators, capable of generating realistic samples in many domains (Ramesh et al., 2021; Brown et al., 2020; van den Oord et al., 2016), but often at the cost of interactivity, restricting users to rigid black-box input-output pairings without interpretable access to the internals of the network. In contrast, structured models chain several stages of interpretable intermediate representations with expressive networks, while still allowing users to interact throughout the hierarchy. For example, these techniques have been especially effective in computer vision, where systems are optimized for both realism and control (Lee et al., 2021b; Chan et al., 2019; Zhang et al., 2019; Wang et al., 2018).

For music generation, despite recent progress, current tools still fall short of this ideal (Figure 1, right). Deep networks can either generate realistic full-band audio (Dhariwal et al., 2020) or provide detailed controls of attributes such as pitch, dynamics, and timbre (Défossez et al., 2018; Engel et al., 2019; 2020a; Hawthorne et al., 2019; Wang & Yang, 2019) but not both. Many existing workflows use the MIDI specification (Association et al., 1996) to Conventional DSP synthesizers (Chowning, 1973; Roads, 1988) provide extensive control but make it difficult to generate realistic instrument timbre, while concatenative samplers (Schwarz, 2007) play back high-fidelity recordings of isolated musical notes, but require manually stitching together performances with limited control over expression and continuity.

In this paper, we propose MIDI-DDSP, a hierarchical generative model of musical performance to provide both realism and control (Figure 1, left). Similar to conventional synthesizers and samplers that use the MIDI standard (Association et al., 1996), MIDI-DDSP converts note timing, pitch, and expression information into fine-grained parameter control of DDSP synthesizer modules.

![](images/e70f104ec8c7b65abe6a18a1f8f5a0c20827338be34b881e83432b3aafdc0be8.jpg)  
Figure 1: (Left) The MIDI-DDSP architecture. MIDI-DDSP extracts interpretable features at the performance and synthesis levels, building a modeling hierarchy by learning feature generation at each level. Red and blue components indicate encoding and decoding respectively. Shaded boxes represent modules with learned parameters. Both expression features and notes are extracted directly from synthesis parameters. (Right) Synthesizers have wide range of control, but struggle to convey realism. Neural audio synthesis and concatenative samplers can produce realistic audio, but they have limited control. MIDI-DDSP enables both realistic neural audio synthesis and detailed user control.

![](images/22a08e7d069df3be9edf8add50822306eba2695d174c7e8cd7bdd8596551f0ba.jpg)

We take inspiration from the hierarchical structure underlying the process of creating music. A composer writes a piece as a series of notes. A performer interprets these notes through a myriad of nuanced, sub-second choices about articulation, dynamics, and expression. These expressive gestures are realized as audio through the short-time pitch and timbre changes of the physical vibration of the instrument. MIDI-DDSP is built on a similar 3-level hierarchy (notes, performance, synthesis) with interpretable representations at each level.

While the efficient DDSP synthesis representation (low-level) allows for high-fidelity audio synthesis (Engel et al., 2020a), users can also control the notes to be played (high-level), and the expression with which they are performed (mid-level). A qualitative example of this is shown in Figure 2, where a given performance on violin is manipulated at all three levels (notes, expression, synthesis parameters) to create a new realistic yet personalized performance.

As seen in Figure 1 (left), MIDI-DDSP can be viewed similarly to a multi-level autoencoder. The hierarchy has three separately trainable modules (DDSP Inference, Synthesis Generator, Expression Generator) and three fixed functions/heuristics (DDSP Synthesis, Feature Extraction, Note Detection). These modules enable MIDI-DDSP to conditionally generate at any level of the hierarchy, providing creative assistance by filling in the details of a performance, synthesizing audio for new note sequences, or even fully automating music generation when paired with a separate note generating model.

It is important to note that the system relies on pitch detection and note detection, so is currently limited to training on recordings of single monophonic instruments, but has no fundamental barrier to adapting to multi-instrument polyphonic recordings as multi-pitch tracking and polyphonic transcription models progress (Hawthorne et al., 2021; Engel et al., 2020b; Bittner et al., 2017). Finally, we also show that each stage can be made conditional on instrument identity, training on 13 separate instruments with a single model.

![](images/5d690cb5ef18755c5d22c80660f8ee7db55d6a4d48b7b12e229e99b213e6332c.jpg)  
Figure 2: An example of detailed user control. Given an initial generation from the full MIDI-DDSP model (top), an expert musician can adjust notes (blue), performance attributes (green), and low-level synthesis parameters (yellow) to craft a personalized expression of a musical piece (bottom). To hear the difference in feeling, we highly recommend readers listen to the sample in the online supplement.

For clarity, we summarize the core contributions of this work:

- We propose MIDI-DDSP, a 3-level hierarchical generative model of music (notes, performance, synthesis), and train a single model capable of realistic audio synthesis for 13 different instruments. (Section 3)  
- Expression Attributes: We introduce heuristics to extract mid-level per-note expression attributes from low-level synthesis parameters. (Figure 4)  
- User Control: Quantitative studies confirm that manipulating the expression attributes creates a corresponding effect in the synthesizer parameters, and we qualitatively demonstrate the detailed control that is available to users manipulating all three levels of the hierarchy. (Table 2 and Figure 2)  
- Assistive Generation: Reconstruction experiments show that MIDI-DDSP can make assistive predictions at each level of the hierarchy, accurately resynthesizing audio, predicting synthesis parameters from note-wise expression attributes, and auto-regressively predicting note-wise expression attributes from a note sequence. (Tables 1a, 1b, 1c)  
- Realistic Note Synthesis: An extensive listening study finds that MIDI-DDSP can synthesize audio from new note sequences (not seen during training) with higher realism than both comparable neural approaches and professional concatenative sampler software. (Figure 5)  
- Automatic Music Generation: We demonstrate that pairing MIDI-DDSP with a pretrained note generation model enables full-stack automatic music generation. As an example, we use Coconet (Huang et al., 2017) to generate and synthesize novel 4-part Bach chorales for a variety of instruments. (Figure 6)

Audio samples of all results and figures are provided in the online supplement<sup>1</sup>. We highly recommend readers to access the online supplement to the paper.

# 2 RELATED WORK

Note Synthesis. Existing neural synthesis models allow either high-level manipulation of note pitch, velocity, and timing (Hawthorne et al., 2019; Kim et al., 2019; Wang & Yang, 2019; Manzelli et al., 2018), or low-level synthesis parameters (Jonason et al., 2020; Castellon et al., 2020; Blaauw & Bonada, 2017). MIDI-DDSP connects these two approaches by enabling both high-level note controls and low-level synthesis manipulation in a single system.

![](images/fd464151af11b33cbbf1cda5820b958738664e4f95f380a35bfdff593d6d396f.jpg)  
Figure 3: Separate training procedures for the three modules in MIDI-DDSP. (Left) The DDSP Inference module predicts synthesis parameters from audio and is trained via an audio reconstruction loss on the resynthesized audio. (Middle) The Synthesis Generator module predicts synthesis parameters from notes and their expression attributes (shown as a 6-dimensional color map) and is trained via a reconstruction loss and an adversarial loss. (Right) The Expression Generator module autoregressively predicts note expression given a note sequence and is trained with teacher forcing. Encoding processes are shown in red, and decoding processes are shown in blue and loss calculations are shown in yellow. Thicker arrows indicate the process that is being trained in each level. Ground-truth data are shown in solid frames, while model predictions are shown in dashed frames.

Most related to this work is MIDI2Params (Castellon et al., 2020), a hierarchical model that autoregressively predicts frame-wise pitch and loudness contours to drive the original DDSP autoencoder (Engel et al., 2020a). MIDI-DDSP builds on this work by adding an additional level of hierarchy for the note expression, training a new more accurate DDSP base model, and explicitly modeling the synthesizer coefficients output by that model, rather than the pitch and loudness inputs to the model. We extensively compare to our reimplementation of MIDI2Params as a baseline throughout the paper.

Hierarchical Audio Modelling. Audio waveforms have dependencies over timescales spanning several orders of magnitude, lending themselves to hierarchical modeling. For example, Dieleman et al. (2018) and Dhariwal et al. (2020) both choose to encode audio as discrete latent codes at different time resolutions, and apply autoregressive models as priors over those codes. MIDI-DDSP applies a similar approach in spirit, but constructs a hierarchy based on semantic musical structure (note, performance, synthesis), allowing interpretable manipulation by users.

Expressive Performance Analysis and Synthesis. Many prior systems pair analysis and synthesis functions to capture expressive performance characteristics (Canazza et al., 2004; Yang et al., 2016; Shih et al., 2017). Such methods often use heuristic functions to generate parameters for driving synthesizers or selecting and modifying sample units. MIDI-DDSP similarly uses feature extraction, but each level is paired with a differentiable neural network function that directly learns the mapping to expression and synthesis controls for more realistic audio synthesis.

# 3 MODEL ARCHITECTURE

# 3.1 DDSP SYNTHESIS AND INFERENCE

Differentiable Digital Signal Processing (DDSP) (Engel et al., 2020a) enables differentiable audio synthesis by using a harmonic plus noise model (Serra & Smith, 1990). Full details are provided in Appendix B.1. Briefly, an oscillator bank synthesizes a harmonic signal from a fundamental frequency  $f_0(t)$ , a base amplitude  $a(t)$ , and a distribution over harmonic amplitudes  $h(t)$ , where the dimensionality of  $h$  is the number of harmonics. The noise signal is generated by filtering uniform noise with linearly spaced filter banks, where  $\eta(t)$  represents the magnitude of noise output from each filter in time. In this study, we use 60 harmonics and 65 noise filter banks, giving 127 total synthesis parameters each time frame  $(s(t) = (f_0(t), a(t), h(t), \eta(t)))$ . The final audio is the addition of harmonic and noise signals.

![](images/808f1303387c207bfef5e2b1ffe61628371584c8f0b3442bbd8478dfa5ff4d70.jpg)  
Figure 4: In MIDI-DDSP, manipulating note-level expression can effectively change the synthesis-level quantities. We show by taking a test-set sample (middle row) and adjusting each expression control value to lowest (bottom row) and highest (upper row), how each synthesis quantities (rightmost legend) would change. The dashed gray line in each plot indicates the note boundary.

Since the synthesis process is differentiable, Engel et al. (2020a) demonstrate that it is possible to train a neural network to predict the other synthesis parameters given  $f_{0}(t)$  and the loudness of the audio, and optimize a multi-scale spectral loss (Wang et al., 2019; Engel et al., 2020a) of the resynthesized audio (Figure 3 left).  $f_{0}(t)$  is extracted by a pre-trained CREPE model (Kim et al., 2018), and the loudness is extracted via an A-weighting of the power spectrum (Hantrakul et al., 2019).

We extend this work for our DDSP Inference module, by providing and additional input features a log-scale Mel-spectrogram of the audio, that produces higher quality resynthesis (Table 1a). Full architectural details are provided in Appendix B.2.

# 3.2 EXPRESSION CONTROLS

We aim to model aspects of expressive performance with a continuous variable. For example, this enables a performer to choose how loud the note should be performed, or how much vibrato to apply. We define six expression controls (detailed in Appendix B.3), scaled within [0, 1]. These are extracted from synthesis parameters  $s(t)$  and applied within the  $i$ th note,  $n_i(t)$ , in a note sequence:

Volume: Controls the volume of a note, extracted by taken average amplitude over a note.

Volume fluctuation: Determines the magnitude of a volume change across a note. Used with the volume peak position, below, this can make a note crescendo or decrescendo. This is extracted by calculating the standard deviation of the amplitude over a note.

Volume peak position: Controls where, over the duration of a note, the peak volume occurs. Zero value corresponds to decrescendo notes, whereas one corresponds to crescendo notes. The volume peak position is extracted by calculating the relative position of maximum amplitude in the note.

Vibrato: Controls the extent of the vibrato of a note. Vibrato is a musical technique defined by pulsating the pitch of a note. Vibrato is extracted by applying Discrete Fourier Transform (DFT) on the fundamental frequency  $f_{0}(t)$  in a note and take the peak amplitude.

Brightness: Controls the timbre of a note where a higher value corresponds to larger high-frequency harmonic. Brightness is extracted by calculating the average harmonic centroid of a note.

Attack Noise: Controls how much noise occurs at the start of the note (the attack), e.g., the fluctuation of string and bow. At certain settings, this determines whether two notes sound consecutively or separately. The attack noise is extracted by taking a note's average noise magnitude in the first ten frames (40ms).

# 3.3 SYNTHESIS GENERATOR

Given the output of the per-note Expression Controls,  $e_i$  for  $i = 1, \dots, I$  notes, and a corresponding note sequence,  $n_i$ , the Synthesis Generator predicts the frame-level synthesis parameters that, in turn, generate audio. Note expression controls are pooled over the duration of the corresponding note to make a conditioning sequence,  $c(t) = [(e_1, n_1), \dots, (e_I, n_I)]$ , with the same length as the fundamental frequency curve,  $f_0(t)$ .

The Synthesis Generator,  $g_{\theta}$ , is an autoregressive recurrent neural net (RNN) is used to predict a fundamental frequency,  $\hat{f}_0(t)$  given conditioning sequence, and a convolutional generative adversarial network (GAN) is used to predict the other synthesis parameters given conditioning sequence and generated fundamental frequency:

$$
\hat {f} _ {0} (t) = g _ {\theta} (\boldsymbol {c} (t)), \quad \hat {a} (t), \hat {\boldsymbol {h}} (t), \hat {\boldsymbol {\eta}} (t) = g _ {\phi} (\boldsymbol {c} (t), \hat {f} _ {0} (t)), \tag {1}
$$

where  $\theta$  denotes trainable parameters in the autoregressive RNN, and  $\phi$  indicates trainable parameters in the convolutional GAN. Architectural details for both of these details is provided in Appendix B.4. The autoregressive RNN is trained using cross-entropy loss  $\mathcal{L}_{ce}$ . The generator of the convolutional GAN is trained by a multi-scale spectral loss  $\mathcal{L}_{spec}$  (Eq. 12) and an adversarial objective consisting of a least-squares GAN (LSGAN)  $\mathcal{L}_{lsgan}$  (Mao et al., 2017) loss and a feature matching loss  $\mathcal{L}_{fm}$  (Kumar et al., 2019) (Eq. 15 to Eq. 18). Thus, the total loss applied to the Synthesis Generator can be written as:

$$
\mathcal {L} = \left(\mathcal {L} _ {c e} + \mathcal {L} _ {s p e c}\right) + \alpha \left(\mathcal {L} _ {l s g a n} + \gamma \mathcal {L} _ {f m}\right). \tag {2}
$$

In training, the ground-truth  $f_0(t)$  is input to the convolutional GAN, thus there is no gradient from the convolutional GAN into the autoregressive RNN.

# 3.4 EXPRESSION GENERATOR

The Expression Generator uses an autoregressive RNN to predict note expression controls from note sequence (Appendix B.6). A single-layer bidirectional GRU extracts context information from input, and a two-layer autoregressive GRU generates note expression. The Expression Generator is trained by mean square error (MSE) loss between ground-truth note expression and teacher-forced prediction (Figure 3 right). At inference time, the output note expression is generated autoregressively and deterministically.

The note sequence used to train the Expression Generator can either be extracted or comes from human labels. To show the full potential of MIDI-DDSP, we use the ground-truth note boundary label from dataset in all experiments for best accuracy. However, note transcription models can be used to provide the note labels.

# 4 EXPERIMENTS

The structured hierarchy and explicit latent representations used in MIDI-DDSP benefit music control as well as music modeling. We design a set of experiments to answer the following questions: First, does the system generate realistic audio, and if so, how does each module contribute? How does this compare to existing systems? And, second, is the system capable of enabling note-level, performance-level, and synthesis-level control? How effective are these controls? We encourage readers to listen to the samples provided in the online supplement.

# 4.1 DATASET

To demonstrate modeling a variety of instruments, we use the URMP dataset (Li et al., 2018), a publicly-available audio dataset containing monophonic solo performances of a variety of instruments. URMP is widely used in music synthesis research (Bitton et al., 2020; Hayes et al., 2021; Zhao et al., 2019; Engel et al., 2020b). The URMP dataset contains solo performance recordings of

Table 1: Each module in MIDI-DDSP produces high-quality reconstruction and prediction. Reconstruction accuracy of each module are shown in table comparing to other methods.  

<table><tr><td>Model</td><td>Spectral Loss</td><td>Model</td><td>RMSE</td><td>Models</td><td>RMSE</td></tr><tr><td>DDSP Inference</td><td>4.28</td><td>Synthesis Generator</td><td>0.19</td><td>Expression Generator</td><td>0.14</td></tr><tr><td>Engel et al. (2020a)</td><td>5.00</td><td>MIDI2Params</td><td>0.26</td><td>MIDI2Params</td><td>0.23</td></tr><tr><td>(a)</td><td></td><td>(b)</td><td></td><td>(c)</td><td></td></tr></table>

![](images/6eb310ec4910b12a85c5719ce4cb05fd011e20e4a62d8e298cffdacc419ae0c1.jpg)  
Ground-truth

![](images/006ac588c75182fcfcaf77dec1c3611bf820fed26dbc7868e9e3c1291d927d44.jpg)  
DDSP Inference

![](images/925ef2faecca470340da906d10b483f5ef41e19a98a58a62728298a5eb6cf23d.jpg)  
MIDI-DDSP

![](images/4e83e2c817b7d3483ffbffa1b69f5b306df317b0afda6d6c8bbfe20faa565c09.jpg)  
Ableton  
Figure 5: (left) Comparing the log-scale Mel spectrograms of synthesis results from test-set note sequences, MIDI-DDSP synthesizes more realistic audio (more similar to ground-truth and DDSP Inference) than prior work score-to-audio method MIDI2Params (enlarged in Figure 7). This is also reflected in the listening study (right), where the MIDI-DDSP synthesis is also perceived as more realistic than the professional concatenative sampler Ableton and the freely available FluidSynth.

![](images/000ce5d0216c0ecdf43acee5aab6bb04f6b4b482734225dfe9ca139046cbe9f2.jpg)  
MIDI2Params

![](images/a8ab359650730fb02ca882c766dc99d4b79bd2c4a6b57e1d78106c6491941495.jpg)  
FluidSynth

![](images/d3155e932f76028470f413282a4092d240c56e1d30418121f619a1fc5ab0c79f.jpg)  
Listening study of pairwise comparisons between methods  
Number of wins

13 string instruments and wind instruments, which allows us to test generalization to different instruments. The recordings in the URMP dataset are played by students, and the performance quality is substantially lower compared to virtuoso datasets used in other work (Hawthorne et al., 2019). The URMP dataset contains 3.75 hours of 117 unique solo recordings, where 85 recordings in 3 hours are used as the training set, and 35 recordings in 0.75 hours are used as the hold-out test set.

# 4.2 MODEL ACCURACY

Modules in MIDI-DDSP can accurately reconstruct output at multiple levels of the hierarchy (Figure 3). We evaluate the reconstruction quality of MIDI-DDSP by evaluating each module. Results are shown in Table 1.

DDSP Inference We measure the difference between reconstruction and ground-truth in the audio spectral loss for our DDSP Inference module and compared it with the original DDSP autoencoder. As shown in Table 1a, with an additional CNN to extract features, the DDSP Inference module can reconstruct audio more accurately than the original DDSP Autoencoder.

Synthesis Generator We predict synthesis parameters from ground-truth note expression and then extract note expression back from the generated synthesis parameters. We measure the root mean square error (RMSE) between note expressions. The prior approach MIDI2Params directly generates synthesis parameters from notes and does not have access to note expressions. However, we can extract note expressions from the generated synthesis parameters and compare them to ground truth. As shown in Table 1b, the Synthesis Generator can faithfully reconstruct the input note expression, whereas without access to note expression, MIDI2Params generates larger error.

Expression Generator We take ground-truth MIDI and evaluate the likelihood of the ground-truth note expressions under the model. As the Expression Generator is auto-regressive, we use teacher-forcing to sequentially accumulate the squared error note by note. The total error thus computed can be interpreted as a log-likelihood. auto-regressive at a much higher temporal resolution. We again compare to MIDI2Params, where we auto-regressively condition its own output within and

Table 2: The note expression outputs are strongly correlated with input adjustment. The Pearson correlation  $r$ -values are shown in the table (all entries  $p < 0.0001$ ). The bold numbers indicate a Pearson  $r$ -value larger than 0.7, which we consider to indicate strong correlation between the input control and the respective output quantity. For simplicity, only four instruments are shown. More results can be found in Table 7.  

<table><tr><td></td><td>Volume</td><td>Vol. Fluc.</td><td>Vol. Peak Pos.</td><td>Vibrato</td><td>Brightness</td><td>Attack Noise</td></tr><tr><td>All instruments</td><td>.97</td><td>.78</td><td>.57</td><td>.70</td><td>.92</td><td>.93</td></tr><tr><td>Violin</td><td>.99</td><td>.84</td><td>.80</td><td>.86</td><td>.96</td><td>.97</td></tr><tr><td>Viola</td><td>.98</td><td>.74</td><td>.70</td><td>.82</td><td>.98</td><td>.97</td></tr><tr><td>Cello</td><td>.97</td><td>.64</td><td>.54</td><td>.74</td><td>.98</td><td>.94</td></tr><tr><td>Double bass</td><td>.98</td><td>.85</td><td>.34</td><td>.84</td><td>.99</td><td>.95</td></tr></table>

on ground-truth across notes to obtain a note-wise metric. That is, MIDI2Params sees the ground truth of past notes, but sees its own output for the current note. As shown in Table 1c, the Expression Generator can accurately predict the note expression. In comparison, MIDI2Params without performance-level modeling suffers from predicting the note expression on a higher level when compared to the frame-wise sequence models.

# 4.3 AUDIO QUALITY EVALUATION BY HUMANS

We evaluate the audio quality of MIDI-DDSP via a listening test. We compare ground truth audio from the URMP dataset to MIDI-DDSP and four other sources: a stripped down version of our system, containing just our DDSP Inference module (Section 3.1), MIDI2Params (Castellon et al., 2020), and two concatenative samplers: FluidSynth and Ableton (detailed in Appendix D.1). DDSP-inference infers synthesis parameters from the ground truth audio; it serves as an upper bound on what is attainable with MIDI-DDSP, which has to predict expression and synthesis parameters from MIDI. MIDI2Params is prior work that synthesizes audio from MIDI by predicting frame-wise loudness and pitch contour, which is fed as input to a DDSP autoencoder.

Participants in the listening test were presented with two 8-second clips, and asked which clip sounded more like a person playing on a real violin, on a 5-point Likert scale. We collected 960 ratings, with each source involved in 320 pair-wise comparisons. Figure 5 shows the number of comparisons in which each source was selected as more realistic. According to a post-hoc analysis using the Wilcoxon signed-rank test with Bonferroni correction (with  $p < 0.01 / 15$ ), the orderings shown in Figure 5 (right) are all statistically significant with the exception of ground truth versus DDSP Inference and MIDI2Params versus FluidSynth (Table 6). In particular, MIDI-DDSP was significantly preferred over MIDI2Params, Ableton and FluidSynth.

The difference among the sources can also be seen from visual inspection of the spectrograms (Figure 5, left). While the DDSP Inference module faithfully re-synthesizes the ground-truth audio, MIDI-DDSP generates coherent performance from a series of notes and has rich, varying expressions across the notes. MIDI2Params failed to generate coherent expression within a note, generating unrealistic pitch and loudness contours. Also, MIDI2Params stopped the note in the middle when generating the fifth note, suggesting that such a frame-wise generation model is limited in modeling long-term dependency even inside a single note. On the contrary, the note expression modeling in MIDI-DDSP allows it to model dependency at the granularity of the note sequence and use synthesis parameters to model the frame-wise parameter changing inside a single note. The two concatenative synthesizers Ableton and FluidSynth generate the same note expression with identical vibrato and volume for all notes. Although the expression is coherent inside a single note, it fails to generate expression dependency among notes automatically.

# 4.4 EFFECTS OF NOTE EXPRESSION CONTROLS

To evaluate the behavior of the note expression controls, we measure how well each control correlates with itself after a roundtrip through synthesis. That is, for each sample in the test set, we interpolate the control from lowest (0) to highest (1) in an interval of 0.1 and generate synthesis pa

![](images/7fcdfa938ed7ae5736bcd5c4396806790f63318ac63b0bc320458393d074ab96.jpg)  
Figure 6: MIDI-DDSP can take input from different sources (human or other models) by designing explicit latent representations at each level. A full hierarchical generative model for music can be constructed by connecting MIDI-DDSP with an automatic composition model. Here, we show MIDI-DDSP taking note input from a score level Bach composition model and automatically synthesizing a Bach quartet by generating explicit latent for each level in the hierarchy.

rameters. Then we extract the note expressions from these synthesis parameters. Table 2 reports the correlation between the value we put in and the value observed after synthesis. All controls exhibit strong correlation as desired, except for volume peak position. A low correlation may stem from characteristics of the instrument, or imbalances of those performance techniques in the dataset.

Figure 4 illustrates how each note expression affects properties of the sound. For example, as we increase vibrato, we see stronger fluctuations in pitch. Similarly, changing the volume peak position changes the shape of the amplitude curve.

# 4.5 FINE GRAINED CONTROL OR FULL END-TO-END GENERATION

The structured modelling approach of MIDI-DDSP enables end users to have as much or as little control over the output as they want. A user can add manipulations at certain levels of the hierarchy or let the model guide the synthesis.

On one end of this spectrum, Figure 2 shows the results of an end user manipulating each level of MIDI-DDSP. Because different levels of the MIDI-DDSP hierarchy correspond with different musical attributes, a user can make manipulations at the note-level to change the attack noise and volume to create staccato notes (second green box in Figure 2) or a user could make adjustments to the synthesis-level to control the pitch contour for making a "pitch bend" (yellow box in Figure 2).

On the other end of the spectrum, MIDI-DDSP can be paired with generative symbolic music models to make fully generated, realistic end-to-end performances. As shown in Figure 6, MIDI-DDSP can be combined with a composition Bach chorales model COCONET (Huang et al., 2017), to form a fully generated musical quartet that sounds like real instruments performance. Readers are encouraged to listen to both the hand-tuned and end-to-end performances on our accompanying website.

# 5 CONCLUSION

We proposed MIDI-DDSP, a hierarchical music modeling system that factorizes the generation of audio to note, performance, and synthesis levels. By proposing explicit representations for each level alongside modeling note expression, MIDI-DDSP enables effective manipulation and realistic automatic generation of music. We show, experimentally, that the input controls for MIDI-DDSP are correlated with desired performance characteristics (e.g., vibrato, volume, etc). We also show that listeners preferred MIDI-DDSP over existing systems, while enabling fine-grained control of these characteristics. MIDI-DDSP can also connect to other models to construct a full audio generation model, where beginners can obtain realistic novel music from scratch, while expert users can manipulate results based on model prediction to realize unique musical design.

# REFERENCES

MIDI Manufacturers Association et al. The complete midi 1.0 detailed specification. Los Angeles, CA, The MIDI Manufacturers Association, 1996.  
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.  
Rachel M Bittner, Brian McFee, Justin Salamon, Peter Li, and Juan Pablo Bello. Deep salience representations for f0 estimation in polyphonic music. In ISMIR, pp. 63-70, 2017.  
Adrien Bitton, Philippe Esling, and Tatsuya Harada. Vector-quantized timbre representation. arXiv preprint arXiv:2007.06349, 2020.  
Merlijn Blaauw and Jordi Bonada. A Neural Parametric Singing Synthesizer Modeling Timbre and Expression from Natural Songs. Applied Sciences, 7(12):1313, dec 2017. ISSN 2076-3417. doi: 10.3390/app7121313. URL http://www.mdpi.com/2076-3417/7/12/1313.  
Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
Sergio Canazza, Giovanni De Poli, Carlo Drioli, Antonio Roda, and Alvise Vidolin. Modeling and control of expressiveness in music performance. Proceedings of the IEEE, 92(4):686-701, 2004.  
Rodrigo Castellon, Chris Donahue, and Percy Liang. Towards realistic midi instrument synthesizers. In NeurIPS Workshop on Machine Learning for Creativity and Design (2020), 2020.  
Caroline Chan, Shiry Ginosar, Tinghui Zhou, and Alexei A Efros. Everybody dance now. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5933-5942, 2019.  
John M Chowning. The synthesis of complex audio spectra by means of frequency modulation. Journal of the audio engineering society, 21(7):526-534, 1973.  
Alexandre Defossez, Neil Zeghidour, Nicolas Usunier, Leon Bottou, and Francis Bach. Sing: Symbol-to-instrument neural generator. arXiv preprint arXiv:1810.09785, 2018.  
Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. arXiv preprint arXiv:2005.00341, 2020.  
Sander Dieleman, Aaron van den Oord, and Karen Simonyan. The challenge of realistic music generation: modelling raw audio at scale. In Advances in Neural Information Processing Systems, pp. 7989-7999, 2018.  
Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, and Adam Roberts. Gansynth: Adversarial neural audio synthesis. In *ICLR*, 2019.  
Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, and Adam Roberts. DDSP: Differentiable digital signal processing. In International Conference on Learning Representations, 2020a.  
Jesse Engel, Rigel Swavely, Lamtharn Hantrakul, Adam Roberts, and Curtis Hawthorne. Self-supervised pitch detection by inverse audio synthesis. In International Conference on Machine Learning, Self-supervised Audio and Speech Workshop, 2020b.  
Lamtharn Hantrakul, Jesse H Engel, Adam Roberts, and Chenjie Gu. Fast and flexible neural audio synthesis. In ISMIR, pp. 524-530, 2019.  
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Olsen, Jesse Engel, and Douglas Eck. Enabling factorized piano music modeling and generation with the MAESTRO dataset. In International Conference on Learning Representations, 2019.  
Curtis Hawthorne, Ian Simon, Rigel Swavely, Ethan Manilow, and Jesse Engel. Sequence-to-sequence piano transcription with transformers. arXiv preprint arXiv:2107.09142, 2021.

Ben Hayes, Charalamos Saitis, and György Fazekas. Neural waveshaping synthesis. arXiv preprint arXiv:2107.05050, 2021.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rygGQyrFvH.  
Cheng-Zhi Anna Huang, Tim Cooijmans, Adam Roberts, Aaron Courville, and Douglas Eck. Counterpoint by convolution. In Proceedings of ISMIR 2017, 2017. URL https://ismir2017.smcnus.org/wp-content/uploads/2017/10/187_Paper.pdf.  
Cheng-Zhi Anna Huang, Hendrik Vincent Koops, Ed Newton-Rex, Monica Dinculescu, and Carrie J. Cai. Ai song contest: Human-ai co-creation in songwriting. *ArXiv*, abs/2010.05388, 2020.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pp. 448-456. PMLR, 2015.  
Nicolas Jonason, Bob Sturm, and Carl Thomé. The control-synthesis approach for making expressive and controllable neural music synthesizers. In 2020 AI Music Creativity Conference, 2020.  
Jong Wook Kim, Justin Salamon, Peter Li, and Juan Pablo Bello. Crepe: A convolutional representation for pitch estimation. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 161-165. IEEE, 2018.  
Jong Wook Kim, Rachel Bittner, Aparna Kumar, and Juan Pablo Bello. Neural music synthesis for flexible timbre control. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 176-180. IEEE, 2019.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2880-2894, 2020.  
Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brébisson, Yoshua Bengio, and Aaron C Courville. Melgan: Generative adversarial networks for conditional waveform synthesis. In Advances in Neural Information Processing Systems, pp. 14910-14921, 2019.  
Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998. doi: 10.1109/5.726791.  
Sang-Hoon Lee, Hyun-Wook Yoon, Hyeong-Rae Noh, Ji-Hoon Kim, and Seong-Whan Lee. Multispectrogan: High-diversity and high-fidelity spectrogram generation with adversarial style combination for speech synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 13198–13206, 2021a.  
Wonkwang Lee, Whie Jung, Han Zhang, Ting Chen, Jing Yu Koh, Thomas Huang, Hyungsuk Yoon, Honglak Lee, and Seunghoon Hong. Revisiting hierarchical approach for persistent long-term video prediction. arXiv preprint arXiv:2104.06697, 2021b.  
Bochen Li, Xinzhao Liu, Karthik Dinesh, Zhiyao Duan, and Gaurav Sharma. Creating a multitrack classical music performance dataset for multimodal music analysis: Challenges, insights, and applications. IEEE Transactions on Multimedia, 21(2):522-535, 2018.  
Pei-Ching Li, Li Su, Yi-Hsuan Yang, Alvin WY Su, et al. Analysis of expressive musical terms in violin using score-informed and expression-based audio features. In ISMIR, pp. 809-815, 2015.

Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. In Proc. icml, volume 30, pp. 3. Citeseer, 2013.  
Rachel Manzelli, Vijay Thakkar, Ali Siahkamari, and Brian Kulis. Conditioning deep generative raw audio models for structured automatic music. In 19th International Society for Music Information Retrieval Conference, 2018.  
Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pp. 2794-2802, 2017.  
Marco Marchini, Rafael Ramirez, Panos Papiotis, and Esteban Maestre. The sense of ensemble: a machine learning approach to expressive performance modelling in string quartets. Journal of New Music Research, 43(3):303-317, 2014.  
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. arXiv preprint arXiv:2102.12092, 2021.  
Curtis Roads. Introduction to granular synthesis. Computer Music Journal, 12(2):11-13, 1988.  
Diemo Schwarz. Corpus-based concatenative synthesis. IEEE signal processing magazine, 24(2): 92-104, 2007.  
Xavier Serra and Julius Smith. Spectral modeling synthesis: A sound analysis/synthesis system based on a deterministic plus stochastic decomposition. Computer Music Journal, 14(4):12-24, 1990.  
Chi-Ching Shih, Pei-Ching Li, Yi-Ju Lin, Yu-Lin Wang, Alvin WY Su, Li Su, and Yi-Hsuan Yang. Analysis and synthesis of the violin playing style of heifetz and oistrakh. In Proceedings of the 20th International Conference on Digital Audio Effects (DAFx-17), 2017.  
Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. In 9th ISCA Speech Synthesis Workshop, pp. 125-125, 2016.  
Bryan Wang and Yi-Hsuan Yang. Performancenet: Score-to-audio music generation with multi-band convolutional residual network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1174-1181, 2019.  
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. High-resolution image synthesis and semantic manipulation with conditional gans. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8798-8807, 2018.  
Xin Wang, Shinji Takaki, and Junichi Yamagishi. Neural source-filter waveform models for statistical parametric speech synthesis. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:402-415, 2019.  
Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li. Empirical evaluation of rectified activations in convolutional network. Proceedings of the International Conference on Machine Learning (ICML) Workshop, 2015.  
Chih-Hong Yang, Pei-Ching Li, AW Su, Li Su, Yi-Hsuan Yang, et al. Automatic violin synthesis using expressive musical term features. In Proceedings of the 19th International Conference on Digital Audio Effects (DAFx-16), pp. 1-7. Brno, Czech Republic, 2016.  
Jiangning Zhang, Xianfang Zeng, Yusu Pan, Yong Liu, Yu Ding, and Changjie Fan. Faceswapnet: Landmark guided many-to-many face reenactment. arXiv preprint arXiv:1905.11805, 2, 2019.  
Hang Zhao, Chuang Gan, Wei-Chiu Ma, and Antonio Torralba. The sound of motions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1735-1744, 2019.

![](images/874e40977e659e952e843133e58ec8b0eb7a07c1cf109b31766ef2cf4a3835f3.jpg)  
A APPENDIX

![](images/c98eca2bb00afd51cd9d40f84d11753fece337065b927fae1984ea20e629929a.jpg)

![](images/4527fa43abea3c49852b1ee90b3b0383dd035eb7db644fb609b570f1a4760d25.jpg)

![](images/e360b307426a4ccbd3c2ba623421a5dd56b8d7ab6f38e025d29c16b7e7d2c71e.jpg)  
Figure 7: The enlarged log-scale Mel spectrograms of synthesis results in Figure 5

![](images/68435464b74db9d44d79e16aba49ec1ea020d401c6cc3b6abe2741bd4daea2ff.jpg)

![](images/e2bf6a5da70d57ec1597b97d738577ff586d0f0b9f4670f3ffd076272446db54.jpg)
