# VoiceMixer: Adversarial Voice Style Mixup

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Although recent advances in voice conversion have shown significant improvement, there still remains a gap between the converted voice and target voice. A key factor that maintains this gap is the insufficient decomposition of content and voice style from the source speech. This insufficiency leads to the converted speech containing source speech style or losing source speech content. In this paper, we present VoiceMixer which can effectively decompose and transfer voice style through a novel information bottleneck and adversarial feedback. With self-supervised representation learning, the proposed information bottleneck can decompose the content and style with only a small loss of content information. Also, for adversarial feedback of each information, the discriminator is decomposed into content and style discriminator with self-supervision, which enable our model to achieve better generalization to the voice style of the converted speech. The experimental results show the superiority of our model in disentanglement and transfer performance, and improve audio quality by preserving content information.

# 1 Introduction

Voice conversion (VC) is the task of transferring the target voice style to the source speech while keeping the content information of the source speech. VC is also called voice style transfer (VST), and it shares a long history with the objective to clone someone's voice. There is even a potential risk of usage in crime such as voice spoofing (Kinnunen et al., 2012), and also in various applications in entertainment (Nachmani and Wolf, 2019), education (Sisman et al., 2020), security (Wu and Li, 2016), and voice restoring (Yamagishi et al., 2012). Although deep learning made the breakthrough in the VC domain, there still remains challenging problems for real-world application such as low audio quality or similarity to target voice style.

Usually, traditional VC systems require the same utterances for different speakers to train properly. However, it is hard to collect such parallel data for many speakers, and extension to many-to-many VC systems becomes a laborious task. To overcome this problem, several methods have been developed. First, generative adversarial networks (GAN) based models (Kaneko and Kameoka, 2018; Kaneko et al., 2019, 2020; Kameoka et al., 2018) use adversarial feedback with cycle-consistent loss to train with non-parallel data. However, it is hard to train these models, and they produce lacking audio quality and transfer performance. The flow-based VC model, Blow (Serrà et al., 2019), is also a non-parallel VC model with normalizing flows using the hyperconditioning mechanism.

Despite effort to transfer the voice style in non-parallel settings, these models are not able to sufficiently disentangle content and style from the source speech, and thus the converted speech still contains the style of source speech. To overcome this limitation, AUTOVC (Qian et al., 2019) utilizes a simple autoencoder. The carefully designed fixed-length based information bottleneck disentangles the content and style information. For better disentanglement, IDE-VC (Yuan et al., 2021) followed the AUTOVC framework with information-theoretic guidance. AdaIN-VC (Chou et al., 2019) and AGAIN-VC (Chen et al., 2020) employs the instance normalization (Ulyanov et al.,

2016) to remove the global style information. Additionally, AGAIN-VC makes use of the activation function as an information bottleneck with a small size of content embedding. However, these models have a trade-off between the audio quality and the disentanglement performance. In the process of disentanglement, the loss of content information results in low audio quality with missing linguistic information. Also, they have to find the proper size of information bottleneck heuristically, and it can be different depending on the data.

Text transcriptions can be used to guide content embedding to learn only linguistic information (Zhang et al., 2019; Park et al., 2020). These models have to be jointly trained with the text-to-speech (TTS) model to encode the linguistic information based on the attention alignment from autoregressive TTS system (Shen et al., 2018). However, they require text transcriptions for training.

Recently, self-supervised representation learning is adopted to extract important representation in speech representation learning task (Oord et al., 2018; Wang et al., 2020). Predicting the future latent representation can make the model learn useful information without labeled data. However, such self-supervised representation learning has not yet gotten the attention in voice conversion task.

In this paper, we present VoiceMixer, which can decompose and transfer voice style through a novel similarity-based information bottleneck and adversarial feedback. We introduce self-supervised representation learning to disentangle and transfer voice style without any text transcription and additional information extracted from the external feature extractor. Self-supervised learned similarity makes the information bottleneck disentangle the content and style without effort to find the proper down-sampling size. Also, we propose an adversarial voice style mixup to learn the latent representation of the converted speech. We first disentangle the discriminator into content and style discriminator. The hidden representations of generator guide each discriminator as conditional information. Through adversarial feedback of disentangled discriminators, the generator has better generalization on the converted speech. The main contributions are as follows:

- We propose the similarity-based information bottleneck with self-supervised representation learning, which can disentangle content and style with only a small loss of content information. This preservation improves the audio quality of converted speech compared to previous methods.  
- For better generalization of the converted speech, we propose an adversarial voice style mixup, which learns the converted speech by adversarial feedback with self-supervised guidance, even though the converted speech does not have ground-truth audio.  
- Through various subjective and objective evaluations, we demonstrate that VoiceMixer has better disentanglement and transfer performance than other baselines in both many-to-many and zero-shot voice style transfer scenarios on the real-world VCTK dataset.

# 2 Background

AUTOVC disentangles content and style information from the source speech, and transfers the voice style of target speech through information bottleneck (Qian et al., 2019). The simple autoencoder framework of AUTOVC consists of three modules; a content encoder  $f_{c}(\cdot)$ , speaker encoder  $f_{s}(\cdot)$ , and a decoder  $g(\cdot, \cdot)$ . During training, this model only requires self-reconstruction with a fixed-length information bottleneck to disentangle the content and style information.

$$
\boldsymbol {C} _ {A} = f _ {c} \left(\boldsymbol {X} _ {1, A}\right), \boldsymbol {S} _ {1} = f _ {s} \left(\boldsymbol {X} _ {1, A}\right), \hat {\boldsymbol {X}} _ {1 \rightarrow 1, A} = g \left(\boldsymbol {C} _ {A}, \boldsymbol {S} _ {1}\right) \tag {1}
$$

Here,  $X_{1,A}$  refers to the utterance "A" from the source speaker "1".  $C_A$  denotes content information of the utterance "A",  $S_1$  denotes speaker information in the speaker "1", and  $\hat{X}_{1\rightarrow 1,A}$  is self-reconstructed speech which contains the content information  $C_A$  and matches the speaker characteristics  $S_1$ . Although it is a very simple way to decompose each information, a proper information bottleneck size  $\tau$  is necessary. Formally, this can be represented as follows:

$$
\boldsymbol {H} (:, \lfloor t / \tau \rfloor) = \boldsymbol {C} _ {A} (:, t) \tag {2}
$$

where  $H$  denotes the downsampled feature for time indices  $t \in \{1, \dots, T\}$ . When the fixed-length information bottleneck size  $\tau$  is "too narrow", the model has higher reconstruction quality but has poor voice style transfer performance. On the other hand, when the  $\tau$  is "too wide", the model has higher voice style transfer performance but has lower reconstruction quality. In the process of separating content and style information, some content information is lost even with proper bottleneck size. Therefore, missing some content information in converted voice is inevitable.

![](images/30965b88ef45447a9cc8f76c917a371123bb55c754297d100bdb1520d2273d9a.jpg)  
(a) Reconstruction

![](images/3e6a4484d993abdbd96ea0229ccc387b9f369953f336b28b8346bb0f3f1653dd.jpg)  
(b) Conversion

![](images/34b02a0efb6b8db32eac2890c695d976e94e7568c9f7ca83ef9e23383c1c1d6a.jpg)  
(c) Content discriminator

![](images/dfe6aabc7b47f3b096e444cb827b36bb75910baa15560d6c70905f0810cc8839.jpg)  
Figure 1: Overall framework of VoiceMixer.  
(d) Style discriminator

# 3 VoiceMixer

In this paper, we propose a similarity-based information bottleneck by self-supervised representation learning. For adversarial feedback, we disentangle the discriminator to train content and style separately with self-supervised guidance. By disentangling the discriminator for each information, it can be possible to train the converted speech which does not have ground-truth audio. It is worth noting again that using other supervised features (e.g., pitch contour or text transcription) help the model to disentangle each information, but our model uses self-supervised representation learning without additional features. We describe the details of our architecture, similarity-based information bottleneck, and the adversarial voice style mixup in the following subsections.

# 3.1 Generator

For the generator, we follow the autoencoder framework of AUTOVC. As shown in Figure 1a, the generator  $G$  consists of a content encoder  $f_{c}(\cdot)$  which extracts the content embedding from speech, a speaker encoder  $f_{s}(\cdot)$  which extracts a speaker embedding from speech, and a decoder  $g(\cdot,\cdot)$  which generates the speech from content and speaker embeddings represented in Equation 1.

# 3.2 Similarity-based information bottleneck

Unlike information bottleneck in Equation 2, we downsample the content embedding according to the similarity between the content embeddings. We assume that the content encoder produces similar content embedding from similar phoneme, and thus we downsample the adjacent phonetic information to be mapped together. We calculate the similarity  $\mathbf{Q} = (\mathbf{q}_1,\dots ,\mathbf{q}_T)$  between content embedding sequence  $\mathbf{C} = (c_{1},\dots ,c_{T})$  and shifted content embedding sequence  $C_{shift} = (c_2,\dots ,c_{T + 1})$  as:

$$
\boldsymbol {q} _ {t} = \operatorname {s i g} \left(\frac {\boldsymbol {c} _ {t} \cdot \boldsymbol {c} _ {t + 1}}{\| \boldsymbol {c} _ {t} \| \| \boldsymbol {c} _ {t + 1} \|} / \rho\right), \tag {3}
$$

where  $sig$  denotes the sigmoid function and  $\rho$  is the temperature parameter. Then, we extract the similarity-based duration  $\pmb{D} = (d_{1},\dots ,d_{N})$  where  $\pmb{d}_n$  is cumulative sum until the similarity  $q_{t}$  is under the average similarity, and the  $\pmb{d}_{n + 1}$  is computed again from  $q_{t + 1}$  until time step  $T$ .

Gaussian down/up-sampling Assume that the center of same content have the largest information of that content, then we apply the Gaussian downsampling to focus attention to the center. Given the content embedding to be downsampled  $C$ , duration  $\mathcal{D}$ , and learnable range parameter  $\sigma = (\sigma_{1}, \dots, \sigma_{N})$  like (Shen et al., 2020), we compute downsampled sequence  $\pmb{H} = (h_1, \dots, h_N)$  as:

$$
\alpha_ {n} = \frac {d _ {n}}{2} + \sum_ {m = 1} ^ {n - 1} d _ {m}, \quad w _ {t} ^ {n} = \frac {\mathcal {N} \left(t ; \alpha_ {n} , \sigma_ {n} ^ {2}\right)}{\sum_ {m = 1} ^ {N} \mathcal {N} \left(t ; \alpha_ {m} , \sigma_ {m} ^ {2}\right)}, \quad \boldsymbol {h} _ {n} = \sum_ {t = 1} ^ {T} w _ {t} ^ {n} \boldsymbol {c} _ {t} \tag {4}
$$

Afterwards, we use Gaussian upsampling as a TTS model following (Shen et al., 2020) to upscale  $\pmb{H}$  to upsampled content sequence  $\tilde{\pmb{C}} = (\tilde{c}_1,\dots ,\tilde{c}_T)$  with the same duration of  $\pmb{D}$ , range parameter for upsampling  $\pmb{\sigma}' = (\sigma_1',\dots ,\sigma_N')$ , and then  $\tilde{\pmb{C}}$  is fed to  $g(\cdot ,\cdot)$  to generate the mel-spectrogram as:

$$
w _ {t} ^ {\prime n} = \frac {\mathcal {N} \left(t ; \alpha_ {n} , \sigma_ {n} ^ {\prime 2}\right)}{\sum_ {m = 1} ^ {N} \mathcal {N} \left(t ; \alpha_ {m} , \sigma_ {m} ^ {\prime 2}\right)}, \quad \tilde {\boldsymbol {c}} _ {t} = \sum_ {n = 1} ^ {N} w _ {t} ^ {\prime n} \boldsymbol {h} _ {n}, \quad \hat {\boldsymbol {X}} _ {1 \rightarrow 1, A} = g (\tilde {\boldsymbol {C}} _ {A}, \boldsymbol {S} _ {1}) \tag {5}
$$

![](images/10b6700e4505e5b9bd42031ec97554af681aa47ad547a24763959d20e35b5bcf.jpg)  
(a)

![](images/746e030f37db5b60b382c307e60c73a0be98cfc4b99be6bc0a43b595aeb78858.jpg)  
(b)

![](images/5680d62a803a1e7a88c98ef7640fb5deeca523da80bdef1a903758d8c6dab84d.jpg)  
Figure 2: (a) Fixed-length information bottleneck. (b) Similarity-based information bottleneck. (c) Context network for self-supervised representation learning on content embedding.  
(c)

# 120 3.3 Auxiliary losses for similarity

Contrastive loss To increase the similarity between the adjacent content embeddings, we train the content encoder with self-supervised representation learning. The content embedding is fed to a context network  $f_{r}$  to learn a content representation illustrated in Figure 2c. To train in non-autoregressive manner, we utilize the masked convolutional blocks (Liu et al., 2020) to predict a content embedding from the adjacent content embeddings, and the contrastive loss for positive sample is defined to minimize distance between predicted content embedding  $\hat{C} = (\hat{c}_1,\dots ,\hat{c}_T)$  and content embedding  $C$ :

$$
\mathcal {L} _ {\text {p o s}} \left(f _ {c}, f _ {r}\right) = \mathbb {E} \left[ - \frac {1}{T} \sum_ {i} ^ {T} \log \operatorname {s i g} \left(\frac {\boldsymbol {c} _ {i} \cdot \hat {\boldsymbol {c}} _ {i}}{\| \boldsymbol {c} _ {i} \| \| \hat {\boldsymbol {c}} _ {i} \|} / \rho\right) \right] \tag {6}
$$

where  $sig$  denotes the sigmoid function and  $\rho$  represents the temperature parameter.

To remove style information on the content embedding in an unsupervised manner, we prevent context network to predict future representation prediction of content embedding. While negative samples are uniformly sampled from the same utterance in (Baevski et al., 2020), we only sample the  $k$ -th future content embedding as a negative sample to prevent maximizing distance between the contents similar to each other. We maximize cosine distance between predicted content embedding and  $k$ -th future representation of content embedding, and the contrastive loss for negative sample is:

$$
\mathcal {L} _ {n e g} \left(f _ {c}, f _ {r}\right) = \mathbb {E} \left[ \frac {1}{T} \sum_ {i} ^ {T} \log s i g \left(\frac {\boldsymbol {c} _ {i + k} \cdot \hat {\boldsymbol {c}} _ {i}}{\| \boldsymbol {c} _ {i + k} \| \| \hat {\boldsymbol {c}} _ {i} \|} / \rho\right) \right]. \tag {7}
$$

Adversarial speaker classification To enforce speaker disentanglement on the content embedding, we apply adversarial speaker classification in a supervised manner (using speaker label  $\pmb{y}_i$ ) as:

$$
\mathcal {L} _ {\text {a d v s c}} \left(f _ {c}\right) = \mathbb {E} \left[ \frac {1}{T} \sum_ {i} ^ {T} \boldsymbol {y} _ {i} \log \left(f _ {\text {c l s}} \left(\boldsymbol {c} _ {i}\right)\right) \right] \tag {8}
$$

where  $f_{cls}$  denotes speaker classifier. To train the entered model jointly with  $f_{cls}$ , we use a gradient reversal layer before the content embedding is fed to  $f_{cls}$  following (Hsu et al., 2019).

# 3.4 Disentangled discriminator with self-supervised guidance

Unlike the previous GAN-based VC model which uses the cycle-consistency training to preserve linguistic information by two-way generation, we follow the autoencoder based reconstruction method for training. For adversarial feedback, we divide the discriminator  $D$  into the content discriminator  $D^{c}(\cdot ,\cdot)$  and style discriminator  $D^{s}(\cdot ,\cdot)$  to disentangle content and style, respectively. To guide each discriminator for each attribute, we condition the content embedding to the content discriminator and style embedding to the style discriminator as a self-supervised conditional information illustrated in Figure 1. For the training objectives, we use the LSGAN (Mao et al., 2017) as followed:

$$
\begin{array}{l} \mathcal {L} _ {a d v} \left(D ^ {c}, D ^ {s}; G\right) = \mathbb {E} \left[ \| D ^ {c} \left(\boldsymbol {X} _ {1, A}, \boldsymbol {C} _ {A}\right) - 1 \| _ {2} + \| D ^ {c} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 1, A}, \boldsymbol {C} _ {A}\right) \| _ {2} \right. \tag {9} \\ \left. + \left\| D ^ {s} \left(\boldsymbol {X} _ {1, A}, S _ {1}\right) - 1 \right\| _ {2} + \left\| D ^ {s} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 1, A}, S _ {1}\right)\right\| _ {2} \right] \\ \end{array}
$$

![](images/9d7b0e6e867d2d7db9b33c2bb10fda17a9ed185af81aaf2f14eafdba495a2e6b.jpg)  
Figure 3: Style feature matching loss for reconstructed and converted mel-spectrogram.

$$
\mathcal {L} _ {a d v} (G; D ^ {c}, D ^ {s}) = \mathbb {E} \left[ \| D ^ {c} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 1, A}, \boldsymbol {C} _ {A}\right) - 1 \| _ {2} + \| D ^ {s} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 1, A}, \boldsymbol {S} _ {1}\right) - 1 \| _ {2} \right] \tag {10}
$$

Feature Matching for reconstruction We use the feature matching loss to train the generator, which minimizes the distance of the discriminator's features between ground truth and generated speech. For each discriminator, we use the content feature matching loss  $\mathcal{L}_{\text{content}}$  from the content discriminator for content and the style feature matching loss  $\mathcal{L}_{\text{style}}$  from the style discriminator for style.

$$
\mathcal {L} _ {\text {c o n t e n t}} (G; D ^ {c}) = \mathbb {E} \left[ \sum_ {i = 1} ^ {K} \frac {1}{N _ {i}} \| D _ {i} ^ {c} \left(\boldsymbol {X} _ {1, A}, \boldsymbol {C} _ {A}\right) - D _ {i} ^ {c} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 1, A}, \boldsymbol {C} _ {A}\right) \| _ {1} \right] \tag {11}
$$

$$
\mathcal {L} _ {\text {s t y l e}} (G; D ^ {s}) = \mathbb {E} \left[ \sum_ {i = 1} ^ {K ^ {\prime}} \frac {1}{N _ {i}} \| D _ {i} ^ {s} \left(\boldsymbol {X} _ {1, A}, \boldsymbol {S} _ {1}\right) - D _ {i} ^ {s} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 1, A}, \boldsymbol {S} _ {1}\right) \| _ {1} \right] \tag {12}
$$

where  $K$  and  $K^{\prime}$  denote the number of blocks in each discriminator, and  $N_{i}$  is the number of features in  $i$ -th discriminator block. The total loss for reconstructed mel-spectrogram is defined as:

$$
\mathcal {L} _ {\text {r e c}} = \mathcal {L} _ {\text {a d v}} (G; D ^ {c}, D ^ {s}) + \lambda_ {c} \mathcal {L} _ {\text {c o n t e n t}} (G; D ^ {c}) + \lambda_ {s} \mathcal {L} _ {\text {s t y l e}} (G; D ^ {s}) + \lambda_ {\text {m e l}} \mathcal {L} _ {\text {m e l}} \tag {13}
$$

where  $\mathcal{L}_{mel}$  is mean absolute error between  $X_{1,A}$  and  $\hat{X}_{1\rightarrow 1,A}$ .

# 3.5 Adversarial Voice Style Mixup

By introducing the disentangled discriminator for each information, we can train the reconstructed speech for each disentangled feature. However, our goal is to convert voice by disentangling the source style and transferring the target style. To learn the latent representations of the converted speech, we propose an adversarial voice style mixup, which can train the converted speech by using the disentangled discriminator with a self-supervised condition. Even though converted speech does not have ground-truth (GT) samples, the converted mel-spectrogram can be trained with adversarial feedback through each discriminator. It is worth noting that the model only uses a self-supervised hidden representation of the generator as conditional features without any external feature extractor for conditional information. The GAN losses for the converted mel-spectrogram are defined as:

$$
\mathcal {L} _ {a d v} ^ {*} \left(D ^ {c}, D ^ {s}; G\right) = \mathbb {E} \left[ \| D ^ {c} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 2, A}, \boldsymbol {C} _ {A}\right) \| _ {2} + \| D ^ {s} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 2, A}, \boldsymbol {S} _ {2}\right) \| _ {2} \right], \tag {14}
$$

$$
\mathcal {L} _ {a d v} ^ {*} (G; D ^ {c}, D ^ {s}) = \mathbb {E} \left[ \| D ^ {c} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 2, A}, \boldsymbol {C} _ {A}\right) - 1 \| _ {2} + \| D ^ {s} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 2, A}, \boldsymbol {S} _ {2}\right) - 1 \| _ {2} \right] \tag {15}
$$

Feature Matching for Mixup We can also use the feature matching loss for a converted speech by the disentangled discriminator. For the content representation, the feature distance of content discriminator between converted speech and source speech is minimized as:

$$
\mathcal {L} _ {\text {c o n t e n t}} ^ {*} (G; D ^ {c}) = \mathbb {E} \left[ \sum_ {i = 1} ^ {K} \frac {1}{N _ {i}} \| D _ {i} ^ {c} \left(\boldsymbol {X} _ {1, A}, \boldsymbol {C} _ {A}\right) - D _ {i} ^ {c} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 2, A}, \boldsymbol {C} _ {A}\right) \| _ {1} \right] \tag {16}
$$

For the style representation, the feature distance of style between converted and target speech is minimized as following:

$$
\mathcal {L} _ {\text {s t y l e} ^ {+}} ^ {*} (G; D ^ {s}) = \mathbb {E} \left[ \sum_ {i = 1} ^ {K ^ {\prime}} \frac {1}{N _ {i}} \| D _ {i} ^ {s} \left(\boldsymbol {X} _ {2, B}, \boldsymbol {S} _ {2}\right) - D _ {i} ^ {s} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 2, A}, \boldsymbol {S} _ {2}\right) \| _ {1} \right] \tag {17}
$$

We call it "Attractive style loss" which minimizes the style feature distance between the same style of the same speaker. As shown in Figure 3, we also introduce "Repulsive style loss" to maximize the style feature distance between the different style of the converted speech and source speech as:

$$
\mathcal {L} _ {\text {s t y l e} ^ {-}} ^ {*} (G; D ^ {s}) = \mathbb {E} \left[ - \sum_ {i = 1} ^ {K ^ {\prime}} \frac {1}{N _ {i}} \| D _ {i} ^ {s} \left(\boldsymbol {X} _ {1, A}, \boldsymbol {S} _ {1}\right) - D _ {i} ^ {s} \left(\hat {\boldsymbol {X}} _ {1 \rightarrow 2, A}, \boldsymbol {S} _ {2}\right) \| _ {1} \right] \tag {18}
$$

When the content encoder does not disentangle the source speaker, the converted speech may contain the style of the source speaker. Thus, this repulsive style loss restricts the converted speech from having the style of source speaker. The total loss for converted mel-spectrogram is defined as:

$$
\mathcal {L} _ {\text {c o n}} = \mathcal {L} _ {\text {a d v}} ^ {*} (G; D ^ {c}, D ^ {s}) + \lambda_ {c} \mathcal {L} _ {\text {c o n t e n t}} ^ {*} (G; D ^ {c}) + \lambda_ {s} \mathcal {L} _ {\text {s t y l e} ^ {+}} ^ {*} (G; D ^ {s}) + \lambda_ {s ^ {-}} \mathcal {L} _ {\text {s t y l e} ^ {-}} ^ {*} (G; D ^ {s}) \tag {19}
$$

Our final objectives for the discriminators and generator are represented as:

$$
\mathcal {L} _ {D} = \mathcal {L} _ {a d v} \left(D ^ {c}, D ^ {s}; G\right) + \lambda_ {\text {c o n}} \mathcal {L} _ {a d v} ^ {*} \left(D ^ {c}, D ^ {s}; G\right) \tag {20}
$$

$$
\mathcal {L} _ {G} = \mathcal {L} _ {r e c} + \lambda_ {c o n} \mathcal {L} _ {c o n} + \lambda_ {p o s} \mathcal {L} _ {p o s} \left(f _ {c}, f _ {r}\right) + \lambda_ {n e g} \mathcal {L} _ {n e g} \left(f _ {c}, f _ {r}\right) + \lambda_ {a d v s c} \mathcal {L} _ {a d v s c} \left(f _ {c}\right) \tag {21}
$$

# 4 Experiment and result

We evaluated our model with the VCTK dataset, which has 46 hours of audio from 109 speakers (Veaux et al., 2017). We divided the dataset into 98 speakers as base speakers for many-to-many VST and 10 speakers as the novel speakers for zero-shot VST. The base speaker is split into train and test sets. For the non-parallel dataset setting, the training set consists of different utterances for all of the speakers, and the test set consists of 25 same utterances. We transform the mel-spectrogram with 80 bins from the audio downsampled at  $22,050\mathrm{Hz}$ . The spectrogram is inverted to a waveform by the pre-trained HiFi-GAN (Kong et al., 2020). For many-to-many VST, we randomly choose 20 speakers with equal distribution of male and female from the base speakers. For zero-shot VST, we randomly choose 10 speakers from the base speakers, and 10 speakers from the novel speakers. For each setting, a single utterance is selected from each speaker, and then all the possible pairs of utterances  $(20 \times 20 = 400)$  are produced. The generated speech is evaluated by the following metrics.

# 4.1 Evaluation metrics

Subjective metrics We conduct the naturalness and similarity mean opinion score test. For the naturalness of speech (Naturalness), converted samples are evaluated by at least 20 raters on a scale of 1 to 5. The Naturalness is reported with  $95\%$  confidence intervals. For the similarity of converted speech (Similarity), both converted speech and the target speech are presented to at least 20 raters, and the raters evaluate on a scale of 1 to 4. We also report the score as a percentage from the binary rating introduced in (Serrà et al., 2019).

Objective metrics For the similarity measurement, we conduct 3 objective metrics; the equal error rate of the automatic speaker verification (ASV EER), the mel-cepstral distance  $(\mathrm{MCD}_{13})$  (Kubichek, 1993), and the  $F0$  root mean square error  $(\mathrm{RMSE}_{f0})$ . We use the pre-trained ASV model (Chung et al., 2020) trained by the large scale dataset, VoxCeleb2 (Chung et al., 2018). We compute the EER at which both acceptance and rejection errors are equal from the sample pairs from the converted and target speech  $(400 \times 20 = 8000)$ . We apply the dynamic time warping (DTW) to calculate the  $\mathrm{MCD}_{13}$  and  $\mathrm{RMSE}_{f0}$  between converted and target speech, which has different time alignment. For the naturalness measurement, we evaluate Fréchet DeepSpeech Distance (FDSD) (Binkowski et al., 2020; Gritsenko et al., 2020), which is the distance between the high-level features of GT and generated audio from the pre-trained DeepSpeech2 (Amodei et al., 2016). Because DeepSpeech2 is a speech recognition model trained with connectionist temporal classification loss to classify the text sequence, the hidden representations are related to linguistic information. Thus, we use FDSD between the converted speech and the source speech for objective naturalness measurement.

# 4.2 Implementation details

The generator consists of a speaker encoder, content encoder, similarity-based information bottleneck, and decoder. We train the entire model jointly. The speaker embedding is extracted from the speaker encoder which has the same architecture as the reference encoder in (Skerry-Ryan et al., 2018). The source speech is fed to the content encoder consisting of a pre-net and three blocks of the multi-receptive field fusion (MRF) (Kong et al., 2020). The pre-net is two linear layers with 384 channels. The output concatenated with source speaker embedding is fed to a 1D convolutional layer with 384 channels, followed by the MRF. The MRF returns the sum of output from 384 channels of multiple convolutional layers with multiple dilations and multiple receptive fields. We use the combination of two dilations of [1, 3], and two receptive fields of [3, 7] for the MRF. Before the features are fed to MRF, the instance normalization (IN) is applied.

The similarity-based information bottleneck has two range predictors for down/up-sampling. Both range predictors consist of three convolutional layers followed by a linear layer with a softmax activation function, similar to (Shen et al., 2020). The range predictor for downsampling uses the similarity-based duration as input. The range predictor for upsampling uses the same duration and downsampled content embedding as input. The content embedding is fed to both adversarial speaker classifier and contrastive encoder. The adversarial speaker classifier consists of five 1D convolutional layers followed by a linear layer to predict speaker identity. To remove the speaker identity on content embedding, a gradient reversal layer is used before the first layer of the adversarial speaker classifier. For the contrastive encoder, we use three masked convolution blocks of (Liu et al., 2020) with 384 channels, receptive field size of 23, and mask sizes of [5, 7, 9]. We set the  $k$  as 24 (about 0.3s) which is over the average duration of consonant-vowel syllables (Steinschneider et al., 2013).

After the similarity-based information bottleneck, the upsampled feature concatenated with the target speaker embedding is fed to the decoder, which consists of a conditional layer, three blocks of MRF, and the linear layer. The conditional layer is a single 1D convolutional layer with 384 channels. The MRF of the decoder has the same architecture with the MRF of the encoder without IN applied to the feature before being fed to the MRF. Finally, the mel-spectrogram is predicted by the linear layer.

The content discriminator consists of four blocks which have a speech-side and content-condition side block following (Lee et al., 2021). Each block has two 1D convolutional layers. The hidden representation of the condition-side block is added to the speech-side hidden representations of [256, 512, 1024, 1024]. The style discriminator consists of 4 blocks which have a speech-side block and style-condition side linear layers. Each output of the linear layer is added to the speech-side hidden representations of [256, 512, 1024, 1024]. We report more details of hyperparameter in Appendix A.

# 4.3 Information bottleneck alignment

To compare the similarity-based information bottleneck using a different combination of auxiliary loss, Figure 4 show the alignment between source speech and downsampled content embedding. The model without contrastive loss  $(\mathcal{L}_{pos}$  and  $\mathcal{L}_{neg})$  nearly shows the diagonal alignment, which implies that content embedding does not only represent content information. The model without  $\mathcal{L}_{advsc}$  shows alignment closer to attention alignment of Tacotron2 (Shen et al., 2018). These results show that contrastive loss is more important to disentangle content and speaker information than  $\mathcal{L}_{advsc}$ . Without using any text transcript and target duration, our model with auxiliary losses produces an alignment similar to phonetic alignment, and shows better performance as shown in Table 3.

![](images/ff3185d67c4902bfd9e8113be13c684d9033f7bb21dbb2f4adaf098554670282.jpg)  
(a)  $\mathrm{w / o}\mathcal{L}_{pos}$  and  $\mathcal{L}_{neg}$

![](images/539e81e1e29c3036a4ef77a3355b702adc078ec9a8b7ca7e8d0575a1bcdccb42.jpg)  
(b) w/o  $\mathcal{L}_{advsc}$

![](images/7e40f4aba3a37431f17f3a834ba901100f19a6f0ea490cad71fca031a6218a50.jpg)  
Figure 4: Alignment of similarity-based information bottleneck  
(c) VoiceMixer

![](images/e38902b6d634610bb77fb9c2ca74967636f1ea99a74299e5c48525bdb5d46675.jpg)  
(d) Phoneme alignment

Table 1: Many-to-many VST evaluation results.  

<table><tr><td>Method</td><td>Naturalness</td><td>Similarity</td><td>ASV EER</td><td>MCD13</td><td>RMSEf0</td><td>FDSD</td></tr><tr><td>GT</td><td>4.07±0.03</td><td>93.7%</td><td>5.5%</td><td>-</td><td>-</td><td>-</td></tr><tr><td>HiFi-GAN (Vocoder)</td><td>3.92±0.03</td><td>92.8%</td><td>7.6%</td><td>3.19</td><td>36.22</td><td>0.546</td></tr><tr><td>StarGAN-VC</td><td>3.48±0.04</td><td>42.5%</td><td>22.7%</td><td>7.97</td><td>33.98</td><td>13.561</td></tr><tr><td>AGAIN-VC</td><td>3.60±0.03</td><td>52.2%</td><td>14.3%</td><td>6.75</td><td>41.39</td><td>3.185</td></tr><tr><td>AUTOVC (τ=16)</td><td>3.65±0.03</td><td>52.1%</td><td>18.3%</td><td>6.70</td><td>44.05</td><td>5.753</td></tr><tr><td>AUTOVC (τ=32)</td><td>3.64±0.03</td><td>52.1%</td><td>14.0%</td><td>6.44</td><td>39.93</td><td>10.703</td></tr><tr><td>AUTOVC + Ladvsc (τ=16)</td><td>3.63±0.03</td><td>54.6%</td><td>14.7%</td><td>6.58</td><td>39.40</td><td>6.036</td></tr><tr><td>Blow</td><td>3.12±0.04</td><td>33.2%</td><td>52.0%</td><td>6.74</td><td>44.55</td><td>15.112</td></tr><tr><td>VoiceMixer (Ours)</td><td>3.78±0.03</td><td>55.9%</td><td>12.5%</td><td>6.77</td><td>42.76</td><td>2.080</td></tr></table>

Table 2: Zero-shot VST evaluation results.  

<table><tr><td>Method</td><td>Naturalness</td><td>Similarity</td><td>ASV EER</td><td>MCD13</td><td>RMSEf0</td><td>FDSD</td></tr><tr><td>GT</td><td>4.08±0.03</td><td>96.3%</td><td>4.4%</td><td>-</td><td>-</td><td>-</td></tr><tr><td>HiFi-GAN (Vocoder)</td><td>4.03±0.03</td><td>95.4%</td><td>6.0%</td><td>3.34</td><td>38.33</td><td>0.576</td></tr><tr><td>AGAIN-VC</td><td>3.29±0.03</td><td>58.2%</td><td>15.0%</td><td>6.96</td><td>44.81</td><td>3.261</td></tr><tr><td>AUTOVC (τ=16)</td><td>3.40±0.03</td><td>46.3%</td><td>25.0%</td><td>6.92</td><td>46.33</td><td>5.227</td></tr><tr><td>AUTOVC (τ=32)</td><td>3.24±0.03</td><td>59.7%</td><td>20.7%</td><td>6.65</td><td>39.93</td><td>9.741</td></tr><tr><td>AUTOVC + Ladvsc (τ=16)</td><td>3.39±0.03</td><td>58.9%</td><td>21.9%</td><td>6.78</td><td>42.08</td><td>5.675</td></tr><tr><td>VoiceMixer (Ours)</td><td>3.75±0.03</td><td>63.4%</td><td>18.5%</td><td>7.04</td><td>44.70</td><td>2.416</td></tr></table>

# 4.4 Audio quality and style transfer performance

For the many-to-many VST evaluation, we compared our model with four VC models; StarGAN-VC (Kameoka et al., 2018), AGAIN-VC (Chen et al., 2020), Blow (Serrà et al., 2019), and AUTOVC (Qian et al., 2019). All models are trained on the same dataset as VoiceMixer, and the implementation details are described in the Appendix A. We trained AUTOVC in various settings with each having different sizes of information bottleneck. For a fair comparison, we also implemented the AUTOVC model with an adversarial speaker classifier. Table 1 shows that our model outperforms other models in Naturalness and FDSD metrics. Our model also shows better transfer performance on the Similarity and ASV EER. When disentangling the content and style information, AGAIN-VC and AUTOVC models lose a lot of content information, and thus the converted speech has lower Naturalness and higher FDSD score. The adversarial speaker classifier on the content embedding can help the disentanglement performance of models, but the naturalness can be degraded.

For the zero-shot VST evaluation, we compared our model with two VC models; AGAIN-VC and AUTOVC as shown in Table 2. We also implemented various AUTOVC trained with different information bottleneck size and adversarial speaker classification loss. Our model has better performance in Naturalness and FDSD. In terms of similarity, even though the AGAIN-VC has higher performance in ASV EER, our model has better performance in Similarity. In terms of AUTOVC, it is hard to select the proper down-sampling factor, which has a trade-off between naturalness and similarity. Thus, it is important to note that our proposed similarity-based information bottleneck need not find the proper factor, which is learned by self-supervised representation learning with a small loss of content information. Our audio samples are available on the demo page.<sup>1</sup>

# 4.5 Ablation study

We conducted ablation studies for the information bottleneck and adversarial feedback in Table 3. We evaluate each model for the same zero-shot VST setting of Table 2. In terms of similarity-based information bottleneck, absence of  $\mathcal{L}_{advsc}$  or using fixed-length information bottleneck (IB) makes it harder to disentangle the content and style in content embedding. For better disentanglement, it is essential to use  $\mathcal{L}_{pos}$  with  $\mathcal{L}_{neg}$ , and it induces the information bottleneck to downsample the

Table 3: Ablation studies for zero-shot voice style transfer  

<table><tr><td>Method</td><td>Naturalness</td><td>Similarity</td><td>ASV EER</td><td>\( MCD_{13} \)</td><td>\( RMSE_{f0} \)</td><td>FDSD</td></tr><tr><td>VoiceMixer (Ours)</td><td>3.72±0.03</td><td>63.7%</td><td>18.5%</td><td>7.04</td><td>44.70</td><td>2.416</td></tr><tr><td>w/o \( \mathcal{L}_{advsc} \)</td><td>3.74±0.03</td><td>41.3%</td><td>32.2%</td><td>7.33</td><td>47.63</td><td>1.217</td></tr><tr><td>w/o \( \mathcal{L}_{pos} \) and \( \mathcal{L}_{neg} \)</td><td>3.21±0.03</td><td>42.3%</td><td>38.0%</td><td>7.92</td><td>54.11</td><td>4.634</td></tr><tr><td>w/o \( \mathcal{L}_{neg} \)</td><td>3.18±0.03</td><td>45.5%</td><td>24.2%</td><td>7.72</td><td>45.05</td><td>11.354</td></tr><tr><td>w fixed-length IB (τ=16)</td><td>3.35±0.03</td><td>51.8%</td><td>28.5%</td><td>7.38</td><td>46.14</td><td>2.827</td></tr><tr><td>w fixed-length IB (τ=32)</td><td>3.28±0.03</td><td>56.8%</td><td>20.7%</td><td>6.97</td><td>44.16</td><td>2.823</td></tr><tr><td>w/o GAN</td><td>3.70±0.03</td><td>58.5%</td><td>20.5%</td><td>6.95</td><td>44.79</td><td>2.666</td></tr><tr><td>w/o disentangled discriminator</td><td>3.68±0.03</td><td>60.4%</td><td>18.3%</td><td>7.02</td><td>45.23</td><td>3.502</td></tr><tr><td>w/o \( \mathcal{L}_{style^-}^* \)</td><td>3.69±0.03</td><td>55.0%</td><td>21.6%</td><td>7.02</td><td>45.66</td><td>2.173</td></tr></table>

content embedding similar to the phoneme alignment as shown in Figure 4. Additionally, when trained without any information bottleneck or all of the auxiliary losses, these models are not able to convert any voice, but only reconstruct source speech. Using GAN makes the model have better performance on all of the metrics. The model trained with a single discriminator (instead of disentangled discriminator) shows lower performance in FDSD. In this regard, disentangled discriminator encourages better generalization for each attribute. Removing  $\mathcal{L}_{\text{style}}^*$  results in lower performance in both subjective and objective similarity metrics even though the FDSD decreases.

# 4.6 Content and speaker disentanglement

We conduct speaker classification on content embedding to evaluate the disentanglement performance compared to AGAIN-VC and AUTOVC. Table 4 represents the classification results. Our model shows the lowest accuracy, which mean our model has better disentanglement performance by removing speaker identity on content embedding despite having the largest feature dimensions of 384 (The content embedding of AUTOVC has 64 dimensions). Even though AUTOVC trained with adversarial speaker classifier has comparable disentanglement performance, AUTOVC loses a lot of content information in their fixed-length based information bottleneck as shown in Table 1. AUTOVC also has to heuristically

find optimal downsampling size for a good balance between content and style. On the other hand, our proposed model finds proper downsampling size based on the similarity of content embedding learned by self-supervised representation learning with only a small loss of content information.

Table 4: Speaker classification accuracy on content embedding of the autoencoder based VC models.  

<table><tr><td>Method</td><td>Accuracy[%]</td></tr><tr><td>AGAIN-VC</td><td>27.31</td></tr><tr><td>AUTOVC (τ=16)</td><td>10.27</td></tr><tr><td>AUTOVC (τ=32)</td><td>4.47</td></tr><tr><td>AUTOVC + Ladvsc (τ=16)</td><td>3.11</td></tr><tr><td>VoiceMixer (Ours)</td><td>1.47</td></tr></table>

# 5 Conclusion

We presented VoiceMixer, which can decompose and transfer voice style by similarity-based information bottleneck and adversarial feedback with self-supervised representation learning. Without effort to find the proper size of information bottleneck carefully, our model is able to learn proper downsampling factor with self-supervised representation learning. We successfully demonstrated that our novel information bottleneck can decompose content and style information from the source speech with a small loss of content information. Moreover, the alignment of information bottleneck is similar to phonetic alignment despite not using any text transcript and target phoneme duration. we also show the adversarial voice style mixup makes it possible to learn the latent representation of converted speech which does not have ground-truth speech, and it improves the overall generalization.

While there remains a gap between target and converted voice, we believe pre-trained speaker encoder with large-scale dataset could improve the VST performance. Moreover, we see our self-supervised learning based speech disentanglement extending to other tasks. For future work, we will apply our speech disentanglement to TTS without text (Dunbar et al., 2019), which is a challenging task to synthesize the speech without any text transcript for use of the untranscribed large-scale speech data.

# References

Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In International Conference on Machine Learning, pages 173-182. PMLR, 2016.  
Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33, 2020.  
Mikołaj Binkowski, Jeff Donahue, Sander Dieleman, Aidan Clark, Erich Elsen, Norman Casagrande, Luis C. Cobo, and Karen Simonyan. High fidelity speech synthesis with adversarial networks. In International Conference on Learning Representations, 2020.  
Yen-Hao Chen, Da-Yi Wu, Tsung-Han Wu, and Hung-yi Lee. Again-vc: A one-shot voice conversion using activation guidance and adaptive instance normalization. arXiv preprint arXiv:2011.00316, 2020.  
Ju-chieh Chou, Cheng-chieh Yeh, and Hung-yi Lee. One-shot voice conversion by separating speaker and content representations with instance normalization. arXiv preprint arXiv:1904.05742, 2019.  
Joon Son Chung, Arsha Nagrani, and Andrew Zisserman. Voxceleb2: Deep speaker recognition. Proc. Interspeech 2018, pages 1086-1090, 2018.  
Joon Son Chung, Jaesung Huh, Seongkyu Mun, Minjae Lee, Hee Soo Heo, Soyeon Choe, Chiheon Ham, Sunghwan Jung, Bong-Jin Lee, and Icksang Han. In defence of metric learning for speaker recognition. In Proc. Interspeech 2020, pages 2977-2981, 2020.  
Ewan Dunbar, Robin Algayres, Julien Karadayi, Mathieu Bernard, Juan Benjumea, Xuan-Nga Cao, Lucie Miskic, Charlotte Dugrain, Lucas Ondel, Alan W. Black, Laurent Besacier, Sakriani Sakti, and Emmanuel Dupoux. The Zero Resource Speech Challenge 2019: TTS Without T. In Proc. Interspeech 2019, pages 1088-1092, 2019. doi: 10.21437/Interspeech.2019-2904.  
Alexey A Gritsenko, Tim Salimans, Rianne van den Berg, Jasper Snoek, and Nal Kalchbrenner. A spectral energy distance for parallel speech synthesis. In Advances in Neural Information Processing Systems, 2020.  
Wei-Ning Hsu, Yu Zhang, Ron J Weiss, Yu-An Chung, Yuxuan Wang, Yonghui Wu, and James Glass. Disentangling correlated speaker and noise for speech synthesis via data augmentation and adversarial factorization. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5901-5905. IEEE, 2019.  
Hirokazu Kameoka, Takuhiro Kaneko, Kou Tanaka, and Nobukatsu Hojo. Stargan-vc: Non-parallel many-to-many voice conversion using star generative adversarial networks. In 2018 IEEE Spoken Language Technology Workshop (SLT), pages 266-273. IEEE, 2018.  
Takuhiro Kaneko and Hirokazu Kameoka. Cyclegan-vc: Non-parallel voice conversion using cycle-consistent adversarial networks. In 2018 26th European Signal Processing Conference (EUSIPCO), pages 2100–2104. IEEE, 2018.  
Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, and Nobukatsu Hojo. Cyclegan-vc2: Improved cyclegan-based non-parallel voice conversion. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6820-6824. IEEE, 2019.  
Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, and Nobukatsu Hojo. Cyclegan-vc3: Examining and improving cyclegan-vcs for mel-spectrogram conversion. arXiv preprint arXiv:2010.11672, 2020.  
Tomi Kinnunen, Zhi-Zheng Wu, Kong Aik Lee, Filip Sedlak, Eng Siong Chng, and Haizhou Li. Vulnerability of speaker verification systems against voice conversion spoofing attacks: The case of telephone speech. In ICASSP 2012-2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4401-4404. IEEE, 2012.

Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 17022-17033, 2020.  
R Kubichek. Mel-cepstral distance measure for objective speech quality assessment. In Proceedings of IEEE Pacific Rim Conference on Communications Computers and Signal Processing, volume 1, pages 125–128, 1993.  
Sang-Hoon Lee, Hyun-Wook Yoon, Hyeong-Rae Noh, Ji-Hoon Kim, and Seong-Whan Lee. Multi-spectrogan: High-diversity and high-fidelity spectrogram generation with adversarial style combination for speech synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.  
Alexander H Liu, Yu-An Chung, and James Glass. Non-autoregressive predictive coding for learning speech representations from local dependencies. arXiv preprint arXiv:2011.00406, 2020.  
Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision, pages 2794-2802, 2017.  
Eliya Nachmani and Lior Wolf. Unsupervised singing voice conversion. pages 2583-2587, 2019.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Seung-won Park, Doo-young Kim, and Myun-chul Joe. Cotatron: Transcription-guided speech encoder for any-to-many voice conversion without parallel data. arXiv preprint arXiv:2005.03295, 2020.  
Kaizhi Qian, Yang Zhang, Shiyu Chang, Xuesong Yang, and Mark Hasegawa-Johnson. Autovc: Zero-shot voice style transfer with only autoencoder loss. In International Conference on Machine Learning, pages 5210-5219. PMLR, 2019.  
Joan Serrà, Santiago Pascual, and Carlos Segura. Blow: a single-scale hyperconditioned flow for non-parallel raw audio voice conversion. arXiv preprint arXiv:1906.00794, 2019.  
Jonathan Shen, Ruoming Pang, Ron J Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, Rj Skerrv-Ryan, et al. Natural tts synthesis by conditioning wavenet on mel spectrogram predictions. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4779-4783. IEEE, 2018.  
Jonathan Shen, Ye Jia, Mike Chrzanowski, Yu Zhang, Isaac Elias, Heiga Zen, and Yonghui Wu. Nonattentive tacotron: Robust and controllable neural ts synthesis including unsupervised duration modeling. arXiv preprint arXiv:2010.04301, 2020.  
Berrak Sisman, Junichi Yamagishi, Simon King, and Haizhou Li. An overview of voice conversion and its challenges: From statistical modeling to deep learning. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:132-157, 2020.  
RJ Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton, Joel Shor, Ron Weiss, Rob Clark, and Rif A. Saurous. Towards end-to-end prosody transfer for expressive speech synthesis with tacotron. In International Conference on Machine Learning, pages 4700-4709, 2018.  
Mitchell Steinschneider, Kirill V. Nourski, and Yonatan I. Fishman. Representation of speech in human auditory cortex: Is it special? Hearing Research, 305:57-73, 2013. ISSN 0378-5955. doi: https://doi.org/10.1016/j.heares.2013.05.013. Communication Sounds and the Brain: New Directions and Perspectives.  
Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016.

Christophe Veaux, Junichi Yamagishi, Kirsten MacDonald, et al. Superseded-cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit. 2017.  
Luyu Wang, Kazuya Kawakami, and Aaron van den Oord. Contrastive predictive coding of audio with an adversary. Proc. Interspeech 2020, pages 826-830, 2020.  
Zhizheng Wu and Haizhou Li. On the study of replay and voice conversion attacks to text-dependent speaker verification. Multimedia Tools and Applications, 75(9):5311-5327, 2016.  
Junichi Yamagishi, Christophe Veaux, Simon King, and Steve Renals. Speech synthesis technologies for individuals with vocal disabilities: Voice banking and reconstruction. Acoustical Science and Technology, 33(1):1-5, 2012.  
Siyang Yuan, Pengyu Cheng, Ruiyi Zhang, Weituo Hao, Zhe Gan, and Lawrence Carin. Improving zero-shot voice style transfer via disentangled representation learning. In International Conference on Learning Representations, 2021.  
Jing-Xuan Zhang, Zhen-Hua Ling, and Li-Rong Dai. Non-parallel sequence-to-sequence voice conversion with disentangled linguistic and speaker representations. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:540-552, 2019.
