# FAST CHIRPLET TRANSFORM TO ENHANCE CNN MACHINE LISTENING - VALIDATION ON ANIMAL CALLS AND SPEECH

# Hervé Glotin

DYNI, LSIS, Machine Learning & Bioacoustics team  
AMU, University of Toulon, ENSAM, CNRS, IUF La Garde, France

glotin@univ-tln.fr

# Julien Ricard

DYNI, LSIS, Machine Learning & Bioacoustics team  
AMU, University of Toulon, ENSAM, CNRS

La Garde, France

julien.ricard@gmail.com

# Randall Balestriero

Department of Electrical and Computer Engineering

Rice University

Houston, TX 77005, USA

randallbalestriero@gmail.com

# ABSTRACT

The scattering framework offers an optimal hierarchical convolutional decomposition according to its kernels. Convolutional Neural Net (CNN) can be seen as an optimal kernel decomposition, nevertheless it requires large amount of training data to learn its kernels. We propose a trade-off between these two approaches: a Chirplet kernel as an efficient Q constant bioacoustic representation to pretrain CNN. First we motivate Chirplet bioinspired auditory representation. Second we give the first algorithm (and code) of a Fast Chirplet Transform (FCT). Third, we demonstrate the computation efficiency of FCT on large environmental data base: months of Orca recordings, and 1000 Birds species from the LifeClef challenge. Fourth, we validate FCT on the vowels subset of the Speech TIMIT dataset. The results show that FCT accelerates CNN when it pretrains low level layers: it reduces training duration by  $-28\%$  for birds classification, and by  $-26\%$  for vowels classification. Scores are also enhanced by FCT pretraining, with a relative gain of  $+7.8\%$  of Mean Average Precision on birds, and  $+2.3\%$  of vowel accuracy against raw audio CNN. We conclude on perspectives on tonotopic FCT deep machine listening, and inter-species bioacoustic transfer learning to generalise the representation of animal communication systems.

# 1 INTRODUCTION

Representation of bioacoustic sequences started with 'Human' speech in the 70'. Speech automatic processing yields to the efficient Mel Filter Cepstral Coefficients (MFCC) representation. Today new bioacoustic representation paradigms arise from environmental monitoring and species classification at weak Signal to Noise Ratio (SNR) and with small amount of data per species.

Several neurobiological evidences suggest that auditory cortex is tuned to complex time varying acoustic features, and consists of several fields that decompose sounds in parallel (Kowalski et al., 1996; Mercado et al., 2000). Therefore it is more than reasonable to investigate the Chirplet time-frequency representation from acoustic and neurophysiological points of view.

Chirps, or transient amplitude and frequency modulated waveforms, are ubiquitous in nature systems (Flandrin (2001)), ranging from bird songs and music, to animal vocalization (frogs, whales) and Speech. Moreover the sinusoidal models are a typical attempt to represent audio signals as a superposition of chirp-like components. Chirp signals are also commonly observed in biosonar systems.

The Chirplet transform subsumes both Fourier analysis and wavelet analysis, providing a broad framework for mapping one-dimensional sound waveforms into a n-dimensional auditory parameter space. It offers the processing described in different auditory fields, i.e. cortical regions with systematically related response sensitivities. Moreover, Chirplet spaces are highly over-complete because there is an infinite number of ways to segment a time-frequency plane, the dictionary is redundant: this corresponds well with the overlapping, parallel signal processing pathways of auditory cortex.

Then we suggest that low level CNN layers shall be pretrained by Chirplet kernels. Thus, we define and code a Fast Chirplet Transform (FCT). We conduct validation on real recordings of whale and birds, and on Speech (vowels subset of TIMIT). We demonstrate that CNN classification benefits from low level layers FCT pretraining. We conclude on the perspectives of tonotopic FCT machine listening and inter-species transfer learning.

# 2 FORMAL DEFINITION OF CHIRPLET

A chirplet can be seen as a complex sinus with increasing or decreasing frequency over time modulated by a Gaussian window to have a localized support in the time and Fourier domain. It is a broad class of filters which includes wavelets and Fourier basis as special cases. As a result, and as presented in (Mann & Haykin, 1991; 1992), the Chirplet transform is a generalization of many known time-frequency representations. We first present briefly the wavelet transform framework to extend it to Chirplets. Given an input signal  $x$  one can compute a wavelet transform (Mallat, 1999) through the application of multiple wavelets  $\psi_{\lambda}$ . A wavelet is an atom with localized support in time and frequency domain which integrates to 0. The analytical support of the wavelets is not compact but they are very well localized. It can be considered compact in the applied case where roundoff error lead to 0 quickly after moving around the center frequency. The whole filter bank is derived from a mother wavelet  $\psi_0$  and a set of dilation coefficients following a geometric progression defined as  $\Lambda = \{2^{1 + j / Q}, j = 0, \dots, JQ - 1\}$  with  $J$  being the number of octave to decompose and  $Q$  the number of wavelets per octave. As a result, one can create the filter-bank as the collection  $\{\psi_0(\frac{t}{\lambda}) := \psi_\lambda, \lambda \in \Lambda\}$ . After application of the filter-bank, one ends up with a time-scale representation, or scalogram,  $Ux(\lambda, t) := |(x \star \psi_\lambda)(t)|$  where the complex modulus was applied in order to remove the phase information and contract the space. It is clear that a wavelet filter-bank is completely characterized by its mother wavelet and the set of scale parameters. Generalizing this framework for Chirplets will be straightforward by now allowing a nonconstant frequency for each filter. As for wavelets, filters are generated from a Gaussian window determining the time support however the complex sinus has nonconstant frequency over time with center-frequency  $f_c$ . Since the scope of the parameters leads infinitely many different possible filters, we have to restrain ourselves, and thus create only a fixed Chirplet filter-bank allowing fast computations. The parameters defining these filters include the time position  $t_c$ , the frequency center  $f_c$ , the duration  $\Delta_t$  and the chirp rate  $c$ :

$$
g _ {t _ {c}, f _ {c}, \log (\Delta t), c} (t) = \frac {1}{\sqrt {\sqrt {\pi} \Delta t}} e ^ {- \frac {1}{2} \frac {(t - t _ {c}) ^ {2}}{\Delta_ {t} ^ {2}}} e ^ {j 2 \pi (c (t - t _ {c}) ^ {2} + f _ {c} (t - t _ {c}))}. \tag {1}
$$

# 3 PROPOSITION OF A FAST CHIRPLET TRANSFORM (FCT)

The parameter space is basically of infinite dimension. Similarly to continuous wavelet transform however, it is possible to use some a priori knowledge in order to create a finite bank-filter. For example, wavelets are generated by knowing the number of wavelets per octave and the number of octave to decompose. As a result, we used the same motivation in order to reduce the number of possible Chirplets required. The goal here is not to compute an invertible transform, but rather provide a redundant transformation highlighting transient structures which are not the same tasks as discussed in (Coifman et al., 1992; Meyer, 1993; Coifman et al., 1994). As a result, we keep the same overall framework as for wavelets with the  $Q$  and  $J$  parameters. For example parameters for bird songs in this paper are  $J = 6$  and  $Q = 16$  with a sampling rate (SR) of  $44100\mathrm{Hz}$ , and  $J = 4$  and  $Q = 16$  on speech and Orca with SR=16 kHz). Finally, since we are interested in frequency modulations, we compute the ascendant and descendant chirp filters as one being the symetrized version of the other. As a result, we use a more straightforward analytical formula defined with a starting frequency  $F_{0}$ , an ending frequency  $F_{1}$ , and the usual wavelet like parameters  $\sigma$  being the

bandwidth. Finally the hyperparameter  $p$  defining the polynomial order of the chirp is constant for the whole bank-filter generation. For example, the case  $p = 1$  leads to a linear chirp,  $p = 2$  to a quadratic chirp. The starting and ending frequencies are chosen to approximately cover one octave and are directly computed from the  $\lambda$  parameters which define the scales. Finally, following the scattering network inspiration from (Bruna & Mallat, 2013), in order to remove unstable noisy pattern, we apply a low-pass filter (a Gaussian blurring) and thus we increase the SNR of the representation.

$$
\Lambda = \{2. 0 ^ {1 + i / Q}, i = 0, \dots , J \times Q - 1 \}, \tag {2}
$$

$$
F _ {0} = \frac {F s}{2 \lambda}, \lambda \in \Lambda , \tag {3}
$$

$$
F _ {1} = \frac {F s}{\lambda}, \lambda \in \Lambda , \tag {4}
$$

$$
\sigma = 2 \frac {d}{\lambda}, \lambda \in \Lambda . \tag {5}
$$

# 4 LOW COMPLEXITY FCT ALGORITHM AND IMPLEMENTATION

We give here our code of Fast Chirplet Transform (FCT), taking advantage of the a priori knowledge for the filter-bank creation and the fast convolution algorithm<sup>1</sup>. Therefore, we first create the Chirplet with the descendant and descendant versions in once (see Annexe Algo 1).

Then we generate the whole filter-bank (see Algo 2 in annexe) with the defined  $\lambda$  and hyperparameters.

Finally, we use the scattering framework (Bruna & Mallat, 2013; Andén & Mallat, 2014): we apply a local low-pass filter to the obtained representation. In fact, the scattering coefficients  $Sx$  result from a time-averaging on the time-frequency representation  $Ux$  bringing local and up to global time-invariance. This time-averaging is computed through the application of the  $\phi$  filter, usually a Gabor atom with specified standard deviation and such that

$$
\int \phi (t) d t = 1. \tag {6}
$$

As a result, one computes these coefficients as:  $Sx(\lambda, t) = (|x \star \psi_{\lambda}| \star \phi)(t)$ , where  $\psi_{\lambda}$  is a Chirplet with  $\lambda$  parameters and  $\phi$ . Similarly, we perform local time-averaging on the Chirplet representation in the same manner.

We present some possible filters in Fig. 2, and some bird features Fig. 3.

The third step in our FCT consists in the reduction of the convolution task. The asymptotic complexity of the Chirplet transform is  $O(N \cdot \log(N))$  with  $N$  being the size of the input signal. This is the same asymptotic complexity as for the continuous wavelet transform and the scattering network. However, it is possible to reach lower asymptotic complexity simply by a division of the convolution task. Usually the convolutions are carried through application of an element-wise multiplication of the signal and the filter in the frequency domain and then compute the inverse Fourier transform to end up with  $x \star \psi_{\lambda}$ . However, if we denote by  $M$  the length of the filter  $\psi_{\lambda}$  it is possible to instead perform multiple times this operation on different overlapping chunks of the signal to then concatenate the results to obtain at the end the same convolution result but now in  $O(N \cdot \log(M))$ . Finally a last improvement induced by this approach is to allow easy tackling of signals with a length just above a power of 2 which otherwise would require to be padded in order to obtain a FFT with real  $O(N \cdot \log(N))$  complexity through the Danielson-Lanczos lemma (Press, 2007). Applying this scheme allowed to compute the convolutions between 3 to 4 times faster. The variations came from the distance between  $N$  and the closest next power of 2 depending on the desired chunk size.

We validate the efficiency of FCT on real bioacoustic recordings. We processed on 10 medium speed CPUs of 4 years old, 100 hours of recording of LifeClef bird challenge (16 kHz Sampling Rate (SR), 16 bits) in 2 days. Second, we processed in 7 days the equivalent of 1 month of

![](images/9a97fab2b08cab7c9cafd2e1795801614a4807cb5002670be0d5ec417379bc4f.jpg)

![](images/d5a3869625f5e6fec4dacab90ee79ab59826d4273096649841010275bcdd75c3.jpg)

![](images/e8c28bb9a00bd184d246de7876d7bf21b1fa602d4476c84b92d98540182fcdc8.jpg)  
Figure 1: Top: Chirplet of Orca call with  $p = 3$ ,  $j = 4$ ,  $q = 16$ ,  $t = 0.001$ ,  $s = 0.01$ , with usual FFT spectrogram below, Sampling Rate (SR) 22 kHz, 16 bits. Waves and Chirplets of Orca are: http://sabiod.univ-tln.fr/orcalab. Bottom: same on bird calls from Amazonia (BIRD10 data set). SR 16 kHz, 16 bits.

Orca whale recordings from Orcalab.org ONG (22 kHz SR, 16 bits), in Fig. 1,2,3 and at http://sabiod.univ-tln.fr/orcalab.

![](images/79f6cc1a464a91197775c1a431f47905c399d2f2cc636cb66f11167e7feb8041.jpg)

![](images/e670cebe83c0c361223a991f7d0257465165b09ec895ce1a318e24d1eb80eca8.jpg)

![](images/e6c5fa759d4d1616fcde4e033d99e9a2eaf4fb120dd9a4e2dfa4bf98485dbbcc.jpg)  
Figure 2: Some FCT displayed in the physical domain and in the time-frequency domain through a spectrogram. The first one reduces to a wavelet since the chirp rate is 0. One can see the importance of the time duration and the chirp rate and well as the center frequency depending on what one wishes to capture.

![](images/8ffb31be060a00b57485850f94e1c762453a9512264854c48927244bab60bf6e.jpg)

![](images/715e7b67fbd20142b0085e3b3608aa10487b96c4d57945bfc3b073cdc3e9d5c6.jpg)

![](images/694b001633a4bfc51c110f4e1f779ca29a271fcacdcac405ffb3943607653275.jpg)

![](images/7fd7a7180c7b5591819a42e0d56d81d9e88d396f46c56360fedc798746c05a2c.jpg)  
Figure 3: FCT of 4 species of amazonian birds LifeClef 2015 challenge including BIRD10 dataset available online. The call patterns are the high SNR (red) regions. The species international codes are, from top to bottom, right to left: nnbhgj, aethwv, aksucy, nipfbr.

![](images/8d38442550741110692fd1e3f8094510e98872182515eb9e8eb6da2fcdb3c339.jpg)

# 5 ENHANCING CNN BIOACOUSTIC REPRESENTATION WITH FCT

A strategy for CNN fine-tuning can be to retrain a classifier on top of a CNN on a new dataset, or to fine-tune the weights of a pretrained network by continuing the backpropagation. It is possible to

fine-tune all the layers of the CNN or to freeze some of the earlier, later or central layers, and to only fine-tune some portion of the network. As the features propagate deeper and deeper in the network layers, they become increasingly invariant and discriminative (Seltzer 2013). Thus usually only the higher level are fine-tuned, the earlier features of a CNN contain more generic features that should be useful to many tasks. As denoted in later layers of the CNN becomes progressively more specific to the details of the classes contained in the original dataset.

In this paper we adapt our parametric Chirplet decomposition to a specific acoustic domain with a specific CNN. We compare a CNN trained on raw audio to one trained on Mel and Chirplet. The best model is the one trained on parametric Chirplet. Second, we show that the CNN can be enhanced by pretraining Chirp in low level layer.

# 5.1 CNN BIRDS CLASSIFICATION ON FCT, RAW AUDIO, VERSUS MEL

The first demonstration is conducted on complex Bird songs. We use the BIRD10 subset of LifeClef 2016 bird classification challenge. It was used as ENS Ulm data challenge 2016, and contains 3 species in a total of 15 minutes of recordings (SR 44100 Hz, 16 bits), and is available (.wav, Mel and FCT features) at http://sabiod.univ-tln.fr/workspace/BIRD10.

We train 3 CNNs (LeCun & Bengio, 1995) on the Lasagne Theano platform. The baseline CNN is trained from the raw audio. A second CNN, with similar topology (see annexe) is trained on a simple log of the simple 64 channels Mel scale of FFT spectrum ( http://pydoc.net/Python/librosa/0.2.0/librosa.feature/ ). We overlap by  $90\%$  the time windows. A third CNN is trained on our FCT. The parameters of both CNN are similar, with 64 frequency bands each (we remove top and bottom band from the Chirplet to set to 64 bands only). Then the input layer is  $64 \times 86$ , the Conv layer has 20 filters of size  $8 \times 10$ . All activation functions are relu. We maxpool  $2 \times 2$ , follow the 20 filters of size  $8 \times 10$ , maxpooling, dense layer (200), dropout at  $10\%$ , with a final softmax dense layer with 3 classes and same dropout. Each CNN is trained by cross-entropy, L2 reg., with a learning rate set of 0.001.

The Fig. 4 gives the MAP of these two CNNs having similar hyperparameters. The CNN on FCT gives the best MAP with  $61.5\%$  at epoch 280 compared to later epoch (820) for Mel with a similar MAP of  $61\%$ . Audio is slower and weaker ( $58\%$  MAP at epoch 1140).

![](images/10871f70da11553724de0411201116c6f085175d950ce93e5f7ffabb2ca1e143.jpg)  
Figure 4: The Mean Average Precision on BIRD10 of the CNNs on Mel, raw audio, or FCT. The training conditions are the same on the three CNNs, and they have similar size and topology (see Annexe). The CNN trained on FCT is slightly better than on Mel or raw audio, and is learning faster.

# 5.2 ENHANCING BIRDS CLASSIFICATION STACKING PRETRAINED CHIRPNET CNN

In order to test the efficiency of the FCT, we pretrain a CNN to encode audio to Chirplets (a.k.a. the Audio2Chirp CNN) and a CNN to convert parametric Chirplet to classes (a.k.a. the Chirp2Class CNN). The topology of these CNNs (Tab. 2, 3) is set for reasonable time of training. We also speed up the training with shorter time overlap of the time windows (only  $30\%$  instead of  $90\%$  in the previous experimentation). We then decrease the average MAP, however the objective here is to compare the gain in MAP and time of convergence in stacked Chirplet deep representations.

We then simply stack at low level layer the audio2chirp with the chirp2class CNN to build a complete audio2class CNN. We train it from random initialization, or from pretrained CNN. Note that the random seed in all the experimentation of this paper is fixed to allow fair comparisons. Results are reported in Tab. 1 for each of the stacked CNN, with the epoch giving the best MAP on the dev. set, and the corresponding MAP on the test set. Results demonstrate that the pretraining of low level layers by FCT enhances CNN. More details are given in Annexe.

<table><tr><td>Model</td><td>BIRD dev
nb. Epoch</td><td>BIRD test
MAP %</td></tr><tr><td>Baseline:
Audio2Class(0) CNN trained from random initialisation (Fig. 6)</td><td>530 (rel.gain)</td><td>51 (rel. gain)</td></tr><tr><td>Audio2Class CNN from stacked
pretrained Audio2Chirp(*) with Chirp2Class(*) CNNs (Fig. 7)</td><td>390 (-26%)</td><td>53 (+3.9%)</td></tr><tr><td>Audio2Class CNN from stacked
pretrained Audio2Chirp(*) with Chirp2Class(*) CNNs
without updating Chirp2Class(*) (Fig. 8)</td><td>380 (-28%)</td><td>55 (+7.8%)</td></tr></table>

Table 1: Summary of the CNN enhanced by our FCT representation, on BIRD. For each model, we detail the time of convergence on dev. and corresponding Mean Average Precision on test set.

# 5.3 ENHANCING VOWELS CLASSIFICATION STACKING PRETRAINED CHIRPNET CNN

In this section we run the same demonstration on the subset of speech vowels of all the TIMIT acoustic-phonetic corpus JS et al. (1993): 3,696 training utterances (sampled at  $16\mathrm{kHz}$ ) from 462 speakers. The cross-validation set consists of 400 utterances from 50 speakers. The core test set of the 8 vowels subset was used to report the results: 192 utterances from 24 speakers, excluding the validation set. There are 61 hand labeled phonetic symbols but the experiments in this paper run on the time windows of 310ms centered on each of the 8 vowels of TIMIT (= iy, ih, eh, ae, aa, ah, uh, uw).

Due to similar bioacoustic voicing dynamics of the two species (near  $4\mathrm{Hz}$ ), we simply set the FCT parameters for vowel to the one used for Orca presented above ( $p = 3, j = 4, q = 16, t = 0.001, s = 0.01$ ). The time windows are set to  $310~\mathrm{ms}$  as recommended in Palaz et al. (2013).

The results of the different training stages of the audio2chirp and chirp2class and stacked model are given in Tab.2 and Annexe. We run due to lack of time the experiment only on vowel classification, which does not really allow comparison with other papers, however this seminal work only aims to study the relative gain between CNN pretrained or not by FCT.

The results demonstrate that FCT pretraining of the audio2class model is improved by  $2.3\%$  of relative gain of accuracy while the training time is decreased by  $26\%$ .

# 6 DISCUSSION AND CONCLUSION

In this paper we propose for the first time at our knowledge the definition and implementation of a Fast Chirplet Transform (FCT). Due to its low complexity, FCT can be computed as fast as FFT.

![](images/b70c946553e6d07b34d70e21728aaf7af198bb598f46ae0856185ce6ba2419f5.jpg)  
Fast Chirplet Transform (top) versus Fast Fourier Transform (bottom) on Speech (TIMIT)

![](images/150c4c1e66f83e9a6c745ede3694daf552c727e217efa4d981cdc0d12afeca6b.jpg)

![](images/df1c0784b39e0c1ea8fa62d6161181ff01d5b782a840e82d6211d82f9a581781.jpg)  
Figure 5: FCT (top) versus Fourier spectrogram (bottom) of two utterances of Speech vowel (TIMIT),  $(p = 3,j = 4,q = 16,t = 0.001,s = 0.01)$ .

![](images/3bbb27a89cbfb92ba12139f7a6eaa87fa321d8508f9d844bc00222e696fe7195.jpg)

![](images/368718f1e1bdaf23540f81d49e3fa29a497684c1ef280dd55d2aaba20d9433dc.jpg)  
Figure 6: Training stacked CNN. Blue: random initialization of audio2chirp(0) and chirp2class(0). Red: Initialization with optimal audio2chirp(*) and chirp2class(*).

Second we show that FCT pretraining accelerates CNN. For Bird10 data set, we have 280 epochs using FCT versus 820 on Mel features, or 1140 on raw audio for same MAP score. The stacked CNN with the chirpnet in low level layer also decreases training from 530 epochs to 380 epochs, while

<table><tr><td>Model
Scores on the 8 vowels of TIMIT</td><td>TIMIT dev
nb. Epoch (rel. gain)</td><td>TIMIT test
AC % (rel. gain)</td></tr><tr><td>Baseline: Audio2Class CNN from scratch</td><td>79 (ref.)</td><td>66.5 (ref.)</td></tr><tr><td>Audio2Class CNN trained from stacked
pretrained Audio2Chirp(*) with Chirp2Class(*) CNNs</td><td>58 (-26%)</td><td>68.0 (+2.3%)</td></tr></table>

Table 2: Summary of the CNN enhanced by our FCT representation on vowel (TIMIT): time of convergence and vowel accuracy on TIMIT test set.

it increases MAP by 4 points (Tab. 1). The experiment on Vowels demonstrates a training of 30 epochs on FCT, versus 60 on raw audio (for same  $65\%$  accuracy level), and an increase of 1.5 point of accuracy (Tab. 2).

These gains may be due to the sparsity of the Chirplet, and the denoising step in the FCT. These experiences bring to light the problem of deep learning for small and biased dataset for which a full learning strategy is sub-optimal due to local optimum convergence. As a result, FCT prior knowledge can be used to mitigate this drawback by reducing the complexity of the deep-net architecture.

Three main perspectives are then opened. Future work will consist on sparse Chirpnet inspired from tonotopic net Strom (1997), auditory nerve and cortex topology Pironkov et al. (2015). The acoustic vibrations are transmitted to the base of the cochlea, thus each region of the basilar membrane are excited by different frequencies. The higher frequencies excite areas closer to the cochlea base, whereas lower frequencies are closer to the apex. This implies that neurons connected to a specific zone of the basilar membrane will be simultaneously stimulated inducing tonotopic representation.

A second perspective is to integrate Chirplet computation into the CNN training itself, as a constrained embedded layer, in a framework similar to a Wavelet Neural Network (Adeli & Jiang, 2006) but with Chirplet activation functions.

Last, we currently work on transfer learning of Chirpnet from animal to speech (and reverse), in order to generalize a deep Chirpnet representation of the animal communication systems.

# 7 ACKNOWLEDGEMENTS

We thank colleagues from ENS Paris Data Team with S. Mallat, and P. Flandrin, for fruitful discussions on Scattering and Chirplet. We thank YLC and YB for advises on CNN. We thank V. Tassan for cleaning the code. We used Theano, Lasagne  $^{2}$ , Librosa  $^{3}$  and Pysoundfile  $^{4}$ .

# REFERENCES

Hojjat Adeli and Xiaomo Jiang. Dynamic fuzzy wavelet neural network model for structural system identification. Journal of Structural Engineering, 132(1):102-111, 2006.  
Joakim Andén and Stéphane Mallat. Deep scattering spectrum. IEEE Transactions on Signal Processing, 62(16):4114-4128, 2014.  
Joan Bruna and Stéphane Mallat. Invariant scattering convolution networks. IEEE transactions on pattern analysis and machine intelligence, 35(8):1872-1886, 2013.  
Ronald R Coifman, Yves Meyer, and Victor Wickerhauser. Wavelet analysis and signal processing. In In Wavelets and their Applications. Citeseer, 1992.  
Ronald R Coifman, Yves Meyer, Steven Quake, and M Victor Wickerhauser. Signal processing and compression with wavelet packets. In Wavelets and their applications, pp. 363-379. Springer, 1994.

Patrick Flandrin. Time frequency and chirps. Proc. SPIE, 4391:161-175, 2001. doi: 10.1117/12.421196. URL http://dx.doi.org/10.1117/12.421196.  
Garofolo JS, LF Lamel, and al. Timit acoustic-phonetic continuous speech corpus. In Linguistic data consortium, Philadelphia, 1993.  
Nina Kowalski, Didier A Depireux, and Shihab A Shamma. Analysis of dynamic spectra in ferret primary auditory cortex. ii. prediction of unit responses to arbitrary dynamic spectra. Journal of Neurophysiology, 76(5):3524-3534, 1996.  
Yann LeCun and Yoshua Bengio. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10):1995, 1995.  
Stéphane Mallat. A wavelet tour of signal processing. Academic press, 1999.  
Steve Mann and Simon Haykin. The chirplet transform: A generalization of gabor's logon transform. In Vision Interface, volume 91, pp. 205-212, 1991.  
Steve Mann and Simon Haykin. Adaptive chirplet transform: an adaptive generalization of the wavelet transform. Optical Engineering, 31(6):1243-1256, 1992.  
Eduardo Mercado, Catherine E Myers, and Mark A Gluck. Modeling auditory cortical processing as an adaptive chirplet transform. Neurocomputing, 32:913-919, 2000.  
Yves Meyer. Wavelets-algorithms and applications. Wavelets-Algorithms and applications Society for Industrial and Applied Mathematics Translation., 142 p., 1, 1993.  
Dimitri Palaz, Ronan Collobert, and Mathew Magimai-Doss. Estimating phoneme class conditional probabilities from raw speech signal using convolutional neural networks. CoRR, abs/1304.1018, 2013. URL http://arxiv.org/abs/1304.1018.  
Gueorgui Pironkov, Stéphane Dupont, and Thierry Dutoit. Investigating sparse deep neural networks for speech recognition. In IEEE ASRU Workshop, pp. 124-129, 2015.  
William H Press. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.  
Nikko Strom. A tonotopic artificial neural network architecture for phoneme probability estimation. In Automatic Speech Rec. and Understanding IEEE Wkp, pp. 156-163, 1997.
