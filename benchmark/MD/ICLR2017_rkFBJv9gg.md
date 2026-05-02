# LEARNING FEATURES OF MUSIC FROM SCRATCH

John Thickstun<sup>1</sup>, Zaid Harchaoui<sup>2</sup> & Sham Kakade<sup>1</sup>

$^{1}$  Department of Computer Science and Engineering,  $^{2}$  Department of Statistics

University of Washington

Seattle, WA 98195, USA

{thickstn, sham}@cs.washington.edu, zaid@uw.edu

# ABSTRACT

We introduce a new large-scale music dataset, MusicNet, to serve as a source of supervision and evaluation of machine learning methods for music research. MusicNet consists of hundreds of freely-licensed classical music recordings by 10 composers, written for 10 instruments, together with instrument/note annotations resulting in over 1 million temporal labels on 34 hours of chamber music performances under various studio and microphone conditions.

We define a multi-label classification task to predict notes in musical recordings, along with an evaluation protocol. We benchmark several machine learning architectures for this task: i) learning from "hand-crafted" spectrogram features; ii) end-to-end learning with a neural net; iii) end-to-end learning with a convolutional neural net. We show that several end-to-end learning proposals outperform approaches based on learning from hand-crafted audio features.

# 1 INTRODUCTION

Music research has benefited recently from the effectiveness of machine learning methods on a wide range of problems from music recommendation (van den Oord et al., 2013; McFee & Lanckriet, 2011) to music generation (Driedger et al., 2015); see also the recent demos of the Google Magenta project<sup>1</sup>. As of today, there is no large publicly available labeled dataset for the simple yet challenging task of note prediction for classical music. The MIREX MultiF0 Development Set (Benetos & Dixon, 2011) and the Bach10 dataset (Duan et al., 2011) together contain less than 7 minutes of labeled music. These datasets were designed for method evaluation, not for training supervised learning methods.

This situation stands in contrast to other application domains of machine learning. For instance, in computer vision, large labeled datasets such ImageNet (Russakovsky et al., 2015) were fruitfully used to train end-to-end learning architectures. Learned feature representations have outperformed traditional hand-crafted low-level visual features and lead to tremendous progress for image classification. In (Humphrey et al., 2012), Humphrey, Bello, and LeCun issued a call to action: "Deep architectures often require a large amount of labeled data for supervised training, a luxury music informatics has never really enjoyed. Given the proven success of supervised methods, MIR would likely benefit a good deal from a concentrated effort in the curation of sharable data in a sustainable manner."

We introduce here a new large labeled dataset, MusicNet, that we make publicly available $^2$  to foster progress learning feature representations of music. MusicNet is a large corpus of aligned labels on freely-licensed classical music recordings, made possible by licensing initiatives of the European Archive, the Isabella Stewart Gardner Museum, Musopen, and various individual artists. The dataset consists of 34 hours of human-verified aligned recordings, containing a total of 1, 299, 329 individual labels on segments of these recordings. Table 1 summarizes statistics of MusicNet.

MusicNet  

<table><tr><td>Minutes</td><td>Labels</td><td>Recordings</td><td>Error Rate</td></tr><tr><td>2,048</td><td>1,299,329</td><td>330</td><td>4.0%</td></tr><tr><td colspan="2">Ensemble</td><td>Minutes</td><td>Labels</td></tr><tr><td colspan="2">Solo Piano</td><td>917</td><td>576,471</td></tr><tr><td colspan="2">String Quartet</td><td>405</td><td>259,702</td></tr><tr><td colspan="2">Accompanied Violin</td><td>148</td><td>124,886</td></tr><tr><td colspan="2">Piano Quartet</td><td>73</td><td>60,362</td></tr><tr><td colspan="2">Accompanied Cello</td><td>63</td><td>37,557</td></tr><tr><td colspan="2">String Sextet</td><td>48</td><td>33,248</td></tr><tr><td colspan="2">Piano Trio</td><td>46</td><td>28,873</td></tr><tr><td colspan="2">Piano Quintet</td><td>25</td><td>27,545</td></tr><tr><td colspan="2">Wind Quintet</td><td>43</td><td>24,820</td></tr><tr><td colspan="2">Horn Piano Trio</td><td>30</td><td>18,799</td></tr><tr><td colspan="2">Wind Octet</td><td>23</td><td>14,635</td></tr><tr><td colspan="2">Clarinet-Cello-Piano Trio</td><td>25</td><td>13,447</td></tr><tr><td colspan="2">Pairs Clarinet-Horn-Bassoon</td><td>24</td><td>12,218</td></tr><tr><td colspan="2">Clarinet Quintet</td><td>26</td><td>11,184</td></tr><tr><td colspan="2">Solo Cello</td><td>49</td><td>10,876</td></tr><tr><td colspan="2">Accompanied Clarinet</td><td>20</td><td>10,049</td></tr><tr><td colspan="2">Solo Violin</td><td>30</td><td>8,837</td></tr><tr><td colspan="2">Violin and Harpsichord</td><td>16</td><td>7,469</td></tr><tr><td colspan="2">Viola Quintet</td><td>15</td><td>4,156</td></tr><tr><td colspan="2">Solo Flute</td><td>8</td><td>2,214</td></tr><tr><td colspan="4">| Piano | Violin | Cello | Viola | Clarinet | Bassoon |</td></tr><tr><td>Notes</td><td>83</td><td>51</td><td>51</td></tr></table>

<table><tr><td>Composer</td><td>Minutes</td><td>Labels</td></tr><tr><td>Beethoven</td><td>1,085</td><td>736,072</td></tr><tr><td>Schubert</td><td>253</td><td>146,648</td></tr><tr><td>Brahms</td><td>192</td><td>133,109</td></tr><tr><td>Mozart</td><td>156</td><td>99,641</td></tr><tr><td>Bach</td><td>184</td><td>62,782</td></tr><tr><td>Dvorak</td><td>56</td><td>46,261</td></tr><tr><td>Cambini</td><td>43</td><td>24,820</td></tr><tr><td>Faure</td><td>33</td><td>22,349</td></tr><tr><td>Ravel</td><td>27</td><td>21,243</td></tr><tr><td>Haydn</td><td>15</td><td>6,404</td></tr></table>

<table><tr><td>Instrument</td><td>Minutes</td><td>Labels</td></tr><tr><td>Piano</td><td>1346</td><td>794,532</td></tr><tr><td>Violin</td><td>874</td><td>230,484</td></tr><tr><td>Viola</td><td>621</td><td>99,407</td></tr><tr><td>Cello</td><td>800</td><td>99,132</td></tr><tr><td>Clarinet</td><td>173</td><td>24,426</td></tr><tr><td>Bassoon</td><td>102</td><td>14,954</td></tr><tr><td>Horn</td><td>132</td><td>11,468</td></tr><tr><td>Oboe</td><td>66</td><td>8,696</td></tr><tr><td>Flute</td><td>69</td><td>8,310</td></tr><tr><td>Harpsichord</td><td>16</td><td>4,914</td></tr><tr><td>String Bass</td><td>38</td><td>3,006</td></tr></table>

Table 1: Summary statistics of the MusicNet dataset.

The focus of this paper is the problem of learning low-level features of music from raw audio data. We define a multi-label classification task to predict notes in musical recordings, along with an evaluation protocol. We benchmark a variety of machine learning architectures for this task: i) learning from "hand-crafted" spectrogram features; ii) end-to-end learning with a neural net; iii) end-to-end learning with a convolutional neural net. We show that several end-to-end learning architectures outperform approaches based on learning from hand-crafted audio features. The experimental results suggest that, for each of the proposed models, modulated sine-like waveform features are stable, optimal low-level features of musical audio. The learned low-level features are visualized in Figure 1.

![](images/be7838c81ac5dbeec275c28c8faccb3d5c0397770f8294d91ccede9244ea58a9.jpg)  
Figure 1: (Left) Bottom-level weights learned by a two-layer ReLU network trained with  $\ell_2$  regularized  $(\lambda = 1)$  square loss for multi-label classification on raw audio recordings. (Middle) Magnified view of the center of each set of weights. (Right) The spectrogram of each set of weights.

![](images/7bccc2507ef8341f0680069acb8e975c77a8fbe36047c6f36ede44b04ef8cb95.jpg)

![](images/aad7ec5d6e722714767e5c1ce3fae93168a563e6b517bd71dacfc2c8c59238d7.jpg)

# 2 MUSICNET

MusicNet is a large collection of freely-licensed recordings together with labels on these recordings exemplified in Table 2. We find that large amounts of data are essential to recovering useful features from music; see Sect. 4.1 for details. The Lakh dataset, released this summer based on the work of Raffel & Ellis (2015), offers note-level annotations for many 30-second clips of pop music in the Million Song Dataset (McFee et al., 2012). Other large-scale music databases are less useful for supervised representation learning. The RWC dataset (Goto et al., 2003) does not have note-level labels. The MAPS dataset (Emiya et al., 2010) consists of synthesized data, which expressive models could overfit. The Mazurka project<sup>3</sup> consists of commercial music; accessing this dataset comes at a cost and inconvenience, requiring researchers to track down a multitude of commercial recordings. Both the MAPS and Mazurka datasets are comprised entirely of piano music.

The MusicNet dataset consists of 330 recordings of a variety of instruments arranged in small chamber ensembles under various studio and microphone conditions. The recordings average 6 minutes in length. The shortest recording in the dataset is 55 seconds and the longest is almost 18 minutes. Table 1 summarizes the statistics of MusicNet with breakdowns into various types of labels. Table 2 demonstrates examples of labels from the MusicNet dataset.

<table><tr><td>Start</td><td>End</td><td>Instrument</td><td>Note</td><td>Measure</td><td>Beat</td><td>Note Value</td></tr><tr><td>45.29</td><td>45.49</td><td>Violin</td><td>G5</td><td>21</td><td>3</td><td>Eighth</td></tr><tr><td>48.99</td><td>50.13</td><td>Cello</td><td>A#3</td><td>24</td><td>2</td><td>Dotted Half</td></tr><tr><td>82.91</td><td>83.12</td><td>Viola</td><td>C5</td><td>51</td><td>2.5</td><td>Eighth</td></tr></table>

Table 2: MusicNet labels on the Pascal String Quartet's recording of Beethoven's Opus 127, String Quartet No. 12 in E-flat major, I - Maestoso - Allegro. Creative commons use of this recording is made possible by the work of the European Archive.

MusicNet labels come from 513 label classes using the most naive definition of a class: distinct instrument/note combinations. The breakdowns reported in Table 1 indicate the number of distinct notes that appear for each instrument in our dataset. For example, while a piano has 88 keys only 83 of them are performed in MusicNet. For many tasks a note's value will be a part of its label, in which case the number of classes will expand by approximately an order of magnitude after taking the cartesian product of the set of classes with the set of values: quarter-note, eighth-note, triplet, etc. We also remark that labels regularly overlap in the time series creating polyphonic multi-labels.

MusicNet is heavily skewed towards Beethoven, thanks to the composer's popularity among performing ensembles. The dataset is also skewed towards Solo Piano due to an abundance of digital scores available for piano works. For training purposes, we expect that researchers may want to augment this dataset to increase coverage of instruments such as Flute and Oboe that are underrepresented in MusicNet. Researchers who do not need to distribute their dataset can make use of immense libraries of commercial recordings. These recordings can be labeled using the alignment protocol described in Sect. 3.

# 3 DATASET CONSTRUCTION

We have collected 158 hours of freely-licensed classical music recordings from the European Archive, the Isabella Stewart Gardner Museum, Musopen, and various artists' collections. We have also collected 1,618 digital scores in the MIDI format from online resources including the Classical Archives (classicalarchives.com) Suzuchan's Classic MIDI (suzumidi.com) and HarfeSoft (harfesoft.de). We can produce an alignment in cases where a digital score in our collection corresponds to a freely-licensed recording. In addition to our aligned scores, we have gathered MIDI scores containing an additional 6,550,760 labels; we make these labels available to researchers who wish to augment MusicNet with commercial recordings.

Music-to-score alignment is a long-standing problem in the music research and signal processing communities (Raphael, 1999). Dynamic time warping (DTW) is a classical approach to this prob

lem. An early reference using DTW is Orio & Schwarz (2001) where music is aligned to a crude synthesis of the score designed to capture some of the structure of an overtone series. We make use of side information from a synthesizer, aligning music to an artificial performance of a score. To the best of our knowledge, commercial synthesis was first used for the purpose of alignment in Turetsky & Ellis (2003).

The majority of previous work on alignment focuses on pop music. This is more challenging than aligning classical music because commercial synthesizers do a poor job reproducing the wide variety of vocal and instrumental timbers that appear in modern pop. Furthermore, pop features anharmonic instruments such as drums for which natural metrics on frequency representations—including  $\ell^2$ —are unmeaningful. We find that a variant of the techniques described in Turetsky & Ellis (2003) works robustly for classical music to score alignment; we discuss our evaluation of this procedure and its error rate on MusicNet in the appendix.

![](images/c39478c405e53661a66bdd17724f2d31771bcb502a06d83c7b52336991ea4f97.jpg)  
Figure 2: (Left) Heatmap visualization of local alignment costs between the synthesized and recorded spectrograms, with the optimal alignment path in red. The block from  $x = 0$  to  $x = 100$  corresponds to silence at the beginning of the recorded performance. The slope of the alignment can be interpreted as an instantaneous tempo ratio between the recorded and synthesized performances. The curvature in the alignment between  $x = 100$  and  $x = 175$  corresponds to an extension of the first notes by the performer. (Right) Annotation of note onsets on the spectrogram of the recorded performance, determined by the alignment shown on the left.

![](images/3e11b92d3209c1bc28281f36c4395afdd9207497a87312e23bfea7f014b22802.jpg)

In order to align the performance with a score, we need to define a metric that compares short segments of the score with segments of a performance. Musical scores can be expressed as binary vectors in  $E \times K$  where  $E = \{1, \dots, n\}$  and  $K$  is a dictionary of notes. Performances reside in  $\mathbb{R}^{T \times p}$ , where  $T \in \{1, \dots, m\}$  is a sequence of time steps and  $p$  is the dimensionality of the spectrogram at time  $T$ . Given some local cost function  $C: (\mathbb{R}^p, K) \to \mathbb{R}$ , a score  $\mathbf{Y} \in E \times K$ , and a performance  $\mathbf{X} \in \mathbb{R}^{T \times p}$ , the alignment problem is to

$$
\begin{array}{l l} \underset {t \in \mathbb {Z} ^ {n}} {\text {m i n i m i z e}} & \sum_ {i = 1} ^ {n} C \left(\mathbf {X} _ {t _ {i}}, \mathbf {Y} _ {i}\right) \\ \text {s u b j e c t t o} & t _ {0} = 0, \\ & t _ {n} = m \\ & t _ {i} \leq t _ {j} \quad \text {i f} i <   j. \end{array} \tag {1}
$$

Dynamic time warping gives an exact solution to the problem in  $\mathcal{O}(mn)$  time and space.

The success of dynamic time warping depends on the metric used to compare the score and the performance. Previous works can be broadly categorized into three groups that define an alignment cost  $C$  between segments of music  $\mathbf{x}$  and score  $\mathbf{y}$  by injecting them into a common normed space via maps  $\Psi$  and  $\Phi$ :

$$
C (\mathbf {x}, \mathbf {y}) = \| \Psi (\mathbf {x}) - \Phi (\mathbf {y}) \| \tag {2}
$$

The most popular approach—which we have adopted—maps the score into the space of the performance (Orio & Schwarz, 2001; Turetsky & Ellis, 2003; Soulez et al., 2003). An alternative approach maps both the score and performance into some third space, commonly a chromogram space (Hu et al., 2003; Izmirli & Dannenberg, 2010; Joder et al., 2013). Finally, some recent methods consider alignment in score space, taking  $\Phi = \mathrm{Id}$  and learning  $\Psi$  (Garreau et al., 2014; Lajugie et al., 2016).

With reference to the general cost (2), we must specify the maps  $\Psi, \Phi$ , and the norm  $\|\cdot\|$ . We compute the cost in the performance feature space  $\mathbb{R}^p$ , hence we take  $\Psi = \mathrm{Id}$ . For our features, we use the log-spectrogram with a window size of 2048 samples. We use a stride of 512 samples between features. Hence adjacent feature frames are computed with  $75\%$  overlap. For audio sampled at  $44.1\mathrm{kHz}$ , this results in a feature representation with  $44,100/512 \approx 86$  frames per second. A discussion of these parameter choices can be found in the appendix. The map  $\Phi$  is computed by a synthetizer: we used Plogue's Sforzando sampler together with Garritan's Personal Orchestra 4 sample library.

For a (pseudo)-metric on  $\mathbb{R}^p$ , we take the  $\ell^2$  norm  $\| \cdot \|_2$  on the low 50 dimensions of  $\mathbb{R}^p$ . Recall that  $\mathbb{R}^p$  represents Fourier components, so we can roughly interpret the  $k$ 'th coordinate of  $\mathbb{R}^p$  as the energy associated with the frequency  $k \times (22,050/1024) \approx k \times 22.5\mathrm{Hz}$ , where  $22,050\mathrm{Hz}$  is the Nyquist frequency of a signal sampled at  $44.1\mathrm{kHz}$ . The 50 dimension cutoff is chosen empirically: we observe that our alignments are much more accurate using a small number of low-frequency bins rather than the full space  $\mathbb{R}^p$ . Synthesizers do not accurately reproduce the high-frequency features of a musical instrument; by ignoring the high frequencies, we align on a part of the spectrum where the synthesis is most accurate. Our choice of cutoff is aggressive compared to usual settings; for instance, Turetsky & Ellis (2003) propose cutoffs in the  $2.5\mathrm{kHz}$  range. The fundamental frequencies of many notes in our dataset are higher than the  $50 \times 22.5\mathrm{Hz} \approx 1\mathrm{kHz}$  cutoff. Nevertheless, we find that all notes align well using only the low-frequency information.

# 4 METHODS

We consider identification of notes in a segment of audio  $\mathbf{x} \in \mathcal{X}$  as a multi-label classification problem, modeled as follows. Assign each audio segment a binary label vector  $\mathbf{y} \in \{0,1\}^{128}$ . The 128 dimensions correspond to frequency codes for notes, and  $\mathbf{y}_n = 1$  if note  $n$  is present at the midpoint of  $\mathbf{x}$ . Let  $f: \mathcal{X} \to \mathcal{H}$  indicate a feature map. We train a multivariate linear regression to predict  $\hat{\mathbf{y}}$  given  $f(\mathbf{x})$ , which we optimize for square loss. The vector  $\hat{\mathbf{y}}$  can be interpreted as a multi-label estimate of notes in  $\mathbf{x}$  by choosing a threshold  $c$  and predicting label  $n$  iff  $\hat{\mathbf{y}}_n > c$ . We search for  $c$  on a sampled subset of MusicNet, optimizing for F-score with grid search.

# 4.1 RELATED WORK

Learning on raw audio has been considered in both the music and speech communities. Supervised learning on music has been driven by access to labeled datasets. Pop music annotations with chord labels (Harte, 2010) have lead to a long line of work on supervised chord recognition, most recently Korzeniowsk & Widmer (2016). Song-level genre labels and various other metadata have also attracted substantial work on representation learning; a recent example is Choi et al. (2016). There is also substantial work modeling raw audio representations of speech; a current example is Tokuda & Zen (2016).

Because access to large labeled datasets was historically limited, much of the work in the music community is unsupervised. Variants of non-negative matrix factorization are popular in the music information retrieval community, for example Khlif & Sethu (2015). Berg-Kirkpatrick et al. (2014) develops a Bayesian model for piano music. Recent work from Google DeepMind explores generative models of raw audio, including music (van den Oord et al., 2016).

# 4.2 MULTI-LAYER PERCEPTRONS

We construct a two-layer ReLU network using the features  $f_{i}(\mathbf{x}) = \max (0,w_{i}^{T}\mathbf{x})$ . Figure 1 illustrates a selection of weights  $w_{i}$  learned by the bottom layer of this network, optimized for multi-label classification using square loss. The weights learned by the network are modulated sinusoids. This explains the effectiveness of spectrograms and related transforms as a low-level representation of musical audio. The weights decay at the boundaries, analogous to Gabor filters in vision. This behavior is explained by our labeling methodology: the audio segments used here are approximately  $1/3$  of a second long, and a segment is given a note label if that note is on in the center of the segment. Therefore information at the boundaries of the segment is less useful for prediction than information nearer to the center.

# 4.3 SPECTROGRAMS

Spectrograms are a popular engineered feature representation for audio signals, which are closely related to the two-layer ReLU network discussed above. If  $\mathbf{x} = (x_{1},\dots ,x_{t})$  denotes a segment of an audio signal of length  $t$  then we can define

$$
\operatorname {S p e c} _ {k} (\mathbf {x}) \equiv \left| \sum_ {s = 1} ^ {t} e ^ {i k s} x _ {s} \right| ^ {2} = \left(\sum_ {s = 1} ^ {t} \cos (k s) x _ {s}\right) ^ {2} + \left(\sum_ {s = 1} ^ {t} \sin (k s) x _ {s}\right) ^ {2}.
$$

These features are not precisely learnable by the two-layer ReLU network. But recall that  $|x| = \max(0, x) + \max(0, -x)$  and if we take weight vectors  $\mathbf{u}, \mathbf{v} \in \mathbb{R}^T$  with  $u_t = \cos(kt)$  and  $v_t = \sin(kt)$  then the ReLU network can learn

$$
f _ {k, \cos} (\mathbf {x}) + f _ {k, \sin} (\mathbf {x}) = | \mathbf {u} ^ {T} \mathbf {x} | + | \mathbf {v} ^ {T} \mathbf {x} | = \left| \sum_ {s = 1} ^ {t} \cos (k s) x _ {s} \right| + \left| \sum_ {s = 1} ^ {t} \sin (k s) x _ {s} \right|.
$$

We call this family of features a ReLUgram and observe that it has a similar form to the spectrogram; we merely replace the  $x \mapsto x^2$  non-linearity of the spectrogram with  $x \mapsto |x|$ . These features achieve similar performance to spectrograms on our classification task (see Table 3).

# 4.4 WINDOW SIZE

When we parameterize a network, we must choose the width of the set of weights in the bottom layer. This width is called the receptive field in the vision community; in the music community it is called the window size. Traditional frequency analyses, including spectrograms, are highly sensitive to the window size. Windows must be long enough to capture relevant information, but not long so long that they lose temporal resolution; this is the classical time-frequency tradeoff. Furthermore, windowed frequency analysis is subject to boundary effects, known as spectral leakage. Classical signal processing attempts to dampen these effects with hand-crafted window functions, which apply a mask that attenuates the signal at the boundaries (Rabiner & Schafer, 2007).

Our models learn good window functions. If we parameterize our models with a large window size then the model will learn that distant information is irrelevant to local prediction, so the magnitude of the learned weights will attenuate at the boundaries (see Figure 1). We therefore focus our attention on two window sizes: 2048 samples, which captures the local content of the signal, and 16,384 samples, which is sufficient to capture almost all relevant context (again we refer to Figure 1; substantially larger window sizes would be a needless computation burden, because the weights at further distances will approximately vanish).

# 4.5 REGULARIZATION

The size of MusicNet is essential to achieving the results in Figure 1. Prior work on end-to-end audio learning was unable to recover clean sinusoidal features from data (Dieleman & Schrauwen, 2014). We encountered similar problems when optimizing on a small subset of MusicNet. In Figure 3 (Left) we optimize a two-layer ReLU network on 65,000 monophonic data points; compare this to similar results in Figure 3 of Dieleman & Schrauwen (2014). We can recover sinusoidal features on the small dataset using heavy regularization, but this destroys classification performance; regularizing with dropout poses a similar tradeoff. By contrast, Figure 3 (Right) shows weights learned on the full MusicNet dataset using no regularization whatsoever. We are still exploring the effects of  $\ell_2$  regularization on the full dataset; preliminary experiments suggest that a modest amount of regularizer stabilizes the optimization and produces even cleaner features without sacrificing performance.

# 4.6 CONVOLUTIONAL NETWORKS

Previously, we estimated  $\hat{\mathbf{y}}$  by regressing against  $f(\mathbf{x})$ . We now consider a convolutional model that regresses against features of a collection of shifted segments  $\mathbf{x}_{\ell}$  near to the original segment  $\mathbf{x}$ . The learned features of this network are visually comparable to those learned by the fully connected network (Figure 1). We have experimented with the stride and number of convolutions in this network. The results reported in Table 3 were achieved using a 64-sample stride and 97 convolutions across a

![](images/5e5ee9a00f99afef73a91ca06fb2bbdec4eb4231d314f89da833daa3bd6260d2.jpg)  
Figure 3: (Left) Features learned by a 2-layer ReLU network trained on small monophonic subset of MusicNet. (Right) Features learned by the same network, trained on the full MusicNet dataset.

![](images/38ea0415115b005a6175ea51e08880dedd6d3f6d87567a7978b51c5346b4ea64.jpg)

window of 16, 384 samples, using a receptive field of 10, 240 samples. Performance correlates with the resolution of the stride and the number of convolutions, but the learned features are consistent across parameterizations. We also experimented with average and max pooling operations. In all cases the learned features are comparable to those of a fully connected network.

# 5 RESULTS

We hold out a test set of 3 recordings for all the results reported in this section:

- Bach's Prelude in D major for Solo Piano. WTK Book 1, No 5. Performed by Kimiko Ishizaka. MusicNet recording id 2303.  
- Mozart's Serenade in E-flat major. K375, Movement 4 - Menuetto. Performed by the Soni Ventorum Wind Quintet. MusicNet recording id 1819.  
- Beethoven's String Quartet No. 13 in B-flat major. Opus 130, Movement 2 - Presto. Released by the European Archive. MusicNet recording id 2382.

Our test set is a representative sampling of MusicNet: it covers most of the instruments in the dataset in small, medium, and large ensembles. The test data points are evenly spaced segments separated by 512 samples, between the 1st and 91st seconds of each recording. For the wider features, there is substantial overlap between adjacent segments. Each segment is labeled with the notes that are on in the middle of the segment.

![](images/6e352a1f1ea8e46ea762f4fe9015bed5b133da91d1ab96e029ebc43db760c22e.jpg)  
Figure 4: Precision-recall curves for the convolutional network on the test set. Curves are evaluated on subsets of the test set consisting of all data points (blue); points with exactly one label (monophonic; green); and points with exactly three labels (red).

We evaluate our models on three scores: precision, recall, and average precision. The precision score is the count of correct predictions by the model (across all data points) divided by the total number

of predictions by the model. The recall score is the count of correct predictions by the model divided by the total number of (ground truth) labels in the test set. Precision and recall are parameterized by the note prediction threshold  $c$  (see Sect. 4). By varying  $c$ , we construct precision-recall curves (see Figure 4). The average precision score is the area under the precision-recall curve.

<table><tr><td>Model</td><td>Features</td><td>Precision</td><td>Recall</td><td>Average Precision</td></tr><tr><td>Linear</td><td>512-point spectrogram</td><td>22.1%</td><td>47.0%</td><td>22.0%</td></tr><tr><td>Linear</td><td>1024-point spectrogram</td><td>28.9%</td><td>52.5%</td><td>30.2%</td></tr><tr><td>Linear</td><td>1024-point ReLUgram</td><td>24.1%</td><td>60.0%</td><td>29.3%</td></tr><tr><td>Linear</td><td>4096-point spectrogram</td><td>35.2%</td><td>63.2%</td><td>40.3%</td></tr><tr><td>Linear</td><td>8192-point spectrogram</td><td>32.1%</td><td>65.6%</td><td>37.5%</td></tr><tr><td>MLP, 500 nodes</td><td>2048 raw samples</td><td>37.5%</td><td>57.5%</td><td>41.0%</td></tr><tr><td>MLP, 2500 nodes</td><td>2048 raw samples</td><td>40.5%</td><td>58.6%</td><td>43.8%</td></tr><tr><td>AvgPool, 5 stride</td><td>2048 raw samples</td><td>38.9%</td><td>59.0%</td><td>43.3%</td></tr><tr><td>MLP, 500 nodes</td><td>16384 raw samples</td><td>40.0%</td><td>63.7%</td><td>45.5%</td></tr><tr><td>CNN, 64 stride</td><td>16384 raw samples</td><td>43.2%</td><td>70.7%</td><td>52.0%</td></tr></table>

Table 3: Benchmark results on MusicNet for models discussed in this paper. All models were optimized using the Tensorflow library (Abadi et al.). The MLP is a 2-layer ReLU network with an unregularized square loss objective. The AvgPool model is parameterized by 500 hidden nodes and 11 convolutions. The CNN was parameterized with 500 hidden nodes and 97 convolutions. We report the precision and recall corresponding to the best  $F_{1}$ -score.

A spectrogram of length  $n$  is computed from  $2n$  samples, so the linear 1024-point spectrogram model is directly comparable to the MLP runs with 2048 raw samples. We find that our learned features<sup>4</sup> significantly beat the performance of spectrograms. Our discussion of windowing in Sect. 4.4 partially explains this. Figure 5 suggests a second reason. Recall (Sect. 4.3) that the spectrogram features can be interpreted as the magnitude of the signal's inner product with sine waves of linearly spaced frequencies. In contrast, our networks learn weights with frequencies distributed similarly to the distribution of notes in our dataset (Figure 5). This gives our network higher resolution in the most critical frequency regions.

![](images/edc04ba298e0b8fd8f97d1a05c594956b06d45ec2845362bb35558bc755973c9.jpg)  
Figure 5: (Left) The frequency distribution of notes in MusicNet. (Right) The frequency distribution of learned nodes in a 500-node, two-layer ReLU network.

![](images/d39a9f2034d77bb8bfea2f84c7f9cbe1b478c04d113637983047d1cbd7b43697.jpg)

In future work, we plan to investigate learned mid-level and high-level features of musical audio. While mid-level features could capture harmonic structure, high-level features could capture the overall structure of a recording. Both mid-level and high-level representations require the low-level features learned in this paper as building blocks to extract short-term and long-term memory temporal structures.

# REFERENCES

M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mane, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viegas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems. URL http://tensorflow.org/.  
E. Benetos and S. Dixon. Joint multi-pitch detection using harmonic envelope estimation for polyphonic music transcription. IEEE Selected Topics in Signal Processing, 2011.  
T. Berg-Kirkpatrick, J. Andreas, and D. Klein. Unsupervised transcription of piano music. NIPS, 2014.  
K. Choi, G. Fazes, and M. Sandler. Automatic tagging using deep convolutional neural networks. ISMIR, 2016.  
S. Dieleman and B. Schrauwen. End-to-end learning for music audio. ICASSP, 2014.  
J. Driedger, T. Pratzlich, and M. Müller. Let It Bee - Towards NMF-inspired audio mosaicing. ISMIR, 2015.  
Z. Duan, B. Pardo, and C. Zhang. Multiple fundamental frequency estimation by modeling spectral peaks and non-peak regions. TASLP, 2011.  
V. Emiya, R. Badeau, and B. David. Multipitch estimation of piano sounds using a new probabilistic spectral smoothness principle. TASLP, 2010.  
D. Garreau, R. Lajugie, S. Arlot, and F. Bach. Metric learning for temporal sequence alignment. NIPS, 2014.  
M. Goto, H. Hashiguchi, T. Nishimura, and R. Oka. RWC music database: Music genre database and musical instrument sound database. ISMIR, 2003.  
C. Harte. Towards Automatic Extraction of Harmony Information from Music Signals. PhD thesis, Department of Electrical Engineering, Queen Mary, University of London, 2010.  
N. Hu, R. B. Dannenberg, and G. Tzanetakis. Polyphonic audio matching and alignment for music retrieval. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics, 2003.  
E. J. Humphrey, J. P. Bello, and Y. LeCun. Moving beyond feature design: Deep architectures and automatic feature learning in music informatics. ISMIR, 2012.  
O. Izmirli and R. B. Dannenberg. Understanding features and distance functions for music sequence alignment. ISMIR, 2010.  
C. Joder, S. Essid, and G. Richard. Learning optimal features for polyphonic audio-to-score alignment. TASLP, 2013.  
A. Khlif and V. Sethu. An iterative multi range non-negative matrix factorization algorithm for polyphonic music transcription. ISMIR, 2015.  
F. Korzeniowsk and G. Widmer. Feature learning for chord recognition: the deep chroma extractor. ISMIR, 2016.  
R. Lajugie, P. Bojanowski, P. Cuvillier, S. Arlot, and F. Bach. A weakly-supervised discriminative model for audio-to-score alignment. ICASSP, 2016.  
B. McFee and G. Lanckriet. Learning multi-modal similarity. JMLR, 2011.  
B. McFee, T. Bertin-Mahieux, D. P. W. Ellis, and G. Lanckriet. The million song dataset challenge. Proceedings of the 21st International Conference on World Wide Web, 2012.  
N. Orio and D. Schwarz. Alignment of monophonic and polyphonic music to a score. International Computer Music Conference, 2001.

L. Rabiner and R. Schafer. Introduction to digital speech processing. Foundations and trends in signal processing, 2007.  
C. Raffel and D. P. W. Ellis. Large-scale content-based matching of MIDI and audio files. ISMIR, 2015.  
C. Raphael. Automatic segmentation of acoustic musical signals using hidden markov models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1999.  
O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. Imagenet large scale visual recognition challenge. IJCV, 2015.  
F. Soulez, X. Rodet, and D. Schwarz. Improving polyphonic and poly-instrumental music to score alignment. ISMIR, 2003.  
K. Tokuda and H. Zen. Directly modeling voiced and unvoiced components in speech waveforms by neural networks. ICASSP, 2016.  
R. J. Turetsky and D. P. W. Ellis. Ground-truth transcriptions of real music from force-aligned midi syntheses. ISMIR, 2003.  
A. van den Oord, S. Dieleman, and B. Schrauwen. Deep content-based music recommendation. NIPS, 2013.  
A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu. WaveNet: A generative model for raw audio. arXiv preprint, 2016.
