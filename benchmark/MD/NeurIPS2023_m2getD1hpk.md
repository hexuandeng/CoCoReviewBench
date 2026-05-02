# FITS: Modeling Time Series with  $10k$  Parameters

Anonymous Author(s)

Affiliation

Address

email

# Abstract

In this paper, we introduce FITS, a lightweight yet powerful model for time series analysis. Unlike existing models that directly process raw time-domain data, FITS operates on the principle that time series can be manipulated through interpolation in the complex frequency domain. By discarding high-frequency components with negligible impact on time series data, FITS achieves performance comparable to state-of-the-art models for time series forecasting and anomaly detection tasks, while having a remarkably compact size of only approximately  $10k$  parameters. Such a lightweight model can be easily trained and deployed in edge devices, creating opportunities for various applications. The anonymous code repo is available in: https://anonymous.4open.science/r/FITS

# 1 Introduction

Time series analysis plays a crucial role in numerous domains, including finance, energy, weather forecasting, and signal processing, where understanding and predicting temporal patterns are essential. Existing time series analysis methods primarily focus on extracting features in the time domain (Zhou et al., 2021; Liu et al., 2022; Zeng et al., 2022; Nie et al., 2023; Zhang et al., 2022). However, due to the inherent complexity and dynamic nature of time series data, the information contained in the time domain tends to be sparse and dispersed. Consequently, researchers design intricate methodologies and complex models to capture and exploit this information, often relying on approaches such as transformer architectures (Zhou et al., 2021; Wu et al., 2021; Zhou et al., 2022a). However, these sophisticated techniques often lead to the proliferation of large-scale and computationally demanding models, posing challenges in terms of efficiency and scalability.

Conversely, the frequency domain representation of time series data offers a more concise and compact representation of its underlying information. Recognizing this potential, previous studies have explored the utilization of frequency domain information in time series analysis. For instance, FEDformer (Zhou et al., 2022a) incorporates spectral information as a supplementary feature, enhancing the modeling capabilities of transformer-based time series models. Another approach, FNet (Lee-Thorp et al., 2022), leverages frequency domain multiplication to replace convolution operations, thereby reducing computational overhead. Moreover, LTSF-Linear (Zeng et al., 2022) has demonstrated that highly accurate predictions can be achieved by solely learning the dominant periodicity. Similarly, methods like TimesNet (Wu et al., 2023) segment the time series based on frequencies with high amplitude and employ CNNs for multi-periodicity feature extraction.

However, existing methodologies often overlook the fundamental nature of the frequency domain representation, which utilizes complex numbers to express both amplitude and phase information. Motivated by the fact that longer time series segments provide a higher-resolution frequency representation, we propose FITS (Frequency Interpolation Time Series Analysis Baseline). The core component of FITS is a complex-valued linear layer that can explicitly learn amplitude scaling and phase shift to perform interpolation in the complex frequency domain. Although FITS conducts interpolation in the frequency domain, it remains an end-to-end time domain model incorporating

the rFFT (Brigham & Morrow, 1967). Specifically, we project the input segment to the complex frequency domain for frequency interpolation using rFFT. We then project the interpolated frequency representation back to the time domain as a longer segment for supervision. This end-to-end design enables FITS to adapt to various downstream tasks with commonly-used time domain supervision, such as forecasting and reconstruction.

Additionally, FITS incorporates a low-pass filter to obtain a compact representation with minimal information loss, resulting in small model volume and minimal computational overhead while maintaining state-of-the-art (SOTA) performance. Notably, under most settings, FITS achieves SOTA performance with under 10k parameters, which is 50 times smaller than the lightweight temporal linear model DLinear (Zeng et al., 2022) and approximately 10,000 times smaller than other mainstream models. The low memory and computation overhead make FITS suitable for deploying or even training on edge devices for forecasting or anomaly detection.

To summarize, our contributions are twofold:

- We introduce FITS, a lightweight model containing merely  $5\mathbf{k} \sim 10\mathbf{k}$  parameters for time series analysis. Despite its compact size which is several orders of magnitude smaller than mainstream models, FITS delivers exceptional performance in various tasks, including long-term forecasting and anomaly detection, achieving state-of-the-art performance in several datasets.  
- FITS employs the complex-valued neural network for time series analysis, which provides a novel perspective that simultaneously captures amplitude and phase information, leading to more comprehensive and efficient modeling of time series data.

# 2 Related Work and Motivation

# 2.1 Frequency-aware Time Series Analysis Models

Recent advancements in time series analysis have witnessed the utilization of frequency domain information to capture and interpret underlying patterns. FNet (Lee-Thorp et al., 2022) leverages a pure attention-based architecture to efficiently capture temporal dependencies and patterns solely in the frequency domain, eliminating the need for convolutional or recurrent layers. On the other hand, FEDFormer (Zhou et al., 2022a) and FiLM (Zhou et al., 2022b) incorporate frequency information as supplementary features to enhance the model's capability in capturing long-term periodic patterns and speed up computation.

The other line of work aims to capture the periodicity inherent in the data. For instance, DLinear (Zeng et al., 2022) adopts a single linear layer to extract the dominant periodicity from the temporal domain and surpasses a range of deep feature extraction-based methods. More recently, TimesNet (Wu et al., 2023) achieves state-of-the-art results by identifying several dominant frequencies instead of relying on a single dominant periodicity. Specifically, they use the Fast Fourier Transform (FFT) to find the frequencies with the largest energy and reshape the original 1D time series into 2D images according to their periods.

However, these approaches still rely on feature engineering to identify the dominant period set. Selecting this set based on energy may only consider the dominant period and its harmonics, limiting the information captured. Moreover, these methodologies are still considered inefficient and prone to overfitting.

# 2.2 Divide and Conquer the Frequency Components

Treating a time series as a signal allows us to break it down into a linear combination of sinusoidal components without any information loss. Each component possesses a unique frequency, initial phase, and amplitude. Forecasting directly on the original time series can be challenging, but forecasting each frequency component is comparatively straightforward, as we only need to apply a phase bias to the sinusoidal wave based on the time shift. Subsequently, we linearly combine these shifted sinusoidal waves to obtain the forecasting result.

This approach effectively preserves the frequency characteristics of the given look-back window while maintaining semantic consistency between the look-back window and the forecasting horizon.

Specifically, the resulting forecasted values maintain the frequency features of the original time series with a reasonable time shift, ensuring that semantic consistency is maintained.  
However, forecasting each sinusoidal component in the time domain can be cumbersome, as the sinusoidal components are treated as a sequence of data points. To address this, we propose conducting this manipulation in the complex frequency domain, which offers a more compact and information-rich representation, as described below.

# 3 Method

# 3.1 Preliminary: FFT and Complex Frequency Domain

The Fast Fourier Transform (FFT, (Brigham & Morrow, 1967)) is a widely used algorithm for efficiently computing the Discrete Fourier Transform (DFT) of a sequence of complex numbers. The DFT is a mathematical operation that converts a discrete-time signal from the time domain to the complex frequency domain. In cases where the input signal is real, such as in time series analysis, the Real FFT (rFFT) is commonly used to obtain a compact representation. With an input of  $N$  real numbers, the rFFT produces a sequence of  $N / 2 + 1$  complex numbers that represent the signal in the complex frequency domain.

# Complex Frequency Domain

In Fourier analysis, the complex frequency domain is a representation of a signal in which each frequency component is characterized by a complex number. This complex number captures both the amplitude and phase of the component, providing a comprehensive description. The amplitude of a frequency component represents the magnitude or strength of that component in the original time-domain signal. In contrast, the phase represents the temporal shift or delay introduced by that component. Mathematically, the complex number associated with a frequency component can be represented as a complex exponential element with a given amplitude and phase:

$$
X (f) = | X (f) | e ^ {j \theta (f)},
$$

where  $X(f)$  is the complex number associated with the frequency component at frequency  $f$ ,  $|X(f)|$  is the amplitude of the component, and  $\theta(f)$  is the phase of the component. As shown in Fig. 1(a), in the complex plane, the complex exponential element can be visualized as a vector with a length equal to the amplitude and angle equal to the phase:

$$
X (f) = | X (f) | (\cos \theta (f) + j \sin \theta (f))
$$

Therefore, the complex number in the complex frequency domain provides a concise and elegant means of representing the amplitude and phase of each frequency component in the Fourier transform.

![](images/1e2079c19b10192ebb5e381f2c00b13c8f16435c1f5ded59f424eaef902d3d34.jpg)  
(a) Complex number on the complex plane

![](images/91093926689095491a1da2b634ea13841b735b98f8355d9a43dd53e48dd2b0fd.jpg)  
Figure 1: Illustration of Complex Number Visualization and Multiplication  
(b) Complex number multiplication

Time Shift and Phase Shift. The time shift of a signal corresponds to the phase shift in the frequency domain. Especially in the complex frequency domain, we can express such phase shift by multiplying a unit complex exponential element with the corresponding phase. Mathematically, if we shift a signal  $x(t)$  forward in time by a constant amount  $\tau$ , resulting in the signal  $x(t - \tau)$ , the Fourier transform is given by:

$$
X _ {\tau} (f) = e ^ {- j 2 \pi f \tau} X (f) = | X (f) | e ^ {j (\theta (f) - 2 \pi f \tau)} = [ c o s (- 2 \pi f \tau) + j s i n (- 2 \pi f \tau) ] X (f)
$$

The shifted signal still has an amplitude of  $|X(f)|$ , while the phase  $\theta_{\tau}(f) = \theta (f) - 2\pi f\tau$  shows a shift which is linear to the time shift.

In summary, the amplitude scaling and phase shifting can be simultaneously expressed as the multiplication of complex numbers, as shown in Fig. 1(b).

# 3.2 FITS Pipeline

Motivated by the fact that a longer time series provides a higher frequency resolution in its frequency representation, we train FITS to generate an extended time series segment by interpolating the frequency representation of the input time series segment. We use a complex-valued linear layer to learn such interpolation. According to the fact that the amplitude scaling and phase shifting can be conveniently expressed as the multiplication of complex numbers, such complex linear combination allows FITS to effectively incorporate both the amplitude scaling and phase shift of frequency components during the interpolation process. As shown in Fig. 2, we use rFFT to project time series segments to the complex frequency domain. After the interpolation, the frequency representation is projected back with inverse rFFT (irFFT).

![](images/6a6fa5f8b600d4063b39f724003fc69dd5991b78b5062fc7ad4f774475651731.jpg)  
Figure 2: Pipeline of FITS, with a focus on the forecasting task. The reconstruction task follows the same pipeline, except for the reconstruction supervision loss.

However, we cannot directly use the frequency representation of the original input time series segment because the mean of such segments will result in a very large 0-frequency component in its complex frequency representation. To eliminate the 0-frequency component, we pass it through reversible instance-wise normalization (RIN) (Kim et al., 2022) to obtain a zero-mean instance. As a result, the normalized complex frequency representation now has a length of  $N / 2$ , where  $N$  represents the original length of the time series.

Furthermore, we incorporate a low-pass filter (LPF) into the FITS model to further reduce its size. The LPF removes high-frequency components above a specified cutoff frequency, resulting in a more compact model representation while retaining the important information of the time series. The rationale behind this design will be elaborated in the subsequent section. Despite operating in the frequency domain, FITS is supervised in the time domain using common loss functions such as Mean Squared Error (MSE) after the irFFT, allowing for diverse supervision tailored to different time series downstream tasks.

In the case of forecasting tasks, we generate the look-back window along with the horizon as shown in Fig. 2. This allows us to provide supervision for forecasting and backcasting, where the model is encouraged to accurately reconstruct the look-back window. Our ablation study reveals that combining backcast and forecast supervision can yield improved performance in certain scenarios.

For reconstruction tasks, we downsample the original time series segment based on a specific downsampling rate. Subsequently, FITS is employed to perform frequency interpolation, enabling the reconstruction of the downsampled segment back to its original form. Thus, direct supervision is applied using reconstruction loss to ensure faithful reconstruction. The reconstruction tasks also follow the pipeline in Fig. 2 with the supervision replaced with reconstruction loss.

# 3.3 Key Mechanisms of FITS

Complex Frequency Linear Interpolation. To control the output length of the model, we introduce an interpolation rate denoted as  $\eta$ , which represents the ratio of the model's output length  $L_{o}$  to its corresponding input length  $L_{i}$ .

It is worth noting that frequency interpolation operates on the normalized complex frequency representation, which has half the length of the original time series. Importantly, this interpolation rate can also be applied to the frequency domain, as indicated by the equation:

$$
\eta_ {f r e q} = \frac {L _ {o} / 2}{L _ {i} / 2} = \frac {L _ {o}}{L _ {i}} = \eta
$$

Based on this formula, with an arbitrary frequency  $f$ , the frequency band  $1 \sim f$  in the original signal is linearly projected to the frequency band  $1 \sim \eta f$  in the output signal. As a result, we define the input length of our complex-valued linear layer as  $L$  and the interpolated output length as  $\eta L$ . Notably, when applying the Low Pass Filter (LPF), the value of  $L$  corresponds to the cutoff frequency (COF) of the LPF. After performing frequency interpolation, the complex frequency representation is zero-padded to a length of  $L_{o} / 2$ , where  $L_{o}$  represents the desired output length. Prior to applying the irFFT, an additional zero is introduced as the representation's zero-frequency component.

Low Pass Filter (LPF). The primary objective of incorporating the LPF within FITS is to compress the model's volume while preserving essential information. The LPF achieves this by discarding frequency components above a specified cutoff frequency (COF), resulting in a more concise frequency domain representation. The LPF retains the relevant information in the time series while discarding components beyond the model's learning capability. This ensures that a significant portion of the original time series' meaningful content is preserved. As demonstrated in Fig. 3, the filtered waveform exhibits minimal distortion even when only preserving a quarter of the original frequency domain representation. Furthermore, the high-frequency components filtered out by the LPF typically comprise noise and trends, which are inherently irrelevant for effective time series modeling.

![](images/676dd2070322f0016ce155509bb0c4bfb8c60cbdd0230e954b022fecdc5e923e.jpg)  
(a) Original

![](images/9831b48caa81945b300d12255cce46ffb3e497f4ed13bdc42ec35a5e4305500f.jpg)  
(b) COF at  $6^{\mathrm{th}}$  harmonic

![](images/49ee775ddb0176c3e3300e2e24f8d119231ecc538ae3cd4a48e5a7b78e164142.jpg)  
Figure 3: Waveform (1 $^{\text{st}}$  row) and amplitude spectrum (2 $^{\text{nd}}$  row) of a time series segment selected from the 'OT' channel of the ETTh1 dataset, spanning from the  $1500^{\text{th}}$  to the  $1980^{\text{th}}$  data point. The segment has a length of 480, and its dominant periodicity is 24, corresponding to a base frequency of 20. The blue lines represent the waveform/spectrum with no applied filter, while the orange lines represent the waveform/spectrum with the filter applied. The filter cutoff frequency is chosen based on a harmonic of the original time series.  
(c) COF at  $3^{\mathrm{rd}}$  harmonic

![](images/91f0b1de078d7fb18e11f02469fb6309a06865f9cf94b237c0c782e1f0350a02.jpg)  
(d) COF at  $2^{\mathrm{nd}}$  harmonic

Selecting an appropriate cutoff frequency (COF) remains a nontrivial challenge. To address this, we propose a method based on the harmonic content of the dominant frequency. Harmonics, which are integer multiples of the dominant frequency, play a significant role in shaping the waveform of a time series. By aligning the cutoff frequency with these harmonics, we keep relevant frequency components associated with the signal's structure and periodicity. This approach leverages the inherent relationship between frequencies to extract meaningful information while suppressing noise and irrelevant high-frequency components. The impact of COF on different harmonics' waveforms is shown in Fig. 3. We further elaborate on the impact of COF in our experimental results.

# 4 Experiments for Forecasting

# 4.1 Forecasting as Frequency Interpolation

Typically, the forecasting horizon is shorter than the given look-back window, rendering direct interpolation unsuitable. Instead, we formulate the forecasting task as the interpolation of a look-back window, with length  $L$ , to a combination of the look-back window and forecasting horizon, with length  $L + H$ . This design enables us to provide more supervision during training. With this approach, we can supervise not only the forecasting horizon but also the backcast task on the look-back window. Our experimental results demonstrate that this unique training strategy contributes to the improved performance of FITS. The interpolation rate of the forecasting task is calculated by:

$$
\eta_ {F o r e} = 1 + \frac {H}{L},
$$

where  $L$  represents the length of the look-back window and  $H$  represents the length of the forecasting horizon.

# 4.2 Experiment Settings

Datasets. All datasets used in our experiments are widely-used and publicly available real-world datasets, including, Traffic, Electricity, Weather, ETT (Zhou et al., 2021). We summarize the characteristics of these datasets in Tab. 1. Apart from these datasets for long-term time series forecasting, we also use the M4 dataset to test the short-term forecasting performance.

Table 1: The statistics of the seven used forecasting datasets.  

<table><tr><td>Dataset</td><td>Traffic</td><td>Electricity</td><td>Weather</td><td>ETTh1&amp;ETTh2</td><td>ETTm1 &amp;ETTm2</td></tr><tr><td>Channels</td><td>862</td><td>321</td><td>21</td><td>7</td><td>7</td></tr><tr><td>Sampling Rate</td><td>1hour</td><td>1hour</td><td>10min</td><td>1hour</td><td>15min</td></tr><tr><td>Total Timesteps</td><td>17,544</td><td>26,304</td><td>52,696</td><td>17,420</td><td>69,680</td></tr></table>

Baselines. To evaluate the performance of FITS in comparison to state-of-the-art time series forecasting models, including PatchTST (Nie et al., 2023), TimesNet (Wu et al., 2023), FEDFormer (Zhou et al., 2022a), FiLM (Zhou et al., 2022b) and LTSF-Linear (Zeng et al., 2023), we directly refer to the reported results in the original papers under the same settings. We report the comparison with other transformer-based methods in the appendix.

Evaluation metrics. We follow the previous works (Zhou et al., 2022a; Zeng et al., 2022; Zhang et al., 2022) to compare forecasting performance using Mean Squared Error (MSE) as the core metrics. Moreover, to evaluate the short-term forecasting, we symmetric Mean Absolute Percentage Error (SMAPE) following TimesNet (Wu et al., 2023).

Implementation details. Following the settings of LTSF-Linear (Zeng et al., 2023), we set the look-back window of FITS as 720 for any forecasting horizon. Further experiments also show that a longer look-back window can result in better performance. To avoid information leakage, We choose the hyper-parameter based on the performance of the validation set.

# 4.3 Comparisons with SOTAs

# Competitive Performance with High Efficiency

We present the results of our experiments on long-term forecasting in Tab. 2 and Tab. 3. The results for short-term forecasting on the M4 dataset are provided in the Appendix. Remarkably, our FITS consistently achieves comparable or even superior performance across all experiments.

Tab. 4 presents the number of trainable parameters for various TSF models using a look-back window of 96 and a forecasting horizon of 720 on the Electricity dataset. The table clearly demonstrates the exceptional efficiency of FITS compared to other models.

Among the listed models, the parameter counts range from millions down to thousands. Notably, large models such as TimesNet and Pyraformer require a staggering number of parameters, with

Table 2: Long-term forecasting results on ETT dataset in MSE. The best result is highlighted in bold, and the second best is highlighted with underline. IMP is the improvement between FITS and the second best/ best result, where a larger value indicates a better improvement.  

<table><tr><td>Dataset</td><td colspan="4">ETTh1</td><td colspan="4">ETTh2</td><td colspan="4">ETTm1</td><td colspan="4">ETTm2</td></tr><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>PatchTST</td><td>0.370</td><td>0.413</td><td>0.422</td><td>0.447</td><td>0.274</td><td>0.341</td><td>0.329</td><td>0.379</td><td>0.293</td><td>0.333</td><td>0.369</td><td>0.416</td><td>0.166</td><td>0.223</td><td>0.274</td><td>0.362</td></tr><tr><td>TimesNet</td><td>0.384</td><td>0.436</td><td>0.491</td><td>0.521</td><td>0.340</td><td>0.402</td><td>0.452</td><td>0.462</td><td>0.338</td><td>0.374</td><td>0.410</td><td>0.478</td><td>0.187</td><td>0.249</td><td>0.321</td><td>0.408</td></tr><tr><td>FEDFormer</td><td>0.376</td><td>0.420</td><td>0.459</td><td>0.506</td><td>0.346</td><td>0.429</td><td>0.496</td><td>0.463</td><td>0.379</td><td>0.426</td><td>0.445</td><td>0.543</td><td>0.203</td><td>0.269</td><td>0.325</td><td>0.421</td></tr><tr><td>FiLM</td><td>0.371</td><td>0.414</td><td>0.442</td><td>0.465</td><td>0.284</td><td>0.357</td><td>0.377</td><td>0.439</td><td>0.302</td><td>0.338</td><td>0.373</td><td>0.420</td><td>0.165</td><td>0.222</td><td>0.277</td><td>0.371</td></tr><tr><td>Dlinear</td><td>0.374</td><td>0.405</td><td>0.429</td><td>0.440</td><td>0.338</td><td>0.381</td><td>0.400</td><td>0.436</td><td>0.299</td><td>0.335</td><td>0.369</td><td>0.425</td><td>0.167</td><td>0.221</td><td>0.274</td><td>0.368</td></tr><tr><td>FITS</td><td>0.375</td><td>0.408</td><td>0.429</td><td>0.427</td><td>0.274</td><td>0.333</td><td>0.340</td><td>0.374</td><td>0.305</td><td>0.339</td><td>0.367</td><td>0.418</td><td>0.164</td><td>0.217</td><td>0.269</td><td>0.347</td></tr><tr><td>IMP</td><td>-0.005</td><td>-0.003</td><td>-0.007</td><td>0.013</td><td>0</td><td>0.008</td><td>-0.011</td><td>0.005</td><td>-0.012</td><td>-0.006</td><td>0.002</td><td>-0.002</td><td>0.002</td><td>0.004</td><td>0.005</td><td>0.015</td></tr></table>

Table 3: Long-term forecasting results on three popular datasets in MSE. The best result is highlighted in bold and the second best is highlighted with underline. IMP is the improvement between FITS and the second best/ best result, where a larger value indicates a better improvement.  

<table><tr><td>Dataset</td><td colspan="4">Electricity</td><td colspan="4">Traffic</td><td colspan="4">Weather</td></tr><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>PatchTST</td><td>0.129</td><td>0.147</td><td>0.163</td><td>0.197</td><td>0.360</td><td>0.379</td><td>0.392</td><td>0.432</td><td>0.149</td><td>0.194</td><td>0.245</td><td>0.314</td></tr><tr><td>TimesNet</td><td>0.168</td><td>0.184</td><td>0.198</td><td>0.220</td><td>0.593</td><td>0.617</td><td>0.629</td><td>0.640</td><td>0.172</td><td>0.219</td><td>0.280</td><td>0.365</td></tr><tr><td>FEDFormer</td><td>0.193</td><td>0.201</td><td>0.214</td><td>0.246</td><td>0.587</td><td>0.604</td><td>0.621</td><td>0.626</td><td>0.217</td><td>0.276</td><td>0.339</td><td>0.403</td></tr><tr><td>FiLM</td><td>0.154</td><td>0.164</td><td>0.188</td><td>0.236</td><td>0.416</td><td>0.408</td><td>0.425</td><td>0.520</td><td>0.199</td><td>0.228</td><td>0.267</td><td>0.319</td></tr><tr><td>Dlinear</td><td>0.140</td><td>0.153</td><td>0.169</td><td>0.203</td><td>0.410</td><td>0.423</td><td>0.435</td><td>0.464</td><td>0.176</td><td>0.218</td><td>0.262</td><td>0.323</td></tr><tr><td>FITS</td><td>0.138</td><td>0.152</td><td>0.166</td><td>0.205</td><td>0.401</td><td>0.407</td><td>0.420</td><td>0.456</td><td>0.145</td><td>0.188</td><td>0.236</td><td>0.308</td></tr><tr><td>IMP</td><td>-0.009</td><td>-0.005</td><td>-0.003</td><td>-0.008</td><td>-0.041</td><td>-0.028</td><td>-0.028</td><td>-0.024</td><td>0.004</td><td>0.006</td><td>0.009</td><td>0.006</td></tr></table>

300.6M and 241.4M, respectively. Similarly, popular models like Transformer, Informer, Autoformer, and FEDformer have parameter counts in the range of 13.61M to 20.68M. Even the lightweight yet state-of-the-art model PatchTST has a parameter count of over 1 million.

In contrast, FITS stands out as a highly efficient model with an impressively low parameter count. With only 4.5K to 16K parameters, FITS achieves comparable or even superior performance compared to these larger models. It is worth highlighting that FITS requires significantly fewer parameters compared to the next smallest model, Dlinear, which has 139.7K parameters. For instance, when considering a 720 look-back window and a 720 forecasting horizon, the Dlinear model requires over 1 million parameters, whereas FITS achieves similar performance with only 10k-50k parameters.

This analysis showcases the remarkable efficiency of FITS. Despite its small size, FITS consistently achieves competitive results, making it an attractive option for time series analysis tasks. FITS demonstrates that achieving state-of-

the-art or close to state-of-the-art performance with a considerably reduced parameter footprint is possible, making it an ideal choice for resource-constrained environments.

Table 4: Number of trainable parameters and MACs of TSF models under lookback window=96 and forecasting horizon=720 on the Electricity dataset.  

<table><tr><td>Model</td><td>Parameters</td><td>MACs</td></tr><tr><td>TimesNet</td><td>301.7M</td><td>1226.49G</td></tr><tr><td>Pyraformer</td><td>241.4M</td><td>0.80G</td></tr><tr><td>Transformer</td><td>13.61M</td><td>4.03G</td></tr><tr><td>Informer</td><td>14.38M</td><td>3.93G</td></tr><tr><td>Autoformer</td><td>14.91M</td><td>4.41G</td></tr><tr><td>FiLM</td><td>14.91M</td><td>5.97G</td></tr><tr><td>FEDformer</td><td>20.68M</td><td>4.41G</td></tr><tr><td>PatchTST</td><td>1.5M</td><td>5.07G</td></tr><tr><td>DLinear</td><td>139.7K</td><td>40M</td></tr><tr><td>FITS (Ours)</td><td>4.5K~10K</td><td>1.6M~8.9M</td></tr></table>

# Case Study on ETTh2 Dataset

We conduct a comprehensive case study on the performance of FITS using the ETTh2 dataset, which further highlights the impact of the look-back window and cutoff frequency on model performance. We provide a case study on other datasets in the Appendix. In our experiments, we observe that increasing the look-back window generally leads to improved performance, while the effect of increasing the cutoff frequency is minor.

Tab. 5 showcases the performance results obtained with different look-back window sizes and cutoff frequencies. Larger look-back windows tend to yield better performance across the board. On the other hand, increasing the cutoff frequency only results in marginal performance improvements. However, it is important to note that higher cutoff frequencies come at the expense of increased computational resources, as illustrated in Tab. 6.

Table 5: The results on the ETTh2 dataset. Values are visualized with a green background, where darker background indicates worse performance. The top-5 best results are highlighted with a red background, and the absolute best result is highlighted with red bold font. F represents supervision on the forecasting task, while  $\mathbf{B} + \mathbf{F}$  represents supervision on backcasting and forecasting tasks.  

<table><tr><td></td><td>Look-back Window</td><td colspan="2">90</td><td colspan="2">180</td><td colspan="2">360</td><td colspan="2">720</td></tr><tr><td>Horizon</td><td>COF/nth Harmonic</td><td>F</td><td>B+F</td><td>F</td><td>B+F</td><td>F</td><td>B+F</td><td>F</td><td>B+F</td></tr><tr><td rowspan="4">96</td><td>2</td><td>0.297687</td><td>0.296042</td><td>0.291606</td><td>0.289387</td><td>0.278644</td><td>0.278403</td><td>0.277708</td><td>0.27696</td></tr><tr><td>3</td><td>0.297796</td><td>0.297377</td><td>0.290061</td><td>0.288239</td><td>0.277512</td><td>0.277746</td><td>0.276537</td><td>0.277068</td></tr><tr><td>4</td><td>0.297106</td><td>0.295624</td><td>0.290725</td><td>0.287993</td><td>0.27624</td><td>0.27693</td><td>0.274207</td><td>0.274498</td></tr><tr><td>5</td><td>0.296168</td><td>0.296698</td><td>0.288518</td><td>0.287375</td><td>0.276367</td><td>0.277935</td><td>0.275989</td><td>0.275636</td></tr><tr><td rowspan="4">192</td><td>2</td><td>0.380163</td><td>0.379868</td><td>0.360591</td><td>0.359769</td><td>0.336552</td><td>0.337976</td><td>0.334854</td><td>0.335887</td></tr><tr><td>3</td><td>0.37983</td><td>0.381802</td><td>0.359088</td><td>0.359498</td><td>0.336384</td><td>0.336358</td><td>0.334666</td><td>0.335507</td></tr><tr><td>4</td><td>0.379657</td><td>0.380439</td><td>0.359087</td><td>0.358536</td><td>0.334803</td><td>0.349995</td><td>0.333522</td><td>0.333382</td></tr><tr><td>5</td><td>0.378556</td><td>0.379883</td><td>0.358809</td><td>0.359376</td><td>0.335451</td><td>0.343227</td><td>0.33384</td><td>0.335053</td></tr><tr><td rowspan="4">336</td><td>2</td><td>0.402706</td><td>0.404805</td><td>0.373257</td><td>0.374678</td><td>0.344241</td><td>0.344414</td><td>0.341869</td><td>0.342549</td></tr><tr><td>3</td><td>0.403238</td><td>0.404878</td><td>0.372231</td><td>0.373948</td><td>0.345578</td><td>0.344976</td><td>0.341436</td><td>0.342793</td></tr><tr><td>4</td><td>0.402702</td><td>0.407712</td><td>0.376199</td><td>0.374435</td><td>0.343004</td><td>0.344167</td><td>0.340795</td><td>0.342245</td></tr><tr><td>5</td><td>0.403484</td><td>0.409516</td><td>0.375102</td><td>0.37462</td><td>0.344333</td><td>0.342731</td><td>0.341043</td><td>0.342214</td></tr><tr><td rowspan="4">720</td><td>2</td><td>0.420072</td><td>0.424272</td><td>0.403985</td><td>0.407392</td><td>0.379822</td><td>0.38519</td><td>0.376871</td><td>0.37677</td></tr><tr><td>3</td><td>0.418323</td><td>0.420538</td><td>0.400986</td><td>0.40686</td><td>0.379638</td><td>0.386397</td><td>0.376236</td><td>0.376004</td></tr><tr><td>4</td><td>0.417485</td><td>0.420982</td><td>0.399987</td><td>0.408128</td><td>0.379096</td><td>0.386409</td><td>0.375865</td><td>0.375637</td></tr><tr><td>5</td><td>0.419122</td><td>0.420355</td><td>0.400776</td><td>0.407871</td><td>0.378665</td><td>0.390754</td><td>0.377138</td><td>0.374586</td></tr></table>

Considering these observations, we find utilizing a longer look-back window in combination with a low cutoff frequency to achieve near state-of-the-art performance with minimal computational cost. For instance, FITS surpasses other methods when employing a 720 look-back window and setting the cutoff frequency to the second harmonic. Remarkably, FITS achieves state-of-the-art performance with a parameter count of only around  $10\mathrm{k}$ . Moreover, by reducing the look-back window to 360, FITS already achieves close-to-state-of-the-art performance by setting the cutoff frequency to the second harmonic, resulting in a further reduction of the model's parameter count to under  $5\mathrm{k}$  (as shown in Tab. 6).

These results emphasize the lightweight nature of FITS, making it highly suitable for deployment and training on edge devices with limited

computational resources. By carefully selecting the look-back window and cutoff frequency, FITS can achieve excellent performance while maintaining computational efficiency, making it an appealing choice for real-world applications.

Table 6: The number of parameters under different settings on ETTh1 & ETTh2 dataset.  

<table><tr><td></td><td></td><td colspan="4">Look-back Window</td></tr><tr><td>Horizon</td><td>COF/nth Harmonic</td><td>90</td><td>180</td><td>360</td><td>720</td></tr><tr><td rowspan="4">96</td><td>2</td><td>703</td><td>1053</td><td>2279</td><td>5913</td></tr><tr><td>3</td><td>1035</td><td>1820</td><td>4307</td><td>12064</td></tr><tr><td>4</td><td>1431</td><td>2752</td><td>6975</td><td>20385</td></tr><tr><td>5</td><td>1922</td><td>3876</td><td>10374</td><td>31042</td></tr><tr><td rowspan="4">192</td><td>2</td><td>1064</td><td>1431</td><td>2752</td><td>6643</td></tr><tr><td>3</td><td>1564</td><td>2450</td><td>5192</td><td>13520</td></tr><tr><td>4</td><td>2187</td><td>3698</td><td>8475</td><td>22815</td></tr><tr><td>5</td><td>2914</td><td>5253</td><td>12558</td><td>34694</td></tr><tr><td rowspan="4">336</td><td>2</td><td>1615</td><td>1998</td><td>3483</td><td>7665</td></tr><tr><td>3</td><td>2392</td><td>3395</td><td>6608</td><td>15704</td></tr><tr><td>4</td><td>3321</td><td>5160</td><td>10725</td><td>26460</td></tr><tr><td>5</td><td>4402</td><td>7293</td><td>15834</td><td>40006</td></tr><tr><td rowspan="4">720</td><td>2</td><td>3078</td><td>3510</td><td>5418</td><td>10512</td></tr><tr><td>3</td><td>4554</td><td>5950</td><td>10266</td><td>21424</td></tr><tr><td>4</td><td>6318</td><td>9030</td><td>16650</td><td>36180</td></tr><tr><td>5</td><td>8370</td><td>12750</td><td>24570</td><td>54780</td></tr></table>

# 5 Experiment for Anomaly Detection

# 5.1 Reconstruction as Frequency Interpolation

As discussed before, we tackle the anomaly detection tasks in the self-supervised reconstructing approach. Specifically, we make a  $N$  time down-sampling on the input and train a FITS network with an interpolation rate of  $\eta_{Rec} = N$  to up-sample it.

# 5.2 Experiment Settings

Datasets. We use five commonly used benchmark datasets: SMD (Server Machine Dataset (Su et al., 2019)), PSM (Polled Server Metrics (Abdulaal et al., 2021)), SWaT (Secure Water Treatment (Mathur & Tippenhauer, 2016)), MSL (Mars Science Laboratory rover), and SMAP (Soil Moisture Active Passive satellite) (Hundman et al., 2018).

Baselines. We compare FITS with models such as TimesNet (Wu et al., 2023), Anomaly Transformer (Xu et al., 2022), THOC (Shen et al., 2020), Omnianomaly (Su et al., 2019). Following TimesNet (Wu et al., 2023), we also compare the anomaly detection performance with other models (Zeng et al., 2023; Zhang et al., 2022; Woo et al., 2022; Zhou et al., 2022a).

Evaluation metrics. Following the previous works (Xu et al., 2022; Shen et al., 2020; Wu et al., 2023), we use Precision, Recall, and F1-score as metrics.

Implementation details. We use a window size of 200 and downsample the time series segment by a factor of 4 to match the original segment during training with the FITS model. Anomaly detection follows the methodology of the Anomaly Transformer (Xu et al., 2022), where time points exceeding a certain reconstruction loss threshold are classified as anomalies. The threshold is selected based on the highest F1 score achieved on the validation set. To handle consecutive abnormal segments, we adopt a widely-used adjustment strategy (Su et al., 2019; Xu et al., 2018; Shen et al., 2020), considering all anomalies within a specific successive abnormal segment as correctly detected when one anomalous time point is identified. This approach aligns with real-world applications, where an abnormal time point often triggers the attention to the entire segment.

Table 7: Anomaly detection result of F1-scores on 5 datasets. The best result is highlighted in bold, and the second best is highlighted with underline. Full results are reported in the Appendix.  

<table><tr><td>Models</td><td>FITS</td><td>TimesNet</td><td>Anomaly Transformer</td><td>THOC</td><td>Omni Anomaly</td><td>Stationary Transformer</td><td>LightTS</td><td>Dlinear</td><td>IMP</td></tr><tr><td>SMD</td><td>99.95</td><td>85.81</td><td>92.33</td><td>84.99</td><td>85.22</td><td>84.72</td><td>82.53</td><td>77.1</td><td>7.62</td></tr><tr><td>PSM</td><td>93.96</td><td>97.47</td><td>97.89</td><td>98.54</td><td>80.83</td><td>97.29</td><td>97.15</td><td>93.55</td><td>-3.93</td></tr><tr><td>SWaT</td><td>98.9</td><td>91.74</td><td>94.07</td><td>85.13</td><td>82.83</td><td>79.88</td><td>93.33</td><td>87.52</td><td>4.83</td></tr><tr><td>SMAP</td><td>70.74</td><td>71.52</td><td>96.69</td><td>90.68</td><td>86.92</td><td>71.09</td><td>69.21</td><td>69.26</td><td>-25.95</td></tr><tr><td>MSL</td><td>78.12</td><td>85.15</td><td>93.59</td><td>89.69</td><td>87.67</td><td>77.5</td><td>78.95</td><td>84.88</td><td>-15.47</td></tr></table>

# 5.3 Comparisons with SOTAs

As shown in Tab. 7, FITS achieves remarkable results on several datasets. Notably, on the SMD and SWaT datasets, FITS exhibits exceptional performance with F1-scores almost reaching perfection at around  $99.95\%$  and  $98.9\%$ , respectively. This demonstrates FITS' ability to accurately detect anomalies and classify them correctly. In comparison, other models, such as TimesNet, Anomaly Transformer, and Stationary Transformer, struggle to match FITS' performance on these datasets.

However, FITS shows comparatively lower performance on the SMAP and MSL datasets. These datasets present a challenge due to their binary event data nature, which may not be effectively captured by FITS' frequency domain representation. While models specifically designed for anomaly detection, such as THOC and Omni Anomaly, achieve higher F1-scores on these datasets.

For a more comprehensive evaluation, waveform visualizations and detailed analysis can be found in the appendix, providing deeper insights into FITS' strengths and limitations in different anomaly detection scenarios. It is important to note that the reported results are achieved with a parameter range of 1-4K and MACs (Multiply-Accumulate Operations) of 10-137K, which will be further detailed in the appendix.

# 6 Conclusions and Discussion

In this paper, we propose FITS for time series analysis, a low-cost model with  $10k$  parameters that can achieve performance comparable to state-of-the-art models that are often several orders of magnitude larger. As a frequency-domain modeling technique, FITS has difficulty handling binary-valued time series and time series with missing data. For the former category, time-domain modeling is preferable as the raw data format is sufficiently compact. For the latter category, we could first employ simple yet effective time-domain imputation techniques and then apply FITS for efficient analysis.

# References

Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. Practical approach to asynchronous multivariate time series anomaly detection and localization. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery; Data Mining, KDD '21, pp. 2485-2494, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383325. doi: 10.1145/3447548.3467174. URL https://doi.org/10.1145/3447548.3467174.  
E. O. Brigham and R. E. Morrow. The fast fourier transform. IEEE Spectrum, 4(12):63-70, 1967. doi: 10.1109/MSPEC.1967.5217220.  
Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining. ACM, jul 2018. doi: 10.1145/3219819.3219845. URL https://doi.org/10.11452F3219819.3219845.  
Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=cGDAkQo1COp.  
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. Fnet: Mixing tokens with fourier transforms, 2022.  
Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet: Time series modeling and forecasting with sample convolution and interaction. In Advances in Neural Information Processing Systems, 2022.  
Aditya P. Mathur and Nils Ole Tippenhauer. Swat: a water treatment testbed for research and training on ics security. In 2016 International Workshop on Cyber-physical Systems for Smart Water Networks (CySWater), pp. 31-36, 2016. doi: 10.1109/CySWater.2016.7469060.  
Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.  
Lifeng Shen, Zhuoong Li, and James Kwok. Timeseries anomaly detection using temporal hierarchical one-class network. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 13016-13026. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf.  
Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery; Data Mining, KDD '19, pp. 2828-2837, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450362016. doi: 10.1145/3292500.3330672. URL https://doi.org/10.1145/3292500.3330672.  
Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshit Kumar, and Steven Hoi. Etsformer: Exponential smoothing transformers for time-series forecasting, 2022.  
Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34:22419-22430, 2021.  
Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023.

Haowen Xu, Yang Feng, Jie Chen, Zhaogang Wang, Honglin Qiao, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, and Dan Pei. Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications. In Proceedings of the 2018 World Wide Web Conference on World Wide Web - WWW '18. ACM Press, 2018. doi: 10.1145/3178876.3185996. URL https://doi.org/10.1145/2F3178876.3185996.  
Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly transformer: Time series anomaly detection with association discrepancy, 2022.  
Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? arXiv preprint arXiv:2205.13504, 2022.  
Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? 2023.  
Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022.  
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11106-11115, 2021.  
Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International Conference on Machine Learning, 2022a.  
Tian Zhou, Ziqing Ma, xue wang, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, and Rong Jin. FiLM: Frequency improved legendre memory model for long-term time series forecasting. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022b. URL https://openreview.net/forum?id=ztQdHSQUWc.