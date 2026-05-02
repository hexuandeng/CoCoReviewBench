# EM-GANSIM: REAL-TIME AND ACCURATE EM SIMULATION USING CONDITIONAL GANS FOR 3D IN-DOOR SCENES

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a novel machine-learning (ML) approach (EM-GANSim) for real-time electromagnetic (EM) propagation that is used for wireless communication simulation in 3D indoor environments. Our approach uses a modified conditional Generative Adversarial Network (GAN) that incorporates encoded geometry and transmitter location while adhering to the electromagnetic propagation theory. The overall physically-inspired learning is able to predict the power distribution in 3D scenes, which is represented using heatmaps. Our overall accuracy is comparable to ray tracing-based EM simulation, as evidenced by lower mean squared error values. Furthermore, our GAN-based method drastically reduces the computation time, achieving a 5X speedup on complex benchmarks. In practice, it can compute the signal strength in a few milliseconds on any location in 3D indoor environments. We also present a large dataset of 3D models and EM ray tracing-simulated heatmaps. To the best of our knowledge, EM-GANSim is the first real-time algorithm for EM simulation in complex 3D indoor environments. We plan to release the code and the dataset.

# 1 INTRODUCTION

Electromagnetic (EM) waves, characterized by the oscillation of electric and magnetic fields, are central to technologies such as visible light, microwave ovens, and wireless communication systems, including Wi-Fi and 5G. Maxwell's equations (Maxwell, 1873) describe the interaction and propagation of these electric and magnetic fields through space, forming the theoretical foundation for understanding EM wave behavior, including reflection, refraction, diffraction, and scattering. These principles are critical when modeling wave propagation in complex environments (Obaidat & Green, 2003; D'Aucelli et al., 2018), such as indoor spaces, where multiple interactions with obstacles and materials occur.

In the field of EM simulation, various methods are employed to understand wave propagation and interaction with media. Path loss and attenuation play crucial roles in these simulations, measuring how much signal power diminishes over distance, due to obstacles or the medium itself. Ray tracing is a widely-used technique for simulating wave interactions with surfaces (Bertoni et al., 1994; Seidel & Rappaport, 1992), as it balances computational efficiency and accuracy. It simulates rays, as narrow beams of EM energy, traveling in straight lines and accounting for key phenomena like reflection and diffraction. The method's efficiency makes it popular for 5G network planning (Hsiao et al., 2017), vehicular communications (Wang & Manocha, 2022), electromagnetic characterization (Egea-Lopez et al., 2021), and ground-penetrating radar (Zhang et al., 2006). Other methods such as wave-based methods that numerically solve Maxwell's equations can provide more accurate results, capturing complex wave behavior such as diffraction and scattering more precisely. However, these methods are often too computationally intensive for real-time or large-scale applications (Coifman et al., 1993; Taftove et al., 2005),

Despite its advantages, current EM simulation systems, particularly those based on ray tracing, have limitations in terms of handling dynamic scenes or complex environments. Ray tracing relies on modeling rays, i.e., narrow beams of EM energy, that travel in straight lines until they encounter an object, tracing their paths from a source and modeling interactions like reflections and diffrac

tions (McKown & Hamilton, 1991). The simulation accuracy depends on detailed environmental models and material properties, making it computationally intensive, and needs significant processing power to simulate the numerous potential ray paths in complex environments. For dynamic scenes and detailed indoor environments, the need to continually update the models and recompute new paths in real-time is a major challenge. Indoor simulations are particularly difficult due to the complexity and density of the obstacles, which further increases the computational load, making current methods inefficient for applications requiring quick responses, such as 5G network planning, where higher frequencies and more complex environments are used (Rappaport et al., 2013; Wang et al., 2020).

Innovative solutions, such as integrating generative adversarial networks (GANs) into EM simulations, are being explored to address these limitations. GAN models include considerations for path loss, reflection, and diffraction in their loss functions, and they also account for material properties and multipath propagation, thereby improving simulation accuracy and heatmap generation for real-world applications.

Main Results: We present a novel GAN-based prediction scheme for real-time EM simulation in 3D indoor scenes. Our formulation uses a physically-inspired generator to predict wireless signal received power heatmaps and ensures high accuracy by incorporating detailed signal propagation mechanisms such as direct propagation, reflection, and diffraction. These physical constraints are embedded within the GAN's loss function to ensure that the generated data adheres to the principles of electromagnetic wave propagation. We use ray tracing techniques to model how signals propagate through an environment, considering reflections off surfaces and diffractions around the obstacles (Sangkusolwong & Apavatjrut, 2017). We evaluate these physical interactions using EM propagation models and the uniform theory of diffraction (UTD) (Kanatas et al., 1997) to predict the path loss for indoor environments accurately. Our approach not only improves the reliability of the heatmap predictions but also enhances the robustness and convergence of the GAN during training. Our main contributions include:

- Accurate Power Distributions: By employing conditional Generative Adversarial Networks (cGANs) and utilizing the strengths of physics-inspired learning, our approach can predict accurate power distributions in 3D indoor environments.  
- Real-Time Performance: We highlight the performance on 15 complex 3D indoor benchmarks. Our approach significantly reduces the computational time needed for simulations compared to prior methods based on ray tracing. Our GAN models streamline the simulation process, achieving 5X faster running time on entire power map generation for various-sized indoor models. Additionally, it enables real-time simulation for individual data points in just a few milliseconds.  
- Dataset: We present a large, comprehensive dataset featuring varied indoor scenarios (2K+ models) and simulated heatmaps (more than 64M) to train our model. This dataset ensures robust and generalized model performance across diverse conditions and is used for training and testing.

# 2 PRIOR WORK

Recent efforts in integrating ML with EM ray tracing and wireless communication systems have highlighted the potential of ML in enhancing wireless communication technologies in various ways. DeepRay (Bakirtzis et al., 2022) uses a data-driven approach that integrates a ray-tracing simulator with deep learning models, specifically convolutional encoder-decoders such as U-Net and SDU-Net, enhancing indoor radio propagation modeling for accurate signal strength prediction in various indoor environments. The model is able to learn from multiple environments and predict unknown geometries with high accuracy. WAIR-D (Huangfu et al., 2022) introduces a comprehensive dataset supporting AI-based wireless research, emphasizing the creation of realistic simulation environments for enhanced model generalization and facilitating fine-tuning for specific scenarios using real-world map data. Huang et al. (Huang et al., 2021) integrate ray tracing and an autoencoding-translation neural network to perform 3-D sound-speed inversion, improving efficiency and accuracy in underwater acoustic applications. Yin et al. (Yin et al., 2022) investigate the use of millimeter wave (mmWave) wireless signals in assisting robot navigation and employ a learning-based classifier for link state classification to enhance robotic movement and decision-making in complex

environments. There are other methods that combine deep reinforcement learning with enhanced ray tracing for antenna tilt optimization and those leveraging 5G MIMO data for beam selection using deep learning techniques to improve cellular network performance through efficient geospatial data processing and precise signal optimization (Zhu et al., 2022; Wang et al., 2023).

ML techniques have also been used to predict the received power in complex indoor and urban environments (Yun & Iskander, 2015). Traditional methods like regression models, decision trees, and support vector machines have been used to model the propagation characteristics of electromagnetic fields. The performance of these methods has been improved by adapting to data from specific environments, thereby enhancing prediction accuracy for both line-of-sight (LoS) and non-line-of-sight (NLoS) conditions (Filosa et al., 2016; Dong et al., 2020; Williams et al., 2015).

Despite their advancements, traditional simulation methods (e.g., ray tracing) face limitations in terms of capturing the highly nonlinear interactions and multipath effects characteristic of indoor and urban EM propagation. The complexity increases with the need to model dynamic changes in the environment such as moving objects and varying channel conditions, which are not always well-addressed by conventional approaches (Marey et al., 2022). On the other hand, cGANs (Creswell et al., 2018) are widely used for tasks requiring the generation of new data instances that resemble a given distribution.

# 3 METHODOLOGY

# 3.1 OVERVIEW

In this section, we present our novel approach for augmenting EM ray tracing techniques with a modified cGAN. Our goal is to design a simulator for 3D indoor scenes, the accuracy of which is similar to that of EM ray tracers but is significantly faster for real-time or dynamic scenarios. Figure 1 shows the overall architecture of our network:

![](images/89f20bdf9d467d82fc857cd411a94401472f810a08255113030c9ccc848453e8.jpg)  
Figure 1: Overall architecture of our cGAN training process. The Generator (G) takes encoded 3D geometry, transmitter location, and a noise vector to output simulated heatmaps. The Discriminator (D) evaluates both the real heatmap from a ray-tracing simulator DCEM and the generated heatmap from G and makes 0/1 decisions.

Our network's formulation can be described as follows:

$$
P _ {\mathrm {r}} = f _ {\mathrm {c G A N}} \left(E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z\right) \tag {1}
$$

where

-  $P_{\mathrm{r}}$  is a 3D vector, representing the received power across the indoor environment, as depicted by the generated heatmap (on some height level). This is the primary output of our cGAN model, representing the simulated EM field distribution.  
-  $f_{\mathrm{cGAN}}$  denotes the function computed using the conditional Generative Adversarial Network. It models the complex relationship between the indoor environment's geometry, material properties, the transmitter's location, and the resulting EM signal heatmap.

-  $E_{\mathrm{g}}$  represents the encoded geometry information of the indoor environment. It encapsulates details such as the spatial layout in 2D with height information and material properties, which are used for accurate EM propagation modeling.  
-  $L_{\mathrm{tx}}$  refers to the precise location of the transmitter within the environment. The transmitter's position, in conjunction with the environment's geometry, significantly impacts EM wave propagation and the distribution of received power.  
-  $z$  represents model or input noise. The noise type is selected through a hyperparameter tuning process. This term accounts for potential discrepancies and uncertainties inherent in the simulation process and is used to improve the accuracy of estimated signal strength throughout the 3D model.

# 3.2 MODIFIED CONDITIONAL GAN

In the context of EM simulations, cGANs offer a unique advantage by not only predicting EM field distributions but also generating potential scenarios that could affect these distributions in dynamic environments (Ratnarajah et al., 2022; Kazeminia et al., 2020). We choose cGANs over other learning methods because of several compelling advantages that cGANs offer (Goodfellow et al., 2014), particularly in terms of generating high-quality synthetic data and improving the accuracy and efficiency of path loss predictions. Unlike traditional methods, cGANs are specifically designed for data generation tasks, making them well-suited for creating heat maps based on complex indoor environments and addressing the challenges associated with received power prediction in wireless communication network design and optimization.

Our approach, EM-GANSim, utilizes GANs for several reasons that align with our objectives for real-time and accurate EM simulation in 3D indoor environments:

1. Data Generation: GANs are adept at generating synthetic data that closely mirrors the distribution of real data. In the context of EM simulation, this capability allows us to create detailed heatmaps that accurately represent the complex interactions of electromagnetic waves with indoor structures. More details are discussed in the Results and Appendix sections.  
2. Efficiency: Traditional ray tracing methods can be computationally intensive, especially for large-scale or real-time applications. GANs, once trained, can generate simulations rapidly, which is crucial for achieving real-time performance. A detailed comparison is provided in Sections 5.1 and 5.2.  
3. Flexibility: GANs can be conditioned on specific parameters, such as geometry and transmitter location, enabling the generation of simulations tailored to particular scenarios without the need for extensive recalculations.  
4. Handling Complexity: GANs are well-suited to capture indoor EM propagation's highly nonlinear and multipath effects, which can be challenging for conventional simulation methods. For example, conventional ray tracing simulators took much longer time to generate heatmap in average scenes as shown in Section 5.2.  
5. Generalization: By learning from a diverse dataset, GANs can generalize to new environments and scenarios, which is essential for creating a versatile simulation tool that can be applied across a wide range of conditions. Our method achieves generalization by training the GAN on a diverse dataset of over 2,000 indoor scenes with various room sizes, configurations, and materials.

We present a modified cGAN architecture for our specific task of simulating wireless communication in 3D indoor environments. Our generator takes as input both the geometry information and a noise vector to generate realistic heatmaps that closely match the distribution of the simulated data. The discriminator's role is to distinguish between the real heatmaps derived from EM simulations and the approximate heatmaps generated by the model. We modify the cGAN architecture to account for the EM propagation models to generate accurate heatmaps. Our modified cGAN Error (Generator Network) is defined as:

$$
\mathcal {L} _ {\mathrm {c G A N}} ^ {\mathrm {G}} = \mathbb {E} _ {E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z} [ \log (1 - D (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, G (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z))) ] \tag {2}
$$

This equation represents the loss for the generator G in the cGAN and aims to minimize the ability of discriminator D to distinguish generated heatmaps from real ones. The Mean Squared Error (MSE) loss measures the discrepancy between the real received power and the power predicted by the generator, given below:

$$
\mathcal {L} _ {\mathrm {M S E}} = \mathbb {E} _ {E _ {\mathrm {g}}, L _ {\mathrm {t x}}, P _ {\mathrm {r}}} [ \| P _ {\mathrm {r}} - G (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z) \| _ {2} ^ {2} ] \tag {3}
$$

# 3.2.1 GENERATOR

Our generator uses a series of convolutional neural network (CNN) layers designed to capture the intricate spatial relationships within indoor environments. Special attention is given to encoding the geometry information effectively, allowing the model to understand how different materials and layouts affect signal propagation. We also incorporate physical constraints into the objective function, ensuring that the generated samples adhere to the fundamental principles of electromagnetic wave propagation. The generator objective function is given as:

$$
\mathcal {L} _ {\mathrm {c G A N}} ^ {\mathrm {G}} = - \mathbb {E} _ {E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z} [ \log D (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, G (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z)) ] + \lambda \mathcal {L} _ {\mathrm {M S E}} + \mu \mathcal {L} _ {\mathrm {p h y}}. \tag {4}
$$

This equation combines the cGAN loss with the MSE loss balanced by a weighting factor  $\lambda$ . Additionally,  $\mathcal{L}_{\mathrm{phy}}$  represents the physical constraints loss, and  $\mu$  is a weighting factor that balances the importance of the physical constraints in the overall objective function. The physical constraints loss  $\mathcal{L}_{\mathrm{phy}}$  includes terms that account for direct propagation, reflection, and diffraction effects:

$$
\mathcal {L} _ {\mathrm {p h y}} = \alpha \mathcal {L} _ {\text {d i r e c t}} + \beta \mathcal {L} _ {\text {r e f l e c t i o n}} + \gamma \mathcal {L} _ {\text {d i f f r a c t i o n}} \tag {5}
$$

Where: -  $\mathcal{L}_{\mathrm{direct}}$  is the loss due to direct path propagation, calculated as:

$$
\mathcal {L} _ {\text {d i r e c t}} = \sum_ {i = 1} ^ {N} \left(P L _ {d} \left(d _ {i}, f\right) - P L _ {d} \left(d _ {i}, f\right)\right) ^ {2} \tag {6}
$$

Here,  $PL_{d}(d_{i},f)$  is the predicted path loss for direct propagation, and  $\hat{PL}_{d}(d_{i},f)$  is the actual path loss based on the formula:

$$
P L _ {d} (d, f) [ d B ] = F S P L (f, d = 1 m) [ d B ] + 1 0 \log_ {1 0} (d) [ d B ] + A T [ d B ] \tag {7}
$$

where  $f$  denotes the carrier frequency in GHz,  $d$  is the 3D T-R separation distance,  $n$  represents the path loss exponent (PLE), and  $AT$  is the attenuation term induced by the atmosphere (Okoro et al., 2021). -  $\mathcal{L}_{\mathrm{reflection}}$  is the loss due to signal reflections, calculated as:

$$
\mathcal {L} _ {\text {r e f l e c t i o n}} = \sum_ {i = 1} ^ {N} \left(P L _ {r} \left(d _ {i}, f\right) - \hat {P L} _ {r} \left(d _ {i}, f\right)\right) ^ {2} \tag {8}
$$

The reflection loss  $PL_{r}$  can be calculated based on reflection coefficients and the geometry of the environment.

-  $\mathcal{L}_{\mathrm{diffraction}}$  is the loss due to signal diffraction, calculated as:

$$
\mathcal {L} _ {\text {d i f f r a c t i o n}} = \sum_ {i = 1} ^ {N} \left(P L _ {\text {d i f f}} \left(d _ {i}, f\right) - \hat {P} L _ {\text {d i f f}} \left(d _ {i}, f\right)\right) ^ {2} \tag {9}
$$

The diffraction loss  $PL_{diff}$  can be calculated using a modified UTD (diffraction model), which considers the edges and round surfaces of obstacles in the environment (Wang et al., 2024).

By incorporating these physical constraints into the generator's loss function, the GAN is guided to produce outputs that are not only visually convincing to the discriminator but also physically accurate in terms of signal propagation characteristics.

# 3.2.2 DISCRIMINATOR

Our discriminator is also based on CNNs, with the addition of condition layers that incorporate the geometry information. This setup ensures that the discrimination process considers not just the accuracy of the heatmaps but also their consistency with the input geometry. This consistency

refers to a check of the alignment of predicted signal strengths with the expected patterns based on EM propagation theory discussed earlier in the generator, such as maintaining the correct spatial distribution and intensity of signals influenced by environmental factors and material properties. The Discriminator Objective Function is given as:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm {c G A N}} ^ {\mathrm {D}} = - \mathbb {E} _ {E _ {\mathrm {g}}, L _ {\mathrm {t x}}, P _ {\mathrm {r}}} [ \log D (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, P _ {\mathrm {r}}) ] \tag {10} \\ - \mathbb {E} _ {E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z} [ \log (1 - D (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, G (E _ {\mathrm {g}}, L _ {\mathrm {t x}}, z))) ]. \\ \end{array}
$$

This function models the discriminator's objective, which seeks to identify real and generated heatmaps correctly, thus ensuring that the generated data is accurate

# 3.3 TRAINING

Training of the modified cGAN is performed using a loss function that balances the fidelity of the generated heatmaps as a function of the input geometric conditions. The training process is carefully monitored to prevent mode collapse and ensure a diverse set of realistic outputs. The proposed method is implemented using PyTorch (Paszke et al., 2019) and uses a GPU for efficient model training and inference. For ease of access, we utilize Google Colab, which provides free GPU resources to facilitate the training process. The primary software and dependencies include Python 3 or higher and essential libraries such as NumPy, SciPy, and matplotlib for data handling and visualization. Our training process on Google Colab takes approximately two days to complete. We optimize the training using hyperparameters such as the learning rate, batch size, and latent space dimensions, which are crucial for achieving the desired model performance and accuracy. A detailed flowchart is presented in the appendix and we plan to release our code at the time of publication.

# 4 IMPLEMENTATION AND PERFORMANCE

We discuss the implementation of our approach and the main issues in terms of obtaining good performance.

# 4.1 DATA ADEQUACY AND QUALITY

Given the complexity of indoor wireless systems, the cGAN would require extensive and high-quality training data that accurately represents the vast array of environmental factors affecting signal propagation. In our process, we generate geometry and power prediction data from WinProp and the DCEM simulator to ensure the diversity and volume of training data, representing different scenarios with high quality.

# 4.2 HYPERPARAMETER TUNING

CGANs are notoriously difficult to train and often sensitive to the choice of hyperparameters, which would require extensive experimentation to fine-tune. In our process, we start with simplified versions of the environment to first train the cGAN before gradually increasing the complexity, which helps the model learn the basic principles before tackling more complex scenarios. By gradually increasing the complexity of the models, we also avoid convergence issues, resulting in a stable solution that provides a realistic simulation of EM ray tracing.

Table 1: Hyperparameters used in GAN Training  

<table><tr><td>Hyperparameter</td><td>Value/Type</td></tr><tr><td>Learning Rate</td><td>0.0002</td></tr><tr><td>Batch Size</td><td>128</td></tr><tr><td>Noise Type</td><td>Gaussian</td></tr><tr><td>Loss Function</td><td>Binary Cross Entropy</td></tr></table>

# 4.3 MODE COLLAPSE

A common issue with cGANs occurs when the generator starts producing a limited range of outputs, which in the case of EM ray tracing could lead to underrepresentation of the solution space. In our work, we included a noise vector in the generator's input to promote feature learning and output variability. A large, diverse training dataset exposed the generator to a range of scenarios, reducing mode collapse risk. Our conditional GAN framework improved output relevance, and adaptive learning rates maintained balanced learning dynamics. These strategies enhanced the model's ability to generate diverse and accurate EM propagation heatmaps.

# 5 RESULTS

This section presents the evaluation of the proposed methodology in terms of accuracy enhancement and efficiency improvement in ray tracing simulations in 3D indoor environments. We have conducted a comparison with WinProp (Jakobus et al., 2018), which is widely recognized as a state-of-the-art solution in EM simulation, as shown in these and more papers (Vaganova et al., 2023) (Wang & Manocha, 2023) (Haron et al., 2021) (Gómez et al., 2023).

We show evaluations in 15 indoor scenes: Scenes 1-15. Detailed specifications of scenes are included in Table 2. On average, the running time of EM-GANSim in any indoor environment is 1 millisecond per data point. However, the models with complex layouts tend to require more computation time than those with single rooms.

Table 2: Detailed Specifications for Various Scenes in terms of size, room configurations, and materials. EM-GANSim is able to predict the signal power strength at any given data location in a few milliseconds.  

<table><tr><td>Scene#</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>Type</td><td>Multiple rooms</td><td>Multiple rooms</td><td>Multiple rooms</td><td>Single room</td><td>Complex floor plan</td></tr><tr><td>Size (m2)</td><td>25</td><td>25</td><td>25</td><td>144</td><td>144</td></tr><tr><td>Materials used</td><td>wood, concrete</td><td>wood, concrete, glass</td><td>wood, concrete</td><td>concrete</td><td>wood, concrete, glass</td></tr></table>

<table><tr><td>Scene#</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>Type</td><td>Complex floor plan</td><td>Complex floor plan</td><td>Single room</td><td>Single room</td><td>Single room</td></tr><tr><td>Size (m2)</td><td>144</td><td>16</td><td>16</td><td>16</td><td>144</td></tr><tr><td>Materials used</td><td>wood, concrete, glass</td><td>wood, concrete, glass</td><td>wood, concrete, glass</td><td>concrete, glass</td><td>concrete, glass</td></tr></table>

<table><tr><td>Scene#</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>Type</td><td>Complex floor plan</td><td>Multiple rooms</td><td>Single room</td><td>Single room</td><td>Multiple rooms</td></tr><tr><td>Size (m2)</td><td>144</td><td>144</td><td>4</td><td>16</td><td>64</td></tr><tr><td>Materials used</td><td>wood, concrete, glass</td><td>wood, concrete, glass</td><td>concrete</td><td>concrete, glass</td><td>wood, concrete, glass</td></tr></table>

# 5.1 ACCURACY OF OUR APPROACH

The accuracy of our method is assessed by comparing the simulated received power distributions against standard RT simulations generated using DCEM Wang & Manocha (2022) and WinProp (Jakobus et al., 2018) simulators on our validation datasets in Fig. 2. More heatmap comparisons are shown in the appendix. These heatmaps serve several purposes in supplementing the results: (1) Validation Across Diverse Conditions: These plots demonstrate the model's ability to generalize across different environments by presenting additional scenarios, validating its robustness and adaptability. (2) Comparative Analysis: The plots include comparisons between the EM-GANSim model predictions and those from benchmark WinProp. This comparative analysis highlights the strengths of EM-GANSim in terms of accuracy. (3) Visualization of EM Interactions: The heatmaps visually depict the power distribution and signal propagation across different room layouts. This visualization aids in understanding how well the model captures physical phenomena such as reflection and diffraction.

The comparison underscores the enhanced accuracy achieved by incorporating GAN into the RT simulations, highlighting the advantage of the proposed method in capturing the intricacies of EM wave interactions with indoor structures.

![](images/33dfafae5e90fb7ac05284cbe4074febba0df7498778aeca4ac4cf621a145e25.jpg)  
Figure 2: Comparative heatmaps displaying received powers in indoor environments of size  $5*5$ $m^2$  (left three columns, Scene 1-3) and  $12*12~m^2$  (right three columns, Scene 4-6). First row: WinProp simulation. Second row: GAN-based simulation. Third row: DCEM simulations. The MSEs of GAN-based and DCEM compared to WinProp are shown in Table 4 in the appendix. We see with GAN-based methods that the heatmaps show less MSE in general captures and exhibit more pronounced areas of both high and low signal strength, suggesting a finer granularity in the simulation of received powers.

These are all new scenes not included in the training dataset. The average MSE of GAN-based results of the training set is approximately  $3dbm^2$  and that of the testing set is around  $8.5dbm^2$ .

In Fig. 3, we show a histogram distribution comparison of the normalized difference in the scenes in the third row of Fig. 2 (Scene 3).

![](images/782b82c8e0b9c259391391f770884d6d8002d51b1276a21ee9788810e93ad976.jpg)  
Figure 3: Left Histogram: Distribution of the normalized differences in received power levels between the GAN-based simulation and the WinProp simulation for the third-row scene. The vertical lines represent the 50th (median), 70th, and 90th percentiles, indicating a central tendency and spread of the differences. Right Histogram: Distribution of the normalized differences in received power levels between the DCEM simulation and WinProp simulation for the third-row scene. The percentiles are marked similarly. We see a tighter distribution in the left graph, suggesting a closer match to WinProp and higher accuracy in the GAN-based simulations.

# 5.2 EFFICIENCY IMPROVEMENT THROUGH GAN

To evaluate the efficiency of using GAN for quick simulations, the computation time was measured and compared between the GAN-based method and the traditional RT approach. The third column in Table 3 is the average generation time of data points in GAN-based predictions, calculated from total time divided by the total number of simulated points. For instance, in a single room of  $2m*2m$ , with a resolution of  $0.05m$ , there are 1600 generated data points. Thus, each data point is generated in approximately 2 milliseconds.

Table 3: Computation Time Comparison Between GAN-based and Traditional RT Approaches  

<table><tr><td></td><td>GAN-based (seconds)</td><td>Traditional RT (seconds)</td><td>Generation time per data point (seconds)</td></tr><tr><td>Single room (~2*2 m2)</td><td>3.2</td><td>12</td><td>0.002</td></tr><tr><td>Multiple rooms (~8*8 m2)</td><td>3</td><td>20</td><td>0.001</td></tr><tr><td>Complex floor plan (~12*12 m2)</td><td>4.3</td><td>22</td><td>0.0009</td></tr></table>

The GAN-based method demonstrated a substantial reduction in computation time, offering near-instantaneous simulation results. This efficiency makes the GAN-based approach particularly suitable for applications requiring real-time data analysis and decision-making.

Based on the comparisons after GAN training, we highlight the benefits of GAN below:

- High-Quality Synthetic Data Generation: cGANs are adept at generating synthetic data that closely mirrors the distribution of real data, an essential capability for accurately predicting heatmaps from limited real-world data.  
- Efficiency in Prediction: The GAN-based method can predict heat maps for an entire target area in a single inference step, offering a significant efficiency advantage over traditional, computation-intensive methods.  
- Accuracy Close to Ray Tracing Simulations: cGANs have the potential to achieve accuracy levels comparable to those of traditional ray tracing simulations by learning to capture the complex variability of path loss across different environments.

# 6 ABLATION EXPERIMENTS

In this section, we first analyze the effect of excluding Gaussian noise from the training process, an element typically introduced to break the symmetry in the model weights, ensuring that different units learn different features. We verify the noise's impact on the generated heatmaps and their respective MSEs. By comparing heatmaps and MSE values, we evaluate the GAN's performance in generating received power distribution in the absence of noise. This ablation study serves not only to reinforce the validity of our methodology but also to offer insights that could refine future implementations of machine learning in EM ray tracing. We define the ablation experiment results as GAN-No-Noise. Observations based on the heatmaps in the appendix, Fig. 7, from the GAN-No-Noise, GAN-based, and DCEM predictions in testing scenes are discussed as follows:

- First Row (GAN-No-Noise): The absence of Gaussian noise results in less varied and uniform heatmaps, indicating potential over-smoothing and reduced accuracy in capturing EM wave interactions within the environment.  
- Second Row (GAN-based with Noise): Inclusion of noise introduces more defined contrasts and a broader range of power levels, suggesting a better representation of the complex nature of EM propagation and environmental features.  
- Cons of GAN-No-Noise: Lack of noise in training leads to simpler patterns, reduced model accuracy, and potential issues in generalizing to new environments, which is critical for applications like network planning.  
- Importance of Noise: Gaussian noise is essential in training to break symmetry in the model, ensuring diverse learning and preventing the network from collapsing into repetitive pattern production.

These observations underline the importance of including noise in the GAN training process to enhance the model's ability to predict received power distributions accurately and robustly, especially when applied to complex indoor EM propagation scenarios.

We also include the corresponding MSE of GAN-No-Noise and GAN-based compared to DCEM in Table 5 in the appendix. For these three testing cases, GAN-based predictions consistently have lower MSE values than GAN-No-Noise, indicating that the inclusion of noise during the training process contributes to a more accurate prediction of received power levels. The improved MSE with noise suggests that Gaussian noise acts as a regularizer, preventing the model from memorizing the training data and instead forcing it to learn the underlying distribution. The presence of noise

also introduces a wider variety of scenarios during training, making the GAN model more robust to unseen environments and better at generalizing from the training data.

Another ablation test is designed to evaluate the impact of incorporating physical constraints into the objective function of our generator, included in the appendix. These physical constraints are integrated to ensure that the generated samples adhere to the fundamental principles of electromagnetic wave propagation, accounting for direct propagation, reflection, and diffraction effects. The ablation tests will involve running the simulator under two distinct conditions:

- With Physics Constraints: The generator's objective function will include the physical constraints loss  $(\mathcal{L}_{\mathrm{phy}})$ , which comprises terms for direct path propagation  $(\mathcal{L}_{\mathrm{direct}})$ , reflections  $(\mathcal{L}_{\mathrm{ref}})$ , and diffractions  $(\mathcal{L}_{\mathrm{diff}})$ .  
- Without Physics Constraints: The physical constraints loss  $(\mathcal{L}_{\mathrm{phy}})$  will be omitted from the objective function, leaving only the cGAN loss and the MSE loss components.

This ablation test demonstrates the impact of incorporating physical constraints by comparing the performance and accuracy of the generator under both conditions. Key performance metrics observed were:

- Signal Propagation Accuracy: The tests revealed that the generator with physical constraints produced more accurate signal propagation characteristics. The predicted path losses (direct, reflection, and diffraction) closely matched the actual path losses, highlighting the effectiveness of the constraints in capturing the physical phenomena of EM wave propagation in indoor environments.  
- Visual and Structural Fidelity: The generated samples with physical constraints exhibited higher visual realism and structural coherence. These samples were more accurate in modeling the indoor environments compared to those generated without the constraints.

# 7 CONCLUSIONS, LIMITATIONS, AND FUTURE WORK

We present a novel approach that uses ML methods along with EM ray tracing to enhance the accuracy and efficiency of wireless communication simulation within 3D indoor environments. We use a modified cGAN that utilizes encoded geometry and transmitter location and can be used for accurate EM wave propagation. We have evaluated its performance on a large number of complex 3D indoor scenes and its performance is comparable to EM ray tracing-based simulations. Furthermore, we observe a 5X performance improvement over prior methods.

Our study enhances wireless communication efficiency and lays the ground for future real-time applications. Our approach has some limitations. Since our training data is based on ray tracing, our prediction scheme may not be able to accurately model low-frequency or other wave interactions. Our current approach is limited to indoor scenes, and we would also like to evaluate it in scenes with multiple dynamic objects. A key challenge is to extend and use these methods for large urban scenes with complex traffic patterns to model wireless signals.

Furthermore, we plan to add dynamic elements, such as movable partitions and furniture, to simulate real-world changes in indoor layouts. By leveraging publicly available architectural data (such as the 3D-Front dataset (Fu et al., 2021)), we will continuously update the dataset with new scenarios that reflect emerging trends in building design and technology. This comprehensive dataset expansion will improve the model's ability to predict EM wave propagation in complex and varied indoor environments, ultimately enhancing its applicability and reliability in practical applications.

# REFERENCES

Stefanos Bakirtzis, Kehai Qiu, Jie Zhang, and Ian Wassell. Deeppray: Deep learning meets raytracing. In 2022 16th European Conference on Antennas and Propagation (EuCAP), pp. 1-5. IEEE, 2022.  
Henry L Bertoni, Walter Honcharenko, Leandro Rocha Maciel, and Howard H Xia. Uhf propagation prediction for wireless personal communications. Proceedings of the IEEE, 82(9):1333-1359, 1994.

Ronald Coifman, Vladimir Rokhlin, and Stephen Wandzura. The fast multipole method for the wave equation: A pedestrian prescription. IEEE Antennas and Propagation magazine, 35(3): 7-12, 1993.  
Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, and Anil A Bharath. Generative adversarial networks: An overview. IEEE signal processing magazine, 35 (1):53-65, 2018.  
Giuseppe Maria D'Aucelli, Nicola Giaquinto, and Gregorio Andria. Linelab-a transmission line simulator for distributed sensing systems: Open-source matlab code for simulating real-world transmission lines. IEEE Antennas and Propagation Magazine, 60(4):22-30, 2018.  
Chunlei Dong, Lixin Guo, and Xiao Meng. Application of CUDA-accelerated go/por method in calculation of electromagnetic scattering from coated targets. IEEE Access, 8:35420-35428, 2020.  
Esteban Egea-Lopez, Jose Maria Molina-Garcia-Pardo, Martine Lienard, and Pierre Degauque. Opal: An open source ray-tracing propagation simulator for electromagnetic characterization. Plos one, 16(11):e0260060, 2021.  
C Filosa, JHM ten Thije Boonkkamp, and WL IJzerman. Ray tracing method in phase space for two-dimensional optical systems. Applied optics, 55(13):3599-3606, 2016.  
Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10933-10942, 2021.  
Josefa Gomez, Carlos J Hellin, Adrián Valledor, Marcos Barranquero, Juan J Cuadrado-Gallego, and Abdelhamid Tayebi. Design and implementation of an innovative high-performance radio propagation simulation tool. IEEE Access, 2023.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.  
Ain Shazleen Haron, Zuhanis Mansor, Izanoordina Ahmad, and Siti Marwangi Mohamad Maharum. The performance of 2.4 ghz and 5ghz wi-fi router placement for signal strength optimization using altair winprop. In 2021 IEEE 7th International Conference on Smart Instrumentation, Measurement and Applications (ICSIMA), pp. 25-29. IEEE, 2021.  
An-Yao Hsiao, Chang-Fa Yang, Te-Shun Wang, Ike Lin, and Wen-Jiao Liao. Ray tracing simulations for millimeter wave propagation in 5g wireless communications. In 2017 IEEE International Symposium on Antennas and Propagation & USNC/URSI National Radio Science Meeting, pp. 1901-1902. IEEE, 2017.  
Wei Huang, Mingliu Liu, Deshi Li, Feng Yin, Haole Chen, Jixuan Zhou, and Huihui Xu. Collaborating ray tracing and ai model for auv-assisted 3-d underwater sound-speed inversion. IEEE Journal of Oceanic Engineering, 46(4):1372-1390, 2021.  
Yourui Huangfu, Jian Wang, Shengchen Dai, Rong Li, Jun Wang, Chongwen Huang, and Zhaoyang Zhang. Wair-d: Wireless ai research dataset. arXiv preprint arXiv:2212.02159, 2022.  
Ulrich Jakobus, Andres G Aguilar, Gerd Woelfle, Johann Van Tonder, Marianne Bingle, Kitty Longtin, and Martin Vogel. Recent advances of feko and winprop. In 2018 IEEE International Symposium on Antennas and Propagation & USNC/URSI National Radio Science Meeting, pp. 409-410. IEEE, 2018.  
Athanasios G Kanatas, Ioannis D Kountouris, George B Kostaras, and Philip Constantinou. A utd propagation model in urban microcellular environments. IEEE Transactions on Vehicular Technology, 46(1):185-193, 1997.  
Salome Kazeminia, Christoph Baur, Arjan Kuijper, Bram van Ginneken, Nassir Navab, Shadi Albarqouni, and Anirban Mukhopadhyay. Gans for medical image analysis. Artificial Intelligence in Medicine, 109:101938, 2020.

Ahmed Marey, Mustafa Bal, Hasan F Ates, and Bahadir K Gunturk. Pl-gan: Path loss prediction using generative adversarial networks. IEEE Access, 10:90474-90480, 2022.  
James Clerk Maxwell. A treatise on electricity and magnetism, volume 1. Oxford: Clarendon Press, 1873.  
John W McKown and R Lee Hamilton. Ray tracing as a design tool for radio networks. IEEE Network, 5(6):27-30, 1991.  
MS Obaidat and DB Green. Simulation of wireless networks. Applied system simulation: methodologies and applications, pp. 115-153, 2003.  
Chukwuemeka Okoro, Charles R Cunningham, Aaron R Baillargeon, Andreas Wartak, and Guillermo J Tearney. Modeling, optimization, and validation of an extended-depth-of-field optical coherence tomography probe based on a mirror tunnel. Applied Optics, 60(8):2393-2399, 2021.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.  
Theodore S Rappaport, Shu Sun, Rimma Mayzus, Hang Zhao, Yaniv Azar, Kevin Wang, George N Wong, Jocelyn K Schulz, Mathew Samimi, and Felix Gutierrez. Millimeter wave mobile communications for 5g cellular: It will work! IEEE access, 1:335-349, 2013.  
Anton Ratnarajah, Zhenyu Tang, Rohith Aralikatti, and Dinesh Manocha. Mesh2ir: Neural acoustic impulse response generator for complex 3d scenes. In Proceedings of the 30th ACM International Conference on Multimedia, pp. 924-933, 2022.  
Wanchai Sangkusolwong and Anya Apavatjrut. Indoor wifi signal prediction using modelized heatmap generator tool. In 2017 21st International Computer Science and Engineering Conference (ICSEC), pp. 1-5. IEEE, 2017.  
Scott Y Seidel and Theodore S Rappaport. 914 mhz path loss prediction models for indoor wireless communications in multifloored buildings. IEEE transactions on Antennas and Propagation, 40 (2):207-217, 1992.  
Allen Taflove, Susan C Hagness, and Melinda Piket-May. Computational electromagnetics: the finite-difference time-domain method. The Electrical Engineering Handbook, 3(629-670):15, 2005.  
Anastasia A Vaganova, Andrey I Panychev, and Natalia N Kisel. Developing an automated design system for radio frequency planning of wireless communication coverage in matcad environment. In 2023 Radiation and Scattering of Electromagnetic Waves (RSEMw), pp. 472-475. IEEE, 2023.  
Cheng-Xiang Wang, Jie Huang, Haiming Wang, Xiqi Gao, Xiaohu You, and Yang Hao. 6g wireless channel measurements and models: Trends and challenges. IEEE Vehicular Technology Magazine, 15(4):22-32, 2020.  
Ruichen Wang and Dinesh Manocha. Dynamic coherence-based em ray tracing simulations in vehicular environments. In 2022 IEEE 95th Vehicular Technology Conference: (VTC2022-Spring), pp. 1-7. IEEE, 2022.  
Ruichen Wang and Dinesh Manocha. Dynamic em ray tracing for large urban scenes with multiple receivers. In 2023 International Wireless Communications and Mobile Computing (IWCMC), pp. 1268-1274. IEEE, 2023.  
Ruichen Wang, Samuel Audia, and Dinesh Manocha. Indoor wireless signal modeling with smooth surface diffraction effects. In 2024 18th European Conference on Antennas and Propagation (EuCAP), pp. 1-5. IEEE, 2024.

Zhangyu Wang, Serkan Isci, Yaron Kanza, Velin Kounev, and Yusef Shaqalle. Cellular network optimization by deep reinforcement learning and ai-enhanced ray tracing. In Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Spatial Big Data and AI for Industrial Applications, pp. 41-50, 2023.  
Kathryn Williams, Luis Tirado, Zhongliang Chen, Borja Gonzalez-Valdes, Jose Angel Martinez, and Carey M Rappaport. Ray tracing for simulation of millimeter-wave whole body imaging systems. IEEE Transactions on Antennas and Propagation, 63(12):5913-5918, 2015.  
Mingsheng Yin, Akshaj Kumar Veldanda, Amee Trivedi, Jeff Zhang, Kai Pfeiffer, Yaqi Hu, Siddharth Garg, Elza Erkip, Ludovic Righetti, and Sundeep Rangan. Millimeter wave wireless assisted robot navigation with link state classification. IEEE Open Journal of the Communications Society, 3:493-507, 2022.  
Zhengqing Yun and Magdy F Iskander. Ray tracing for radio propagation modeling: Principles and applications. IEEE access, 3:1089-1100, 2015.  
Jianzhong Zhang, Guohui Yang, and Feng Li. Ray tracing method for ground penetrating radar waves. In 2006 7th International Symposium on Antennas, Propagation & EM Theory, pp. 1-4. IEEE, 2006.  
Fusheng Zhu, Weiwen Cai, Zhigang Wang, and Fang Li. Ai-empowered propagation prediction and optimization for reconfigurable wireless networks. Wireless Communications and Mobile Computing, 2022:1-10, 2022.
