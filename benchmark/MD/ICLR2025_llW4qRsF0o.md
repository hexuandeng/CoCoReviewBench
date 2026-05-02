# PHYSICS-TRANSFER LEARNING: A FRAMEWORK TO ADDRESS THE ACCURACY-PERFORMANCE DILEMMA IN MODELING COMPLEXITY PROBLEMS IN ENGINEERING SCIENCES

Anonymous authors

Paper under double-blind review

# ABSTRACT

The development of theoretical sciences traditionally adheres to an observation-assumption-model paradigm, which is effective in simple systems but challenged by the 'curse of complexity' in modern engineering sciences. Advancements in artificial intelligence (AI) and machine learning (ML) offer a data-driven alternative, capable of interpolating and extrapolating scientific inference where direct solutions are intractable. Moreover, feature engineering in ML resembles dimensional analysis in classical physics, suggesting that data-driven ML methods could potentially extract new physics behind complex data. Here we propose a physics-transfer (PT) learning framework to learn physics across digital models of varying fidelities and complexities, which addresses the accuracy-performance dilemma in understanding representative multiscale problems. The capability of our approach is showcased through screening metallic alloys by their strengths and predicting the morphological development of brains. The physics of crystal plasticity is learned from low-fidelity molecular dynamics simulation and the model is then fed by material parameters from high-fidelity, electronic structures level, density functional theory calculations, offering chemically accurate strength predictions with several orders lower computational costs. The physics of bifurcation in the evolution of brain morphologies is learned from simple sphere and ellipsoid models and then applied to predict the morphological development of human brains, showing excellent agreement with longitudinal magnetic resonance imaging (MRI) data. The learned latent variables are shown to be highly relevant to uncovered physical descriptors, explaining the effectiveness of the PT framework, which holds great potential in closing the gaps in understanding complexity problems in engineering sciences.

# 1 INTRODUCTION

The development of theoretical frameworks in engineering sciences has traditionally adhered to an observation-assumption-model paradigm, exemplified by Galileo's studies on beam bending to the formulation of dislocation theory in the mechanical behaviors of materials. This method is particularly effective in problems with a low-dimensional parameter space, where the complexity can often be captured by analytical models. However, as we expand into the multiscale understanding of matter, the 'curse of complexity' emerges, making it increasingly challenging to capture the intricate physics with purely analytical methods (Fish et al., 2021). For instance, material strength is governed by phenomena across multiple length and time scales. Even for single crystals, dislocations can be nucleated under mechanical loads, evolving cooperatively into complex networks (Oh et al., 2009). Brain development involves gene expression, cellular behaviors, and mechanical instabilities across various spatiotemporal scales, as reflected in the evolving morphologies (Llinares-Benadero & Borrell, 2019a). First-principles theories offer high accuracy but are challenging to scale. Empirical models, while highly efficient, are constrained by the limitations of their assumptions and uncertainties in parameterization. This is the accuracy-performance dilemma in modeling complexities of multiscale physics in engineering sciences.

Recent advancements in machine learning (ML) and artificial intelligence (AI) present a promising, data-driven alternative (Fig. 1a). This emerging approach, while still constrained by the density and coverage of data, offers an increasingly powerful tool as data quality and quantity improve. The ability of ML models to interpolate and extrapolate improves accordingly, suggesting that these tools can complement traditional theories where direct solutions become impractical or intractable (Zhang et al., 2018; Li et al., 2022). Moreover, the process of feature engineering in ML bears a resemblance to dimensional analysis in classical physics, offering a systematic way to uncover and utilize internal correlations within complex data (Xu et al., 2022b). This parallel suggests that ML, through its data-driven methods, could potentially extract and transfer physical insights across digital models of varying fidelity and complexity.

Inspired by these thoughts, we propose a physics-transfer (PT) framework to learn physics across digital models of varying fidelities and complexities (Fig. 1b). The learned physics is used for scientific inference with high accuracy and performance to address the dilemma in modeling the complexity. Two representative cases are chosen to demonstrate the capabilities of PT framework including materials strength screening and predicting the development of brain morphologies, which encompass inorganic matter and organs and involve multiscale physics (Fig. 1c). The physics of crystal plasticity is initially learned through low-fidelity molecular dynamics simulations, and these insights are subsequently utilized in high-fidelity density functional theory computations of material parameters, enabling chemically accurate strength predictions. The physics of bifurcation in brain morphologies is initially learned using spherical models with simple geometries and then applied to predict the evolutionary behaviors of human brains. The proposed framework holds great potential for enhancing our comprehension of complexity problems in engineering sciences and bridging gaps between understandings from modeling and experimental data.

![](images/1a4c743e25ad4fbfbea3bb79ee19c802359d54b2e7e8b153032a46d8b7b9e1cf.jpg)

![](images/050346c7ba70d912e61cdb2a6cbb44a329f8629efc41efc9845e5a983f77de49.jpg)

![](images/7702c3af0a30770eada294499152c0517038c9219a7849dfe64436eae3fcadf7.jpg)

![](images/6adc9d07e378906cdaaae039e7a60c51ef899a6f130e9d25a434b398671cef1e.jpg)  
brains

![](images/80155b273ea9c942ec7f72569fe7edc0ed1ea06f6b27b6a56fa0ca322323005b.jpg)  
Figure 1: Accuracy-performance dilemma in modeling multiscale physics and proposed physics-transfer (PT) learning framework. (a) Machine learning, constrained by data density and coverage, serves as a potent complement to traditional theories for interpolating and extrapolating solutions, especially as data quality and quantity increase. (b) The PT learning framework learns physics across digital models of varying fidelity and complexity, enabling extrapolation to effectively address the accuracy-performance dilemma. (c) The 'curse of complexity' in multiscale physics of inorganic matter and organs.  
ellipsoids

![](images/c114dc2abbe4f565b6c8d73eb639534fcc7fe0c511e635e2013aa8754b5ea0c6.jpg)  
spheres

# 2 PHYSICS-TRANSFER LEARNING FRAMEWORK

Models with different fidelity  $(\mathcal{F})$  in multi-scale modeling exhibit distinct parameters distributions  $p(\theta|\mathcal{F})$ , where  $\theta$  are model parameters, and  $p(\cdot|\cdot)$  is conditional probability. ML and AI models

can provide general model with parameters distributions  $p(\theta | \mathcal{D})$  based on data  $(\mathcal{D})$ . Typically, data with different fidelity (e.g., low fidelity  $(\mathcal{D}_{\mathrm{LF}})$  in molecular dynamics (MD), high fidelity  $(\mathcal{D}_{\mathrm{HF}})$  in density functional theory (DFT)) will result in ML models having different parameter distributions, that is:

$$
p \left(\theta \mid \mathcal {D} _ {\mathrm {L F}}\right) \neq p \left(\theta \mid \mathcal {D} _ {\mathrm {H F}}\right), \tag {1}
$$

which limits the transferability and extrapolation of models trained on data with different fidelity.

The physics  $(\mathcal{P})$  behind the  $\mathcal{D}$  can assist the extrapolation with a 'physics-transfer' paradigm. Specifically, if there is a physical relationship between features  $(\mathbf{x})$  and target  $(\mathcal{O})$  in  $\mathcal{D}$ , that is:

$$
\mathbf {x} \xrightarrow {\mathcal {P}} \mathcal {O}, \tag {2}
$$

$$
\mathbf {x} \cap \mathcal {O} = \mathcal {D} ^ {\prime} \subset \mathcal {D}, \tag {3}
$$

the designed ML models  $(h\in \mathcal{H})$  can be trained on the data, and models trained on data with different fidelity would have similar parameter distributions, that is:

$$
p \left(\theta \mid \mathcal {D} _ {\mathrm {L F}} ^ {\prime}\right) \approx p \left(\theta \mid \mathcal {D} _ {\mathrm {H F}} ^ {\prime}\right), \tag {4}
$$

which makes the transferability and extrapolation of models with different fidelity possible and bridge gaps between different modeling methods in multi-scale modeling. To validate the effectiveness of the PT learning framework, in the next sections, we train models on low-fidelity or simple-geometry data and then perform zero-shot extrapolation directly to high-fidelity and high-complexity data. We assess the accuracy of the prediction results and demonstrate its role in addressing the accuracy-performance dilemma.

# 3 EXPERIMENTS

To demonstrate the effectiveness of the PT learning framework, we perform experiments on the problems of materials strength screening and predicting the development of brain morphologies. These two issues encompass the multiscale complexity of inorganic matter and organs suffering from the accuracy-performance.

# 3.1 PHYSICS-TRANSFER LEARNING FOR MATERIALS STRENGTH SCREENING

The strength of materials, like many problems in the natural sciences, spans multiple length and time scales, and the solution has to balance accuracy and performance. In the crystal plasticity (CP) theory, plastic flow and hardening behaviors during material deformation are modeled in a multiscale framework bridging the atomic-scale lattice dynamics and continuum-level stress/strain fields (Roters et al., 2011). One of the key material parameters in CP models is the critical resolved shear stress (CRSS),  $\tau_{\mathrm{c}}$ , which determines the activation of specific slip systems. In CP models, CRSS is a phenomenological parameter often obtained by fitting experimental results (Salem et al., 2005; Gong et al., 2015). Alternatively, the Peierls stress  $(\tau_{\mathrm{P}})$  defined as the minimum shear stress required to move a single dislocation of unit length in a perfect crystal in the absence of thermal activation is also used in the literature for CRSS (Shimanek et al., 2022). The Peierls stress can be obtained from full-atom MD simulations. However, the strain inhomogeneity induced by a dislocation usually spans  $10 - 20\mathrm{nm}$ , which cannot be directly calculated from first-principles calculations. Previous studies are thus limited by the use of empirical force fields (Soleymani et al., 2014). In practice, the Peierls-Nabarro (PN) model offers a simplified and approximating approach to derive the Peierls stress with the assumptions of sinusoidal interfacial restoring stress and a rigidly shifting dislocation, where the structure of a dislocation core is determined by minimizing the elastic energies and lattice misfit (Nabarro, 1947). The success of the PN model suggests that the Peierls stress is controlled by the elastic responses of the crystals and the energy landscape of interfacial slips (Bulatov & Kaxiras, 1997; Nabarro, 1997; Lu et al., 2000; Rodney et al., 2017).

By assuming the existence of such correspondence, we use the PT learning framework to predict the Peierls stress for a wide spectrum of metallic alloys and inorganic crystals at the first-principles level (Fig. 2a). The maps between the Peierls stress  $(\mathcal{O})$  and characteristic materials parameters  $(\mathbf{x})$  are trained from empirical or machine-learning force-field (MLFFs) MD simulations with a designed neural network  $(h \in \mathcal{H})$  and obtain the posterior probability of the model parameters  $(p(\theta | \mathcal{D}_{\mathrm{LF}}'))$ .

Then the well-trained models are extrapolated to DFT-calculated parameters to make predictions at the chemically accurate level. This mapping transfers the physics from low-fidelity but efficient force field models to the first-principles methods, successfully resolving the accuracy-performance dilemma.

![](images/eb04dd8f1a06875e140c6febef447c9ffcc1800cf872b71afb9269df33ed27a3.jpg)  
a

![](images/cbe7d8a0f597f48f321c9e3139ed82a15a6f9deebdf07ace397bcab1b53ac44a.jpg)  
b

![](images/ec4a8ca98d176cb1c300baa24e6e5e15d9216a1e7e145dca237a4776b055d605.jpg)  
C

![](images/b47ca219c9248cd93a44d34b1552c2f8a3f14f9c7d3fc80e8c480a8226d31c0f.jpg)  
d

![](images/c62c9b7379749f2897b7cc43aebac1e3290f1528049efe1997899b029bf6680b.jpg)  
e

![](images/5f9917fc586f003b50c9960371cb96485c69760b0e30b6d36c9bdbe8ab81cdc0.jpg)

![](images/0e1f1626faf2383b1e1e6648353e19b1bd239e257716b854bf99aa1dae164913.jpg)

![](images/a5639477e1fc72a2f8dc14c1affddc5448b29224000a182273decab7ec1bcb05.jpg)  
f

![](images/046319ab0af611752b43ff22814ba2e2b88e573df38b490a2ffff02174a7ed33.jpg)  
9

![](images/8206de0a990913c89574fd76a9c8cdf375a950b7606b07b9e7023ab4175f32d7.jpg)  
h

![](images/c36ded1dc07461077d690d59a0b8a3284b0630940debb6c1b397adc2b7ab27c9.jpg)  
i  
j

![](images/c94675a5ff7e54a2b88140cf5ffc35a83599cbd4af849aea53a55fae859f2091.jpg)  
Figure 2: PT predictions for materials strength and uncertainty quantification. (a) PT framework transfers the physics from low-fidelity force field models to chemically accurate first-principles methods, effectively addressing the trade-off between accuracy and computational expense. (b) Well-trained neural networks learn the physical mapping between the Peierls stress and characteristic materials parameters obtained from atomistic simulation datasets using empirical force fields. (c, d) PT framework predicts the Peierls stress with high accuracy and efficiency. The PT predictions are closely aligned with the outcomes of density functional theory (DFT) and machine-learning force-field (MLFF) calculations, with a difference below  $48.91\%$ , while the results obtained using embedded atom methods (EAM) models deviate substantially from the DFT predictions, with a discrepancy of  $221.27\%$  (c). The PT approach also reduces the computational time notably by statistical inference, in comparison with atomistic simulations using DFT, MLFFs, or EAM (d). (e-h) PT predictions for different slip systems (e). The PT predictions show good consistency compared to MLFF simulation results (with errors  $e = 12.55\%$ ,  $48.09\%$ ,  $4.30\%$  for  $\mathrm{Cu}\{111\} \langle \overline{1} 10\rangle$ , Fe  $\{110\} \langle 111\rangle$ , Ti  $\{10\overline{1} 0\} \langle 11\overline{2} 0\rangle$  in prediction, respectively), and superior accuracy compared to EAM ( $e = 33.07\%$ ,  $72.02\%$ ,  $13.89\%$  ((f),(g),(h)), respectively). (i) Uncertainty quantification shows that the PT predictions eliminate physical and system uncertainties. 'L' denotes the large-supercell system with  $\sim 0.8\times 10^{6}$  atoms ( $160~\mathrm{nm}\times 2~\mathrm{nm}\times 40~\mathrm{nm}$ ). (j) Uncertainty decomposition shows that the inference errors are smaller compared to the physical and system uncertainties. The standard deviation is reported in the error bars.

# 3.1.1 DATASETS

To construct the digital libraries, a wide spectrum of metals with crystalline structures of fcc (Cu, Ni, Al, Au, Pd, Pt), bcc (Fe, Mo, Ta, W), and hcp (Ti, Mg, Zr, Co) is explored. The elastic constants,  $\gamma$  surfaces, and the Peierls stress are calculated using empirical force fields such as EAM and modified EAM (MEAM) with parameters reported from different sources (Becker et al., 2013; Hale et al., 2018), as well as the lattice mismatch energy and slip resistance. The primary slip systems of fcc ( $\{111\}\langle \overline{1}10\rangle$ ) and bcc ( $\{110\}\langle 111\rangle$ ), and the prismatic slip systems of hcp ( $\{10\overline{1}0\}\langle 11\overline{2}0\rangle$ ) are considered. Finally, the digital libraries composed of characteristic materials parameters and the Peierls stress are established to learn the physics of crystal plasticity.

# 3.1.2 ARCHITECTURE AND MODEL SETUP

To effectively learn the physical mapping between elastic constants,  $\gamma$  surfaces, and the Peierls stress, we employ a convolutional neural network (CNN) to extract features from the  $\gamma$  surfaces (He et al., 2016), an feedforward neural network (FNN) to extract the features related to elastic properties and merge them in the latent features space. Then, we use another FNN to predict the Peierls stress. The FNN for extracting elastic properties contains two layers with neuron numbers of 6 (number of elastic features), and 32 (dimension of extracted features), respectively. Following that, the FNN for predicting the Peierls stress has 3 layers with neuron numbers of 64, 32, and 1, respectively. We use the stochastic gradient descent (SGD) optimizer with learning rates of  $10^{-4}$  (Hardt et al., 2016).

# 3.1.3 EVALUATION METRICS

Recent progress in computational hardware and software promotes the development of MLFFs, which harness neural networks to model the potential energy surfaces (PES) with the precision of the training set, mostly from first-principles calculations (Ko & Ong, 2023; Hedman et al., 2023; Gong et al., 2023). MLFFs learn the dependence of the potential energy of a system on the atomic positions. The size effects in direct DFT calculations can be mitigated if this mapping accommodates all atomic environments encountered in the MLFF simulations, and the locality holds well (Zhang et al., 2018). The Peierls stress predicted by the MLFFs thus serves as a benchmark to validate the PT predictions. However, the accuracy of the state-of-the-art MLFF predictions for non-equilibrium structures such as those containing dislocations is usually limited in comparison with the equilibrium features (Takamoto et al., 2022), and the MD simulations to predict the Peierls stress using MLFFs still need careful design of the models and simulation parameters to mitigate the effects of sample sizes, loading geometries, and kinetics (Morrow et al., 2023). In addition, MLFFs have higher computational costs than common empirical force fields such as EAM and MEAM. A direct mapping between the DFT-derived  $\gamma$  surface and the Peierls stress can thus have an advantage in facilitating fast material screening, especially for vast material space.

# 3.1.4 RESULTS

To assess the accuracy of the PT predictions, we first calculate the Peierls stress directly by utilizing different methods of calculations, including EAM, MLFFs, DFT, PT models trained on EAM data (PT-EAM), and PT models trained on MLFFs data (PT-MLFFs) for small systems (annotated as 'S', containing 244 atoms in a  $3.48\mathrm{nm}\times 0.41\mathrm{nm}\times 1.90\mathrm{nm}$  supercell) of the fcc system that suffer from strong size effects in predicting the plasticity of bulk materials. The results in Fig. 2b indicate that the well-trained neural networks effectively learn the physical mapping between the Peierls stress and the characteristic elastic and surface parameters. The PT-EAM predictions are quantitatively close  $(< 48.91\%)$  to those from DFT and MLFF calculations. In comparison, those obtained from EAM models show a significant deviation by  $221.27\%$  from the DFT predictions (Fig. 2c for Cu  $\{111\} \langle 110\rangle$ ). The time cost of statistical inference in the PT approach is within several milliseconds on an Intel(R) Core(TM) i5-8250U computer with 4 cores), which is significantly lower than that of simulations based on DFT, MLFFs, and EAM (Fig. 2d). These results obtained for small systems successfully demonstrate advantages in the accuracy and efficiency of the PT approach to predict the Peierls stress.

We then consider large models ( $\mathrm{L}$ ,  $\sim 0.8$  million atoms in a  $160 \, \mathrm{nm} \times 2 \, \mathrm{nm} \times 40 \, \mathrm{nm}$  supercell) for 3 crystalline structures (fcc, bcc, hcp) with their associated specific slip systems (Fig. 2e), where direct DFT calculations are intractable. MD simulations using MLFFs are performed to validate

the accuracy of PT predictions from EAM and MEAM models. The results show good consistency (with errors  $e = 12.55\%$ ,  $48.09\%$ ,  $4.30\%$  for Cu  $\{111\} \langle \overline{1} 10\rangle$ , Fe  $\{110\} \langle 111\rangle$ , Ti  $\{10\overline{1} 0\} \langle 11\overline{2} 0\rangle$  in prediction, respectively) and superior performance compared to the EAM results with  $e = 33.07\%$ ,  $72.02\%$ ,  $13.89\%$  (Figs. 2f-h). By comparing the results obtained for the small and large systems, we also noted that the size effects are more significant for the empirical force fields. The PT framework thus demonstrates high efficiency compared to DFT and MLFF calculations that can mitigate the size effects, and chemically accurate predictions compared to empirical force fields such as EAM.

For the Peierls stress predictions, uncertainties exist among different theoretical approaches. Uncertainty quantification (UQ) of these methods is of crucial importance in evaluating and selecting the models. Fig. 2i shows the error maps for various calculation methods. The predictions of small systems with EAM (EAM-S) contain physical uncertainties on the potentials and system uncertainties in size effect. The calculations of small systems with DFT (DFT-S) eliminate physical uncertainties but still suffer from system uncertainties. The predictions of large systems using EAM (EAM-L) with weaker size effects reduce system uncertainties but retain the physical uncertainties. Both PT-EAM predictions and MLFFs calculations eliminate physical and system uncertainties, but PT predictions are superior in computational efficiency, in both the training and inference processes. The uncertainties of different approaches are quantitatively decomposed in Fig. 2j. The uncertainty of prediction using EAM-S contains physical, system errors (99.05% in total) and the inference error (0.95%) by considering the MLFF results as the ground truth. For EAM-L, their contributions are 62.85% and 37.15%, respectively. The PT-EAM prediction only involves uncertainty of inference ( $e = 12.55\%$ ). The low uncertainty of inference compared to the physical and system errors demonstrates the power of the PT framework and can be estimated from the theory of machine learning (Abu-Mostafa et al., 2012; Feng et al., 2023).

![](images/81915dbfda5301284f9e5720c1735e1588d95c7b64095157a797f868c57e5b5b.jpg)

![](images/d71ace03779aaece6263e92896b22f375f2805050f22d2cc7d1d612011400ca1.jpg)

![](images/5d90dcb22746690f5820459b9226455c86ae80ee20edb0b90d24ca474e6ab9bb.jpg)

![](images/a5cf101bd67bb2a187e9c5c1dabf44265fce92c3285e6ec533a06bb11cbd633b.jpg)  
Figure 3: Material strength screening using PT approach. (a) The material strength database constructed by PT learning, which covers 88 elements across the periodic table. (b) The distribution of  $\tau_{\mathrm{P}}$  in the material strength database. (c) Distribution of  $\tau_{\mathrm{P}}$  in the space of chemical compositions, visualized by t-distributed stochastic neighbor embedding (t-SNE). The crystals are represented by the sum of the one-hot encodings of their constituent elements. The t-SNE reduces the high-dimensional representations of crystals to two principal features (crystal embedding features 1 and 2,  $\mathrm{CEF}_1$  and  $\mathrm{CEF}_2$ ) (Van der Maaten & Hinton, 2008) (d) High-strength material screening from the extensive space of metastable materials in GNoME. (e) High-strength materials screened using PT learning and the corresponding yield strengths  $(\sigma_{\mathrm{Y}})$  reported in experiments (extracted from MatWeb (Ross, 2013)).

![](images/80196402941227cd5e14a3322eb0f6d1adf21e1772475530442514d0e89576cc.jpg)

![](images/f5b327b1c5091c3d50abd988bf7fa208de0432f01a48f11923b7e245efdc7936.jpg)

![](images/3e2394c678794ab5389f9c8aa65681e8018d9e246e39d2ab4e73b057a0a73b6f.jpg)

The high accuracy and efficiency of the PT framework allow for single-crystal strength screening and the implementation of mesoscale physics such as the grain texture into the paradigm of high-

throughput materials screening and discovery. For a given material genome including the elements and lattice types, the characteristic materials parameters can be estimated by the equilibrium properties reported in the Materials Project. For example, the  $\gamma$  surface can be fitted from a few single-point energy calculations (e.g., intrinsic stacking fault energy (SFE)  $\gamma_{\mathrm{isf}}$ , unstable SFE  $\gamma_{\mathrm{usf}}$ , aligned SFE  $\gamma_{\mathrm{asf}}$ , and the energies of their intermediate configurations) and interpolated using the Fourier series (Su et al., 2019). The elastic constants can be determined by the slope of the linear region in stress-strain curves. The generated characteristic parameters can be used to screen materials by their strengths, a Holly Grail in materials science, through the predicted Peierls stress, and extension by implementing mesoscale physics models such as CP (Roters et al., 2011). Recent advances in theoretical materials science accelerated by artificial intelligence significantly expanded the space of scientific exploration. The graph networks for materials exploration (GNoME) model enlarges the library of inorganic crystals from 48k to 2.2M, many of which are metastable materials that have not been synthesized by existing methods and thus cannot be assessed by experiments (Merchant et al., 2023). The PT model can efficiently screen materials in such a huge library at the chemically accurate level, especially for non-equilibrium material properties and processes inaccessible by conventional approaches due to the accuracy-performance dilemma. Specifically, 3,471 fcc (Fm3m), bcc (Im3m) or hcp (P63/mmc) crystals of the 2.2M inorganic crystalline compounds in GNoME are supplemented with calculated elastic properties and chosen for material strength screening (Figs. 3a-c). A product material strength database is finally constructed (Figs. 3a-c). High-strength metal materials (Os, Ru, Tc, Re) screened out from the database are verified by their experimentally measured yield strengths ( $\sigma_{\mathrm{Y}}$ ) (Ross, 2013), and are much stronger than the metals in the training set (e.g., fcc Cu, bcc Fe, hcp Ti) (Figs. 3d and 3e).

# 3.2 PHYSICS-TRANSFER LEARNING FOR THE PREDICTION OF BRAIN MORPHOLOGY DEVELOPMENT

Brain development involves complex multiscale physical processes, encompassing gene expression, protein folding, and cellular behaviors such as cell division, differentiation, and migration, as well as macroscopic morphological instabilities (Llinares-Benadero & Borrell, 2019a). The continuum mechanics theory that incorporates growth tensor parameters is widely used to describe the morphological evolution of tissue growth (Tallinen et al., 2016; Striedter et al., 2015; Darayi et al., 2022; Budday & Steinmann, 2018; da Costa Campos et al., 2021; Alenyà et al., 2022). These growth tensor parameters can be linked to micro-scale cellular behaviors, providing a multiscale modeling framework for modeling morphological instabilities. The intricate geometry of the brain and the nonlinear nature of brain morphological development involving materials and contact result in low computational efficiency and poor convergence in finite element analysis (FEA) (Tallinen et al., 2016). Consequently, there is limited work directly simulating the morphological development of the brain, with most studies discussing that on simplified geometries, such as two-dimensional shell-substrates geometries, or three-dimensional spheres and ellipsoids (Fig. 4a) (Darayi et al., 2022; Budday & Steinmann, 2018; da Costa Campos et al., 2021; Wang et al., 2021). Indeed, the growth of spheres or ellipsoids shares similar spatiotemporal characteristics with brain morphological development, such as ridge-valley networks and bifurcation behaviors. By designing neural network architecture  $(h\in \mathcal{H})$ , one can learn the physics of bifurcation and morphological features from simple geometries. The well-trained models with parameter distributions of  $(p(\theta |\mathcal{D}_{\mathrm{LF}}^{'}))$  can be directly extrapolated to predict the morphological development of the high-complexity brain.

# 3.2.1 DATASETS

The experimental data of human brain morphologies is rare, especially for individual brain morphologies (Bethlehem et al., 2022; Ciceri et al., 2024). We collect the currently available open-source brain structural magnetic resonance imaging (MRI) atlases from the source (Ciceri et al., 2024). The pipeline involving cortical and sub-cortical volume segmentation and cortical surface extraction is adopted to obtain brain morphologies from MRI data (Makropoulos et al., 2018). The collected experimental data of human brain morphologies are used to validate the effectiveness of PT learning. We construct digital libraries of morphological patterns involving spheres, ellipsoids, and human brains with increasing geometrical complexities. For spheres and ellipsoids with simpler geometries, a representative core-shell model is used (Tallinen et al., 2014; Wang et al., 2021; Xu et al., 2022a; Yin et al., 2008), as implemented to explore the mechanical instability in cortical folding (Tallinen et al., 2016; Striedter et al., 2015; Darayi et al., 2022; Budday & Steinmann,

2018; da Costa Campos et al., 2021; Alenyà et al., 2022). The outer spherical shell represents the cerebral cortex (gray matter), and the inner core for the white matter. The core and shell structures are modeled as modestly compressible hyperelastic Neo-Hookean material with different growth rates (Tallinen et al., 2016). Following the experimental evidence (Fischl & Dale, 2000; Chang et al., 2007; Xu et al., 2010; Dervaux & Amar, 2008; Budday et al., 2015), the cortical thickness ranges from  $0.03 - 1.63\mathrm{mm}$  according to the abnormal and normal human cerebral cortex measurements and the scale factor (Fischl & Dale, 2000; Chang et al., 2007), and the relative shear modulus  $(\mu_{\mathrm{grey}} / \mu_{\mathrm{white}})$  ranges from  $0.65 - 1$  (Xu et al., 2010; Dervaux & Amar, 2008; Budday et al., 2015). The tangential growth (TG) model is used to simulate the cellular mechanisms that create the growth stresses and lead to the pattern evolution (Tallinen et al., 2014; 2016; Llinares-Benadero & Borrell, 2019b).

# 3.2.2 ARCHITECTURE AND MODEL SETUP

In FEA, morphological data are meshed into discretized tetrahedral element, which can be represented as graphs, where the nodes of the graph correspond to the vertices of the elements, and the edges of the graph correspond to the edges of the elements. Consequently, graph neural networks (GNN) are suitable for extracting features from morphologies represented as graphs. Specifically, we utilize an encoder-decoder architecture to learn the complexity of morphological development, constraining the model with the 3D coordinates of the morphologies and global feature such as gyration index (Fig. 4b). The input to the model is a graph representation of the morphology, where the node features include positional coordinates and the normal direction. The output is the local feature curvature of the morphology.

![](images/f5541ea7f7d0e67ac2e0453fbbc223b7a0d9e7bbac9fb9ae9689c40e40f9e164.jpg)  
Figure 4: Brain development prediction using PT approach. (a) The accuracy of predicting the development of brain morphologies improves with the increase in the geometric complexity of the model. (b) An encoder-decoder architecture constrained by multiscale morphological features is used to resolve the morphological complexity. (c) The interpolation predictions for spherical data. (d,e) The extrapolation predictions for ellipsoidal data (d) and the development of brain morphologies (e).

# 3.2.3 RESULTS

We train our models on spherical and ellipsoidal data, which are then applied to the morphological development of human brains. In our ablation study, the features of normal directions are removed and only morphological data is retained, referred to as statistical learning, since the curvatures contain essential physics of the bifurcation processes, which is well known in the nonlinear elasticity community. Our results show that, for the inference of spherical data, both traditional statistical and PT learning yield satisfactory results (Fig. 4c). However, when extrapolating to ellipsoids and human brains using the model trained on the sphere data, PT learning excels while statistical learning fails, highlighting the generalizability of PT learning (Figs. 4d and 4e).

# 4 RELATED WORK

Our PT framework shares some conceptual features with existing ML methods developed to combine multi-fidelity data (Ramakrishnan et al., 2015; Batra et al., 2019; Smith et al., 2019).  $\Delta$ -learning predicts high-fidelity properties by learning the discrepancies in predictions from models at different levels of fidelity (Ramakrishnan et al., 2015). The objective properties are calculated by correcting low-fidelity calculations following a statistical treatment. In a similar spirit, the low-fidelity as a feature (LFAF) method learns the relation between properties obtained from models with different fidelities, and predicts the high-fidelity properties using objective properties and other materials parameters obtained from low-fidelity models as the input (Batra et al., 2019). Transfer learning pre-trains neural networks on low-fidelity data and fine-tunes the parameters on high-fidelity ones to achieve high accuracy in predictions (Smith et al., 2019). However, these methods are statistical in nature and their applications mainly focus on the properties at equilibrium. In supervised learning, it is necessary and beneficial to label data obtained from high-fidelity models in the training process, which are not available for most non-equilibrium properties such as the Peierls stress with chemical accuracy at the DFT level. Our PT framework resolves this constraint from the accuracy-performance dilemma by going beyond the statistical approach and transferring the physics across models with different fidelities, which is characterized by materials parameters that can be obtained from single-point, unit-cell calculations. For example, the Peierls stress is predicted accurately and efficiently utilizing the learned physics and chemically accurate materials parameters. Longitudinal MRI data of brains are rare, limiting the use of traditional statistical learning methods to directly predict the development of human brain morphologies (Bethlehem et al., 2022; Ciceri et al., 2024). PT learning approach can learn the physics of bifurcation from data of morphological development obtained for simple geometries, and then be applied to the human brain with elevated morphological complexities.

# 5 DISCUSSION AND LIMITATIONS

Our work 'digitalizes' the observation-assumption-model practice in engineering sciences using neural network representations. As the ML model learns physics from data, physical features naturally emerge in the space of latent variables (Fig. 5). In the case of material strength screening, after the model has learned the physics of crystal plasticity, the principal components of latent features show a weak correlation with the input variables such as the elastic constants (Fig. 5a), but a strong correlation with the dislocation width, another important physical quantity in crystal plasticity (Fig. 5b). In the study of brain morphologies, the ML model of PT learning exhibits similar weight distribution  $(p(\theta | \mathcal{D}_{\mathrm{LF}}^{'}) \approx p(\theta | \mathcal{D}_{\mathrm{HF}}^{'}))$  after learning from spherical and ellipsoidal data (Fig. 5c), whereas the ML model of statistical learning using the morphology data only shows a significant difference in parameter distribution  $(p(\theta | \mathcal{D}_{\mathrm{LF}}) \neq p(\theta | \mathcal{D}_{\mathrm{HF}}))$  compared to PT learning (Fig. 5d). The preserved features of weight distribution across data of varying complexity demonstrate the generalizability to complex geometries.

These observations indicate that the PT learning framework captures the essential physics of problems with high complexities, and explains its outstanding performance in addressing the accuracy-performance dilemma. The learned physics in the PT approach is limited by the fidelity of digital libraries, which depend on the completeness of theoretical descriptions and experimental data. Specifically, in materials strength screening, databases constructed with well-trained MLFFs are expected to offer more accurate physics than EAM or MEAM, although their computational costs are

high, and a full set of MLFFs for all metal alloys is not available at present. Our studies show that the error of PT-MLFF predictions using the physics learned from MLFF simulations is reduced to  $e = 1.51\%$  (Fig. 2j). This few-shot fine-tuning approach utilizing well-trained MLFFs substantially improves the accuracy of the learned physics compared to the database constructed with EAM potentials (Figs. 2c, 2f-h, 2j). For the prediction of human brain morphologies, the rareness of MRI data could be resolved by the output of ongoing projects such as the Developing Human Connectome Project (dHCP) (Makropoulos et al., 2018) or adding animal data.

The advancement of engineering sciences has often been marked by key moments where fundamental physics is distilled to form theoretical frameworks. Our PT approach continues these efforts by leveraging data representation from real-world problems. By reducing the dimensionality of latent variable spaces and abstracting data correlations, this method has the potential to reveal new theoretical insights, which will be a focus of ongoing research.

![](images/5c9753eec00fa4a464ab4e4d791b63376726500f022e0a8aca520f00f65fb1a3.jpg)  
Figure 5: Neural networks analysis for latent space features and weights parameters. (a,b) Emergence of physics in the latent space. The model that has learned the physics of crystal plasticity shows low correlation with input variables (a), but high correlation with the key physical variable of dislocation width in crystal plasticity (b). (c,d) The weights parameters distribution of ML models trained on the spherical and ellipsoidal data for PT learning (c) and statistical learning (d).

![](images/b5bc761ec51a21a01406d12d6eacd63d91308147290ccf97c7cf08060dfa48de.jpg)

![](images/749adbd57fbc97c7b02189a292c35f678604910028769ee681a57699735d89db.jpg)

![](images/efa890ceab69f9f1073eeeab23fb53863e92eec3123777ef0fda9c05d3c5e4ae.jpg)

# REFERENCES

Yaser S Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin. Learning from Data, volume 4. AMLBook New York, 2012.  
Mireia Alenyà, Xiaoyu Wang, Julien Lefèvre, Guillaume Auzias, Benjamin Fouquet, Elisenda Eixarch, François Rousseau, and Oscar Camara. Computational pipeline for the generation and validation of patient-specific mechanical models of brain development. Brain Multiphys., 3:100045, 2022. doi: 10.1016/j.brain.2022.100045.  
Rohit Batra, Ghanshyam Pilania, Blas P Uberuaga, and Rampi Ramprasad. Multifidelity information fusion with machine learning: A case study of dopant formation energies in hafnia. ACS Appl. Mater. Interfaces, 11(28):24906-24918, 2019.  
Chandler A Becker, Francesca Tavazza, Zachary T Trautt, and Robert A Buarque de Macedo. Considerations for choosing and using force fields and interatomic potentials in materials science and engineering. Curr. Opin. Solid State Mater. Sci., 17(6):277-283, 2013.  
Ted Belytschko, Wing Kam Liu, Brian Moran, and Khalil Elkhodary. *Nonlinear Finite Elements for Continua and Structures*. John Wiley & Sons, New Jersey, 2014.  
Richard AI Bethlehem, Jakob Seidlitz, Simon R White, Jacob W Vogel, Kevin M Anderson, Chris Adamson, Sophie Adler, George S Alexopoulos, Evdokia Anagnostou, Ariosky Areces-Gonzalez, et al. Brain charts for the human lifespan. Nature, 604(7906):525-533, 2022.  
Peter E Blöchl. Projector augmented-wave method. Phys. Rev. B, 50(24):17953, 1994.  
Silvia Budday and Paul Steinmann. On the influence of inhomogeneous stiffness and growth on mechanical instabilities in the developing brain. Int. J. Solids Struct., 132:31-41, 2018. doi: 10.1016/j.ijsolstr.2017.08.010.

Silvia Budday, Richard Nay, Rijk De Rooij, Paul Steinmann, Thomas Wyrobek, Timothy C Ovaert, and Ellen Kuhl. Mechanical properties of gray and white matter brain tissue by indentation. J. Mech. Behav. Biomed. Mater., 46:318-330, 2015. doi: 10.1016/j.jmbbm.2015.02.024.  
Vasily Bulatov and Wei Cai. Computer Simulations of Dislocations. Oxford University Press, 2006.  
Vasily V Bulatov and Efthimios Kaxiras. Semidiscrete variational Peierls framework for dislocation core properties. Phys. Rev. Lett., 78(22):4221, 1997.  
Bernard S Chang, Fusun Duzcan, Seonhee Kim, Mine Cinbis, Abha Aggarwal, Kira A Apse, Osman Ozdel, Munevver Atmaca, Sevil Zencir, Huseyin Bagci, et al. The role of reln in lissencephaly and neuropsychiatric disease. Am. J. Med. Genet. B Neuropsychiatr. Genet., 144:58-63, 2007. doi: 10.1002/ajmg.b.30392.  
Tommaso Ciceri, Luca Casartelli, Florian Montano, Stefania Conte, Letizia Squarcina, Alessandra Bertoldo, Nivedita Agarwal, Paolo Brambilla, and Denis Peruzzo. Fetal brain mri atlases and datasets: a review. NeuroImage, pp. 120603, 2024.  
Lucas da Costa Campos, Raphael Hornung, Gerhard Gompper, Jens Elgeti, and Svenja Caspers. The role of thickness inhomogeneities in hierarchical cortical folding. NeuroImage, 231:117779, 2021. doi: 10.1016/j.neuroimage.2021.117779.  
Mohsen Darayi, Mia E Hoffman, John Sayut, Shuolun Wang, Nagehan Demirci, Jack Consolini, and Maria A Holland. Computational models of cortical folding: A review of common approaches. J. Biomech., 139:110851, 2022. doi: 10.1016/j.jbiomech.2021.110851.  
Julien Dervaux and Martine Ben Amar. Morphogenesis of growing soft tissues. Phys. Rev. Lett., 101(6):068101, 2008. doi: 10.1103/PhysRevLett.101.068101.  
Christer Ericson. Real-Time Collision Detection. CRC Press, Florida, 2004.  
Zheyong Fan, Zezhu Zeng, Cunzhi Zhang, Yanzhou Wang, Keke Song, Haikuan Dong, Yue Chen, and Tapio Ala-Nissila. Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport. Phys. Rev. B, 104(10): 104309, 2021.  
Yu Feng, Wei Zhang, and Yuhai Tu. Activity-weight duality in feed-forward neural networks reveals two co-determinants for generalization. Nat. Mach. Intell., 5(8):908-918, 2023.  
Bruce Fischl and Anders M Dale. Measuring the thickness of the human cerebral cortex from magnetic resonance images. Proc. Natl. Acad. Sci. USA, 97(20):11050-11055, 2000. doi: 10.1073/pnas.200033797.  
Jacob Fish, Gregory J Wagner, and Sinan Keten. Mesoscopic and multiscale modelling in materials. Nat. Mater., 20(6):774-786, 2021.  
Jicheng Gong, T Benjamin Britton, Mitchell A Cuddihy, Fionn PE Dunne, and Angus J Wilkinson. <a> Prismatic, <a> basal, and <c + a> slip strengths of commercially pure Zr by micro-cantilever tests. Acta Mater., 96:249-257, 2015.  
Xiaoguo Gong, Zhuoyuan Li, ASL Pattamatta, Tongqi Wen, and David J Srolovitz. An accurate machine learning interatomic potential for fcc and hcp nickel. arXiv preprint arXiv:2312.17596, 2023.  
Lucas M Hale, Zachary T Trautt, and Chandler A Becker. Evaluating variability with atomistic simulations: The effect of potential and calculation methodology on the modeling of lattice and elastic constants. Model. Simul. Mat. Sci. Eng., 26(5):055003, 2018.  
Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In Int. Conf. Mach. Learn. (ICML), pp. 1225-1234, 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 770-778, 2016.

Daniel Hedman, Ben McLean, Christophe Bichara, Shigeo Maruyama, J Andreas Larsson, and Feng Ding. Dynamics of growing carbon nanotube interfaces probed by machine learning-enabled molecular simulations. arXiv preprint arXiv:2302.09542, 2023.  
Esther Klingler, Fiona Francis, Denis Jabaudon, and Silvia Cappello. Mapping the molecular and cellular complexity of cortical malformations. Science, 371(6527):eaba4517, 2021. doi: 10.1126/science.aba4517.  
Tsz Wai Ko and Shyue Ping Ong. Recent advances and outstanding challenges for machine learning interatomic potentials. Nat. Comput. Sci., 3:998-1000, 2023.  
Georg Kresse and Daniel Joubert. From ultrasoft pseudopotentials to the projector augmented-wave method. Phys. Rev. B, 59(3):1758, 1999.  
He Li, Zun Wang, Nianlong Zou, Meng Ye, Runzhang Xu, Xiaoxun Gong, Wenhui Duan, and Yong Xu. Deep-learning density functional theory hamiltonian for efficient ab initio electronic-structure calculation. Nat. Comput. Sci., 2(6):367-377, 2022.  
Hojun Lim, LM Hale, JA Zimmerman, CC Battaile, and CR Weinberger. A multi-scale model of dislocation plasticity in  $\alpha$ -Fe: Incorporating temperature, strain rate and non-Schmid effects. Int. J. Plast., 73:100-118, 2015.  
Cristina Llinares-Benadero and Víctor Borrell. Deconstructing cortical folding: genetic, cellular and mechanical determinants. Nat. Rev. Neurosci., 20(3):161-176, 2019a.  
Cristina Llinares-Benadero and Víctor Borrell. Deconstructing cortical folding: Genetic, cellular and mechanical determinants. Nat. Rev. Neurosci., 20(3):161-176, 2019b. doi: 10.1038/s41583-018-0112-2.  
Gang Lu, Nicholas Kioussis, Vasily V Bulatov, and Efthimios Kaxiras. Generalized-stacking-fault energy surface and dislocation properties of aluminum. Phys. Rev. B, 62(5):3099, 2000.  
Antonios Makropoulos, Emma C Robinson, Andreas Schuh, Robert Wright, Sean Fitzgibbon, Jelena Bozek, Serena J Counsell, Johannes Steinweg, Katy Vecchiato, Jonathan Passerat-Palmbach, et al. The developing human connectome project: A minimal processing pipeline for neonatal cortical surface reconstruction. Neuroimage, 173:88-112, 2018.  
Amil Merchant, Simon Batzner, Samuel S Schoenholz, Muratahan Aykol, Gowoon Cheon, and Ekin Dogus Cubuk. Scaling deep learning for materials discovery. Nature, 624:80-85, 2023.  
Hendrik J Monkhorst and James D Pack. Special points for Brillouin-zone integrations. Phys. Rev. B, 13(12):5188, 1976.  
Joe D Morrow, John LA Gardner, and Volker L Deringer. How to validate machine-learned interatomic potentials. J. Chem. Phys., 158(12), 2023.  
FRN Nabarro. Dislocations in a simple cubic lattice. Proc. Phys. Soc., 59(2):256, 1947.  
FRN Nabarro. Fifty-year study of the Peierls-Nabarro stress. Mater. Sci. Eng. A, 234:67-76, 1997.  
Sang Ho Oh, Marc Legros, Daniel Kiener, and Gerhard Dehm. In situ observation of dislocation nucleation and escape in a submicrometer aluminium single crystal. Nat. Mater., 8(2):95-100, 2009.  
John P Perdew, Kieron Burke, and Matthias Ernzerhof. Generalized gradient approximation made simple. Phys. Rev. Lett., 77(18):3865, 1996.  
Steve Plimpton. Fast parallel algorithms for short-range molecular dynamics. J. Comput. Phys., 117 (1):1-19, 1995.  
Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Big data meets quantum chemistry approximations: The  $\Delta$ -machine learning approach. J. Chem. Theory Comput., 11(5):2087-2096, 2015.

David Rodney, L Ventelon, E Clouet, Laurent Pizzagalli, and F Willaime. Ab initio modeling of dislocation core properties in metals and semiconductors. Acta Mater., 124:633-659, 2017.  
Robert B Ross. *Metallic Materials Specification Handbook*. Springer Science & Business Media, 2013.  
Franz Roters, Philip Eisenlohr, Thomas R Bieler, and Dierk Raabe. *Crystal Plasticity Finite Element Methods*. John Wiley & Sons, 2011.  
AA Salem, SR Kalidindi, and SL Semiatin. Strain hardening due to deformation twinning in  $\alpha$ -titanium: Constitutive relations and crystal-plasticity modeling. Acta Mater., 53(12):3495-3502, 2005.  
John D Shimanek, Shipin Qin, Shunli Shang, Zikui Liu, and Allison M Beese. Predictive crystal plasticity modeling of single crystal nickel based on first-principles calculations. JOM, 74(4): 1423-1434, 2022.  
Justin S Smith, Benjamin T Nebgen, Roman Zubatyuk, Nicholas Lubbers, Christian Devereux, Kipton Barros, Sergei Tretiak, Alexandr Isayev, and Adrian E Roitberg. Approaching coupled cluster accuracy with a general-purpose neural network potential through transfer learning. Nature Commun., 10(1):2903, 2019.  
M Soleymani, MH Parsa, and H Mirzadeh. Molecular dynamics simulation of stress field around edge dislocations in aluminum. Comput. Mater. Sci., 84:83-96, 2014.  
Keke Song, Rui Zhao, Jiahui Liu, Yanzhou Wang, Eric Lindgren, Yong Wang, Shunda Chen, Ke Xu, Ting Liang, Penghua Ying, et al. General-purpose machine-learned potential for 16 elemental metals and their alloys. arXiv preprint arXiv:2311.04732, 2023.  
Georg F Striedter, Shyam Srinivasan, and Edwin S Monuki. Cortical folding: When, where, how, and why? Annu. Rev. Neurosci., 38:291-307, 2015. doi: 10.1146/annurev-neuro-071714-034128.  
Yangqing Su, Shuozhi Xu, and Irene J Beyerlein. Density functional theory calculations of generalized stacking fault energy surfaces for eight face-centered cubic transition metals. J. Appl. Phys., 126(10), 2019.  
So Takamoto, Chikashi Shinagawa, Daisuke Motoki, Kosuke Nakago, Wenwen Li, Iori Kurata, Taku Watanabe, Yoshihiro Yayama, Hiroki Iriguchi, Yusuke Asano, et al. Towards universal neural network potential for material discovery applicable to arbitrary combination of 45 elements. Nat. Commun., 13(1):2991, 2022.  
Tuomas Tallinen, Jun Young Chung, John S Biggins, and L Mahadevan. Gyrification from constrained cortical expansion. Proc. Natl. Acad. Sci. USA, 111(35):12667-12672, 2014. doi: 10.1073/pnas.1406015111.  
Tuomas Tallinen, Jun Young Chung, François Rousseau, Nadine Girard, Julien Lefèvre, and Lakshminarayanan Mahadevan. On the growth and form of cortical convolutions. Nat. Phys., 12(6): 588-593, 2016. doi: 10.1038/nphys3632.  
Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-SNE. J. Mach. Learn. Res., 9(11), 2008.  
Xiaoyu Wang, Julien Lefèvre, Amine Bohi, Mariam Al Harrach, Mickaël Dinomais, and François Rousseau. The influence of biophysical parameters in a biomechanical model of cortical folding patterns. Sci. Rep., 11:7686, 2021. doi: 10.1038/s41598-021-87124-y.  
Fan Xu, Yangchao Huang, Shichen Zhao, and Xi-Qiao Feng. Chiral topographic instability in shrinking spheres. Nat. Comput. Sci., 2(10):632-640, 2022a. doi: 10.1038/s43588-022-00332-y.  
Gang Xu, Andrew K Knutsen, Krikor Dikranian, Christopher D Kroenke, Philip V Bayly, and Larry A Taber. Axons pull on the brain, but tension does not drive cortical folding. J. Biomech. Eng., 132(7):071013, 2010. doi: 10.1115/1.4001683.

Zhaoyue Xu, Xinlei Zhang, Shizhao Wang, and Guowei He. Artificial neural network based response surface for data-driven dimensional analysis. J. Comput. Phys., 459:111145, 2022b.  
Jie Yin, Zexian Cao, Chaorong Li, Izhak Sheinman, and Xi Chen. Stress-driven buckling patterns in spheroidal core/shell structures. Proc. Natl. Acad. Sci. USA, 105(49):19132-19135, 2008. doi: 10.1073/pnas.0810443105.  
Linfeng Zhang, Jiequn Han, Han Wang, Roberto Car, and EJPRL Weinan. Deep potential molecular dynamics: A scalable model with the accuracy of quantum mechanics. Phys. Rev. Lett., 120(14): 143001, 2018.
