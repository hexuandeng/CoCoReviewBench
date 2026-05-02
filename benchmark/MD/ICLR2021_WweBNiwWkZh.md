# SKINNING A PARAMETERIZATION OF THREE-DIMENSIONAL SPACE FOR NEURAL NETWORK CLOTH

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a novel learning framework for cloth deformation by embedding virtual cloth into a tetrahedral mesh that parametrizes the volumetric region of air surrounding the underlying body. In order to maintain this volumetric parameterization during character animation, the tetrahedral mesh is constrained to follow the body surface as it deforms. We embed the cloth mesh vertices into this parameterization of three-dimensional space in order to automatically capture much of the nonlinear deformation due to both joint rotations and collisions. We then train a convolutional neural network to recover ground truth deformation by learning cloth embedding offsets for each skeletal pose. Our experiments show significant improvement over learning cloth offsets from body surface parameterizations, both quantitatively and visually, with prior state of the art having a mean error five standard deviations higher than ours. Without retraining, our neural network generalizes to other body shapes and T-shirt sizes, giving the user some indication of how well clothing might fit. Our results demonstrate the efficacy of a general learning paradigm where high-frequency details can be embedded into low-frequency parameterizations.

# 1 INTRODUCTION

Cloth is particularly challenging for neural networks to model due to the complex physical processes that govern how cloth deforms. In physical simulation, cloth deformation is typically modeled via a partial differential equation that is discretized with finite element models ranging in complexity from variational energy formulations to basic masses and springs, see e.g. Baraff & Witkin (1998); Bridson et al. (2002; 2003); Grinspun et al. (2003); Baraff et al. (2003); Selle et al. (2008). Mimicking these complex physical processes and numerical algorithms with machine learning inference has shown promise, but still struggles to capture high-frequency folds/wrinkles. PCA-based methods De Aguiar et al. (2010); Hahn et al. (2014) remove important high variance details and struggle with nonlinearities emanating from joint rotations and collisions. More recently, Gundogdu et al. (2019); Santesteban et al. (2019); Patel et al. (2020); Jin et al. (2020) leverage body skinning Magnenat-Thalmann et al. (1988); Lander (1998); Lewis et al. (2000) to capture some degree of the nonlinearity; the cloth is then represented via learned offsets from a co-dimension one skinned body surface. Building on this prior work, we propose replacing the skinned co-dimension one body surface parameterization with a skinned (fully) three-dimensional parameterization of the volume surrounding the body.

We parameterize the three-dimensional space corresponding to the volumetric region of air surrounding the body with a tetrahedral mesh. In order to do this, we leverage the work of Lee et al. (2018; 2019), which proposed a number of techniques for creating and deforming such a tetrahedral mesh using a variety of skinning and simulation techniques. The resulting kinematically deforming skinned mesh (KDSM) was shown to be beneficial for both hair animation/simulation Lee et al. (2018) and water simulation Lee et al. (2019). Here, we only utilize the most basic version of the KDSM, assigning skinning weights to its vertices so that it deforms with the underlying joints similar to a skinned body surface (alternatively, one could train a neural network to learn more complex KDSM deformations). This allows us to make a very straightforward and fair comparison between learning offsets from a skinned body surface and learning offsets from a skinned parameterization of three-dimensional space. Our experiments showed an overall reduction in error of approximately

$50\%$  (see Table 2 and Figure 8) as well as the removal of visual/geometric artifacts (see e.g. Figure 9) that can be directly linked to the usage of the body surface mesh, and thus we advocate the KDSM for further study. The neural network we trained for a particular body can also be used to infer cloth with unique wrinkle patterns on different body shapes and T-shirt sizes without retraining (see supplemental material). In order to further illustrate the efficacy of our approach, we show that the KDSM is amenable to being used with recently proposed works on texture sliding for better three-dimensional reconstruction Wu et al. (2020b) as well as in conjunction with networks that use a postprocess for better physical accuracy in the  $\mathrm{L}^{\infty}$  norm Geng et al. (2020) (see Figure 10).

In summary, our specific contributions are: 1) a novel three-dimensional parameterization for virtual cloth adapted from the KDSM, 2) an extension (enabling plastic deformation) of the KDSM to accurately model cloth deformation, and 3) a learning framework to efficiently infer such deformations from body pose. The mean error of the cloth predicted in Jin et al. (2020) is five standard deviations higher than the mean error of our results.

# 2 RELATED WORK

Cloth: Data-driven cloth prediction using deep learning has shown significant promise in recent years. To generate clothing on the human body, a common approach is to reconstruct the cloth and body jointly Alldieck et al. (2018a;b); Xu et al. (2018); Alldieck et al. (2019a,b); Habermann et al. (2019); Natsume et al. (2019); Saito et al. (2019); Yu et al. (2019); Bhatnagar et al. (2019); Onizuka et al. (2020); Saito et al. (2020). In such cases, human body models such as SCAPE Anguelov et al. (2005) and SMPL Loper et al. (2015) can be used to reduce the dimensionality of the output space. To predict cloth shape, a number of works have proposed learning offsets from the body surface Guan et al. (2012); Neophytou & Hilton (2014); Pons-Moll et al. (2017); Lahner et al. (2018); Yang et al. (2018); Gundogdu et al. (2019); Santesteban et al. (2019); Patel et al. (2020); Jin et al. (2020) such that body skinning can be leveraged. There are a variety of skinning techniques used in animation; the most popular approach is linear blend skinning (LBS) Magnenat-Thalmann et al. (1988); Lander (1998). Though LBS is efficient and computationally inexpensive, it suffers from well-known artifacts addressed in Kavan & Zara (2005); Kavan et al. (2007); Jacobson & Sorkine (2011); Le & Hodgins (2016). Since regularization often leads to overly smooth cloth predictions, additional wrinkles/folds can be added to initial network inference results Popa et al. (2009); Mirza & Osindero (2014); Robertini et al. (2014); Lahner et al. (2018); Wu et al. (2020b); Patel et al. (2020). Most recently, Patel et al. (2020) parameterized cloth as a submesh of the SMPL body mesh and decomposed cloth deformation into low-frequency and high-frequency components. However, this parameterization limits cloth to be bound by the topology of SMPL, and the high-frequency folds/wrinkles added by the network are not constrained to match those in the ground truth data. In contrast, our method allows one to predict cloth deformation independent of a predefined PCA basis, and using Geng et al. (2020) ensures that folds/wrinkles are physically consistent.

3D Parameterization: Parameterizing the air surrounding deformable objects is a way of treating collisions during physical simulation Sifakis et al. (2008); Müller et al. (2015); Wu & Yuksel (2016). For hair simulation in particular, previous works have parameterized the volume enclosing the head or body using tetrahedral meshes Lee et al. (2018; 2019) or lattices Volino & Magnenat-Thalmann (2004; 2006). These volumes are animated such that the embedded hairs follow the body as it deforms enabling efficient hair animation, simulation, and collisions. Interestingly, deforming a low-dimensional reference map that parameterizes high-frequency details has been explored in computational physics as well, particularly for fluid simulation, see e.g. Bellotti & Theillard (2019).

# 3 SKINNING A 3D PARAMETERIZATION

We generate a KDSM using red/green tetrahedralization Molino et al. (2003); Teran et al. (2005a) to parameterize a three-dimensional volume surrounding the body. Starting with the body in the T-pose, we surround it with an enlarged bounding box containing a three-dimensional Cartesian grid. As is typical for collision bodies in computer graphics Bridson et al. (2003), we generate a level set representation separating the inside of the body from the outside (see e.g. Osher & Fedkiw (2002)). See Figure 1a. Next, a thickened level set is computed by subtracting a constant value from the current level set values (Figure 1b). Then, we use red/green tetrahedralization as outlined in Molino

et al. (2003); Teran et al. (2005a) to generate a suitable tetrahedral mesh (Figure 1c). Optionally, this mesh could be compressed to the level set boundary using either physics or optimization, but we forego this step because the outer boundary is merely where our parameterization ends and does not represent an actual surface as in Molino et al. (2003); Teran et al. (2005a).

Skinning weights are assigned to the KDSM using linear blend skinning (LBS) Magnenat-Thalmann et al. (1988); Lander (1998), just as one would skin a co-dimension one body surface parameterization. In order to skin the KDSM so that it follows the body as it moves, each vertex  $v_{k}$  is assigned a nonzero weight  $w_{kj}$  for each joint  $j$  it is associated with. Then, given a pose  $\theta$  with joint transformations  $T_{j}(\theta)$ , the world space position of each vertex is given by  $v_{k}(\theta) = \sum_{j} w_{kj} T_{j}(\theta) v_{k}^{j}$  where  $v_{k}^{j}$  is the untransformed location of vertex  $v_{k}$  in the local reference space of joint  $j$ . See Figure 1d. Importantly, it can be quite difficult to significantly deform tetrahedral meshes without having some tetrahedra invert Irving et al. (2004); Teran et al. (2005b); thus, we address inversion and robustness issues/details in Section 5.

![](images/04d3a1effad86769ab27e6d0e24fb984d288106e2e076838828a7cb50c758139.jpg)  
(a)

![](images/e04ae69e02494f2436bb66d7c5a98cc1100e8915741315ab2c2090a88ecafd61.jpg)  
(b)

![](images/00ff62317dee8f0296cbc16b7dc6f8253142fdb57be28500b8b63f11bc430404.jpg)  
Figure 1: We build a tetrahedral mesh surrounding the body to parameterize the enclosed three-dimensional space. First, a level set representation of the body (a) is generated and subsequently thickened (b) to contain the clothing worn on the body. Then, we use red/green tetrahedralization Molino et al. (2003); Teran et al. (2005a) to create a tetrahedral mesh (c) from the thickened level set. This tetrahedral mesh is skinned to follow the body as it moves (d). Note that the tetrahedral mesh surrounds the whole upper body to demonstrate that this parameterization can also be used for long-sleeve shirts.  
(c)

![](images/e1aa9f261083f88dd79045240edb688d7bebe115dbba90da5e36de6cf2198dc4.jpg)  
(d)

# 4 EMBEDDING CLOTH IN THE KDSM

In continuum mechanics, deformation is defined as a mapping from a material space to the world space, and one typically decomposes this mapping into purely rigid components and geometric strain measures, see e.g. Bonet & Wood (1997). Similar in spirit, we envision the T-pose KDSM as the material space and the skinned KDSM as being defined by a deformation mapping to world space for each pose  $\theta$ . As such, we denote the position of each cloth vertex in the material space (i.e. T-pose, see Figure 2a) as  $u_{i}^{m_{o}}$ . We embed each cloth vertex  $u_{i}^{m_{o}}$  into the tetrahedron that contains it via barycentric weights  $\lambda_{ik}^{m_{o}}$ , which are only nonzero for the parent tetrahedron's vertices. Then, given a pose  $\theta$ , a cloth vertex's world space location is defined as  $u_{i}(\theta) = \sum_{k}\lambda_{ik}^{m_{o}}v_{k}(\theta)$  so that it is constrained to follow the KDSM deformation, assuming linearity in each tetrahedron (see Figure 2b). Technically, this is an indirect skinning of the cloth with its skinning weights computed as a linear combination of the skinning weights of its parent tetrahedron's vertices, and leads to the obvious errors one would expect (see e.g. Figure 3, second row).

The KDSM approximates a deformation mapping for the region surrounding the body. This approximation could be improved via physical simulation (see e.g. Lee et al. (2018; 2019)), which is computationally expensive but could be made more efficient using a neural network. However, the tetrahedral mesh is only well suited to capture deformations of a volumetric three-dimensional space and as such struggles to capture deformations intrinsic to codimension one surfaces/shells including the bending, wrinkling, and folding important for cloth. Thus, we take further motivation from constitutive mechanics (see e.g. Bonet & Wood (1997)) and allow the cloth vertices to move in material space (the T-pose) akin to plastic deformation. That is, we use plastic deformation in the material space in order to recapture elastic deformations (e.g. bending) lost/recovered when embedding cloth into a tetrahedral mesh. These elastic deformations are encoded as a pose-dependent plastic displacement for each cloth vertex, i.e.  $d_{i}(\theta)$ ; then, the pose-dependent, plastically deformed material space position of each cloth vertex is given by  $u_{i}^{m}(\theta) = u_{i}^{m_{o}} + d_{i}(\theta)$ .

Given a pose  $\theta$ ,  $u_{i}^{m}(\theta)$  will not necessarily have the same parent tetrahedron or barycentric weights as  $u_{i}^{m_o}$ ; thus, a new embedding is computed for  $u_{i}^{m}(\theta)$  obtaining new barycentric weights  $\lambda_{ik}^{m}(\theta)$ . Using this new embedding, the position of the cloth vertex in pose  $\theta$  will be  $u_{i}(\theta) = \sum_{k}\lambda_{ik}^{m}(\theta)v_{k}(\theta)$ . Ideally, if the  $d_{i}(\theta)$  are computed correctly,  $u_{i}(\theta)$  will agree with the ground truth location of cloth vertex  $i$  in pose  $\theta$ . The second row of Figure 4 shows cloth in the material space T-pose plastically deformed such that its skinned location in pose  $\theta$  (Figure 4, first row) well matches the ground truth shown in the first row of Figure 3. Learning  $d_{i}(\theta)$  for each vertex can be accomplished in exactly the same

fashion as learning displacements from the skinned body surface mesh, and thus we use the same approach as proposed in Jin et al. (2020). Afterwards, an inferred  $d_{i}(\theta)$  is used to compute  $u_{i}^{m}(\theta)$  followed by  $\lambda_{ik}^{m}(\theta)$ , and finally  $u_{i}(\theta)$ . Addressing efficiency, note that only the vertices of the parent tetrahedra of  $u^{m}(\theta)$  need to be skinned, not the entire tetrahedral mesh.

![](images/6bbaad4acb88b01118163a5e1caff132955b59da99160f84eb9a42981217902a.jpg)  
(a)

![](images/d43bfbe8814ad38108c760cc925723bca3ff6c08e3e1cb91b7f9530d2df77247.jpg)  
Figure 2: One can embed the cloth into the T-posed KDSM (a) and fix this embedding as the KDSM deforms (b). However, this results in undesired artifacts in the cloth (see e.g. Figure 3, second row).  
(b)

![](images/f33ea5b5c9d2bfa48c4135f060ad9ee7d791a1f83f0d24abce7b2f89945b13cc.jpg)  
(a)

![](images/fe1fd9941df914924b8a577533373011dc8ae79b08380e671c3779fd01925345.jpg)

![](images/b4c53fcdca8ae4800283c4c14aca20cf4aa2e02c18b824235a2a45ddea758803.jpg)

![](images/4ac053cfcb9f316b7136bd9acc1c6a2d7271b2e1a424071c7b92e33308ab8e94.jpg)

![](images/290071ff8a4a322c80915d4f25afaf67e0dc9945cf80651f16076b5ca211bc1e.jpg)  
(b)

![](images/ace55b0c118433ee0ccbb4d418581af4c0b2a3dee9360d35ab0099d9b7f7ba58.jpg)  
Figure 3: (a) The ground truth cloth and (b) skinning the cloth using a fixed tetrahedral embedding. Note how poorly this naive embedding of the cloth into the KDSM matches the ground truth (especially as compared to a more sophisticated embedding using our plastic deformation as shown in Figure 4).

![](images/e4c2414c2e61375b54741450c6c89051df9e9e6e24f415adb237f5a1d08b6404.jpg)

![](images/4e90fb9ba6cc77679e887dac4eb9e901f35b9f17d5dad201a0b4a515670505aa.jpg)

![](images/fc327ac5f060194759b402e555943882ffb6c82a4f6cfa27453a5b51da9931a3.jpg)  
(a)

![](images/479ca0ee1876886338586b90f50172c9826eb22169bef00840a3cf9b1ce42b07.jpg)

![](images/0e75794f3e806aaa97c68ff97c3098b2c79cd74fd9590084d1379ecf7b641b8e.jpg)

![](images/096dd199f2479971b2768c1e1840d5d3dd42d705e2baf47d6fff0e72b883b708.jpg)

![](images/915d8c81fdff5c49bb08b0d37f63b7a87e442b8bcf1566f2a5ab4b0b186d7fd2.jpg)  
(b)

![](images/0b39df67d855eecdc9f987ac94d898b0dd7cde556bac457095c22dc5cb67ce94.jpg)

![](images/924ad62fb767c7482ed0f6c2eeec754ccb595676005b4216970ca5083d5d8da9.jpg)  
Figure 4: (a) The hybrid cloth embedding method (see Section 5) produces cloth  $u(\theta)$  that closely matches the ground truth shown in the first row of Figure 3. (b) This is accomplished, for each pose, by plastically deforming the cloth in material space (the T-pose) before embedding it to follow the deformation of the KDSM.

![](images/579352c2e02fabd4f2616f63b7e15ba1c33d18da4a43812e9bd080efc7261bfa.jpg)

In order to compute each training example  $(\theta, d(\theta))$ , we examine the ground truth cloth in pose  $\theta$ , i.e.  $u^{GT}(\theta)$ . For each cloth vertex  $u_{i}^{GT}(\theta)$ , we find the deformed tetrahedron it is located in and

compute barycentric weights  $\lambda_{ik}^{GT}(\theta)$  resulting in  $u_i^{GT}(\theta) = \sum_k\lambda_{ik}^{GT}(\theta)v_k(\theta)$ . Then, that vertex's material space (T-posed) location is given by  $u_i^m (\theta) = \sum_k\lambda_{ik}^{GT}(\theta)v_k^m$  where  $v_{k}^{m}$  are the material space (T-posed) positions of the tetrahedral mesh (which are the same for all poses, and thus not a function of  $\theta$ ). Finally, we define  $d_{i}(\theta) = u_{i}^{m}(\theta) - u_{i}^{m_{o}}$ .

# 5 INVERSION AND ROBUSTNESS

Unfortunately, the deformed KDSM will generally contain both inverted and overlapping tetrahedra, both of which can cause a ground truth cloth vertex  $u_{i}^{GT}(\theta)$  to be contained in more than one deformed tetrahedron, leading to multiple candidates for  $u_{i}^{m}(\theta)$  and  $d_{i}(\theta)$ . Although physical simulation can be used to reverse some of these inverted elements Irving et al. (2004); Teran et al. (2005b) as was done in Lee et al. (2018; 2019), it is typically not feasible to remove all inverted tetrahedra. Additionally, overlapping tetrahedra occur quite frequently between the arm and the torso, especially because the KDSM needs to be thick enough to ensure that it contains the cloth as it deforms.

Before resolving which parent tetrahedron each vertex with multiple potential parents should be embedded into, we first robustly assemble a list of all such candidate parent tetrahedra as follows. Given a deformed tetrahedral mesh  $v(\theta)$  in pose  $\theta$ , we create a bounding box hierarchy acceleration structure Hahn (1988); Webb & Gigante (1992); Barequet et al. (1996); Gottschalk et al. (1996); Lin & Gottschalk (1998) for the tetrahedral mesh built from a slightly thickened bounding box around each tetrahedron. Then given a ground truth cloth vertex,  $u_i^{GT}(\theta)$ , we robustly find all tetrahedra containing (or almost containing) it using a minimum barycentric weight of  $-\epsilon$  with  $\epsilon > 0$ . We prune this list of tetrahedra, keeping only the most robust tetrahedron near each element boundary where numerical precision could cause a vertex to erroneously be identified as inside multiple or no tetrahedra. This is done by first sorting the tetrahedra on the list based on their largest minimum barycentric weight, i.e. preferring tetrahedra the vertex is deeper inside. Starting with the first tetrahedron on the sorted list, we identify the face across from the vertex with the smallest barycentric weight and prune all of that face's vertex neighbors (and thus face/edge neighbors too) from the remainder of the list. Then, the next (non-deleted) tetrahedron on the list is considered, and the process is repeated, etc.

Method 1: Any of the parent tetrahedra that remain on the list may be chosen to obtain training examples with zero error as compared to the ground truth, although different choices lead to higher/ lower variance in  $d(\theta)$  and thus higher/ lower demands on the neural network. To establish a baseline, we first take the naive approach of randomly choosing  $u_{i}^{m}(\theta)$  when multiple candidates exist. This can lead to high variance in  $d(\theta)$  and subsequent ringing artifacts during inference. See Figure 5.

![](images/1890337ce182b5c475ea7aa1864d9dae17cbc87354ef226d6bc65583e34653ca.jpg)  
(a)

![](images/c4d7354bbe84315aebf1c2193a375e9bc30b3ba74bfadba1911349d9e3647640.jpg)  
(b)

![](images/8eca2a71d6f34415bac605ff39d998a853bc67089a5ff1a9ef9ac2b6c50cf9d4.jpg)  
(c)  
Figure 5: (a) shows a training example where overlapping tetrahedra led to cloth torso vertices being embedded into arm tetrahedra, resulting in high variance in  $d(\theta)$ . Although there are various ad hoc approaches for remedying this situation, it is difficult to devise a robust strategy in complex regions such as the armpit. (b) shows that the ground truth  $u^{GT}(\theta)$  is still correctly recovered in spite of this high variance in  $u^m (\theta)$  and  $d(\theta)$ ; however, (c) shows that this high variance leads to spurious ringing oscillations during subsequent inference.

Method 2: Aiming for lower variance in the training data, we leverage the method of Jin et al. (2020) where UV texture space and normal direction offsets from the skinned body surface are calculated for each pose  $\theta$  in the training examples. These same offsets can be used in any pose, since the UVN coordinate system is still defined (albeit deformed) in every pose. Thus, we utilize these UVN offsets in our material space (T-pose) in order to define  $u^{m}(\theta)$  and subsequently  $d(\theta)$ . In particular, given the shrinkwrapped cloth in the T-pose, we apply UVN offsets corresponding to pose  $\theta$ . Although this results in lower variance than that obtained from Method 1, the resulting  $d(\theta)$  do not exactly recover the ground truth cloth  $u^{GT}(\theta)$ . See Figure 6.

![](images/350507625eba462a9f9564d302ab52f9e2cd5b9bdf23da6efe5d6a6b58115a13.jpg)  
(a)

![](images/c1ce8bb0ac9fed27c7c9b61c362b0dec20d74189b902ffe76721c237b0aa2f8d.jpg)  
(b)

![](images/32d62a296dd42014f42dc1f70190df43624b9e27af7081b620c506b774738eb7.jpg)  
(c)  
Figure 6: (a) shows the result obtained using Method 2 to compute  $u_{m}(\theta)$  in material space (the T-pose) for a pose  $\theta$ . (b) shows the result obtained using this embedding to compute  $u(\theta)$  as compared to the ground truth  $u^{GT}(\theta)$  (c). Although the variance in  $u^{m}(\theta)$  and  $d(\theta)$  is lower than that obtained using Method 1, the training examples now contain errors (shown with a heat map) when compared to the ground truth.

**Hybrid Method:** When a vertex has only one candidate parent tetrahedron, Method 1 is used. When there is more than one candidate parent tetrahedron, we choose the parent that gives an embedding closest to the result of Method 2 (in the T-pose) as long as the disagreement is below a threshold (1 cm). As shown (for a particular training example) in Figure 7a, this can leave a number of potentially high variance vertices undefined. Aiming for smoothness, we use the Poisson morph from Cong et al. (2015) to morph from the low variance results of Method 2 to the partially-defined cloth mesh shown in Figure 7a, utilizing the already defined/valid vertices as Dirichlet boundary conditions. See Figure 7b. Although smooth, the resulting predictions may contain significant errors, and thus we only validate those that are within a threshold (1 cm) of the results of Method 2. See Figure 7c. The Poisson equation morph guarantees smoothness, while only utilizing the morphed vertices close to the results of Method 2 limits errors (as compared to the ground truth) to some degree. This process is repeated until no newly newly morphed vertices are within the threshold (1 cm). At that point, the remaining vertices are assigned their morphed values despite any errors they might contain. See Figure 7d.

![](images/2a6c9d98c49cf2e779e64e7c0fdf86e62c1721b8be04ba01bcbb751506f61af4.jpg)  
(a)

![](images/206681f980d53e27c12796e44c01bdfce94d486e6972c9582f6bb7ae7771949c.jpg)  
Figure 7: (a) Subset of vertices for which some choice of a parent tetrahedron using Method 1 reasonably agrees with Method 2. (b) The rest of the mesh can be filled in with the 3D morph proposed in Cong et al. (2015). (c) Subset of vertices from (b) that reasonably agree with Method 2. (d) Final result of our hybrid method (after repeated morphing).  
(b)

![](images/2c33101f08a6b24b4570c6c8ce70b637645a33402babd86e243e1bc141b420be.jpg)  
(c)

![](images/b5758d531a57d9175d3bbf1a32f995d94a01ddbcdf2b344cc1754c20fcea5483.jpg)  
(d)

# 6 EXPERIMENTS

Dataset Generation: Our cloth dataset consists of T-shirt meshes corresponding to about 10,000 poses for a particular body Wu et al. (2020a) (the same as in Jin et al. (2020)). We applied an 80-10-10 split to obtain training, validation, and test datasets, respectively. Table 1 compares the maximum  $\mathrm{L}^2$  and  $\mathrm{L}^\infty$  norms as compared to the ground truth for each of the three methods used to generate training examples. While Method 1 minimizes cloth vertex errors, the resulting  $d(\theta)$  contains high variance. Method 2 has significant vertex errors, but significantly lower variance in  $d(\theta)$ . We leverage the advantages of both using the hybrid method.

![](images/02f11dc62d266812010a3ba9abb4f60f795a8f76462aa45798fbe63e56abf9a4.jpg)  
Figure 8: Histogram of average vertex errors over every example in the test dataset.

Network Training: We adapt the network architecture

from Jin et al. (2020) for learning the displacements  $d(\theta)$ , i.e. by storing the displacements  $d(\theta)$  as pixel-based cloth images for the front and back sides of the T-shirt. Given joint transformation matrices of shape  $1 \times 1 \times 90$  for pose  $\theta$ , the network applies transpose convolution, batch normalization, and ReLU activation layers. The output of the network is  $128 \times 128 \times 6$ , where the first three dimensions represent the predicted displacements for the front side of the T-shirt, and the last three dimensions

Table 1: Dataset generation analysis (in cm). To measure variance in  $d(\theta)$ , we calculate the change in  $d(\theta)$  between any two vertices that share an edge in the triangle mesh, denoted by  $\Delta d(\theta)$ .  

<table><tr><td>Method</td><td>Max Vertex Error</td><td>Avg Vertex Error</td><td>Max ||Δd||</td><td>Avg ||Δd||</td></tr><tr><td>Method 1</td><td>8.9 × 10-6</td><td>9.8 × 10-7</td><td>136.5</td><td>9.35</td></tr><tr><td>Method 2</td><td>12.7</td><td>0.549</td><td>14.9</td><td>0.75</td></tr><tr><td>Hybrid Method</td><td>11.6</td><td>0.021</td><td>14.7</td><td>0.79</td></tr></table>

represent those for the back side. We train with an  $L^2$  loss on the difference between the ground truth displacements  $d(\theta)$  and network predictions  $\hat{d}(\theta)$ , using the Adam optimizer Kingma & Ba (2014) with a  $10^{-3}$  learning rate in PyTorch Paszke et al. (2017).

Network Inference: From the network output  $\hat{d}(\theta)$ , we define  $\hat{u}^m(\theta) = u^{m_o} + \hat{d}(\theta)$ , which is then embedded into the material space (T-posed) tetrahedral mesh and subsequently skinned to world space to obtain the cloth mesh prediction  $\hat{u}(\theta)$ . Table 2 summarizes the network inference results on the test dataset (not used in training). While all three methods detailed in Section 5 outperform the method proposed in Jin et al. (2020), the hybrid method achieved the lowest average vertex error and standard deviation. Figure 8 shows histograms of the average vertex error over all examples in the test dataset for the hybrid method and Jin et al. (2020). Note that the mean error of Jin et al. (2020) is five standard deviations above the mean of the hybrid method. Table 3 shows the errors in volume enclosed by the cloth (after capping the neck/sleeves/torso).

Table 2: Test dataset, average vertex errors (cm).  

<table><tr><td>Network</td><td>Vertex Error</td></tr><tr><td>Jin et al. (2020)</td><td>1.19 ± 0.20</td></tr><tr><td>KDSM (Method 1)</td><td>1.06 ± 0.63</td></tr><tr><td>KDSM (Method 2)</td><td>0.78 ± 0.17</td></tr><tr><td>KDSM (Hybrid)</td><td>0.52 ± 0.12</td></tr></table>

Table 3: Test dataset, average volume errors  $\left( {\mathrm{{cm}}}^{3}\right)$  .  

<table><tr><td>Network</td><td>Volume Error</td></tr><tr><td>Jin et al. (2020)</td><td>2991 ± 715</td></tr><tr><td>KDSM (Hybrid)</td><td>194 ± 161</td></tr></table>

There are significant visual improvements as well, see e.g. Figure 9. In addition, we evaluate the hybrid method network on a motion capture sequence from cmu and compare the inferred cloth to the results in Jin et al. (2020). The hybrid method is able to achieve greater temporal consistency; see the supplemental video. To demonstrate the efficacy of our approach in conjunction with other approaches, we apply texture sliding from Wu et al. (2020b) and the physical post process from Geng et al. (2020) to the results of the hybrid method network predictions, see Figure 10.

# 7 DISCUSSION

In this paper, we presented a framework for learning cloth deformation using a volumetric parameterization of the air surrounding the body. This parameterization was implicitly defined via a tetrahedral mesh that was skinned to follow the body as it animates, i.e. KDSM. A neural network was used to predict offsets in material space (the T-pose) such that the result well matched the ground truth after skinning the KDSM. The cloth predicted using the hybrid method detailed in Section 5 exhibits half the error as compared to state-of-the-art; in fact, the mean error from Jin et al. (2020) is five standard deviations above the mean resulting from our hybrid approach. Our results demonstrate that the KDSM is a promising foundation for learning virtual cloth and potentially for hair and solid/fluid interactions as well. Moreover, the KDSM should prove useful for treating cloth collisions, multiple garments, and interactions with external physics.

The KDSM intrinsically provides a more robust parameterization of three-dimensional space, since it contains a true extra degree of freedom as compared to the degenerate co-dimension one body surface. In particular, embedding cloth into a tetrahedral mesh has stability guarantees that do not exist when computing offsets from the body surface. See Figure 11. We believe that the significant decrease in network prediction errors is at least partially attributable to increased stability from using a volumetric parameterization.

![](images/b431d483621a955091e52fd30ee57c35ce9306ec77cf1a7ee41a1db5ef8f1b12.jpg)  
(a)  $u^{GT}$

![](images/f79fe3349c4a6b17a667ff878b246edf623b2ee3efe3605a4561922e487c06ae.jpg)

![](images/10fc1b856bf8404810af6db10c468c6852965132fc869c7aeb85c3366f34adcc.jpg)

![](images/d8ca18a1a449a24dc56c3a44d8751032a359e734ea1b0f2f051701b995ab223b.jpg)

![](images/c650974d5f651f16e7fa465efba9b553f6c8020d5f1d88ecb3a9e9cc09dd3258.jpg)  
(b)  $\hat{u}$

![](images/9a9af89953091f34f8bd769c5d4194608dc6b55b36cd23f0c07ff9b63b1e9310.jpg)

![](images/feb7e087a75f4f57f1e9990e02b8fbb8f729e2885e1d17b43a6e3e373063169a.jpg)

![](images/1dc9a92113a2c979ea65a3c2988875c3b1e3e397105017b3fd00c71d46f3ef02.jpg)

![](images/232c42ecbcb037cea8528e24765dae1ada4da357cc878471060aa251014ccb5b.jpg)  
(c) Jin et al.

![](images/fdd67aa938bf6c1c08c380b900ccf4e5eb1295ec3338e25f9d2f020dc964718b.jpg)  
Figure 9: Test dataset example predictions (b) compared to the ground truth cloth in (a) and the results from Jin et al. (2020) in (c). Regularization can smooth the body surface offsets predicted using Jin et al. (2020) and as such reveals the underlying body shape, e.g. the belly button (indicated with a red square).

![](images/d31f259d1bd6e7252922fa71e6533727a602636c471748d43829305ac40e064b.jpg)

![](images/3f81c85a1d907599c9839b02d961bcff854846164139616bc3c86981cceb8439.jpg)

![](images/b1237feebfd6d08a91765a0614069d31659eabe7962ac84d891ad9b03b44b357.jpg)  
(a)

![](images/5e9ef94edfa4d6f8b92c4c2ed109c223f4733d139f31c8fa9c308e4efb41dc97.jpg)  
(b)

![](images/e7b7a79a5aca16395ffec8f18940885af158e5e8f7ef546fa1808f4b32004002.jpg)  
Figure 10: Given the hybrid method network prediction in (a), we apply texture sliding from Wu et al. (2020b) and the physics postprocess from Geng et al. (2020) as shown in (b), compared to the ground truth (c). The shown example is the same as in Figure 14 of Wu et al. (2020b).  
(c)

![](images/ea80442bb57e46cdd92f59851bb4ce7c3ce704457051ca4e024f6d906e8bd4fc.jpg)  
(a)

![](images/614f927c3f2c54f90e1d11f0121bca3e9dd924d7a717bceef6db1567e6054fed.jpg)  
(b)  
Figure 11: (a) Embedding cloth in a tetrahedral mesh guarantees that each transformed vertex will remain inside and thus be bounded by the displacement of its parent tetrahedron. (b) However, no such bounds exist when the cloth is defined via UVN offsets from the body surface, since angle perturbations of the surface cause the cloth to move along an arclength  $C = \psi r$  where even small  $\psi$  can lead to large  $C$  for large enough  $r$ .

# REFERENCES

Cmu graphics lab motion capture database. http://mocap.cs.cmu.edu/.  
Thiemo Alldieck, Marcus Magnor, Weipeng Xu, Christian Theobalt, and Gerard Pons-Moll. Detailed human avatars from monocular video. In 2018 International Conference on 3D Vision (3DV), pp. 98-109. IEEE, 2018a.  
Thiemo Alldieck, Marcus Magnor, Weipeng Xu, Christian Theobalt, and Gerard Pons-Moll. Video based reconstruction of 3d people models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8387-8397, 2018b.  
Thiemo Alldieck, Marcus Magnor, Bharat Lal Bhatnagar, Christian Theobalt, and Gerard Pons-Moll. Learning to reconstruct people in clothing from a single rgb camera. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1175-1186, 2019a.  
Thiemo Alldieck, Gerard Pons-Moll, Christian Theobalt, and Marcus Magnor. Tex2shape: Detailed full human body geometry from a single image. In Proceedings of the International Conference on Computer Vision (ICCV). IEEE, 2019b.  
Dragomir Anguelov, Praveen Srinivasan, Daphne Koller, Sebastian Thrun, Jim Rodgers, and James Davis. Scape: shape completion and animation of people. In ACM transactions on graphics (TOG), volume 24, pp. 408-416. ACM, 2005.  
David Baraff and Andrew Witkin. Large steps in cloth simulation. In Proceedings of the 25th annual conference on Computer graphics and interactive techniques, pp. 43-54, 1998.  
David Baraff, Andrew Witkin, and Michael Kass. Untangling cloth. In ACM SIGGRAPH 2003 Papers, SIGGRAPH '03, pp. 862-870, New York, NY, USA, 2003. ACM. ISBN 1-58113-709-5.  
Gill Barequet, Bernard Chazelle, Leonidas J Guibas, Joseph SB Mitchell, and Ayellet Tal. Boxtree: A hierarchical representation for surfaces in 3d. In Computer Graphics Forum, volume 15, pp. 387-396. Wiley Online Library, 1996.  
Thomas Bellotti and Maxime Theillard. A coupled level-set and reference map method for interface representation with applications to two-phase flows simulation. Journal of Computational Physics, 392:266-290, 2019.  
Bharat Lal Bhatnagar, Garvita Tiwari, Christian Theobalt, and Gerard Pons-Moll. Multi-garment net: Learning to dress 3d people from images. In Proceedings of the IEEE International Conference on Computer Vision, pp. 5420-5430, 2019.  
Javier Bonet and Richard D Wood. Nonlinear continuum mechanics for finite element analysis. Cambridge university press, 1997.  
R. Bridson, S. Marino, and R. Fedkiw. Simulation of clothing with folds and wrinkles. In Proceedings of the 2003 ACM SIGGRAPH/Eurographics Symposium on Computer Animation, SCA '03, pp. 28-36, Aire-la-Ville, Switzerland, Switzerland, 2003. Eurographics Association. ISBN 1-58113-659-5.  
Robert Bridson, Ronald Fedkiw, and John Anderson. Robust treatment of collisions, contact and friction for cloth animation. ACM Trans. Graph., 21(3):594-603, July 2002. ISSN 0730-0301.  
Matthew Cong, Michael Bao, Jane L. E, Kiran S. Bhat, and Ronald Fedkiw. Fully automatic generation of anatomical face simulation models. In Proceedings of the 14th ACM SIGGRAPH / Eurographics Symposium on Computer Animation, SCA '15, pp. 175-183, New York, NY, USA, 2015. Association for Computing Machinery.  
Edilson De Aguiar, Leonid Sigal, Adrien Treuille, and Jessica K Hodgins. Stable spaces for real-time clothing. ACM Transactions on Graphics (TOG), 29(4):1-9, 2010.  
Zhenglin Geng, Daniel Johnson, and Ronald Fedkiw. Coercing machine learning to output physically accurate results. Journal of Computational Physics, 406:109099, 2020.

Stefan Gottschalk, Ming C Lin, and Dinesh Manocha. Obbtree: A hierarchical structure for rapid interference detection. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pp. 171-180, 1996.  
Eitan Grinspun, Anil N Hirani, Mathieu Desbrun, and Peter Schroder. Discrete shells. In Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer animation, pp. 62-67. Eurographics Association, 2003.  
Peng Guan, Loretta Reiss, David A Hirshberg, Alexander Weiss, and Michael J Black. Drape: Dressing any person. ACM Transactions on Graphics (TOG), 31(4):1-10, 2012.  
Erhan Gundogdu, Victor Constantin, Amrollah Seifoddini, Minh Dang, Mathieu Salzmann, and Pascal Fua. Garnet: A two-stream network for fast and accurate 3d cloth draping. In Proceedings of the IEEE International Conference on Computer Vision, pp. 8739-8748, 2019.  
Marc Habermann, Weipeng Xu, Michael Zollhoefer, Gerard Pons-Moll, and Christian Theobalt. Livecap: Real-time human performance capture from monocular video. ACM Transactions on Graphics (TOG), 38(2):14, 2019.  
Fabian Hahn, Bernhard Thomaszewski, Stelian Coros, Robert W Sumner, Forrester Cole, Mark Meyer, Tony DeRose, and Markus Gross. Subspace clothing simulation using adaptive bases. ACM Transactions on Graphics (TOG), 33(4):1-9, 2014.  
James K Hahn. Realistic animation of rigid bodies. Acm Siggraph Computer Graphics, 22(4): 299-308, 1988.  
Geoffrey Irving, Joseph Teran, and Ronald Fedkiw. Invertible finite elements for robust simulation of large deformation. In Proceedings of the 2004 ACM SIGGRAPH/Eurographics symposium on Computer animation, pp. 131-140, 2004.  
Alec Jacobson and Olga Sorkine. Stretchable and twistable bones for skeletal shape deformation. In Proceedings of the 2011 SIGGRAPH Asia Conference, pp. 1-8, 2011.  
Ning Jin, Yilin Zhu, Zhenglin Geng, and Ronald Fedkiw. A pixel-based framework for data-driven clothing. In Proceedings of the 19th ACM SIGGRAPH / Eurographics Symposium on Computer Animation, volume 39. Association for Computing Machinery, 2020.  
Ladislav Kavan and Jiří Žára. Spherical blend skinning: a real-time deformation of articulated models. In Proceedings of the 2005 symposium on Interactive 3D graphics and games, pp. 9-16, 2005.  
Ladislav Kavan, Steven Collins, Jiří Žára, and Carol O'Sullivan. Skinning with dual quaternions. In Proceedings of the 2007 symposium on Interactive 3D graphics and games, pp. 39-46, 2007.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Zorah Lahner, Daniel Cremers, and Tony Tung. Deepwrinkles: Accurate and realistic clothing modeling. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 667-684, 2018.  
Jeff Lander. Skin them bones: Game programming for the web generation. Game Developer Magazine, 5(1):10-18, 1998.  
Binh Huy Le and Jessica K Hodgins. Real-time skeletal skinning with optimized centers of rotation. ACM Transactions on Graphics (TOG), 35(4):1-10, 2016.  
Minjae Lee, David Hyde, Michael Bao, and Ronald Fedkiw. A skinned tetrahedral mesh for hair animation and hair-water interaction. IEEE transactions on visualization and computer graphics, 25(3):1449-1459, 2018.  
Minjae Lee, David Hyde, Kevin Li, and Ronald Fedkiw. A robust volume conserving method for character-water interaction. In Proceedings of the 18th annual ACM SIGGRAPH/Eurographics Symposium on Computer Animation, pp. 1-12, 2019.

John P Lewis, Matt Cordner, and Nickson Fong. Pose space deformation: a unified approach to shape interpolation and skeleton-driven deformation. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pp. 165-172, 2000.  
Ming Lin and Stefan Gottschalk. Collision detection between geometric models: A survey. In Proc. of IMA conference on mathematics of surfaces, volume 1, pp. 602-608, 1998.  
Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multi-person linear model. ACM transactions on graphics (TOG), 34(6):248, 2015.  
Nadia Magnenat-Thalmann, Richard Laperrire, and Daniel Thalmann. Joint-dependent local deformations for hand animation and object grasping. In Proceedings on Graphics Interface '88, pp. 26-33, 1988.  
Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.  
Neil Molino, Robert Bridson, Joseph Teran, and Ronald Fedkiw. A crystalline, red green strategy for meshing highly deformable objects with tetrahedra. In IMR, pp. 103-114. CiteSeer, 2003.  
Matthias Müller, Nuttapong Chentanez, Tae-Yong Kim, and Miles Macklin. Air meshes for robust collision handling. ACM Transactions on Graphics (TOG), 34(4):1-9, 2015.  
Ryota Natsume, Shunsuke Saito, Zeng Huang, Weikai Chen, Chongyang Ma, Hao Li, and Shigeo Morishima. Siclope: Silhouette-based clothed people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4480-4490, 2019.  
Alexandros Neophytou and Adrian Hilton. A layered model of human body and garment deformation. In 2014 2nd International Conference on 3D Vision, volume 1, pp. 171-178. IEEE, 2014.  
Hayato Onizuka, Zehra Hayirci, Diego Thomas, Akihiro Sugimoto, Hideaki Uchiyama, and Rin-ichiro Taniguchi. Tetratsdf: 3d human reconstruction from a single image with a tetrahedral outer shell. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6011–6020, 2020.  
Stanley Osher and Ronald Fedkiw. Level Set Methods and Dynamic Implicit Surfaces. Springer, New York, 2002.  
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Chaitanya Patel, Zhouyingcheng Liao, and Gerard Pons-Moll. Tailornet: Predicting clothing in 3d as a function of human pose, shape and garment style. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7365-7375, 2020.  
Gerard Pons-Moll, Sergi Pujades, Sonny Hu, and Michael J Black. Clothcap: Seamless 4d clothing capture and retargeting. ACM Transactions on Graphics (TOG), 36(4):1-15, 2017.  
Tiberiu Popa, Quan Zhou, Derek Bradley, Vladislav Kraevoy, Hongbo Fu, Alla Sheffer, and Wolfgang Heidrich. Wrinkling captured garments using space-time data-driven deformation. In Computer Graphics Forum, volume 28, pp. 427-435. Wiley Online Library, 2009.  
Nadia Robertini, Edilson De Aguiar, Thomas Helten, and Christian Theobalt. Efficient multi-view performance capture of fine-scale surface detail. In 2014 2nd International Conference on 3D Vision, volume 1, pp. 5-12. IEEE, 2014.  
Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In Proceedings of the International Conference on Computer Vision (ICCV). IEEE, 2019.  
Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 84-93, 2020.

Igor Santesteban, Miguel A Otaduy, and Dan Casas. Learning-based animation of clothing for virtual try-on. In Computer Graphics Forum, volume 38, pp. 355-366. Wiley Online Library, 2019.  
Andrew Selle, Jonathan Su, Geoffrey Irving, and Ronald Fedkiw. Robust high-resolution cloth using parallelism, history-based collisions, and accurate friction. IEEE transactions on visualization and computer graphics, 15(2):339-350, 2008.  
Eftychios Sifakis, Sebastian Marino, and Joseph Teran. Globally coupled collision handling using volume preserving impulses. In Proceedings of the 2008 ACM SIGGRAPH/Eurographics Symposium on Computer Animation, pp. 147-153, 2008.  
Joseph Teran, Neil Molino, Ronald Fedkiw, and Robert Bridson. Adaptive physics based tetrahedral mesh generation using level sets. Engineering with computers, 21(1):2-18, 2005a.  
Joseph Teran, Eftychios Sifakis, Geoffrey Irving, and Ronald Fedkiw. Robust quasistatic finite elements and flesh simulation. In Proceedings of the 2005 ACM SIGGRAPH/Eurographics symposium on Computer animation, pp. 181-190, 2005b.  
Pascal Volino and Nadia Magnenat-Thalmann. Animating complex hairstyles in real-time. In Proceedings of the ACM symposium on Virtual reality software and technology, pp. 41-48, 2004.  
Pascal Volino and Nadia Magnenat-Thalmann. Real-time animation of complex hairstyles. IEEE Transactions on Visualization and Computer Graphics, 12(2):131-142, 2006.  
Robert Webb and Mike Gigante. Using dynamic bounding volume hierarchies to improve efficiency of rigid body simulations. In Visual Computing, pp. 825-842. Springer, 1992.  
Jane Wu, Zhenglin Geng, Ning Jin, and Ronald Fedkiw. Physbam virtual cloth dataset, 2020a. http://physbam.stanford.edu.  
Jane Wu, Yongxu Jin, Zhenglin Geng, Hui Zhou, and Ronald Fedkiw. Recovering geometric information with learned texture perturbations. arXiv preprint arXiv:2001.07253, 2020b.  
Kui Wu and Cem Yuksel. Real-time hair mesh simulation. In Proceedings of the 20th ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games, pp. 59-64, 2016.  
Weipeng Xu, Avishek Chatterjee, Michael Zollhöfer, Helge Rhodin, Dushyant Mehta, Hans-Peter Seidel, and Christian Theobalt. Monoperfcap: Human performance capture from monocular video. ACM Transactions on Graphics (ToG), 37(2):27, 2018.  
Jinlong Yang, Jean-Sébastien Franco, Franck Hétroy-Wheeler, and Stefanie Wuhrer. Analyzing clothing layer deformation statistics of 3d human motions. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 237-253, 2018.  
Tao Yu, Zerong Zheng, Yuan Zhong, Jianhui Zhao, Qionghai Dai, Gerard Pons-Moll, and Yebin Liu. Simulcap: Single-view human performance capture with cloth simulation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.