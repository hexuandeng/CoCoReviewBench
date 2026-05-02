# NEUROGENESIS-INSPIRED DICTIONARY LEARNING: ONLINE MODEL ADPTION IN A CHANGING WORLD

Sahil Garg

The Department of Computer Science, University of Southern California, Los Angeles, CA USA sahilgar@usc.edu

Irina Rish, Guillermo Cecchi, Aurelie Lozano

IBM Thomas J. Watson Research Center, Yorktown Heights, NY USA

{rish, gcecchi, aclozano}@us.ibm.com

# ABSTRACT

In this paper, we focus on online representation learning in non-stationary environments which may require continuous adaptation of model's architecture. We propose a novel online dictionary-learning (sparse-coding) framework which incorporates the addition and deletion of hidden units (dictionary elements), and is inspired by the adult neurogenesis phenomenon in the dentate gyrus of the hippocampus, known to be associated with improved cognitive function and adaptation to new environments. In the online learning setting, where new input instances arrive sequentially in batches, the "neuronal birth" is implemented by adding new units with random initial weights (random dictionary elements); the number of new units is determined by the current performance (representation error) of the dictionary, higher error causing an increase in the birth rate. "Neuronal death" is implemented by imposing  $l_{1} / l_{2}$ -regularization (group sparsity) on the dictionary within the block-coordinate descent optimization at each iteration of our online alternating minimization scheme, which iterates between the code and dictionary updates. Finally, hidden unit connectivity adaptation is facilitated by introducing sparsity in dictionary elements. Our empirical evaluation on several real-life datasets (images and language) as well as on synthetic data demonstrates that the proposed approach can considerably outperform the state-of-art fixed-size (non-adaptive) online sparse coding of Mairal et al. (2009) in the presence of non-stationary data. Moreover, we identify certain properties of the data (e.g., sparse inputs with nearly non-overlapping supports) and of the model (e.g., dictionary sparsity) associated with such improvements.

# 1 INTRODUCTION

The ability to adapt to a changing environment is essential for successful functioning in both natural and artificial intelligent systems. In human brains, adaptation is achieved via neuroplasticity, which takes different forms, including synaptic plasticity, i.e. changing connectivity strength among neurons, and neurogenesis, i.e. the birth and maturation of new neurons (accompanied with the death of some new or old neurons). Particularly, adult neurogenesis (Kempermann, 2006) (i.e., neurogenesis in the adult brain) in the dentate gyrus of the hippocampus is associated with improved cognitive functions such as pattern separation (Sahay et al., 2011), and is often implicated as a "candidate mechanism for the specific dynamic and flexible aspects of learning" (Stuchlik, 2014).

In the machine-learning context, synaptic plasticity is analogous to parameter tuning (e.g., learning neural net weights), while neurogenesis can be viewed as an online model selection via addition (and deletion) of hidden units in specific hidden-variable models used for representation learning (where hidden variables represent extracted features), from linear and nonlinear component analysis methods such as PCA, ICA, sparse coding (dictionary learning), nonlinear autoencoders, to deep neural nets and general hidden-factor probabilistic models. However, optimal model selection in large-scale hidden-variable models (e.g., adjusting the number of layers, hidden units, and their

connectivity), is intractable due to enormous search space size. Growing a model gradually can be a more feasible alternative; after all, every brain's "architecture" development process starts with a single cell. Furthermore, the process of adapting the model's architecture to dynamically changing environments is necessary for achieving lifelong, continual learning. Finally, an online approach to dynamically expanding and contracting model's architecture can serve as a potential alternative to currently popular network compression (distillation) approaches (Hinton et al., 2015; Srivastava et al., 2014; Ba & Caruana, 2014; Bucilu et al., 2006), which have to first select an architecture and train a large-scale model, such as a deep neural network with millions of parameters, only to compress it later to a smaller, simpler model with similarly good performance.

In this paper, we focus on dictionary learning, a.k.a. sparse coding (Olshausen & Field, 1997; Kreutz-Delgado et al., 2003; Aharon et al., 2006; Lee et al., 2006) – a representation learning approach which finds a set of basis vectors (atoms, or dictionary elements) and representations (encodings) of the input samples as sparse linear combinations of those elements<sup>1</sup>. More specifically, our approach builds upon the computationally efficient online dictionary-learning method of Mairal et al. (2009), where the data samples are processed sequentially, one at a time (or in small batches). Online approaches are particularly important in large-scale applications with millions of potential training samples, where off-line learning can be infeasible; furthermore, online approaches are a natural choice for building systems capable of continual, lifelong learning.

Herein, we propose a novel online dictionary learning approach inspired by adult neurogenesis, which extends the state-of-art method of Mairal et al. (2009) to nonstationary environments by incorporating online model adaption, i.e. the addition and deletion of dictionary elements (i.e., hidden units) in response to the dynamically changing properties of the input data. More specifically, at each iteration of online learning (i.e., for every batch of data samples), we add a group of random dictionary elements (modeling neuronal birth), where the group size depends on the current representation error, i.e. the mismatch between the new input samples and their approximation based on the current dictionary: higher error triggers more neurogenesis. The neuronal death, which involves removing "useless" dictionary elements, is implemented as an  $l_{1} / l_{2}$  group-sparsity regularization; this step is essential in neurogenesis-inspired learning, since it reduces a potentially uncontrolled growth of the dictionary, and helps to avoid overfitting (note that neuronal death is also a natural part of the adult neurogenesis process, where neuronal survival depends on multiple factors, including the complexity of a learning environment (Kempermann, 2006)). Moreover, we introduce sparsity in dictionary elements, which reflects sparse connectivity between hidden units/neurons and their inputs; this is a more biologically plausible assumption than the fully-connected architecture of standard dictionary learning, and it also works better in our experiments. Thus, adaptation in our model involves not only the addition/deletion of the elements, but adapting their connectivity as well.

We demonstrate on both simulated and two real-life data (natural images and language processing) that, in the presence of non-stationary input, our approach can significantly outperform the non-adaptive, fixed-dictionary-size online method of Mairal et al. (2009). Moreover, we identify certain data properties and parameter settings associated with such improvements. Finally, we demonstrate that the novel approach not only improves the representation accuracy, but also can boost the classification accuracy based on the extracted features.

Note that, though the group-sparsity constraint enforcing deletion of some dictionary elements was already introduced before in the group sparse coding by Bengio et al. (2009), it was only implemented and tested in the off-line rather than online setting, and, most importantly, it was not accompanied by the neurogenesis. On the other hand, while some prior work considered online node addition in hidden-variable models (specifically, in neural networks, from cascade correlations (Fahlman & Lebiere, 1989) to the recent work by Draelos et al. (2016)), it did not incorporate any model pruning to balance the model expansion. To the best of our knowledge, our work is the first to evaluate the interplay between the birth and death of hidden units, especially in the context of online dictionary learning.

# 2 BACKGROUND ON DICTIONARY LEARNING

Traditional off-line dictionary learning (Olshausen & Field, 1997; Aharon et al., 2006; Lee et al., 2006) aims at finding a dictionary  $D \in \mathbb{R}^{m \times k}$ , which allows for an accurate representation of a training data set  $X = \{\pmb{x}_1, \dots, \pmb{x}_n \in \mathbb{R}^m\}$ , where each sample  $\pmb{x}_i$  is approximated by a linear combination  $\pmb{x}_i \approx D\pmb{\alpha}_i$  of the columns of  $D$ , called dictionary elements  $\{\pmb{d}_1, \dots, \pmb{d}_k \in \mathbb{R}^m\}$ . Here  $\pmb{\alpha}_i$  is the encoding (code vector, or simply code) of  $\pmb{x}_i$  in the dictionary. Dictionary learning is also referred to as sparse coding, since it is assumed that the code vectors are sparse, i.e. have a relatively small number of nonzeros; the problem is formulated as minimizing the objective

$$
f _ {n} (D) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1}{2} \left\| \boldsymbol {x} _ {i} - \boldsymbol {D} \boldsymbol {\alpha} _ {i} \right\| _ {2} ^ {2} + \lambda_ {c} \| \boldsymbol {\alpha} _ {i} \| _ {1} \tag {1}
$$

where the first term is the mean square error loss incurred due to approximating the input samples by their representations in the dictionary, and the second term is the  $l_{1}$ -regularization which enforces the codes to be sparse. The joint minimization of  $f_{n}(D)$  with respect to the dictionary and codes is non-convex; thus, a common approach is alternating minimization involving convex subproblems of finding optimal codes while fixing a dictionary, and vice versa.

However, the classical dictionary learning does not scale to very large datasets; moreover, it is not immediately applicable to online learning from a continuous stream of data. The online dictionary learning (ODL) method proposed by Mairal et al. (2009) overcomes both of these limitations, and serves as a basis for our proposed approach, presented in Alg. 1 in the next section. While the highlighted lines in Alg. 1 represent our extension of ODL, the non-highlighted ones are common to both approaches, and are discussed first. The algorithms start with some dictionary  $D^0$ , e.g. a randomly initialized one (other approaches include using some of the inputs as dictionary elements (Mairal et al., 2010; Bengio et al., 2009)). At each iteration  $t$ , both online approaches consider the next input sample  $x_{t}$  (more generally, a batch of samples) as in the step 3 of Alg. 1 and compute its sparse code  $\alpha_{t}$  by solving the LASSO (Tibshirani, 1996) problem (the step 4 in Alg. 1), with respect to the current dictionary. In Alg. 1, we simply use  $D$  instead of  $D^{(t)}$  to simplify the notation. Next, the standard ODL algorithm computes the dictionary update,  $D^{(t)}$ , by optimizing the surrogate objective function  $\hat{f}_t(D)$  which is defined just as the original objective in eq. (1), for  $n = t$ , but with one important difference: unlike the original objective, where each code  $\alpha_{i}$  for sample  $x_{i}$  is computed with respect to the same dictionary  $D$ , the surrogate function includes the codes  $\alpha_{1},\alpha_{2},\dots ,\alpha_{t}$  computed at the previous iterations, using the dictionaries  $D^{(0)},\ldots ,D^{(t - 1)}$ , respectively; in other words, it does not recompute the codes for previously seen samples after each dictionary update. This speeds up the learning without worsening the (asymptotic) performance, since the surrogate objective converges to the original one in (1), under certain assumptions, including data stationarity (Mairal et al., 2009). Note that, in order to prevent the dictionary entries from growing arbitrarily large, Mairal et al. (2009; 2010) impose the norm constraint, i.e. keep the columns of  $D$  within the convex set  $\mathcal{C} = \{D\in \mathbb{R}^{m\times k}\quad s.t.\quad \forall j\ d_j^T u_j\leq 1\}$ . Then the dictionary update step computes  $D^{(t)} = \arg \min_{D\in \mathcal{C}}\hat{f}_t(D)$ , ignoring  $l_{1}$ -regularizer over the code which is fixed at this step, as

$$
\arg \min  _ {\boldsymbol {D} \in \mathcal {C}} \frac {1}{t} \sum_ {i = 1} ^ {t} \frac {1}{2} \left\| \boldsymbol {x} _ {i} - \boldsymbol {D} \boldsymbol {\alpha} _ {i} \right\| _ {2} ^ {2} = \arg \min  _ {\boldsymbol {D} \in \mathcal {C}} \frac {1}{2} T r (\boldsymbol {D} ^ {T} \boldsymbol {D} \boldsymbol {A}) - T r (\boldsymbol {D} ^ {T} \boldsymbol {B}), \tag {2}
$$

where  $A = \sum_{i=1}^{t} \alpha_i \alpha_i^T$  and  $B = \sum_{i=1}^{t} x_i \alpha_i^T$  are the "bookkeeping" matrices (we also call them "memories" of the model), compactly representing the input samples and encoding history. At each iteration, once the new input sample  $\pmb{x}_i$  is encoded, the matrices are updated as  $A \gets A + \alpha_t \alpha_t^T$  and  $B \gets B + x_t \alpha_t^T$  (see the step 11 of Alg. 1). In (Mairal et al., 2009; 2010), a block coordinate descent is used to optimize the convex objective in eq. 2; it iterates over the dictionary elements in a fixed sequence, optimizing each while keeping the others fixed as shown in eq. (3) (essentially, the steps 14 and 17 in Alg. 1; the only difference is that our approach will transform  $\pmb{u}_j$  into  $\pmb{w}_j$  in order to impose additional regularizer before computing step 17), until convergence.

$$
\boldsymbol {u} _ {j} \leftarrow \frac {\boldsymbol {b} _ {j} - \sum_ {k \neq j} \boldsymbol {d} _ {k} a _ {j k}}{a _ {j j}}; \quad \boldsymbol {d} _ {j} \leftarrow \frac {\boldsymbol {u} _ {j}}{\max  (1 , | | \boldsymbol {u} _ {j} | | _ {2})} \tag {3}
$$

Herein, when the off-diagonal entries  $\mathbf{a}_{jk}$  in  $\mathbf{A}$  are as large as the diagonal  $a_{jj}$ , the dictionary elements get "tied" to each other, playing complementary roles in the dictionary, thereby constraining the updates of each other.

It is important to note that, for the experiment settings where we consider dictionary elements to be sparse in our algorithm NODL (discussed next in Sec. 3), we will actually use as a baseline algorithm a modified version of the fixed-size ODL, which allows for sparse dictionary elements, i.e. includes the sparsification step 15 in Alg. 1, thus optimizing the following objective in dictionary update step instead of the one in eq. (2):

$$
\arg \min  _ {\boldsymbol {D} \in \mathcal {C}} \frac {1}{t} \sum_ {i = 1} ^ {t} \frac {1}{2} \left\| \boldsymbol {x} _ {i} - \boldsymbol {D} \boldsymbol {\alpha} _ {i} \right\| _ {2} ^ {2} + \sum_ {j} \lambda_ {j} \| \boldsymbol {d} _ {j} \| _ {1}. \tag {4}
$$

From now on, ODL will refer to the above extended version of the fixed-size method of Mairal et al. (2009) wherever we have sparsity in dictionary elements (otherwise, the standard method of Mairal et al. (2009) is the baseline); in our experiments, dictionary sparsity of both the baseline and the proposed method (discussed in the next section) will be matched. Note that Mairal et al. (2010) mention that the convergence guaranties for ODL hold even with the sparsity constraints on dictionary elements.

# 3 OUR APPROACH: NEUROGENIC ONLINE DICTIONARY LEARNING (NODL)

Our objective is to extend online dictionary learning, designed for stationary input distributions, to an even more adaptive framework which could effectively handle nonstationary data, learning to represent new types of data without forgetting how to represent the old ones. Towards this end, we propose a novel algorithm, called Neurogenetic Online Dictionary Learning (see Alg. 1), which can flexibly extend and reduce a dictionary in response to the changes in an input distribution (and the inherent representation complexity of data). The key extensions w.r.t. to the non-adaptive, fixed-dictionary-size algorithm of Mairal et al. (2009), are highlighted in Alg. 1; the two parts involve (1) neurogenesis, i.e. the addition of dictionary elements (hidden units, or "neurons") and (2) the death of old and/or new elements which are "less useful" than other elements for the task of data reconstruction.

At each iteration in Alg. 1, the next batch of samples is received and the corresponding codes, in the dictionary, are computed; next, we add  $k_{n}$  new dictionary elements sampled at random from  $\mathbb{R}^{m}$  (i.e.,  $k_{n}$  random linear projections of the input sample). The choice of the parameter  $k_{n}$  is important; one approach is to tune it (e.g., by cross-validation), while another is to adjust it dynamically, based on the dictionary performance: e.g., if the environment is changing, the old dictionary may not be able to represent the new input well, leading to decline in the representation accuracy, which triggers neurogenesis. Herein, we use as the performance measure the Pearson correlation between a new sample and its representation in the current dictionary  $r(\pmb{x}_t, \pmb{D}^{(t-1)}\pmb{\alpha}_t)$ , i.e. denoted as  $p_c(\pmb{x}_t, \pmb{D}^{(t-1)}, \pmb{\alpha}_t)$  (for a batch of data, the average over  $p_c(\cdot)$  is taken). If it drops below a certain pre-specified threshold  $\gamma$  (where  $0 \ll \gamma \leq 1$ ), the neurogenesis is triggered (the step 5 in Alg. 1). The number  $k_{n}$  of new dictionary elements is proportional to the error  $1 - p_c(\cdot)$ , so that worse performance will trigger more neurogenesis, and vice versa; the maximum number of new elements is bounded by  $c_k$  (the step 6 in Alg. 1). We refer to this approach as conditional neurogenesis as it involves the conditional birth of new elements. Next,  $k_{n}$  random elements are generated and added to the current dictionary (the step 7), and the memory matrices  $A, B$  are updated, respectively, to account for larger dictionary (the step 8). Finally, the sparse code is recomputed for  $x_t$  (or, all the samples in the current batch) with respect to the extended dictionary (the step 9).

The next step is the dictionary update, which uses, similarly to the standard online dictionary learning, the block-coordinate descent approach. However, the objective function includes additional regularization terms, as compared to (2):

$$
\boldsymbol {D} ^ {(t)} = \arg \min  _ {\boldsymbol {D} \in \mathcal {C}} \frac {1}{t} \sum_ {i = 1} ^ {t} \frac {1}{2} \left\| \boldsymbol {x} _ {i} - \boldsymbol {D} \boldsymbol {\alpha} _ {i} \right\| _ {2} ^ {2} + \lambda_ {g} \sum_ {j} \left\| \boldsymbol {d} _ {j} \right\| _ {2} + \sum_ {j} \lambda_ {j} \left\| \boldsymbol {d} _ {j} \right\| _ {1}. \tag {5}
$$

The first term is the standard reconstruction error, as before. The second term,  $l_{1} / l_{2}$ -regularization, promotes group sparsity over the dictionary entries, where each group corresponds to a column, i.e. a dictionary element. The group-sparsity (Yuan & Lin, 2006) regularizer causes some columns in  $D$  to be set to zero (i.e. the columns less useful for accurate data representation), thus effectively

Algorithm 1 Neurogenetic Online Dictionary Learning (NODL)  
Require: Data stream  $x_{1},x_{2},\dots ,x_{n}\in \mathbb{R}^{m}$  ; initial dictionary  $D\in \mathbb{R}^{m\times k}$  ; conditional neurogenesis threshold,  $\gamma$  ; max number of new elements added per data batch,  $c_{k}$  ; group sparsity regularization parameter,  $\lambda_{g}$  ; number of non-zeros in a dictionary element,  $\beta_{d}$  ; number of non-zeros in a code,  $\beta_{c}$    
1: Initialize:  $A\gets 0,B\gets 0$  % reset the `memory' assuming single data in a batch, for the simpler exposition   
2: for  $t = 1$  to n do   
3: Input  $x_{t}$  % representing the  $t_\mathrm{th}$  batch of data % Sparse coding of data:   
4:  $\alpha_{t} = \arg_{\alpha \in \mathbb{R}^{k}}\min \frac{1}{2} ||\pmb{x}_{t} - D\pmb{\alpha}||_{2}^{2} + \lambda_{c}||\pmb{\alpha}||_{1}$  %  $\lambda_{c}$  tuned to have  $\beta_{c}$  non-zeros in  $\alpha_{t}$    
% Conditional neurogenesis: if accuracy below threshold, add more elements (should not be more than the number of data in a batch if  $pc(x_t,D,\alpha_t)\leq \gamma$  then   
6:  $k_{n} = (1 - p_{c}(x_{t},D,\alpha_{t}))c_{k}$  % the count of the births of neurons   
7:  $D_{n}\gets$  initializeRand  $(k_n)$ $D\gets [D\quad D_n]$    
8:  $A\gets \left[ \begin{array}{cc}A & 0\\ 0 & 0 \end{array} \right],B\gets [B\quad 0],k\gets k + k_n$  % Repeat sparse coding, now including the new dictionary elements   
9:  $\alpha_{t} = \arg_{\alpha \in \mathbb{R}^{k}}\min \frac{1}{2} ||\pmb{x}_{t} - D\pmb{\alpha}||_{2}^{2} + \lambda_{c}||\pmb{\alpha}||_{1}$    
10: end if % End of neurogenesis   
% `Memory' update:   
11:  $A\gets A + \alpha_{t}\alpha_{t}^{T}$ $B\gets B + x_{t}\alpha_{t}^{T}$    
% Dictionary update by block-coordinate descent with  $l_1 / l_2$  group sparsity   
repeat   
for  $j = 1$  to k do   
 $u_j\gets \frac{b_j - \sum_{k\neq j}d_ka_{jk}}{a_{jj}}$    
% Sparsifying elements (optional):   
 $v_{j}\gets Prox_{\lambda_{j}||.||_{1}}(u_{j}) = sgn(u_{j})(|u_{j}| - \lambda_{j})_{+},$  %  $\lambda_{j}$  tuned to get  $\beta_d$  non-zeros in  $v_{j}$    
% Killing useless elements with  $l_1 / l_2$  group sparsity   
 $w_j\gets v_j\left(1 - \frac{\lambda_g}{||v_j||_2}\right)_+$ $d_j\gets \frac{w_j}{\max(1,||w_j||_2)}$    
18: end for   
19: until convergence   
20: end for   
21: return D

eliminating the corresponding dictionary elements from the dictionary ("killing" the corresponding hidden units). As it was mentioned previously, Bengio et al. (2009) used the  $l_{1} / l_{2}$ -regularizer in dictionary learning, though not in online setting, and without neurogenesis.

Finally, the third term imposes  $l_{1}$ -regularization on dictionary elements thus promoting sparse dictionary, besides the sparse coding. Introducing sparsity in dictionary elements, corresponding to the sparse connectivity of hidden units in the neural net representation of a dictionary, is motivated by both their biological plausibility (neuronal connectivity tends to be rather sparse in multiple brain networks), and by the computational advantages this extra regularization can provide, as we observe later in experiments section (Sec. 4).

As in the original algorithm of Mairal et al. (2009), the above objective is optimized by the block-coordinate descent, where each block of variables corresponds to a dictionary element, i.e., a column in  $D$ ; the loop in steps 12-19 of the Alg. 1 iterates until convergence, defined by the magnitude of change between the two successive versions of the dictionary falling below some threshold. For each column update, the first and the last steps (the steps 14 and 17) are the same as in the original method of Mairal et al. (2009), while the two intermediate steps (the steps 15 and 16) are implementing additional regularization. Both steps 15 and 16 (sparsity and group sparsity regularization) are implemented using the standard proximal operators as described in Jenatton et al. (2011). Note that we actually use as input the desired number of non-zeros, and determine the corresponding sparsity parameter  $\lambda_{g}$ ,  $\lambda_{c}$  and  $\lambda_{j}$  using a binary search procedure (see Appendix).

Overall, the key features of our algorithm is the interplay of both the (conditional) birth and (group-sparsity) death of dictionary elements in an online setting.

# 4 EXPERIMENTS

We now evaluate empirically the proposed approach, NODL, against ODL, the standard (non-adaptive) online dictionary learning of Mairal et al. (2009). Moreover, in order to evaluate separately the effects of either only adding, or only deleting dictionary elements, we also evaluate two restricted versions of our method: NODL+ involves only addition but no deletion (equivalent to NODL with no group-sparsity, i.e.  $\lambda_{g} = 0$ ), and NODL- which, vice versa, involves deletion only but no addition (equivalent to NODL with the number of new elements  $c_{k} = 0$ ). The above algorithms are evaluated in a non-stationary setting, where a sequence of training samples from one environment (first domain) is followed by another sequence from a different environment (second domain), in order to test their ability to adapt to new environments without "forgetting" the previous ones.

# 4.1 REAL-LIFE IMAGES

Our first domain includes the images of Oxford buildings  $^{2}$  (urban environment), while the second uses a combination of images from Flowers  $^{3}$  and Animals  $^{4}$  image databases (natural environment); examples of both types of images are shown in Fig. 6(a) and 6(b). We converted the original color images into black&white format and compressed them to smaller sizes, 32x32 and 100x100. Note that, unlike (Mairal et al., 2009), we used full images rather than image patches as our inputs.

We selected 5700 images for training and another 5700 for testing; each subset contained 1900 images of each type (i.e., Oxford, Flowers, Animals). In the training phase, as mentioned above, each online dictionary learning algorithm receives a sequence of 1900 samples from the first, urban domain (Oxford), and then a sequence of 3800 samples from the second, natural domain (1900 Flowers and 1900 Animals, permuted randomly). At each iteration, a batch of 200 images is received as an input. (For comparison, Mairal et al. (2009) used a batch of size 256, though image patches rather than full images.) The following parameters are used by our algorithm: Pearson correlation threshold  $\gamma = 0.9$ , group sparsity parameter  $\lambda_{g} = 0.03$  and  $\lambda_{g} = 0.07$ , for  $32\times 32$  and  $100\times 100$  images, respectively. The upper bound on the number of new dictionary elements at each iteration is  $c_{k} = 50$ . (We observed that the results are only mildly sensitive to the specified parameter values.)

Once the training phase is completed, the resulting dictionary is evaluated on test images from both the first (urban) and the second (natural) domains; for the second domain, separate evaluation is performed for flowers and animals. First, we evaluate the reconstruction ability of the resulting dictionary  $D$ , comparing the actual inputs  $x$  versus approximations  $x^{*} = D\alpha$ , using the mean square error (MSE), Pearson correlation, and the Spearman correlation. We present the results for Pearson correlations between the actual and reconstructed inputs, since all the three metrics show consistent patterns (for completeness, MSE results are shown in Appendix). Moreover, we evaluate the dictionaries in a binary classification setting (e.g., flowers vs animals), using as features the codes of test samples in a given dictionary. Finally, we explored a wide range of sparsity parameters for both the codes and the dictionary elements.

Our key observations are that: (1) the proposed method frequently often outperforms (or is at least as good as) its competitors, on both the new data (adaptation) and the old ones (memory); (2) it is most beneficial when dictionary elements are sparse; (3) vice versa, when dictionary elements are dense, neurogenic approach matches the baseline, fixed-size dictionary learning. We now discuss the results in detail.

# Sparse Dictionary Elements

In Fig. 1, we present the results for sparse dictionaries, where each column (an element in the dictionary) has 5 nonzeros out of the 1024 dimensions; the codes are relatively dense, with at most 200 nonzeros out of  $k$  (the number of dictionary elements), and  $k$  ranging from 5 to 1000 (i.e. the codes are not sparse for  $k \leq 200$ ). Due to space limitations, we put in the Appendix (Sec. B.2) our results on a wider range of values for the dictionary and code sparsity (Fig. 11). In Fig. 1(a), we compare the dictionary size for different methods: the final dictionary size after completing the training phase (y-axis) is plotted against the initial dictionary size (x-axis). Obviously, the baseline (fixed-size) ODL method (magenta plot) keeps the size constant, deletion-only NODL- approach

![](images/b5f90e6f64298203df6a4f508ed096ac8898adf93e64189a0f4f5691f253a229.jpg)  
(a) Learned Dictionary Size

![](images/7b4d4db953059fac46adf2d7d2c2d498cadebd6de88a1c6210fb70e4e212546b.jpg)  
(b) 1st domain (Oxford)

![](images/46e5425117739a449696ce7a0166b1460ad6e23e42c403bac3e8ac2ef67ac33a.jpg)  
(c) 2nd domain (Flowers)

![](images/a71b3e7a0c9e641349470b3c7aec88c639a432085f9d75d435b107736cf0ac7e.jpg)  
Figure 1: Reconstruction accuracy of NODL and ODL on  $32 \times 32$  images (sparse dictionary).  
(a) 1st domain (Oxford)  
Figure 2: Reconstruction accuracy of NODL and ODL on  $100 \times 100$  images with sparse dictionary elements (50 non-zeros) and non-sparse codes.

![](images/b5dccb19a86034f6323522d6a53bc7ec6900ee3da64274ebff2eac6f0ffbf342.jpg)  
(b) 2nd domain (Flowers)

![](images/698cfe720b956b0079ca9907f39903d8ec640660a8070c60e11dfb5e2c434d4e.jpg)  
(c) Classification Error

reduces the initial size (red plot), and addition-only NODL+ increases the size (light-blue plot). However, the interplay between the addition and deletion in our NODL method (dark-blue) produces a more interesting behavior: it tends to adjust the representation complexity towards certain balanced range, i.e. very small initial dictionaries are expanded, while very large ones are, vice versa, reduced.

Our main results demonstrating the advantages of the proposed NODL method are shown next in Fig. 1(b) and Fig. 1(c), for the "old" (Oxford) and "new" (Flowers) environment (domain), respectively. (Very similar result are shown for Animals as well, in the Appendix). The x-axis shows the final dictionary size, and the y-axis is the reconstruction accuracy achieved by the trained dictionary on the test samples, measured by Pearson correlation between the actual and reconstructed data. NODL clearly outperforms the fixed-size ODL, especially on smaller dictionary sizes; remarkably, this happens on both domains, i.e. besides improved adaptation to the new data, NODL is also better at preserving the "memories" of the old data, without increasing the representation complexity, i.e. for the same dictionary size.

Interestingly, just deletion would not suffice, as deletion-only version, NODL-, is inferior to our NODL method. On the other hand, addition-only, or NODL+, method is as accurate as NODL, but tends to increase the dictionary size too much. The interplay between the addition and deletion processes in our NODL seems to achieve the best of the two worlds, achieving superior performance while keeping the dictionary size under control, in a narrower range (400 to 650 elements), expanding, as necessary, small dictionaries, while compressing large ones<sup>5</sup>.

We will now focus on comparing the two main methods, the baseline ODL and the proposed NODL method. The advantages of our approach become even more pronounced on larger input sizes, e.g.  $100 \times 100$  images, in similar sparse-dictionary, dense-code settings. (We keep the dictionary elements at the same sparsity rate, 50 nonzeros out of 10,000 dimensions, and just use completely non-sparse codes). In Fig. 2(a) and Fig. 2(b), we see that NODL considerably outperforms ODL on both the first (Oxford) and the (part of the ) second domain (Flowers); the results for Animals are very similar and are given in the Appendix in Fig. 9. In Appendix Sec. B.6, Fig. 16 depicts examples of actual animal images and the corresponding reconstructions by the fixed-size ODL and our NODL methods (not included here due to space restrictions). A better reconstruction quality of our method can be observed (e.g., a more visible dog shape, more details such as dog's legs, as opposed to a collection clusters produced by the ODL methods note however that printer resolution may reduce the visible difference, and looking at the images in online version of this paper is recommended).

Moreover, NODL can be also beneficial in classification settings. Given a dictionary, i.e., a sparse linear autoencoder trained in an unsupervised setting, we use the codes (i.e., feature vectors) computed on the test data from the second domain (Animals and Flowers) and evaluate multiple classifiers learned on those features in order to discriminate between the two classes. In Fig. 2(c), we show the logistic regression results using 10-fold cross-validation; similar results for several other classifiers are presented in the Appendix, Fig. 9. Note that we also perform filter-based feature subset selection, using the features statistical significance as measured by its p-value as the ranking function, and selecting subsets of top  $k$  features, increasing  $k$  from 1 to the total number of features (the code length, i.e. the number of dictionary elements). The x-axis in Fig. 2(c) shows the value of  $k$ , while the y-axis plots the classification error rate for the features derived by each method. We can see that our NODL method (blue) yields lower errors than the baseline ODL (magenta) for relatively small subsets of features, although the difference is negligible for the full feature set. Overall, this suggests that our NODL approach achieves better reconstruction performance of the input data, without extra overfitting in classification setting, since it generalizes at least as good as, and often better than the baseline ODL method.

# Non-sparse dictionary elements

When exploring a wide range of sparsity settings (see appendix), we observed quite different results for non-sparse dictionaries as opposed to those presented above. Fig. 7(b) (in Appendix, due to space constraints) summarizes the results for a particular setting of fully dense dictionaries (no zero entries), but sparse codes (50 non-zeros out of up to 600 dictionary elements; however, the codes are still dense when dictionary size is below 50). In this setting, unlike the previous one, we do not observe any significant improvement in accuracy due to neurogenetic approach, neither in reconstruction nor in classification accuracy; both methods perform practically the same. (Also, note a somewhat surprising phenomenon: after a certain point, i.e. about 50 elements, the reconstruction accuracy of both methods actually declines rather than improves with increasing dictionary size.)

It is interesting to note, however, that the overall classification errors, for both methods, are much higher in this setting (from 0.4 to 0.52) than in the sparse-dictionary setting (from 0.22 to 0.36). Even using non-sparse codes in the non-sparse dictionary setting still yields inferior results when compared to sparse dictionaries (see the results in the Appendix).

In summary, on real-life image datasets we considered herein, our NODL approach is often superior (and never inferior) to the standard ODL method; also, there is a consistent evidence that our approach is most beneficial in sparse dictionary settings.

# 4.2 SPARSE ORTHOGONAL INPUTS: NLP AND SYNTHETIC DATA

So far, we explored some conditions on methods properties (e.g., sparse versus dense dictionaries, as well as code sparsity/density) which can be beneficial for the neurogenetic approach. Our further question is: what kind of specific data properties would best justify neurogenetic versus traditional, fixed-size dictionary learning? As it turns out, the fixed-size ODL approach has difficulties adapting to a new domain in nonstationary settings, when the data in both domains are sparse and, across the domains, the supports (i.e., the sets of non-zero coordinates) are almost non-overlapping (i.e., datasets are nearly orthogonal). This type of data properties is related to a natural language processing problem considered below. Furthermore, pushing this type of structure to the extreme, we used simulations to better understand the behavior of our method. Herein, we focused, again, on sparse dictionary elements, as a well-suited basis for representing sparse data. Moreover, our empirical results confirm that using dense dictionary elements does not yield good reconstruction of sparse data, as expected.

# Sparse Natural Language Processing Problem

We consider a very sparse word co-occurrence matrix (on average, about 14 non-zeros in a column of size 12,883) using the text from two different domains, biology and mathematics, with the total vocabulary size of approximately 12,883 words. The full matrix was split in two for illustration purposes and shown in Fig. 3(c) and 3(d), where math terms correspond to the first block of columns and the biology terms correspond to the second one (though it might be somewhat hard to see in the picture, the average number of nozeros per row/column is indeed about 14).

We use the sparse columns (or rows) in the matrix, indexed by the vocabulary words, as our input data to learn the dictionary of sparse elements (25 non-zeros) with sparse codes (38 non-zeros). The

![](images/18cd32397f21af57a8f18011c49ba7dbf38daa6171587c90178946ef89fd5a4b.jpg)  
(a) 1st domain (Biology)

![](images/3cf014650a0fe5d8c321aec2b11eed0a6be2ec6ddbf1adc9c317eeb4cb3647d2.jpg)  
(b) 2nd Domain (Mathematics)

![](images/c62734c1a90ece5fc90195b9d7d4d9fbc39e8d73098177bfdd78b846bd9ab234.jpg)  
(c) Biology

![](images/66512e1ce7be125e14d26f6ababc5d8b63d51b9baacb8abff85126188db243b2.jpg)  
(d) Math  
Figure 3: Reconstruction accuracy for the sparse NLP data.

corresponding word codes in the learned dictionary can be later used as word embeddings, or word vectors, in various NLP tasks such as information extraction, semantic parsing, and others Yogatama et al. (2015); Faruqui et al. (2015); Sun et al. (2016). (Note that many of the non-domain specific words were removed from the vocabulary to obtain the final size of 12,883.) Herein, we evaluate our NODL method (i.e. NODL (sparse) in the plots) versus baseline ODL dictionary learning approach (i.e. ODL (sparse)) in the settings where the biology domain is processed first and then one have to switch to the the mathematics domain. We use 2750 samples from each of the domains for training and the same number for testing. The evaluation results are shown in Fig. 3. For the first domain (biology), both methods perform very similarly (i.e., remember the old data equally well), while for the second, more recent domain, our NODL algorithm is clearly outperforming its competitor. Moreover, as we mention above, non-sparse (dense) dictionaries are not suited for the modeling of highly sparse data such as our NLP data. In the Fig. 3, both random dense dictionaries (random-D) and the dense dictionaries learned with ODL (i.e. ODL (dense)) do poorly in the biology and mathematics domains.

However, the reconstruction accuracy as measured by Pearson correlation was not too high, overall, i.e. the problem turned out to be more challenging than encoding image data. It gave us an intuition about the structure of sparse data that may be contributing to the improvements due to neurogenesis. Note that the word co-occurrence matrix from different domains such as biology and mathematics tends to have approximately block-diagonal structure, where words from the same domain are occurring together more frequently than they co-occur with the words from the different domain. Pushing this type of structure to extreme, we studied next the simulated sparse dataset where the samples from the two different domains are not only sparse, but have completely non-overlapping supports, i.e. the data matrix is block-diagonal (see Fig. 6(c) in Appendix).

# Synthetic Sparse Data

We generated a synthetic sparse dataset with 1024 dimension, and only 50 nonzeros in each sample. Moreover, we ensured that the data in both domains had non-overlapping supports (i.e., non-intersecting sets of non-zero coordinates), by always selecting nonzeros in the first domain from the first 512 dimensions, while only using the last 512 dimensions for the second domain Fig. 6(c) in Appendix). For the evaluation on the synthetic data, we use the total of 200 samples for the training and testing purposes each (100 samples for each of the two domains), and smaller batches for online training, containing 20 samples each (instead of 200 samples used earlier for images and language data).

Since the data is sparse, we accordingly adjust the sparsity of dictionary elements (50 nonzeros in an element; for the code sparsity, we will present the results with 50 nonzeros as well). In Fig. 4, we see reconstruction accuracy, for the first and second domain data. For the first domain, the baseline ODL method (i.e. ODL (sparse) in the plots) and our NODL (i.e. NODL (sparse)) perform equally well. On the other hand, for the second domain, the ODL algorithm's performance degrades significantly compared to the first domain. This is because the data from the second domain have non-overlapping support w.r.t. the data from the first domain. Our method is able to perform very well on the second domain (almost as good as the first domain). It is further interesting to analyze the case of random non-sparse dictionary (random-D) which even performs better than the baseline ODL method, for the second domain. This is because random dictionary elements remain non-sparse in all the dimensions thereby doing an average job in both of the domains. Along the same lines, ODL (dense) performs better than the ODL (sparse) in the second domain. Though, the performance of non-sparse dictionaries should degrade significantly with an increase in the sparsity of data, as

![](images/de36b607870f0f0d19e589fb8fe1ddf104443bcc4cb6f656a9438814ccf5a898.jpg)  
(a) Pearson- First Domain

![](images/5a2d7cb88d1f97cb45dd11ad027165bf59ebcce4ffe7dd5cc2dc115730e8b707.jpg)  
(b) Pearson- Second Domain

![](images/b5bd32ca3ea9672f17777503128460fb05f86d736006212088a6513d71df22b0.jpg)  
(c)  $D$  -ODL  
Figure 4: Reconstruction accuracy for the sparse synthetic data.

![](images/82fbbea7c55605f7775763a0ae738799f9a4ebf92e8e8689c33acd0d8dd86e46.jpg)  
(d)  $D$ -NODL (ours)

we see above for the NLP data. Clearly, our NODL (sparse) gives consistently better reconstruction accuracy, compared to the other methods, across the two domains.

In Fig. 4(c) and Fig. 4(d), we see the sparsity structure of the dictionary elements learned using the baseline ODL method and our NODL method respectively. From these plots, we get better insights on why the baseline method does not work. It keeps same sparsity structure as it used for the data from the first domain. Our NODL adapts to the second domain data because of its ability to add new dictionary elements, that are randomly initialized with non-zero support in all the dimensions.

Next, in Sec. 5, we discuss our intuitions on why NODL performs better than the ODL algorithm under certain conditions.

# 5 WHEN NEUROGENESIS CAN HELP, AND WHY

In the Sec. 4, we observed that our NODL method outperforms the ODL algorithm in two general settings, both involving sparse dictionary elements: (i) non-sparse data such as real-life images, and (ii) sparse data with (almost) non-overlapping supports. In this section, we attempt to analyze what contributes to the success of our approach in these settings, starting with the last one.

# Sparse data with non-overlapping supports, sparse dictionary

As discussed above, in this scenario, the data from both the first and the second domain are sparse, and their supports (non-zero dimensions) are non-overlapping, as shown in the Fig. 6(c). Note that, when training a dictionary using the fixed-size, sparse-dictionary ODL method, we observe only a minor adaptation to the second domain after training on the first domain, as shown in Fig. 4(c).

Our empirical observations are supported by the theoretical result summarized in Lemma 1 below. Namely, we prove that when using the ODL algorithm in the above scenario, the dictionary trained on the first domain can not adapt to the second domain. (The minor adaptation, i.e., a few nonzeros, observed in our results in Fig. 4(c) occurs only due to implementation details involving normalization of sparse dictionary elements when computing codes in the dictionary – the normalization introduces non-zeros of small magnitude in all dimensions (see Appendix for the experiment results with no normalization of the elements, conforming to the Lemma 1)).

Lemma 1. Let  $\pmb{x}_1, \pmb{x}_2, \dots, \pmb{x}_{t-1} \in \mathbb{R}^m$  be a set of samples from the first domain, with non-zeros (support) in the set of dimensions  $P \subset M = \{1, \dots, m\}$ , and let  $\pmb{x}_t, \pmb{x}_{t+1}, \dots, \pmb{x}_n \in \mathbb{R}^m$  be a set of samples from the second domain, with non-zeros (support) in dimensions  $Q \subset M$ , such that  $P \cap Q = \emptyset$ ,  $|P| = |Q| = l$ . Let us denote as  $\pmb{d}_1, \pmb{d}_2, \dots, \pmb{d}_k \in \mathbb{R}^m$  dictionary elements learned by ODL algorithm, with the sparsity constraint of at most  $l$  nonzero in each element<sup>6</sup>, on the data from the first domain,  $\pmb{x}_1, \dots, \pmb{x}_{t-1}$ . Then (1) those elements have non-zero support in  $P$  only, and (2) after learning from the second domain data, the support (nonzero dimensions) of the corresponding updated dictionary elements will remain in  $P$ .

Proof Sketch. Let us consider processing the data from the first domain. At the first iteration, a sample  $\pmb{x}_1$  is received, its code  $\alpha_1$  is computed, and the matrices  $A$  and  $B$  are updated, as shown in

![](images/ab41ad51c01e08bd8436046ad4f467e2bede2a5b5dbb867b327c5198444d73bd.jpg)  
(a)  $D$  with ODL method

![](images/fb6aca54b9a463b6555b2fa848971e880f5fbd09ebd2ae95e24d0932872cfe34.jpg)  
(b)  $\mathbf{A}$  with ODL method  
Figure 5: Visualization of the sparse dictionary and the matrix  $\mathbf{A}$  learned on the first imaging domain (Oxford images), using the baseline ODL method and our method.

![](images/38980af9b3408db2f88acb5f1623ebaa03365fc48579deaef4256344dde0d26e.jpg)  
(c)  $A$  with our method

Alg. 1 (non-highlighted part); next, the dictionary update step is performed, which optimizes

$$
\boldsymbol {D} ^ {(1)} = \arg \min  _ {\boldsymbol {D} \in \mathcal {C}} \frac {1}{2} T r (\boldsymbol {D} ^ {T} \boldsymbol {D} \boldsymbol {A}) - T r (\boldsymbol {D} ^ {T} \boldsymbol {B}) + \sum_ {j} \lambda_ {j} \| \boldsymbol {d} _ {j} \| _ {1}. \tag {6}
$$

Since the support of  $x_{1}$  is limited to  $P$ , we can show that optimal dictionary  $D^{*}$  must also have all columns/elements with support in  $P$ . Indeed, assuming the contrary, let  $\mathbf{d}_{\mathbf{j}}(i) \neq 0$  for some dictionary element/column  $j$ , where  $i \notin P$ . But then it is easy to see that setting  $\mathbf{d}_{\mathbf{j}}(i)$  to zero reduces the sum-squared error and the  $l_{1}$ -norm in (6), yielding another dictionary that achieves a lower overall objective; this contradicts our assumption that  $\mathbf{D}^{*}$  was optimal. Thus, the dictionary update step must produce a dictionary where all columns have their support in  $P$ . By induction, this statement will also be true for the dictionary obtained after processing all samples from the first domain. Next, the samples from the second domain start arriving; note that those samples belong to a different subspace, spanning the dimensions within the support set  $Q$ , which is not intersecting with  $P$ . Thus, using the current dictionary, the encoding  $\alpha_{t}$  of first sample  $x_{t}$  from the second domain (i.e. the solution of the LASSO problem in step 4 of the Alg. 1) will be a zero vector. Therefore, the matrices  $A$  and  $B$  remains unchanged during the update in step 11, and thus the support of each  $b_{j}$  and, consequently,  $u_{j}$  and the updated dictionary elements  $d_{j}$  will remain in  $P$ . By induction, every dictionary update in response to a new sample from the second domain will preserve the support of the dictionary elements, and thus the final dictionary elements will also have their support only in  $P$ .

![](images/26072b274f140cfbb4c43c6393a1ba042b755a236983c41f1958b5e60274cd1a.jpg)

# Non-sparse data, sparse dictionary

We will now discuss an intuitive explanation behind the success of neurogenetic approach in this scenario, leaving a formal theoretical analysis as a direction for future work. When learning sparse dictionaries on non-sparse data such as natural images, we observed that many dictionary elements have non-overlapping supports with respect to each other; see, for example, Fig. 5(a), where each column corresponds to a 10000-dimensional dictionary element with nonzero dimensions shown in black color. Apparently, the non-zeros dimensions of an element tend to cluster spatially, i.e. to form a patch in an image. The non-overlapping support of dictionary elements results into a specific structure of the matrix  $\mathbf{A}$ . As shown in Fig. 5(b), for ODL approach, the resulting matrix  $\mathbf{A}$  includes many off-diagonal nonzero elements of large absolute values (along with high values on the diagonal). Note that, by definition,  $\mathbf{A}$  is an empirical covariance of the code vectors, and it is easy to see that a nonzero value of  $a_{jk}$  implies that the  $j$ -th and the  $k$ -th dictionary elements were used jointly to explain the same data sample(s). Thus, the dense matrix structure with many non-zero off-diagonal elements, shown in Fig. 5(b), implies that, when the dictionary elements are sparse, they will be often used jointly to reconstruct the data. On the other hand, in the case of non-sparse dictionary elements, the matrix  $\mathbf{A}$  has an almost diagonally-dominant structure, i.e. only a few dictionary elements are used effectively in the reconstruction of each data sample even with non-sparse codes (see Appendix for details).

Note that in the dictionary update expression  $\pmb{u}_j \gets \frac{\pmb{b}_j - \sum_{k \neq j} \pmb{d}_k a_{jk}}{a_{jj}}$  in (3), when the values  $a_{jk} / a_{jj}$  are large for multiple  $k$ , the  $j_{th}$  dictionary element becomes tightly coupled with other dictionary elements, which reduces its adaptability to new, non-stationary data. In our algorithm, the values  $a_{jk} / a_{jj}$  remain high if both elements  $j$  and  $k$  have similar "age"; however, those values are much

lower if one of the elements is introduced by neurogenesis much more recently than the other one. In 5(c), the upper left block on the diagonal, representing the oldest elements (added during the initialization), is not diagonally-dominant (see the sub-matrices of  $\mathbf{A}$  with NODL in Fig. 13 in the Appendix). The lower right block, corresponding to the most recently added new elements, may also have a similar structure (though not visible due to relatively low magnitudes of the new elements; see the Appendix). Overall, our interpretation is that the old elements are tied to each other whereas the new elements may also be tied to each other but less strongly, and not tied to the old elements, yielding a block-diagonal structure of  $\mathbf{A}$  in case of neurogenetic approach, where blocks correspond to dictionary elements adapted to particular domains. In other words, neurogenesis allows for an adaptation to a new domain without forgetting the old one.

# 6 CONCLUSIONS

In this work, we proposed a novel algorithm, Neurogenetic Online Dictionary Learning (NODL), for the problem of learning representations in non-stationary environments. Our algorithm builds a dictionary of elements by learning from an online stream of data while also adapting the dictionary structure (the number of elements/hidden units and their connectivity) via continuous birth (addition) and death (deletion) of dictionary elements, inspired by the adult neurogenesis process in hippocampus, which is known to be associated with better adaptation of an adult brain to changing environments. Moreover, introducing sparsity in dictionary elements allows for adaptation of the hidden unit connectivity and further performance improvements.

Our extensive empirical evaluation on both real world and synthetic data demonstrated that the interplay between the birth and death of dictionary elements allows for a more adaptive dictionary learning, better suited for non-stationary environments than both of its counterparts, such as the fixed-size online method of Mairal et al. (2009) (no addition and no deletion), and the online version of the group-sparse coding method by Bengio et al. (2009) (deletion only). Furthermore we evaluated, both empirically and theoretically, several specific conditions on both method's and data properties (involving the sparsity of elements, codes, and data) where our method has significant advantage over the standard, fixed-size online dictionary learning. Overall, we can conclude that neurogenetic dictionary learning typically performs as good as, and often much better than its competitors. In our future work, we plan to explore the non-linear extension of the dictionary model, as well as a stacked auto-encoder consisting of multiple layers.

# REFERENCES

Michal Aharon, Michael Elad, and Alfred Bruckstein. K-svd: An algorithm for designing overcomplete dictionaries for sparse representation. Signal Processing, IEEE Transactions on, 2006.  
Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In Advances in neural information processing systems, 2014.  
Samy Bengio, Fernando Pereira, Yoram Singer, and Dennis Strelow. Group sparse coding. In Advances in Neural Information Processing Systems 22. 2009.  
Cristian Bucilu, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, 2006.  
Timothy J. Draelos, Nadine E. Miner, Jonathan A. Cox, Christopher C. Lamb, Conrad D. James, and James B. Aimone. Neurogenic deep learning. In ICLR 2016 Workshop Track, 2016.  
Scott E Fahlman and Christian Lebiere. The cascade-correlation learning architecture. 1989.  
Manaal Faruqui, Yulia Tsvetkov, Dani Yogatama, Chris Dyer, and Noah Smith. Sparse overcomplete word vector representations. arXiv preprint arXiv:1506.02004, 2015.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Rodolphe Jenatton, Julien Mairal, Guillaume Obozinski, and Francis Bach. Proximal methods for hierarchical sparse coding. Journal of Machine Learning Research, 2011.  
Gerd Kempermann. Adult neurogenesis: stem cells and neuronal development in the adult brain. 2006.

Kenneth Kreutz-Delgado, Joseph F Murray, Bhaskar D Rao, Kjersti Engan, Te-Won Lee, and Terrence J Sejnowski. Dictionary learning algorithms for sparse representation. Neural computation, 2003.  
Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y Ng. Efficient sparse coding algorithms. In Advances in neural information processing systems, 2006.  
Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro. Online dictionary learning for sparse coding. In Proceedings of the 26th annual international conference on machine learning, 2009.  
Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro. Online learning for matrix factorization and sparse coding. Journal of Machine Learning Research, 2010.  
Bruno A Olshausen and David J Field. Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision research, 1997.  
Amar Sahay, Kimberly N Scobie, Alexis S Hill, Colin M O'Carroll, Mazen A Kheirbek, Nesha S Burghardt, Andre A Fenton, Alex Dranovsky, and René Hen. Increasing adult hippocampal neurogenesis is sufficient to improve pattern separation. Nature, 2011.  
Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 2014.  
Ales Stuchlik. Dynamic learning and memory, synaptic plasticity and neurogenesis: an update. Frontiers in behavioral neuroscience, 2014.  
Fei Sun, Jiafeng Guo, Yanyan Lan, Jun Xu, and Xueqi Cheng. Sparse word embeddings using 11 regularized online learning. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, 2016.  
Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society. Series B (Methodological), 1996.  
Dani Yogatama, Manaal Faruqui, Chris Dyer, and Noah A Smith. Learning word representations with hierarchical sparse coding. In Proc. of ICML, 2015.  
Ming Yuan and Yi Lin. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 2006.
