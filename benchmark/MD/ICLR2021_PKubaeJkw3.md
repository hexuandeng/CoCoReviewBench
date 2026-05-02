# RETHINKING ARCHITECTURE SELECTION IN DIFFER-ENTIABLE NAS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Differentiable Neural Architecture Search is one of the most popular Neural Architecture Search (NAS) method for its search efficiency and simplicity, accomplished by jointly optimizing the model weight and architecture parameters in a weight-sharing supernet via gradient-based algorithms. At the end of the search phrase, the operations with the largest architecture parameters will be selected to form the final architecture, with the implicit assumption that the values of architecture parameters reflect the operation strength. While much has been discussed about the optimization of the supernet, the architecture selection process has received little attention. We provide both empirical and theoretical analysis to show that the magnitude of architecture parameters do not necessarily indicate how much the operation contributes to the performance of the supernet. We propose an alternative perturbation-based architecture selection that directly measures each operation's influence on the supernet. We re-evaluate several differentiable NAS methods with the proposed architecture selection and find that it is able to consistently extract significantly improved architectures from the underlying supernets. Furthermore, we find that several failure modes of Darts can be greatly alleviated with the proposed selection method, indicating that much of the poor generalization observed in Darts can be attributed to the failure of magnitude-based architecture selection rather than entirely the optimization of its supernet. Our code will be made publically available shortly.

# 1 INTRODUCTION

Neural Architecture Search (NAS) has been drawing increasing attention in both academia and industry for its potential to automatize the process of discovering high-performance architectures, which have long been handcrafted. Early works on NAS deploy Evolutionary Algorithm (Stanley & Miikkulainen, 2002; Real et al., 2017; Liu et al., 2017) and Reinforcement Learning (Zoph & Le, 2017; Pham et al., 2018; Zhong et al., 2018) to guide the architecture discovery process. Recently, several one-shot methods have been proposed that significantly improve the search efficiency (Brock et al., 2018; Guo et al., 2019; Bender et al., 2018).

As a particularly popular instance of one-shot methods, Darts (Liu et al., 2019) enables the search process to be performed with a gradient-based optimizer in an end-to-end manner. They apply continuous relaxation that transforms the categorical choice of architectures into continuous architecture parameters  $\alpha$ . The resulting supernet can be optimized via gradient-based methods, and the operations associated with the largest architecture parameters are selected to form the final architecture. Despite its simplicity, several works cast doubt on the effectiveness of Darts. For example, simple randomized search (Li & Talwalkar, 2019) outperforms the original Darts; Zela et al. (2020) observes that Darts degenerates to networks filled with parametric-free operations such as skip connection or even random noise, leading to the poor performance of the selected architecture.

While the majority of previous research attributes the failure of Darts to its supernet optimization (Zela et al., 2020; Chen & Hsieh, 2020; Chen et al., 2020), little has been discussed about the validity of another important assumption: the value of  $\alpha$  reflects the strength of the underlying operations. In this paper, we conduct an in-depth analysis of this problem. Surprisingly, we find that in many cases  $\alpha$  does not really indicate operation importance in a supernet. Firstly, the operation associated with the largest  $\alpha$  often does not result in the highest validation accuracy after discretization. Secondly,

as an important example, we show mathematically that the domination of skip connection observed in Darts (i.e.  $\alpha_{skip}$  becomes larger than other operations.) is in fact a reasonable outcome of the supernet optimization, but becomes problematic when we rely on  $\alpha$  to select the best operation.

If  $\alpha$  is not a good indicator of operation strength, how should we select the final architecture from a pretrained supernet? Our analysis indicates that the strength of each operation should be evaluated based on its contribution to the supernet performance instead. To this end, we propose an alternative perturbation-based architecture selection method. Given a pretrained supernet, the best operation on an edge is selected and discretized based on how much it perturbs the supernet accuracy; and the final architecture is derived edge by edge, with fine-tuning in between so that the supernet remains converged for every operation decision. We re-evaluate several differentiable NAS methods (Darts (Liu et al., 2019), SDarts (Chen & Hsieh, 2020), SGAS (Li et al., 2020)) and show that the proposed selection method is able to consistently extract significantly improved architectures from the supernet than magnitude-based counterparts. Furthermore, we find that the robustness issues of Darts can be greatly alleviated by replacing the magnitude-based selection with our selection method.

# 2 BACKGROUND AND RELATED WORK

# 2.1 PRELIMINARIES OF DIFFERENTIABLE ARCHITECTURE SEARCH (DARTS)

We start by reviewing the formulation of Darts. Darts' search space consists of repetitions of cell-based micro structures. Every cell can be viewed as a DAG with N nodes and E edges, where each node represents a latent feature map  $x^{i}$  and each edge is associated with an operation  $o$  (e.g. skip_connect, sep_conv_3x3) from the search space  $\mathcal{O}$ . Continuous relaxation is then applied to this search space. Concretely, every operation on an edge is activated during search phase, with their outputs mixed by the architecture parameter  $\alpha$  to form the final mixed output of that edge  $\bar{m}(x^{i}) = \sum_{o \in \mathcal{O}} \frac{\exp \alpha_{o}}{\sum_{o^{\prime}} \exp \alpha_{o^{\prime}}} o(x^{i})$ . This particular formulation allows architecture search to be performed in a differentiable manner: Darts jointly optimizes  $\alpha$  and model weight  $w$  with the following bilevel objective via alternative gradient updates

$$
\min  _ {\alpha} \mathcal {L} _ {v a l} \left(w ^ {*}, \alpha\right) \text {s . t .} w ^ {*} = \underset {w} {\arg \min } \mathcal {L} _ {t r a i n} (w, \alpha). \tag {1}
$$

We refer to the continuous relaxed network used in the search phase as the supernet of Darts. At the end of the search phase, the operation associated with the largest  $\alpha_{o}$  on each edge will be selected from the supernet to form the final architecture.

# 2.2 FAILURE MODE ANALYSIS OF DARTS

Several works cast doubt on the robustness of Darts. Zela et al. (2020) tests Darts on four different search spaces and observes significantly degenerated performance. They empirically find that the selected architectures tend to perform poorly when Darts' supernet falls into high curvature areas of validation loss (captured by large dominant eigenvalues of the Hessian  $\nabla_{\alpha, \alpha}^2 \mathcal{L}_{val}(w, \alpha)$ ). While Zela et al. (2020) relates this problem to the failure of supernet training in Darts, we examine it from the architecture selection aspects of Darts, and show that much of the robustness issue of Darts can be alleviated by a better architecture selection method than magnitude-based selection based on  $\alpha$ .

# 2.3 PROGRESSIVE SEARCH SPACE SHRINKING

There is a line of research on NAS that focuses on reducing the search cost and aligning model size of the search and evaluation phases via progressive search space shrinking (Liu et al., 2018; Li et al., 2019; Chen et al., 2020; Li et al., 2020). The general scheme of these methods is to prune out weak operations and edges sequentially during the search phrase, based on the magnitude of  $\alpha$  following Darts. Our method is orthogonal to them in this aspect since we select operations based on how much it contributes to the supernet's performance rather than the value of  $\alpha$ . Although we also discretize edges greedily and fine-tune the network in between, the purpose is to let the supernet to recover from loss of accuracy after discretization so as to accurately evaluate operation strength on the next edge, rather than to reduce the search cost.

# 3 THE PITFALL OF MAGNITUDE-BASED ARCHITECTURE SELECTION IN DARTS

In this section, we put forward the opinion that the architecture parameters  $\alpha$  do not necessarily represent the strength of the underlying operation in general, backed by both empirical and theoretical evidences. As an important example, we mathematically justify that the skip connection domination phenomena observed in Darts is reasonable by itself, and becomes problematic when combined with the magnitude-based architecture selection.

# 3.1  $\alpha$  MAY NOT REPRESENT THE OPERATION STRENGTH

![](images/aa99bd3ba384a9389f37d89234c53fc06c220763dbbe71d8ae0ab782a04bddce.jpg)  
Figure 1:  $\alpha$  vs discretization accuracy at convergence of all the operations on three randomly selected edges from a pretrained Darts supernet. Operations associated with the largest  $\alpha$  constantly deviate from the one that lead to highest discretization accuracy at convergence.

![](images/75f1eef93b3661f37a3cbbe593a42ca8dd5d35ca19de861d8c9baaed1e8ceb22.jpg)

![](images/45a421f7e885212fe93194e3a0fdc6b666f5e5dcd73fd7e209f5c74851cef446.jpg)

![](images/196262592cbf969d03f5daee55b7876d7edf785df43415d746f36780bb4ee936.jpg)  
(a) Magnitude

![](images/5dbeabff681a706e7547e86fed425d4fcebcae2ca1279c97d949c519b8a0f175.jpg)  
Figure 2: Operation strength on each edge of S2 (skip_connect, sep_conv_3x3). (a). Operation associated with the largest  $\alpha$ . (b). Operation that results in highest discretization validation accuracy at convergence. Parameterized operations are marked red.  
(b) Strength

Following Darts, existing differentiable NAS methods use the value of architecture parameters  $\alpha$  to select the final architecture from the supernet, with the implicit assumption that  $\alpha$  represents the strength of the underlying operations. In this section we study the validity of this assumption in detail.

Consider one edge on a pretrained supernet, the strength of an operation on the edge can be naturally defined as the supernet accuracy after we discretize to an operation and fine-tune the remaining network till it converges again, which we refer to as "discretization accuracy at convergence" for short. The operation that achieves the best discretization accuracy at convergence can be considered as the best operation for the given edge. Figure 1 shows the comparison of  $\alpha$  (blue) and operation strength (orange) of randomly select edges on Darts supernet. As we can see, the operation associated with the largest  $\alpha$  does not necessarily match the one that achieves the highest discretization validation accuracy at convergence. Moreover, operations assigned with small  $\alpha$  are sometimes strong ones that lead to high discretization accuracy at convergence. To further verify the mismatch, we investigate the operation strength on search space S2 where Darts fails dramatically due to the excess of skip connections (Zela et al., 2020). S2 is a variant of Darts search space that only contains two operations per edge (skip_connect, sep_conv_3x3). Figure 2 shows the selected operations based on  $\alpha$  (left) and operation strength (right) on all edges on S2. From Figure 2a we can see that  $\alpha_{skip\_connect} > \alpha_{sep\_conv\_3x3}$  on 12 of 14 edges. Consequently, the derived child architecture will lack representation ability and perform poorly due to too many skip connections. However, as

shown in Figure 2b, the supernet benefits more from discretizing to sep_conv_3x3 than skip_connect on half of the edges.

# 3.2 A CASE STUDY: SKIP CONNECTION

Several works point out that Darts tends to assign large  $\alpha$  to skip connections, resulting in shallow architectures with poor generability (Zela et al., 2020; Liang et al., 2019; Bi et al., 2019). This "skip domination" issue is generally attributed to the failure of Darts' supernet optimization. In contrast, we draw inspiration from research on ResNet (He et al., 2016) and show that this phenomena by itself is a reasonable outcome while Darts refines its estimations of the optimal feature map, rendering  $\alpha_{skip}$  ineffective in architecture selection.

In vanilla networks (VGG), each layer computes a new level of feature map from the output feature map of the predecessor layer, thus reordering layers at test time would dramatically hurt the performance (Veit et al., 2016). Unlike VGG, Greff et al. (2017) and Veit et al. (2016) discover that successive layers in ResNet with compatible channel sizes are in fact estimating the same optimal feature map, so that the out

Table 1: Test accuracy before and after layer (edge) shuffling on cifar10. For ResNet and VGG, we randomly swap two layers in each stage (defined as successive layers between two downsampling block. For Darts supernet, we randomly swap two edges in every cell.

<table><tr><td></td><td>VGG</td><td>ResNet</td><td>Darts</td></tr><tr><td>Before</td><td>92.69</td><td>93.86</td><td>88.44</td></tr><tr><td>After</td><td>9.83 ± 0.33</td><td>83.2015 ± 2.03</td><td>81.09 ± 1.87</td></tr></table>

puts of these layers stay relatively close to each other at convergence; As a result, ResNet's test accuracy remains robust against layer reordering. Greff et al. (2017) refers to this unique way of feature map estimation in ResNet as "unrolled estimation".

Darts' supernet resembles ResNet, rather than VGG, in both appearance and behavior. Appearance-wise, within a cell of Darts' supernet, edges with skip connection are in direct correspondence with successive residual layers in ResNet. Behavior-wise, Darts' supernet also exhibits high degree of robustness under edge shuffling. As shown in Table [1], randomly reordering edges on a pretrained Darts' supernet at test time also has little effect on its performance. This evidence indicates that Darts performs unrolled estimation like ResNet as well, i.e. edges within a cell share the same optimal feature map that they try to estimate. In the following proposition, we apply this finding and provide the optimal solution of  $\alpha$  in the sense of minimizing the variance of feature map estimation.

Proposition 1. Without loss of generality, consider one cell from a simplified search space consists of two operations: (skip, conv). Let  $m^*$  denotes the optimal feature map, which is shared across all edges according to the unrolled estimation view (Greff et al., 2017). Let  $o_e(x_e)$  be the output of convolution operation, and let  $x_e$  be the skip connection (i.e. the input feature map of edge  $e$ ). Assume  $m^*$ ,  $o_e(x_e)$  and  $x_e$  are normalized to the same scale. The current estimation of  $m^*$  can then be written as:

$$
\overline {{m}} _ {e} \left(x _ {e}\right) = \frac {\exp \left(\alpha_ {\text {c o n v}}\right)}{\exp \left(\alpha_ {\text {c o n v}}\right) + \exp \left(\alpha_ {\text {s k i p}}\right)} o _ {e} \left(x _ {e}\right) + \frac {\exp \left(\alpha_ {\text {s k i p}}\right)}{\exp \left(\alpha_ {\text {c o n v}}\right) + \exp \left(\alpha_ {\text {s k i p}}\right)} x _ {e}, \tag {2}
$$

where  $\alpha_{conv}$  and  $\alpha_{skip}$  are the architecture parameters defined in Darts. The optimal  $\alpha_{conv}^{*}$  and  $\alpha_{skip}^{*}$  minimizing  $var(\overline{m}_e(x_e) - m^*)$ , the variance of the difference between the optimal feature map  $m^{*}$  and its current estimation  $\overline{m}_e(x_e)$ , is given by:

$$
\alpha_ {c o n v} ^ {*} \propto \operatorname {v a r} \left(x _ {e} - m ^ {*}\right) \tag {3}
$$

$$
\alpha_ {s k i p} ^ {*} \propto \operatorname {v a r} \left(o _ {e} \left(x _ {e}\right) - m ^ {*}\right) \tag {4}
$$

We refer the reader to Appendix A.4 for the detailed proof. From eq. (3) and eq. (4), we can see that the relative magnitudes of  $\alpha_{skip}$  and  $\alpha_{conv}$  comes down to which one of  $x_{e}$  and  $o_{e}(x_{e})$  is closer to  $m^{*}$  in variance:

-  $x_{e}$  (input of edge  $e$ ) comes from the mixed output of previous edge. Since the goal of every edge is to estimate  $m^{*}$  (unrolled estimation),  $x_{e}$  is also directly estimating  $m^{*}$ .

-  $o_{e}(x_{e})$  is the output of a single convolution operation instead of the complete mixed output of edge  $e$ , and thus it will deviate from  $m^{*}$  even at convergence.

Therefore, in a well optimized supernet,  $x_{e}$  will naturally be closer to  $m^{*}$  than  $o_{e}(x_{e})$ , causing  $\alpha_{skip}$  to be greater than  $\alpha_{conv}$ .

Our analysis above indicates that the better the supernet, the larger the gap of  $(\alpha_{skip} - \alpha_{conv})$  (after softmaxed) will become, since  $x_{e}$  gets closer and closer to  $m^{*}$  as the supernet is optimized. This result is evidenced by Figure 3 where mean  $(\alpha_{skip} - \alpha_{conv})$  continues to grow as the supernet gets better. In this case, although  $\alpha_{skip} > \alpha_{conv}$  is reasonably by itself, it becomes an inductive bias to NAS if we were to select the architecture based on  $\alpha$ .

![](images/c5479198cec625e76b2db09954f4d98c51403fd3b1fbf6aabd776cbb5f6e1614.jpg)  
Figure 3: mean  $(\alpha_{skip} - \alpha_{conv})$  (after softmax) v.s. supernet validation accuracy. The gap of  $(\alpha_{skip} - \alpha_{conv})$  increases as supernet gets better.

# 4 PERTURBATION-BASED ARCHITECTURE SELECTION

Instead of relying on  $\alpha$  values to select the best operation, we propose to directly evaluate operation strength in

terms of its contribution to the supernet performance. The operation selection criteria is laid out in section 4.1 In section 4.2 we describe the entire architecture selection process.

# 4.1 EVALUATING THE STRENGTH OF EACH OPERATION

In section 3.1, we define the strength of each operation on a given edge as how much it contributes to the performance of the supernet, measured by discretization accuracy. To avoid inaccurate evaluation due to large disturbance of the supernet during discretization, we fine-tune the remaining supernet till it converges again and then compute its validation accuracy (discretization accuracy at convergence). The fine-tuning process needs to be carried out for evaluating each operation on an edge, leading to extremely large computation costs.

To alleviate the computational overhead, we consider a more practical measure of operation strength: for each operation on a given edge, we mask it out while keeping all other operations, and re-evaluate the supernet. The one that results in the largest drop in validation accuracy of the supernet will be considered as the most important operation on that edge. This alternative criteria incurs much less perturbation to the supernet than discretization since it only deletes one operation from the supernet at a time. As a result, the validation accuracy of the supernet after deletion stays close to the unmodified supernet, and thus alleviates the requirement of tuning the remaining supernet to convergence. Therefore, we implement this measurement for the operation selection in this work.

Algorithm 1: Perturbation-based Architecture Selection  
Input: A pretrained Supernet  $S$  , Set of Edges  $\mathcal{E}$  from  $S$  , Set of Nodes  $\mathcal{N}$  from  $S$    
Result: Set of selected operations  $\{o_e^*\}_{e\in \mathcal{E}}$    
while  $|\mathcal{E}| > 0$  do randomly select an edge  $e\in \mathcal{E}$  (and remove it from  $\mathcal{E}$  ); forall operation o on edge e do evaluate the validation accuracy of  $S$  when o is removed  $(ACC_{\backslash o})$  end select the best operation for e:  $o_e^* \gets \arg \min_o ACC_{\backslash o}$  . discretize edge e to  $o_e^*$  and tune the remaining supernet for a few epochs;   
end

# 4.2 THE COMPLETE ARCHITECTURE SELECTION PROCESS

Our method operates directly on top of Darts' pretrained supernet. Given a supernet, we randomly iterate over all edges, evaluate each operation on an edge, and select the best one to be discretized to based on the measurement described in section 4.1. After that, we tune the supernet for a few epochs

so that the lost accuracy during discretization is recovered. The above steps are repeated until all edges are decided. Algorithm  $\boxed{1}$  summarizes the operation selection process. We refer the reader to Appendix A.3 for the full algorithm, including deciding the cell topology. This simple method is termed "perturbation-based architecture selection (PT)" in the following section.

# 5 EXPERIMENTAL RESULTS

In this section, we demonstrate that perturbation-based architecture selection method is able to consistently find better architectures than those selected based on the values of  $\alpha$ . The evaluation is based on the search space of Darts and NAS-Bench-201 (Dong & Yang, 2020), and we show the perturbation-based architecture selection method can be applied to several variants of Darts.

# 5.1 RESULTS ON DARTS CNN SEARCH SPACE

We keep all the search and retrain settings identical to Darts since our method only modifies the architecture selection part. After the search phase, we perform perturbation-based architecture selection following Algorithm 1 on the pretrained supernet. Between two selections, we tune the supernet for 5 epochs as it is enough for the supernet to recover from the drop of accuracy after discretization. We run the search and architecture selection phase with four random seeds, and report both the best and average test errors of the obtained architectures.

As shown in Table 2 the proposed method (Darts+PT) improves Darts' test error from  $3.00\%$  to  $2.61\%$ , with manageable search cost (0.8 GPU days). Note that by only changing the architecture selection method, Darts performs significantly better than many other differentiable NAS methods that enjoy carefully designed optimization process of the supernet, such as GDAS (Dong & Yang, 2019), and SNAS (Xie et al., 2019). This empirical result suggests that architecture selection is crucial to Darts: with the proper selection algorithm, Darts remains a very competitive method.

Our method is also able to improve the performance of other variants of Darts. To show this, we evaluate our method on SDarts(rs) and SGAS (Chen & Hsieh, 2020; Li et al., 2020). SDarts(rs) is a variant of Darts that regularizes the search phase by applying Gaussian perturbation to  $\alpha$ . Unlike Darts and SDarts, SGAS performs progressive search space shrinking. Concretely, SGAS progressively discretizes its edges with the order from most to least important, based on a novel edge importance score. For a fair comparison, we keep its unique search space shrinking process unmodified and only replace its magnitude-based operation selection with ours. As we can see from Table 2 our method consistently achieves better average test errors compared with its magnitude-based counterpart. Concretely, the proposed method improves SDarts' test error from  $2.67\%$  to  $2.54\%$  and SGAS' test error from  $2.66\%$  to  $2.56\%$ . Moreover, the best architecture discovered in our experiments achieves a test error of  $2.44\%$ , ranks top among other NAS methods.

# 5.2 PERFORMANCE ON NAS-BENCH-201 SEARCH SPACE

![](images/9dbde6462aed087f2b171c5bd2b52815ad08e547c9d5ff61cf5b15dad67c40c9.jpg)  
Figure 4: Trajectory of test accuracy on space NAS-Bench-201 and three datasets (Left: CIFar10, Middle: CIFar100, Right:Imagenet16-120). The test accuracy of our method is plotted by taking the snapshots of Darts' supernet at corresponding epochs and run our selection method on top of it.

![](images/4d69bd197ec10511fe107f8bf6b8845f9af18cf68f8cd4835ceba4d5f844dec3.jpg)

![](images/46ea9ed0f12d4bc7ed79290038baa1395e66419791849e5d8f712dee6b0008fc.jpg)

To further verify the effectiveness of the proposed perturbation-based architecture selection, we conduct experiments on NAS-Bench-201. NAS-Bench-201 provides a unified cell-based search space similar to Darts. Every architecture in the search space is trained under the same protocol on three datasets (cifar10, CIFar100 andImagenet16-120) and their performance can be obtained by querying

Table 2: Comparison with state-of-the-art image classifiers on CIFAR-10.  

<table><tr><td>Architecture</td><td>Test Error (%)</td><td>Params (M)</td><td>Search Cost (GPU days)</td><td>Search Method</td></tr><tr><td>DenseNet-BC (Huang et al., 2017)</td><td>3.46</td><td>25.6</td><td>-</td><td>manual</td></tr><tr><td>NASNet-A (Zoph et al., 2018)</td><td>2.65</td><td>3.3</td><td>2000</td><td>RL</td></tr><tr><td>AmoebaNet-A (Real et al., 2019)</td><td>3.34 ± 0.06</td><td>3.2</td><td>3150</td><td>evolution</td></tr><tr><td>AmoebaNet-B (Real et al., 2019)</td><td>2.55 ± 0.05</td><td>2.8</td><td>3150</td><td>evolution</td></tr><tr><td>PNAS (Liu et al., 2018)*</td><td>3.41 ± 0.09</td><td>3.2</td><td>225</td><td>SMBO</td></tr><tr><td>ENAS (Pham et al., 2018)</td><td>2.89</td><td>4.6</td><td>0.5</td><td>RL</td></tr><tr><td>NAONet (Luo et al., 2018)</td><td>3.53</td><td>3.1</td><td>0.4</td><td>NAO</td></tr><tr><td>SNAS (moderate) (Xie et al., 2019)</td><td>2.85 ± 0.02</td><td>2.8</td><td>1.5</td><td>gradient</td></tr><tr><td>GDAS (Dong &amp; Yang, 2019)</td><td>2.93</td><td>3.4</td><td>0.3</td><td>gradient</td></tr><tr><td>BayesNAS (Zhou et al., 2019)</td><td>2.81 ± 0.04</td><td>3.4</td><td>0.2</td><td>gradient</td></tr><tr><td>ProxylessNAS (Cai et al., 2019)†</td><td>2.08</td><td>5.7</td><td>4.0</td><td>gradient</td></tr><tr><td>NASP (Yao et al., 2020)</td><td>2.83 ± 0.09</td><td>3.3</td><td>0.1</td><td>gradient</td></tr><tr><td>P-DARTS (Chen et al., 2019)</td><td>2.50</td><td>3.4</td><td>0.3</td><td>gradient</td></tr><tr><td>PC-DARTS (Xu et al., 2020)</td><td>2.57 ± 0.07</td><td>3.6</td><td>0.1</td><td>gradient</td></tr><tr><td>R-DARTS (L2) (Zela et al., 2020)</td><td>2.95 ± 0.21</td><td>-</td><td>1.6</td><td>gradient</td></tr><tr><td>DARTS (Liu et al., 2019)</td><td>3.00 ± 0.14</td><td>3.3</td><td>0.4</td><td>gradient</td></tr><tr><td>SDARTS-RS (Chen &amp; Hsieh, 2020)</td><td>2.67 ± 0.03</td><td>3.4</td><td>0.4</td><td>gradient</td></tr><tr><td>SGAS (Cri 1. avg) (Li et al., 2020)</td><td>2.66 ± 0.24</td><td>3.7</td><td>0.25</td><td>gradient</td></tr><tr><td>DARTS+PT (avg)*</td><td>2.61 ± 0.08</td><td>3.0</td><td>0.8‡</td><td>gradient</td></tr><tr><td>DARTS+PT (best)</td><td>2.48</td><td>3.3</td><td>0.8‡</td><td>gradient</td></tr><tr><td>SDARTS-RS+PT (avg)*</td><td>2.54 ± 0.10</td><td>3.3</td><td>0.8‡</td><td>gradient</td></tr><tr><td>SDARTS-RS+PT (best)</td><td>2.44</td><td>3.2</td><td>0.8‡</td><td>gradient</td></tr><tr><td>SGAS+PT (Crit.1 avg)*</td><td>2.56 ± 0.10</td><td>3.9</td><td>0.29‡</td><td>gradient</td></tr><tr><td>SGAS+PT (Crit.1 best)</td><td>2.46</td><td>3.9</td><td>0.29‡</td><td>gradient</td></tr></table>

${}^{ \dagger  }$  Obtained on a different space with PyramidNet (Han et al.,2017) as the backbone.  
$\ddagger$  Recorded on a single GTX 1080Ti GPU.  
* Obtained by running the search and retrain under four different seeds and computing the average test error of the derived architectures.

the database. As in section 5.1 we take the pretrained supernet from Darts, and apply our method on top of it. All other settings are kept unmodified. Figure 4 shows the performance trajectory of Darts+PT compared with Darts. While the architectures found by magnitude-based selection degenerates overtime, perturbation-based method is able to stably extract better architectures from exactly the same underlying supernets. The result implies that the degenerated performance observed in Darts comes from the failure of magnitude based architecture selection.

# 6 ANALYSIS

# 6.1 ISSUE WITH THE ROBUSTNESS OF DARTS

Zela et al. (2020) observes that Darts tends to yield degenerate architectures with very poor performance. We conject that the robustness issue of Darts can in large be explained by the failure of magnitude-based architecture selection.

To show this, we test Darts' performance with perturbation-based architecture selection on four spaces proposed by Zela et al. (2020) (S1-S4). The complete specifications of these spaces can be found in Appendix A.2. Given a supernet, the architecture selected based on  $\alpha$  performs poorly across spaces and datasets (column 3 in Table 3). However, our method is able to consistently extract meaningful architectures that achieve significantly improved performance (Column 4 Table 3).

Table 3: Darts+PT on S1-S4 (test error (%)).  

<table><tr><td>Dataset</td><td>Space</td><td>DARTS</td><td>Darts+PT (Ours)</td><td>Darts+PT (fix α)*</td></tr><tr><td rowspan="4">C10</td><td>S1</td><td>3.84</td><td>3.50</td><td>2.86</td></tr><tr><td>S2</td><td>4.85</td><td>2.79</td><td>2.59</td></tr><tr><td>S3</td><td>3.34</td><td>2.49</td><td>2.52</td></tr><tr><td>S4</td><td>7.20</td><td>2.64</td><td>2.58</td></tr><tr><td rowspan="4">C100</td><td>S1</td><td>29.46</td><td>24.48</td><td>24.40</td></tr><tr><td>S2</td><td>26.05</td><td>23.16</td><td>23.30</td></tr><tr><td>S3</td><td>28.90</td><td>22.03</td><td>21.94</td></tr><tr><td>S4</td><td>22.85</td><td>20.80</td><td>20.66</td></tr><tr><td rowspan="4">SVHN</td><td>S1</td><td>4.58</td><td>2.62</td><td>2.39</td></tr><tr><td>S2</td><td>3.53</td><td>2.53</td><td>2.32</td></tr><tr><td>S3</td><td>3.41</td><td>2.42</td><td>2.32</td></tr><tr><td>S4</td><td>3.05</td><td>2.42</td><td>2.39</td></tr></table>

* This column will be explained later in Section 7

Notably, Darts+PT is able to find meaningful architecture on S2 (skip_connect, sep_conv_3x3) and S4 (noise, sep_conv_3x3) where Darts failed dramatically. As shown in Figure 5 on S2, while magnitude-based selection degenerates to architectures filled with skip connections, Darts+PT is able to find architecture with 4 convolutions; On S4, Darts+PT consistently favors sep_conv_3x3 on edges where  $\alpha$  selects noise.

![](images/75753ce2196ae588cce0921da3e50bf0a9183b4b4646d5c1c90999d9991533c8.jpg)  
(a) S2 (Darts)

![](images/815f7e907c8ed068f2eb87bc7c1b89b28b8fda25e46b7136754eac9dc429801a.jpg)  
Figure 5: Comparison of normal cells found on S2 and S4. Perturbation-based architecture selection (Darts+PT) is able to find reasonable architectures in cases where magnitude-based method (Darts) fails dramatically. The complete architecture can be found in Appendix A.5. Non-trivial operations are marked red.  
(b) S2 (Darts+PT)

![](images/f6a412b55f832cd1fe8c0be75b65978bfe703475ba83d3e9b1fded79c62cd724.jpg)  
(c) S4 (Darts)

![](images/59ee5cd3761af8c045d8c0e5e9ff4982da56c7a7de561f22981c5e32c694010d.jpg)  
(d) S4 (Darts+PT)

# 6.2 PROGRESSIVE TUNING

In addition to operation selection, we also tune the supernet after an edge is discretized so that the supernet could regain the lost accuracy. To measure the effectiveness of our operation selection criteria alone, we conduct ablation study on the progressive tuning part. Concretely, we test a baseline by combining progressive tuning with magnitude-based operation selection instead of our selection criteria, which we code-named Darts+PT-Mag. Figure 6 plots the change of validation accuracy of Darts+PT and Darts+PT-Mag during the operation selection phase. As we can see, Darts+PT is able to identify better operations that leads to higher validation accuracy than magnitude-based method, revealing the effectiveness of our operation selection criteria. Moreover, Darts+PT-Mag is only able to obtain a test error of  $2.85\%$  on

Darts space on cifar10, much worse than Darts+PT (2.61%), indicating that the operation selection part plays a crucial role in our method.

![](images/655d9b5f5e8f4852ea457a508e5b1e10e594c05639b3ee406f67470eace8eb4f.jpg)  
Figure 6: Trend of validation accuracy in the operation selection phase on S2. Darts+PT is able to select better operations that lead to higher accuracy of the supernet than Darst+PT-Mag.

# 7 CONCLUSION AND DISCUSSION

This paper attempts to understand Differentiable NAS from the architecture selection perspective. We re-examine the magnitude-based architecture selection process of Darts and provide empirical and theoretical evidence on why it does not indicate the underlying operation strength. We introduce an alternative perturbation-based architecture selection method which measures the operation strength directly via its contribution to the supernet performance.

The proposed selection method is able to consistently extract improved architecture from supernets trained identical to the respective base methods on several spaces and datasets.

Table 4: Darts+PT v.s. Darts+PT (fixed  $\alpha$  ) on more spaces (test error  $\%$  ) on CIFar10.  

<table><tr><td>Space</td><td>DARTS</td><td>Darts+PT</td><td>Darts+PT (fix α)</td></tr><tr><td>Darts Space</td><td>3.00</td><td>2.61</td><td>2.87</td></tr><tr><td>NasBench201</td><td>45.7</td><td>11.89</td><td>6.20</td></tr></table>

Our method brings more freedom in supernet training as it does not rely on  $\alpha$  to derive the final architecture. For example, we find that by fixing  $\alpha = 0$  (uniform weights for all the operations) while training supernet and apply perturbation-based architecture selection, the resulting method performs on-par with Darts+PT, and in some cases even better (the last column on Table3 and Table4). This surprising finding suggests that even the most naive approach, simply training a supernet without  $\alpha$ , will be a competitive method when combining with the proposed perturbation-based architecture selection. We therefore hope the perturbation-based architecture selection can bring a new perspective to the NAS community to rethink the role of  $\alpha$  in Differential NAS.

# REFERENCES

Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding and simplifying one-shot architecture search. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 550-559, Stockholm, Sweden, 10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/bender18a.html.  
Kaifeng Bi, Changping Hu, Lingxi Xie, Xin Chen, Longhui Wei, and Qi Tian. Stabilizing darts with amended gradient estimation on architectural parameters, 2019.  
Andrew Brock, Theo Lim, J.M. Ritchie, and Nick Weston. SMASH: One-shot model architecture search through hypernetworks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rydeCEhs-  
Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=HylVB3AqYm  
Xiangning Chen and Cho-Jui Hsieh. Stabilizing differentiable architecture search via perturbation-based regularization. In Proceedings of the 37th International Conference on Machine Learning, 2020.  
Xiangning Chen, Ruochen Wang, Minhao Cheng, Xiaocheng Tang, and Cho-Jui Hsieh. Drnas: Dirichlet neural architecture search, 2020.  
Xin Chen, Lingxi Xie, Jun Wu, and Qi Tian. Progressive differentiable architecture search: Bridging the depth gap between search and evaluation. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1294-1303, 2019.  
Xuanyi Dong and Yi Yang. Searching for a robust neural architecture in fourgpu hours. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1761-1770, 2019.  
Xuanyi Dong and Yi Yang. Nas-bench-201: Extending the scope of reproducible neural architecture search. In International Conference on Learning Representations (ICLR), 2020.  
Klaus Greff, Rupesh K. Srivastava, and Jürgen Schmidhuber. Highway and residual networks learn unrolled iterative estimation. In International Conference on Learning Representations (ICLR), 2017.  
Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling, 2019.  
Dongyoon Han, Jiwhan Kim, and Junmo Kim. Deep pyramidal residual networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jul 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778, 06 2016. doi: 10.1109/CVPR.2016.90.  
Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jul 2017. doi: 10.1109/cvpr.2017.243. URL http://dx.doi.org/10.1109/CVPR.2017.243  
Guilin Li, Xing Zhang, Zitong Wang, Zhenguo Li, and Tong Zhang. Stacnas: Towards stable and consistent differentiable neural architecture search, 2019.  
Guohao Li, Guocheng Qian, Itzel C. Delgadillo, Matthias Muller, Ali Thabet, and Bernard Ghanem. Sgas: Sequential greedy architecture search. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1620-1630, 2020.

Liam Li and Ameet Talwalkar. Random search and reproducibility for neural architecture search, 2019.  
Hanwen Liang, Shifeng Zhang, Jiacheng Sun, Xingqiu He, Weiran Huang, Kechen Zhuang, and Zhenguo Li. Darts+: Improved differentiable architecture search with early stopping, 2019.  
Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. Lecture Notes in Computer Science, pp. 19-35, 2018.  
Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray Kavukcuoglu. Hierarchical representations for efficient architecture search, 2017.  
Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In International Conference on Learning Representations, 2019.  
Renqian Luo, Fei Tian, Tao Qin, Enhong Chen, and Tie-Yan Liu. Neural architecture optimization. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 7816-7827. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/8007-neural-architecture-optimization.pdf  
Hieu Pham, Melody Guan, Barret Zoph, Quoc Le, and Jeff Dean. Efficient neural architecture search via parameters sharing. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 4095-4104, Stockholm, Sweden, 10-15 Jul 2018. PMLR.  
Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan, Quoc V. Le, and Alexey Kurakin. Large-scale evolution of image classifiers. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, pp. 2902-2911. JMLR.org, 2017.  
Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Regularized evolution for image classifier architecture search. Proceedings of the AAAI Conference on Artificial Intelligence, 33: 4780-4789, Jul 2019. ISSN 2159-5399. doi: 10.1609/aaai.v33i01.33014780. URL http://dx.doi.org/10.1609/aaai.v33i01.33014780.  
Kenneth O. Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. Evolutionary Computation, 10(2):99-127, 2002. doi: 10.1162/106365602320169811. URL https://doi.org/10.1162/106365602320169811.  
Andreas Veit, Michael Wilber, and Serge Belongie. Residual networks behave like ensembles of relatively shallow networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 29, pp. 550-558. Curran Associates, Inc., 2016.  
Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. SNAS: stochastic neural architecture search. In International Conference on Learning Representations, 2019.  
Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian, and Hongkai Xiong. PC-DARTS: Partial channel connections for memory-efficient architecture search. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=BJ1S634tPr  
Quanming Yao, Ju Xu, Wei-Wei Tu, and Zhanxing Zhu. Efficient neural architecture search via proximal iterations. In AAAI, 2020.  
Arber Zela, Thomas Elsken, Tonmoy Saikia, Yassine Marrakchi, Thomas Brox, and Frank Hutter. Understanding and robustifying differentiable architecture search. In International Conference on Learning Representations, 2020.  
Zhao Zhong, Junjie Yan, Wei Wu, Jing Shao, and Cheng-Lin Liu. Practical block-wise neural network architecture generation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, 2018.

Hongpeng Zhou, Minghao Yang, Jun Wang, and Wei Pan. Bayesnas: A bayesian approach for neural architecture search. In ICML, pp. 7603-7613, 2019. URL http://proceedings.mlr.press/v97/zhou19e.html  
Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In International Conference on Learning Representations (ICLR), 2017. URL https://arxiv.org/abs/1611.01578  
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Jun 2018. doi: 10.1109/cvpr.2018.00907. URL http://dx.doi.org/10.1109/CVPR.2018.00907