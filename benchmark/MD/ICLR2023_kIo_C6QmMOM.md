# COUPLED MULTIWAVELET NEURAL OPERATOR LEARNING FOR COUPLED PARTIAL DIFFERENTIAL EQUATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Coupled partial differential equations (PDEs) are key tasks in modeling the complex dynamics of many physical processes. Recently, neural operators have shown the ability to solve PDEs by learning the integral kernel directly in Fourier/Wavelet space, so the difficulty for solving the coupled PDEs depends on dealing with the coupled mappings between the functions. Towards this end, we propose a coupled multiwavelets neural operator (CMWNO) learning scheme by decoupling the coupled integral kernels during the multiwavelet decomposition and reconstruction procedures in the Wavelet space. The proposed model achieves significantly higher accuracy compared to previous learning-based solvers in solving the coupled PDEs including Gray-Scott (GS) equations and the non-local mean field game (MFG) problem. According to our experimental results, the proposed model exhibits a  $2X - 4X$  improvement relative  $L2$  error compared to the best results from the state-of-the-art models.

# 1 INTRODUCTION

Human perception relies on detecting and processing waves. While our eyes detect waves of electromagnetic radiation, our ears detect waves of compression in the surrounding air. Going beyond waves, from complex dynamics of blood flow to sustain tissue growth and life, to navigating underwater, ground and aerial vehicles at high speeds requires discovering, learning and controlling the partial differential equations (PDEs) governing individual or webs of biological, physical and chemical phenomena (Lacasse et al., 2007; Henriquez, 1993; Laval & Leclercq, 2013; Ghanavati et al., 2017; Radmanesh et al., 2020). Within this context, neural operators have been successfully used to learn and solve various PDEs. By representing the integral kernel termed as Green's function in the Fourier or Wavelet spaces, the fourier neural operator (Li et al., 2020b) and the multiwavelet-based neural operator (Gupta et al., 2021b,a)) exhibit significant improvements on solving PDEs compared with previous work. However, when it comes to coupled systems characterized by coupled differential equations such as mean field games (MFGs) (Lasry & Lions, 2007; Achdou & Capuzzo-Dolcetta, 2010) or analysis of the surface currents in the tropical Pacific Ocean (Bonjean & Lagerloef, 2002), the interactions between the variables / functions need to be considered to decouple the system. Without the knowledge of underlying PDEs, the complex interactions can be hardly represented in the data-driven model. To build a data-driven model that can give a general representation of the interactions to efficiently solve coupled differential equations, we propose the coupled multiwavelets neural operator (CMWNO).

Neural Operators. Neural operators (Li et al., 2020b; Gupta et al., 2021b; Bhattacharya et al., 2020; Patel et al., 2021) focus on learning the mapping between infinite-dimensional spaces of functions. The critical feature for neural operators is to model the integral operator namely the Green's function through various neural network architectures. The graph neural operators (Li et al., 2020b; use the graph kernel to model the integral operator inspired by graph neural networks; the Fourier neural operator (Li et al., 2020b) uses an iterative architecture to learn the integral operator in Fourier space. The multiwavelet neural operators (Gupta et al., 2021b; a) utilize the non-standard form of the multiwavelets to represent the integral operator through 4 neural networks in the Wavelet space. The neural operators are completely data-driven and resolution independent by learning the mapping between the functions directly, which can achieve the state-of-the-art performance on

solving PDEs and initial value problems (IVPs). To deal with coupled PDEs in the coupled system and be data-efficient, we aim to decode the various interaction scenarios inside the neural operators.

Multiwavelet Transform. In contrast to wavelets, multiwavelets (refer to Appendix C) use more than one scaling functions which are orthogonal. The multiwavelets exploit the advantages of wavelets, such as  $(i)$  the vanishing moments,  $(ii)$  the orthogonality, and  $(iii)$  the compact support. Along the essence of wavelet transform, a series of wavelet bases are introduced with scaled / shifted versions in multiwavelets to construct the basis of the coarsest scale polynomial subspace. The multiwavelet bases have been proved to be successful for representing integral operators as shown in (Alpert et al., 1993) (the discrete version of multiwavelets) and (Alpert, 1993b). In our proposed model, to develop compactly supported multiwavelets, we use the Legendre polynomials (Appendix D) which are non-zero only over a finite interval as the basis. The differential  $(\partial/\partial t)$  and the integral  $(\iint_{\Omega})$  operators can be represented by the first-order multiwavelet coefficients  $(s$  and  $d)$  of orthogonal bases via decomposition in the Wavelet space.

Mean Field Games (MFGs). As a representative problem for coupled systems in the real world, MFGs gains raising attentions in various areas, including economics (Achdou et al., 2014; 2022), finance (Gueant et al., 2011; Huang et al., 2019) and engineering (De Paola et al., 2019; Gomes et al., 2021), etc. Building on statistical mechanics principles and infusing them into the study of strategic decision making, MFGs investigate the dynamics of a large population of interacting agents seen as particles in a thermodynamic gas. Simply speaking, MFGs consist of  $(i)$  a Fokker-Planck equation (or related PDE) that describes the dynamics of the aggregate distribution of agents, which is coupled to  $(ii)$  a Hamilton-Jacobi-Bellman equation (another PDE) prescribing the optimal control of an individual (Lasry & Lions, 2006; 2007; Huang et al., 2006; 2007). Among different types of MFGs, the class of non-potential MFGs system with mixed couplings is particularly important as it is more reflective of the real world with a continuum of agents in a differential game.

Solutions on MFGs. Previous works either only restrict to systems without non-local coupling, such as alternating direction method of multipliers (ADMM) (Benamou & Carlier, 2015; Benamou et al., 2017) and primal-dual hybrid gradient (PDHG) algorithm (Briceno-Arias et al., 2019; 2018) or use general purpose numerical methods for solving the MFG problems (Achdou et al., 2013a,b; Achdou & Capuzzo-Dolcetta, 2010), which misses specific information from the target structure. In addition, the aforementioned works are not parallelizable with linear computational cost under the coupled MFGs settings. Recently, (Liu & Nurbekyan, 2020) considers dual variables of nonlocal couplings in Fourier or feature space. Furthermore, (Liu et al., 2021) expands the feature-space in the kernel-based representations of machine learning methods and uses expansion coefficients to decouple the mean field interactions. However, both dual variables and expansion coefficients need to bound the interactions of coupled system in a reasonable interval with prior knowledge. In our work, we first introduce the neural operator into coupled MFG fields, which can decouple the various interactions inside the multiwavelet domain.

Novel Contributions. The main novel contributions of our work are summarized as follows:

- For coupled differential equations, we propose a coupled neural operator learning scheme, named CMWNO. To the best of our knowledge, CMWNO is the first work using pure data-driven method to solve coupled differential equations.  
- Utilizing multiwavelet transform, CMWNO can deal with the interactions between the kernels of coupled differential equations in the Wavelet space. Specifically, we first yield the representation of coupled information during the decomposition process of multiwavelet transform. Then, the decoupled representation can interact separately to help the operators' reconstruction process. In addition, we propose a dice strategy to mimic the information interaction during the training process.  
- The proposed model successfully learns the interaction between the coupled variables when the couple degree is increasing and thus it could open new directions for studying complex coupled systems via data-driven methods. Experimentally, the proposed CMWNO framework offers the state-of-the-art performance on both Gray-Scotts (GS) equations and non-local MFGs. Specifically, CMWNO outperforms the best baseline  $(\mathrm{MWT}_c)$  by  $54.0\%$

on GS equations with various resolutions and outperforms the best baseline  $(\mathrm{FNO}_c)$  by  $61.4\%$  on non-local MFGs with different time steps.

# 2 COUPLED MULTIWAVELET NEURAL OPERATORS LEARNING

To solve a coupled control system characterized by coupled state equations in control theory, a popular way is to use the Laplace operator  $s$  to represent differential and integral operators (Gilbarg et al., 1977). Therefore, the coupled high-order differential equations can be transformed into the first-order differential equations in the Laplace space which will reduce the decoupling difficulty. Inspired by the use of the Laplace operator and the properties of the multiwavelets, we assume that the interactions between kernels can be used to approximate the coupled information by reducing the degree of high-order operators in multiwavelet bases. With this assumption, we are able to build the coupled multiwavelet neural operators (CMWNO) learning scheme, which utilizes decomposition representation from the operator and mimic the interaction via a dice strategy.

# 2.1 COUPLED DIFFERENTIAL EQUATIONS

To provide a simple example of the coupled kernels,  $\kappa_{1}$  and  $\kappa_{2}$ , let us consider a general coupled system with 2 coupled variables  $u(x,t)$  and  $v(x,t)$  with the given initial conditions  $u_0(x)$  and  $v_{0}(x)$ . Given  $\mathcal{A}$  and  $\mathcal{U}$  as two Sobolev spaces  $\mathcal{H}^{s,p}$  with  $s > 0$ ,  $p = 2$ , let  $T$  denote a generic operator such that  $T:\mathcal{A}\to \mathcal{U}$ . Without the knowledge of how these two variables are coupled, to solve for  $u(x,\tau)$  and  $v(x,\tau)$ , we need two operators  $T_{1}$  and  $T_{2}$  such that  $T_{1}u_{0}(x) = u(x,\tau)$  and  $T_{2}v_{0}(x) = v(x,\tau)$ . The coupled kernels termed as Green's function can be written as follows:

$$
\begin{array}{l} T _ {1} u _ {0} (x) = \int_ {D} \kappa_ {1} (x, y, u _ {0} (x), u _ {0} (y), v _ {0} (x), v _ {0} (y), \kappa_ {2}) u _ {0} (y) d y, \\ T _ {2} v _ {0} (x) = \int_ {D} \kappa_ {2} (x, y, u _ {0} (x), u _ {0} (y), v _ {0} (x), v _ {0} (y), \kappa_ {1}) v _ {0} (y) d y, \tag {1} \\ u (x, 0) = u _ {0} (x); \quad v (x, 0) = v _ {0} (x), \quad x \in D, \\ \end{array}
$$

where  $D \subset \mathbb{R}^d$  is a bounded domain. The interacted kernels cannot be directly solved without considering the interference from the other kernel, and our idea is to simplify the kernels first and deal with the interactions in the multiwavelet domain.

# 2.2 MULTIWAVELET OPERATOR

To briefly introduce the multiwavelet operator, we explain how the neural networks are used to represent the kernel in this section. The basic concept of multiresolution analysis (MRA) and multiwavelets (Alpert et al., 1993; Alpert, 1993a,b) are provided in the Appendix C

Notation For  $k \in \mathbb{Z}$  and  $n \in \mathbb{N}$ , the space of piecewise polynomial functions is defined as:  $\mathbf{V}_n^k = \{f|_{\text{the restriction of } f \text{ to the interval } (2^{-n}l, 2^{-n}(l + 1)) \text{ is a polynomial of degree } < k, \text{ for all } l = 0, 1, \ldots, 2^n - 1, \text{ and } f \text{ vanishes elsewhere}\}$ .  $\mathbf{V}_0^k$  consists of the orthogonal scaling functions  $\varphi_i$  with  $i = 0, \ldots, n$ , and  $\mathbf{V}_n^k$  can be spanned by shifting and scaling these functions as  $\varphi_{jl}^n(x) = 2^{n/2} \varphi_j(2^n x - l)$ , where  $j = 0, \ldots, k - 1$  and  $l = 0, \ldots, 2^n - 1$ . The coefficients of  $\varphi_{jl}^n(x)$  are called scaling coefficients marked as  $s_{jl}^n$ . The multiwavelet subspace  $\mathbf{W}_n^k$  is defined as the orthogonal complement of  $\mathbf{V}_n^k$  in  $\mathbf{V}_{n+1}^k$  such that  $\mathbf{V}_n^k \oplus \mathbf{W}_n^k = \mathbf{V}_{n+1}^k$ ,  $\mathbf{V}_n^k \perp \mathbf{W}_n^k$ .  $\mathbf{W}_0^k$  consists of the orthogonal wavelet functions  $\psi_i$  with  $i = 0, \ldots, n$ . Similar to  $\mathbf{V}_n^k$ ,  $\mathbf{W}_n^k$  is composed of the wavelets functions  $\psi_{jl}^n(x)$  with wavelets coefficients  $d_{jl}^n$ .

To represent the functions and learn the mapping in multiwavelet space, the nonstandard form is used to represent the integral operator. According to (Beylkin et al., 1991; Alpert et al., 2002b), an orthogonal projection operator  $P_{n}^{k}: \mathcal{H}^{s,2} \to \mathbf{V}_{n}^{k}$ , and  $Q_{n}^{k}: \mathcal{H}^{s,2} \to \overline{\mathbf{W}}_{n}^{k}$  with  $Q_{n}^{k} = P_{n+1}^{k} - P_{n}^{k}$ , then an single operator  $T$  in our coupled system can be represented as:

$$
T = \bar {T} _ {0} ^ {k} + \sum_ {n = 0} ^ {\infty} \left(A _ {n} ^ {k} + B _ {n} ^ {k} + C _ {n} ^ {k}\right), \tag {2}
$$

where  $\bar{T}_0^k = P_0^k TP_0^k, A_n^k = Q_n^k TQ_n^k, B_n^k = Q_n^k TP_n^k, C_n^k = P_n^k TQ_n^k$ ,  $Q_{n}^{k}$  is the multiwavelet operator. Therefore, the nonstandard forms of the operator is a collection of triplets  $\{\bar{T}_0^k,(A_i^k,B_i^k,C_i^k)_{n = 0,1,\ldots}\}$ . For a given operator  $T:Tu_0(x) = u_\tau (x)$ , the map under wavelet space can be written as:

$$
T _ {d l} ^ {i} = A _ {i} ^ {k} d _ {l} ^ {i} + B _ {i} ^ {k} s _ {l} ^ {i}, \quad T _ {\hat {s} l} ^ {i} = C _ {i} ^ {k} d _ {l} ^ {i}, \quad T _ {s l} ^ {0} = \bar {T} s _ {l} ^ {0}, \quad i = 0, 1, \dots , n \tag {3}
$$

where,  $(T_{sl}^{i}, T_{dl}^{i}) / (s_{l}^{i}, d_{l}^{i})$  are the scaling/wavelet coefficients of  $u_{\tau}(x) / u_0(x)$  in subspace  $\mathbf{V}_{i + 1}^k$ . In our model, one kernel is approximated using 4 simple neural networks  $A, B, C$  and  $\bar{T}$  such that  $T_{dl}^{i} \approx A_{\theta_A}(d_l^i) + B_{\theta_B}(s_l^i), T_{\hat{s} l}^{i} \approx C_{\theta_C}(d_l^i)$ , and  $T_{sl}^{0} \approx \bar{T}_{\theta_{\bar{T}}} (s_l^L)$ .

# 2.3 COUPLED MULTIWAVELETS MODEL

This section introduces a coupled multiwavelets model to provide a general solution on coupled differential equations. First, we make a mild assumption to decouple two coupled operators given in Section 2.1. To simplify eq. 1 without loss of generality, we assume that we can build two operators  $T_{u}$  and  $T_{v}$  to approximate  $u(x,\tau)$  and  $v(x,\tau)$ , where  $T_{u}$  and  $T_{v}$  are decoupled and do not carry any interference from each other. In other words, we can write  $T_{u}u_{0}(x) = u^{\prime}(x,\tau);T_{v}v_{0}(x) = v^{\prime}(x,\tau)$  where  $u^{\prime}(x,\tau)$  and  $v^{\prime}(x,\tau)$  are the approximations of  $u(x,\tau)$  and  $v(x,\tau)$  without coupling. The assumption is mild and easy to get satisfied in the Wavelet space since the operators can be represented by the first-order multiwavelet coefficients. According to this assumption, we can derive the following relations:

$$
u (x, \tau) = T _ {u} u (x, 0) + \epsilon_ {1} \left(T _ {v}\right), \quad x \in D \tag {4}
$$

$$
v (x, \tau) = T _ {v} v (x, 0) + \epsilon_ {2} (T _ {u}), \quad x \in D
$$

where  $\epsilon_{1}(T_{u})$  quantifies the interference from operator  $T_{v}$  to solve  $u(x,\tau)$  and  $\epsilon (T_v)$  represents the measurable interaction from operator  $T_{u}$ . Therefore, the integral operators can be written as:

$$
T _ {u} u _ {0} (x) = \int_ {D} \kappa_ {u} (x, y) u _ {0} (y) d y, \tag {5}
$$

$$
T _ {v} v _ {0} (x) = \int_ {D} \kappa_ {v} (x, y) v _ {0} (y) d y,
$$

the kernels  $\kappa_{u}$  and  $\kappa_{v}$  termed as Green's functions can be learned through neural operators, where  $\kappa_{u}$  can be learned using the data of  $u$  while the kernel  $\kappa_{v}$  is learned from  $v$ . To model  $\epsilon_1(T_u)$  and  $\epsilon_2(T_v)$ , we transform the operators into multiwavelet coefficients in the Wavelet space and embed it through simple linear combination after the decomposition steps.

Based on the concept of multiwavelets (Appendix Section  $\mathbb{C}$ ), here we simply explain the decomposition step and reconstruction step of multiwavelets in our coupled system. Since  $\mathbf{V}_n^k = \mathbf{V}_{n - 1}^k\oplus \mathbf{W}_{n - 1}^k$  according to Section 2.2, the bases of  $V_{n}^{k}$  can be written as a linear combination of the scaling functions  $\varphi_i^{n - 1}$  and the wavelet functions  $\psi_i^{n - 1}$ . The linear coefficients  $(H^{(0)},H^{(1)},G^{(0)},G^{(1)})$  are termed as multiwavelet decomposition filters, transforming representation between subspaces  $\mathbf{V}_{n - 1}^k,\mathbf{W}_{n - 1}^k$ , and  $\mathbf{V}_n^k$ . For a given function  $f(x)$ , the scaling/wavelet coefficients  $s_{jl}^{n} / d_{jl}^{n}$  of scaling/wavelet functions  $\varphi_{jl}^{n} / \psi_{jl}^{n}$  are computed as:

$$
s _ {j l} ^ {n} = \int_ {2 ^ {- n} l} ^ {2 ^ {- n} (l + 1)} f (x) \varphi_ {j l} ^ {n} (x) d x; \quad d _ {j l} ^ {n} = \int_ {2 ^ {- n} l} ^ {2 ^ {- n} (l + 1)} f (x) \psi_ {j l} ^ {n} (x) d x \tag {6}
$$

Using the multiwavelet decomposition filters, the relations between the coefficients on two consecutive levels  $n$  and  $n + 1$  are computed as (decomposition step):

$$
\mathbf {s} _ {l} ^ {n} = H ^ {(0)} \mathbf {s} _ {2 l} ^ {n + 1} + H ^ {(1)} \mathbf {s} _ {2 l + 1} ^ {n + 1}; \quad \mathbf {d} _ {l} ^ {n} = G ^ {(0)} \mathbf {s} _ {2 l} ^ {n + 1} + G ^ {(1)} \mathbf {s} _ {2 l + 1} ^ {n + 1}. \tag {7}
$$

Therefore, starting with the coefficients  $s_l^n$ , we repeatedly apply the decomposition step in eq. 7 to compute the scaling/wavelet coefficients on coarser levels. Similarly, the reconstruction step can be represented as:

$$
\mathbf {s} _ {2 l} ^ {n + 1} = H ^ {(0) T} \mathbf {s} _ {l} ^ {n} + G ^ {(0) T} \mathbf {d} _ {l} ^ {n}, \quad \mathbf {s} _ {2 l + 1} ^ {n + 1} = H ^ {(1) T} \mathbf {s} _ {l} ^ {n} + G ^ {(1) T} \mathbf {d} _ {l} ^ {n}. \tag {8}
$$

Repeatedly applying the reconstruction step, we can compute the coefficients  $s_l^n$  from  $s_l^0$  and  $d_l^i$ ,  $i = 0, \dots, n$ . In general, the function can be parameterized as the scaling/wavelet coefficients in

![](images/b2e7954cc695cd51b4305b9fb81b8cb20a9b06aa58b1db4cf99bdfad5551523f.jpg)  
Figure 1: Architecture of CMWNO. Note that there are two coupled operators,  $T_{u}$  and  $T_{v}$ , in our system, which aligns the number of coupled variables. The network  $\bar{T}$  is only applied for the coarsest scale  $L$  (0 in this system). The dashed arrows correspond to the auxiliary information from the unused operator without gradient during training process. For the interaction between operators, when we update the operator  $T_{u}$ , the decomposed ingredients from  $T_{v}$  will be equipped into the reconstruction module of  $T_{u}$  in the Wavelet domain, vice versa.

the Wavelet space after the decomposition steps, and the coefficients can be mapped to the function after reconstruction steps. In our work, to model the interference  $\epsilon_{1}(T_{u})$  and  $\epsilon_{2}(T_{v})$ , we obtain the multiwavelets coefficients of each kernel during the decomposition steps and embed them into the other kernel in the reconstruction step. Note that we will elaborate the detailed training strategy of how to mimic interactions inside our system in Section 2.4

Our idea is to represent the functions and operators in Wavelet space to decouple the system using simple linear combinations. Considering the example in Section 2.1, according to the eq. 4 and 5, we first build two operators  $T_{u}$  and  $T_{v}$  such that  $T_{u}u_{0}(x) = u^{\prime}_{\tau}(x)$ ;  $T_{v}v_{0}(x) = v^{\prime}_{\tau}(x)$ . For the operators  $T_{u}$  and  $T_{v}$ , we denote their scaling/wavelet coefficients in wavelet domain as  $T_{u,sl}^{i} / T_{u,dl}^{i}$  and  $T_{v,sl}^{i} / T_{v,dl}^{i}$  respectively. For the input  $u_{0}(x) / v_{0}(x)$  and the output  $u_{\tau}(x) / v_{\tau}(x)$ , we denote their coefficients as  $U_{0,s(d)l}^{i} / V_{0,s(d)l}^{i}$  and  $U_{\tau,s(d)l}^{i} / V_{\tau,s(d)l}^{i}$ . According to eqs. 3 and 5, the multiwavelet coefficients of  $T_{u}$  and  $T_{v}$  can be calculated as:

$$
T _ {u, d l} ^ {i} = A _ {u, i} ^ {k} U _ {0, d l} ^ {i} + B _ {u, i} ^ {k} U _ {0, s l} ^ {i}, \quad T _ {u, \hat {s} l} ^ {i} = C _ {u, i} ^ {k} U _ {0, d l} ^ {i}, \quad T _ {u, s l} ^ {0} = \bar {T} U _ {0, s l} ^ {0}; \tag {9}
$$

$$
T _ {v, d l} ^ {i} = A _ {v, i} ^ {k} V _ {0, d l} ^ {i} + B _ {v, i} ^ {k} V _ {0, s l} ^ {i}, \quad T _ {v, \hat {s} l} ^ {i} = C _ {v, i} ^ {k} V _ {0, d l} ^ {i}, \quad T _ {v, s l} ^ {0} = \bar {T} V _ {0, s l} ^ {0},
$$

where  $i = 0,1,\dots ,n$ . Considering the interference from the other operators, the coefficients of the solutions  $u_{\tau}(x)$  and  $v_{\tau}(x)$  in the Wavelet space can be written as:

$$
U _ {\tau , d l} ^ {i} = T _ {u, d l} ^ {i} + \hat {T} _ {v, d l} ^ {i}, \quad U _ {\tau , \hat {s} l} ^ {i} = T _ {u, \hat {s} l} ^ {i} + \hat {T} _ {v, \hat {s} l} ^ {i}, \quad U _ {\tau , s l} ^ {0} = T _ {u, s l} ^ {0} + \hat {T} _ {v, s l} ^ {0};
$$

$$
V _ {\tau , d l} ^ {i} = T _ {v, d l} ^ {i} + \hat {T} _ {u, d l} ^ {i}, \quad V _ {\tau , \hat {s} l} ^ {i} = T _ {v, \hat {s} l} ^ {i} + \hat {T} _ {u, \hat {s} l} ^ {i}, \quad V _ {\tau , s l} ^ {0} = T _ {v, s l} ^ {0} + \hat {T} _ {u, s l} ^ {0}; \tag {10}
$$

where  $i = 0,1,\ldots ,n$ . In the training process, the inputs of the neural networks  $\{A_{u / v},B_{u / v},C_{u / v},\bar{T}_{u / v}\}$  are the multiwavelet coefficients of  $u_0(x) / v_0(x)$ , and the outputs are the multiwavelet coefficients of  $T_{u} / T_{v}$ . When the neural networks  $\{A_u,B_u,C_u,\bar{T}_u\}$  are trained for  $T_{u}$ , the neural networks  $\{A_v,B_v,C_v,\bar{T}_v\}$  output  $T_{v,[sl,dl]}^i$  without backpropagation, we use  $\hat{T}_{[u,v],[sl,dl]}^i$  to mark the coefficients without gradient. Utilizing the orthogonality of the multiwavelets, the coefficients embedding the information of the operators  $T_{u} / T_{v}$  can be directly added to  $T_{v} / T_{u}$  in the same Wavelet space  $V_{n}^{k}$ , then the neural networks with backpropagation can learn the information from the other operator. In that way, the complex coupled equations can be solved via reducing the order of the functions and directly approximate decoupled functions at each iteration.

The architecture of the CMWNO is shown in Fig. ① which illustrates the mapping process inside the wavelet space of layer  $n$ . The operations inside the wavelet space can be matched by the order of layers in the models, which means the decomposition operations for different resolutions are done independently. After decomposing  $s^n$  via eq. ⑦ we can get the transferred information of input where each component will be used to reconstruct the original input at the layer  $n$ .

# 2.4 DICE STRATEGY

![](images/c90662ef59f2f34826f9b2428caca92c0eeca73b31c230076d2a7a91c6fa7441.jpg)  
Figure 2: Dice strategy. For each sample, one only needs to go through a specific path (round diagonal corner rectangle). Inside each path, the order of updating is from left to right, where the darker block indicates the operator we want to update and the lighter blocks provide decomposition information from the fixed operator.

Inspired by scheduled sampling (Bengio et al. 2015), which is designed to gently bridge the discrepancy between training and inference samples, we propose rolling the dice to randomly decide the interaction order between each neural operators, which is named dice strategy. Specifically, we roll the dice for every sample to decide which path to use, which can effectively mitigate the imbalance update problem for each kernel caused by the fixed training order. As illustrated in Fig. 2 when the dice tells the model to use path 1 (upper path), we will update operator  $T_{u}$  by equipping the coupled information from the other operator  $T_{v}$  first. Note that,  $T_{v}$  is learned by previous samples and have not updated yet. Then we use the updated operator  $T_{u}$  to decompose the initial state  $u_{0}$ , which can be used to

update  $T_{v}$ . Inside the Wavelet space with well-defined basis, where we are able to utilize varying orthogonal information from each initial state jointly. Note that this strategy is scalable to more operators and we left the design of this strategy for future work.

# 3 EXPERIMENTS

In this section, we empirically evaluate the proposed model on famous coupled PDEs such as the Gray-Scott (GS) equations and the non-local mean field game (MFG) problem characterized by coupled PDEs. Note that we compare against the state-of-the-art data-driven models which fits for our research goal to build efficient coupled operators for general downstream data-driven applications without sufficient expert knowledge. The experiments show that CMWNO not only achieves the lowest  $L2$  relative errors when solving coupled PDEs, but also works consistently great under different input conditions. For the data structure, since our datasets are functions, we apply point-wise evaluations on the input and output data. For example, for the function  $f(x), x \in D$ , we discretize the domain as  $x_{1}, \ldots, x_{s} \in D$ , where  $x_{i}$  are s-point discretization of the domain. Unless stated otherwise, we train on 1000 samples and test on 200 samples.

Model architecture. In our proposed model, for each operator, the neural networks  $A$ ,  $B$  and  $C$  use a single-layered convolutional neural networks while  $\bar{T}$  uses a single linear layer. Our model is extensible and each kernel constructed by 4 neural networks  $\{A,B,C,\bar{T}\}$  learning the mapping in wavelet space. The number of the kernels can be chosen based on the number of coupled variables or the number of explicit operators.

Benchmark models. We compare our model with physics-informed neural networks (PINN (Raissi et al., 2019)) and the state-of-the-art neural operators including Fourier neural operator (FNO), Multiwavelet-based neural operator (MWT), and Padé exponential model (Padé), which show the best performance on solving PDEs according to the experiment results in (Li et al., 2020b; Gupta et al., 2021b). The PINN combines the advantages of data-driven machine learning and physical modeling to train a model that automatically satisfies physical constraints with insufficient training data, and has comparable generalization performance to predict important physical parameters of the model while ensuring accuracy. One can incorporate the differential form constraints from PDEs into the design of the loss function of the neural network with automatic differentiation technique

in deep neural networks. Experimentally, the fully connected PINN should be slow or even nonconvergence when the solution of the PDE has high frequency or multi-resolution, which aligns with the previous works (Fuits & Tchelepi, 2020; Raissi, 2018; Zhu et al., 2019; Raissi & Karniadakis, 2018). In addition, PINN cannot directly be used to a complete data-driven scenario without exact PDE structure so we build the general loss function for coupled PDE according to (Connors et al., 2009).

For the benchmark neural operator models, since we have the coupled functions as input and output (e.g.,  $u$  and  $v$ ), we concatenate  $u$  and  $v$  for the models and marked the models as  $\mathrm{FNO}_c$ ,  $\mathrm{MWT}_c$ ,  $\mathrm{Padé}_c$ , and  $\mathrm{PINN}_c$ . We also use two single multiwavelet-based neural operators to learn  $u_{\tau}(x)$  ( $v_{\tau}(x)$ ) from  $v_{\tau}(x)$  ( $u_{\tau}(x)$ ) independently and mark the model as  $\mathrm{MWT}_s$ .

Training parameters. The neural operators and PINN are trained using Adam optimizer with a learning rate of 0.001 and decay of 0.95 after every 100 steps. The models are trained for a total of 500 epochs which is the same with training CMWNO for fair comparison. All experiments are done on an Nvidia A100 40GB GPUs.

# 3.1 GRAY-SCOTT (GS) EQUATIONS

The GS equations are coupled differential equations which model the underlying reaction and diffusion patterns of chemical species. It is also able to generate a wide range of patterns which exist in nature, such as bacteria, spirals and coral patterns. Each variable (i.e.,  $u$  and  $v$ ) diffuses independently with a linear growth or decay term, while coupled together by  $\pm uv^2$  (Trefethen & Embree, 2001; Driscoll et al., 2014). For a given field  $u(x,t) / v(x,t)$ , the GS equations take the form:

$$
\begin{array}{l} \partial_ {t} u (x, t) = \epsilon_ {1} \partial_ {x x} u (x, t) + F (1 - u (x, t)) - \lambda u (x, t) v ^ {2} (x, t), \quad x \in (0, 1 0), t \in (0, 1 ] \\ \partial_ {t} v (x, t) = \epsilon_ {2} \partial_ {x x} v (x, t) - (K + F) v (x, t) + \lambda u (x, t) v ^ {2} (x, t), \quad x \in (0, 1 0), t \in (0, 1 ] \tag {11} \\ u (x, 0) = u _ {0} (x); \quad v (x, 0) = v _ {0} (x), \quad x \in (0, 1 0) \\ \end{array}
$$

where  $\epsilon_1 = 1, \epsilon_2 = 10^{-2}, K = 6.62 \times 10^{-2}, F = 2 \times 10^{-2}$ . We use the coupling coefficient  $\lambda \in (0,1]$  to control the degree of coupling of  $u$  and  $v$ . We aim to learn the operators (i) mapping the initial condition  $u(x,0)$  to the solution  $u(x,t = 1)$  with the interference of  $v(x,t)$ ; (ii) mapping the initial condition  $v(x,0)$  to the solution  $v(x,t = 1)$  considering the interference of  $u(x,t)$ . The initial conditions are generated in Gaussian random fields (GRF) according to  $u_0(x), v_0(x) \sim \mathcal{N}(0,7^4(-\Delta + 7^2I)^{-2.5})$  with periodic boundary conditions. We also use a different scheme to generate  $u_0(x)$  by using the smooth random functions (Rand) in chebfun package (Driscoll et al., 2014) which returns a band-limited function defined by a Fourier series with independent random coefficients; the parameter  $\gamma$  specifies the minimal wavelength and here we choose  $\gamma = 0.5$ . Therefore, generating the initial conditions by different schemes, we have two combinations of the initial conditions (i.e.,  $u_0(x)$  and  $v_0(x)$ ) and we mark them as (U-GRF, V-GRF) and (U-Rand, V-GRF) respectively according to the generating schemes. Given the initial conditions, we solve the equations using a fourth-order stiff time-stepping scheme named as ETDRK4 (Cox & Matthews, 2002) with a resolution of  $2^{10}$ , and sub-sample this data to obtain the datasets with the lower resolutions.

Table 1: Gray-Scott (GS) equation benchmarks for different input resolution  $s$  at initial condition (U-GRF,V-GRF). The relative  $L2$  errors are shown for each model. Bolded values are the best results, and underlined values are the second best. Set the same below.  

<table><tr><td rowspan="2">Models</td><td colspan="2">s=256</td><td colspan="2">s=512</td><td colspan="2">s=1024</td></tr><tr><td>u</td><td>v</td><td>u</td><td>v</td><td>u</td><td>v</td></tr><tr><td>CMWNO</td><td>0.00468</td><td>0.00464</td><td>0.00492</td><td>0.00434</td><td>0.00471</td><td>0.00450</td></tr><tr><td>MWTs</td><td>0.08075</td><td>0.07308</td><td>0.08041</td><td>0.07382</td><td>0.07996</td><td>0.07213</td></tr><tr><td>MWTc</td><td>0.01445</td><td>0.00742</td><td>0.01408</td><td>0.00744</td><td>0.01334</td><td>0.00779</td></tr><tr><td>FNOc</td><td>0.01431</td><td>0.00812</td><td>0.01542</td><td>0.00819</td><td>0.01545</td><td>0.00885</td></tr><tr><td>Padéc</td><td>0.01904</td><td>0.00964</td><td>0.02070</td><td>0.01022</td><td>0.02233</td><td>0.01055</td></tr><tr><td>PINNc</td><td>0.44480</td><td>0.52605</td><td>0.45376</td><td>0.52903</td><td>0.45576</td><td>0.53183</td></tr></table>

Varying resolution. The results of our experiments on GS equations with different resolutions (i.e.,  $s = 256, 512, 1024$ ) are shown in Table [1]. As shown in the results, all the models exhibit the

![](images/146a61ea8247e1a9eb8fced49b8a21790caaca3ce88ea947b0d3471d111cfc01.jpg)  
Figure 4: Comparing the models by varying the coupling coefficient  $\lambda$  at the initial condition (U-GRF, V-GRF) with resolution  $s = 1024$ .

![](images/6f9c2add469b1afbc2bee1d8923a9402c534b3fbcf8e8a2e665717b550393475.jpg)

resolution independence, and the neural operators performs better than PINN. The model  $\mathrm{MWT}_c$  with concatenated data performs better than model  $\mathrm{MWT}_s$  using two independent single MWT models to train  $u$  and  $v$  separately, which indicates the information from  $v_0(x) / u_0(x)$  benefits the model predicting for  $u_{\tau}(x) / v_{\tau}(x)$ . For solving  $u_{\tau}(x) / v_{\tau}(x)$ , our proposed CMWNO outperforms  $3X / 2X$  improvements compared with the best benchmark with respect to relative  $L2$  error. The learning curve of the neural operators solving  $u_{\tau}(x)$  at resolution  $s = 1024$  is shown in Fig. 3

![](images/9bd462fa1da27608f003882481a19148d3b21be52da8bcede6466aa4fc961ded.jpg)  
Figure 3: Learning curve - Relative  $L2$  error vs epochs for neural operators.

Varying coupling coefficient By varying the coupling coefficient  $\lambda$  in the GS equations, we can get different degrees of coupling between  $u$  and  $v$  according to eq. [1] The higher value of  $\lambda$  means higher degree of coupling between  $u$  and  $v$ . Given the same initial conditions  $u_0(x)$  and  $v_0(x)$ , the outputs with different  $\lambda$  (i.e.,  $\lambda = 0.2, 0.4, 0.6, 0.8, 1$ ) are shown in Fig. [4]. The numerical results are in the Tables [3] and [4] (see Appendix [F]). It shows that as  $\lambda$  increases, all the models perform worse. For solving  $u_{\tau}(x)$ , compared with at  $\lambda = 0.2$ , the relative  $L2$  errors at  $\lambda = 1$  of the models increase by  $18.9\%$  (CMWNO);

459.6% (MWT $_s$ ); 105.9% (MWT $_c$ ); 107.4% (FNO $_c$ ); 107.4% (Padé $_s$ ). In terms of  $v_{(x,\tau)}$ , the numbers are 11.6% (CMWNO); 326.8% (MWT $_s$ ); 34.5% (MWT $_c$ ); 44.8% (FNO $_c$ ); 44.3% (Padé $_s$ ). As we can see, the MWT $_s$  works the worst since the model cannot learn the interaction between  $u$  and  $v$ . The models learning coupled operators through concatenated data works better than the single model but still do not perform well on high coupling data. On the contrary, our CMWNO outperforms well consistently with both low / high coupling coefficient, which indicates that our architecture is able to decouple the coupled kernels.

Varying initial conditions In addition to experimenting with both the initial conditions  $u_{0}(x)$  and  $v_{0}(x)$  generated in the GRF as marked (U-GRF, V-GRF), we also perform the models on (U-Rand,V-GRF). The numerical results are shown in Table 5 (see Appendix F). Our CMWNO achieves the lowest relative  $L2$  error on both  $u$  and  $v$  with  $3X$  and  $2X$  improvements respectively. We provide a sample of initial conditions in Fig.6 (see Appendix E), and Fig. 5 shows its predicted outputs from models CMWNO, MWT  $s$  and MWT  $c$ . It shows that our proposed CMWNO can give a precise prediction in a smooth way while MWT  $s$  and MWT  $c$  can only fit the true curve roughly.

# 3.2 MEAN FIELD GAME PROBLEM

For local interactions, directly discretizing interaction terms is economical. However, non-local MFG requires each player in making decisions to take into account the global information rather than local information, which will increase the amount of computation in the process of calculation. In other words, we need matrix multiplication on a full grid to calculate the interaction terms by evaluating the expressions  $\int_{\omega} K(x,y) \rho(y,t) dy$ . In this work, we propose a more general framework, CMWNO,

![](images/e433a552c1d2c6f57dc5fb256a424fbc48db698e7d5f655aaa4217ef6006f27f.jpg)  
Figure 5: The output of GS couple equations at the initial condition (U-Rand, V-GRF). (Left) The predicted output of the models to  $u(x,\tau = 1)$ . (Right) The predicted output of the models to  $v(x,\tau = 1)$ .

![](images/ba5d2025d65d5227f211cf3ebe68ad45a28cd4f2a530af559d779d7813363059.jpg)

Table 2: The relative  $L2$  errors for predicting  $\rho (x,t) / \varphi (x,t)$  with  $t = 0.2,0.4,0.6$ , and 0.8.  

<table><tr><td rowspan="2">Models</td><td colspan="2">t=0.2</td><td colspan="2">t=0.4</td><td colspan="2">t=0.6</td><td colspan="2">t=0.8</td></tr><tr><td>ρ</td><td>φ</td><td>ρ</td><td>φ</td><td>ρ</td><td>φ</td><td>ρ</td><td>φ</td></tr><tr><td>CMWNO</td><td>0.00083</td><td>0.00073</td><td>0.00154</td><td>0.00252</td><td>0.00543</td><td>0.00467</td><td>0.02417</td><td>0.00305</td></tr><tr><td>MWTc</td><td>0.00328</td><td>0.00646</td><td>0.00916</td><td>0.02244</td><td>0.02245</td><td>0.02768</td><td>0.06011</td><td>0.01622</td></tr><tr><td>FNOc</td><td>0.00241</td><td>0.00278</td><td>0.00473</td><td>0.00667</td><td>0.01329</td><td>0.01096</td><td>0.04950</td><td>0.00818</td></tr><tr><td>Padéc</td><td>0.00213</td><td>0.00320</td><td>0.00473</td><td>0.01307</td><td>0.01189</td><td>0.02466</td><td>0.03676</td><td>0.01171</td></tr><tr><td>PINNc</td><td>0.09519</td><td>0.00527</td><td>0.04045</td><td>0.01986</td><td>0.02737</td><td>0.02008</td><td>0.05523</td><td>0.01122</td></tr></table>

to model the interactions in the Wavelet space and the results show that our model can be used to deal with the coupled systems. Here we solve the non-local MFG which can be characterized as:

$$
\partial_ {t} \rho (x, t) + \nabla \cdot (\rho (x, t) \nabla \varphi (x, t)) = 0, \quad x \in [ 0, 1 ], t \in (0, 1)
$$

$$
\partial_ {t} \varphi (x, t) - \frac {1}{2} \| \varphi (x, t) \| ^ {2} + \int_ {D} K (x, y) \rho (y, t) d t = 0, \quad x \in [ 0, 1 ], t \in (0, 1) \tag {12}
$$

where  $\rho(x, t)$  is the density distribution of the players, and  $\varphi(x, t)$  is the cost function. In a forward-forward MFG setting (Gomes & Sedjro, 2017), we can obtain the value of  $\rho(x, 0)$  and  $\varphi(x, 0)$ . We aim to learn the operators: (i) mapping the initial condition  $\rho(x, 0)$  to the solution  $\rho(x, t = \tau)$  with the interference of  $\varphi(x, t)$ ; (ii) mapping the initial condition  $\varphi(x, 0)$  to the solution  $\varphi(x, t = \tau)$  considering the interference of  $\rho(x, t)$ . To obtain the datasets, we generate  $\rho(x, 0) / \rho(x, t = 1)$  by using the random functions in chebfun package with the wavelength parameter  $\gamma = 0.3 / 0.1$ , respectively. The coupled equations are numerically solved by the primal-dual hybrid gradient (PDHG) algorithm (Briceno-Arias et al., 2019; 2018) with the resolution  $s = 256$ . The initial conditions of  $\rho(x, 0)$  and  $\varphi(x, 0)$  are used as the input while the  $\rho(x, t)$  and  $\varphi(x, t)$  ( $t = 0.2, 0.4, 0.6, 0.8$ ) are taken as the output.

We perform all the models working for coupled datasets mentioned above to solve this MFG coupled PDEs, and the results with different  $t$  are shown in Table 2. Compared to the existing model with the best results, our proposed CMWNO yields  $34.2\% \sim 67.4\%$  improvements in terms of  $\rho$  and  $57.4\% \sim 73.7\%$  in terms of  $\varphi$  with respect to the relative  $L2$  error. It is worth noting that  $\mathrm{MWT}_c$  performs the worst in most cases which indicates that the interactions between  $\rho$  and  $\varphi$  can not be learned through a single multiwavelet kernel. By interacting two kernels in the Wavelet space after decomposition steps, our proposed CMWNO can better decouple the interactions between  $\rho$  and  $\varphi$  to solve the MFG PDEs.

# 4 CONCLUSION

In this work, we propose a coupled multiwavelets neural operator using multiwavelet discretization of the spatial domain. Solving for coupled equations requires an information entanglement across operators for individual process. We found that combining operators in the projected domain of multiwavelets is effective. Numerical experiments using representative coupled PDEs including Gray-Scott and mean field game problem show that our coupling mechanism effectively learns the two processes in comparison with standalone operators.

# REFERENCES

Yves Achdou and Italo Capuzzo-Dolcetta. Mean field games: numerical methods. SIAM Journal on Numerical Analysis, 48(3):1136-1162, 2010.  
Yves Achdou, Guy Barles, Hitoshi Ishii, and Grigorii Lazarevich Litvinov. Hamilton-jacobi equations: approximations, numerical analysis and applications. 2013a.  
Yves Achdou, Fabio Camilli, and Italo Capuzzo-Dolcetta. Mean field games: convergence of a finite difference method. SIAM Journal on Numerical Analysis, 51(5):2585-2612, 2013b.  
Yves Achdou, Francisco J Buera, Jean-Michel Lasry, Pierre-Louis Lions, and Benjamin Moll. Partial differential equation models in macroeconomics. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 372(2028):20130397, 2014.  
Yves Achdou, Jiequn Han, Jean-Michel Lasry, Pierre-Louis Lions, and Benjamin Moll. Income and wealth distribution in macroeconomics: A continuous-time approach. The review of economic studies, 89(1):45-86, 2022.  
B. Alpert, G. Beylkin, D. Gines, and L. Vozovoi. Adaptive solution of partial differential equations in multiwavelet bases. Journal of Computational Physics, 182(1):149-190, 2002a. ISSN 0021-9991.  
Beylkin Alpert, Gregory Beylkin, David Gines, and Lev Vozovoi. Adaptive solution of partial differential equations in multiwavelet bases. Journal of Computational Physics, 182(1):149-190, 2002b.  
Bradley Alpert, Gregory Beylkin, Ronald Coifman, and Vladimir Rokhlin. Wavelet-like bases for the fast solution of second-kind integral equations. SIAM journal on Scientific Computing, 14(1): 159-184, 1993.  
Bradley K. Alpert. A class of bases in  $L^2$  for the sparse representation of integral operators. SIAM Journal on Mathematical Analysis, 24(1):246-262, 1993a. doi: 10.1137/0524016.  
Bradley K Alpert. A class of bases in  $1^{\prime}2$  for the sparse representation of integral operators. SIAM journal on Mathematical Analysis, 24(1):246-262, 1993b.  
Jean-David Benamou and Guillaume Carlier. Augmented lagrangian methods for transport optimization, mean field games and degenerate elliptic equations. Journal of Optimization Theory and Applications, 167(1):1-26, 2015.  
Jean-David Benamou, Guillaume Carlier, and Filippo Santambrogio. Variational mean field games. In Active Particles, Volume 1, pp. 141-171. Springer, 2017.  
Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.  
Gregory Beylkin, Ronald Coifman, and Vladimir Rokhlin. Fast wavelet transforms and numerical algorithms i. Communications on pure and applied mathematics, 44(2):141-183, 1991.  
Kaushik Bhattacharya, Bamdad Hosseini, Nikola B. Kovachki, and Andrew M. Stuart. Model reduction and neural networks for parametric pdes, 2020.  
Fabrice Bonjean and Gary SE Lagerloef. Diagnostic model and analysis of the surface currents in the tropical pacific ocean. Journal of Physical Oceanography, 32(10):2938-2954, 2002.  
Luis Briceno-Arias, Dante Kalise, Ziad Kobeissi, Mathieu Lauriere, A Mateos Gonzalez, and Francisco J Silva. On the implementation of a primal-dual algorithm for second order time-dependent mean field games with local couplings. ESAIM: Proceedings and Surveys, 65:330-348, 2019.  
Luis M Briceno-Arias, Dante Kalise, and Francisco J Silva. Proximal methods for stationary mean field games with local couplings. SIAM Journal on Control and Optimization, 56(2):801-836, 2018.

Tianping Chen and Hong Chen. Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. IEEE Transactions on Neural Networks, 6(4):911-917, 1995.  
Jeffrey M. Connors, Jason S. Howell, and William J. Layton. Partitioned time stepping for a parabolic two domain problem. SIAM Journal on Numerical Analysis, 47(5):3526-3549, 2009. doi: 10.1137/080740891. URL https://doi.org/10.1137/080740891  
Steven M Cox and Paul C Matthews. Exponential time differencing for stiff systems. Journal of Computational Physics, 176(2):430-455, 2002.  
Antonio De Paola, Vincenzo Trovato, David Angeli, and Goran Strbac. A mean field game approach for distributed control of thermostatic loads acting in simultaneous energy-frequency response markets. IEEE Transactions on Smart Grid, 10(6):5987-5999, 2019.  
T. A Driscoll, N. Hale, and L. N. Trefethen. Chebfun Guide. Pafnuty Publications, 2014.  
Olga Fuks and Hamdi A Tchelepi. Limitations of physics informed machine learning for nonlinear two-phase transport in porous media. Journal of Machine Learning for Modeling and Computing, 1(1), 2020.  
Meysam Ghanavati, Animesh Chakravarthy, and Prathyush P Menon. Analysis of automotive cyberattacks on highways using partial differential equation models. IEEE Transactions on Control of Network Systems, 5(4):1775-1786, 2017.  
David Gilbarg, Neil S Trudinger, David Gilbarg, and NS Trudinger. Elliptic partial differential equations of second order, volume 224. Springer, 1977.  
Diogo Gomes and Marc Sedjro. One-dimensional, forward-forward mean-field games with congestion. arXiv preprint arXiv:1703.10029, 2017.  
Diogo A Gomes et al. A mean-field game approach to price formation. Dynamic Games and Applications, 11(1):29-53, 2021.  
Somdatta Goswami, Aniruddha Bora, Yue Yu, and George Em Karniadakis. Physics-informed neural operators. arXiv preprint arXiv:2207.05748, 2022.  
Olivier Guéant, Jean-Michel Lasry, and Pierre-Louis Lions. Mean field games and applications. In Paris-Princeton lectures on mathematical finance 2010, pp. 205-266. Springer, 2011.  
Gaurav Gupta, Xiongye Xiao, Radu Balan, and Paul Bogdan. Non-linear operator approximations for initial value problems. In International Conference on Learning Representations, 2021a.  
Gaurav Gupta, Xiongye Xiao, and Paul Bogdan. Multiwavelet-based operator learning for differential equations, 2021b.  
Craig S Henriquez. Simulating the electrical behavior of cardiac tissue using the bidomain model. Critical reviews in biomedical engineering, 21(1):1-77, 1993.  
Minyi Huang, Roland P Malhamé, and Peter E Caines. Large population stochastic dynamic games: closed-loop mckean-vlasov systems and the nash certainty equivalence principle. Communications in Information & Systems, 6(3):221-252, 2006.  
Minyi Huang, Peter E. Caines, and Roland P. Malhame. Large-population cost-coupled lqq problems with nonuniform agents: Individual-mass behavior and decentralized  $\varepsilon$ -nash equilibria. IEEE Transactions on Automatic Control, 52(9):1560-1571, 2007. doi: 10.1109/TAC.2007.904450.  
Xuancheng Huang, Sebastian Jaimungal, and Mojtaba Nourian. Mean-field game strategies for optimal execution. Applied Mathematical Finance, 26(2):153-185, 2019.  
Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces. arXiv preprint arXiv:2108.08481, 2021.

David Lacasse, André Garon, and Dominique Pelletier. Mechanical hemolysis in blood flow: user-independent predictions with the solution of a partial differential equation. Computer Methods in Biomechanics and Biomedical Engineering, 10(1):1-12, 2007.  
Jean-Michel Lasry and Pierre-Louis Lions. Jeux à champ moyen. i-le cas stationnaire. Comptes Rendus Mathématique, 343(9):619-625, 2006.  
Jean-Michel Lasry and Pierre-Louis Lions. Mean field games. Japanese journal of mathematics, 2 (1):229-260, 2007.  
Jorge A Laval and Ludovic Leclercq. The hamilton-jacobi partial differential equation and the three representations of traffic flow. Transportation Research Part B: Methodological, 52:17-30, 2013.  
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations, 2020a.  
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Graph kernel network for partial differential equations, 2020b.  
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Andrew Stuart, Kaushik Bhattacharya, and Anima Anandkumar. Multipole graph neural operator for parametric partial differential equations. In Advances in Neural Information Processing Systems, volume 33, pp. 6755-6766, 2020c.  
Siting Liu and Levon Nurbekyan. Splitting methods for a class of non-potential mean field games. arXiv preprint arXiv:2007.00099, 2020.  
Siting Liu, Matthew Jacobs, Wuchen Li, Levon Nurbekyan, and Stanley J Osher. Computational methods for first-order nonlocal mean field games with applications. SIAM Journal on Numerical Analysis, 59(5):2639-2668, 2021.  
Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218-229, 2021.  
Lu Lu, Xuhui Meng, Shengze Cai, Zhiping Mao, Somdatta Goswami, Zhongqiang Zhang, and George Em Karniadakis. A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. Computer Methods in Applied Mechanics and Engineering, 393:114778, 2022.  
Chuizheng Meng, Sungyong Seo, Defu Cao, Sam Griesemer, and Yan Liu. When physics meets machine learning: A survey of physics-informed machine learning. arXiv preprint arXiv:2203.16797, 2022.  
Ravi G. Patel, Nathaniel A. Trask, Mitchell A. Wood, and Eric C. Cyr. A physics-informed operator regression framework for extracting data-driven continuum models. Computer Methods in Applied Mechanics and Engineering, 373:113500, 2021. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2020.113500.  
Mohammadreza Radmanesh, Manish Kumar, and Donald French. Partial differential equation-based trajectory planning for multiple unmanned air vehicles in dynamic and uncertain environments. Journal of Dynamic Systems, Measurement, and Control, 142(4), 2020.  
M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686-707, 2019. ISSN 0021-9991.  
Maziar Raissi. Deep hidden physics models: Deep learning of nonlinear partial differential equations. The Journal of Machine Learning Research, 19(1):932-955, 2018.  
Maziar Raissi and George Em Karniadakis. Hidden physics models: Machine learning of nonlinear partial differential equations. Journal of Computational Physics, 357:125-141, 2018.

Chen Tang, Lin Han, Hongwei Ren, Tao Gao, Zhifang Wang, and Ke Tang. The oriented-couple partial differential equations for filtering in wrapped phase patterns. Optics Express, 17(7):5606-5617, 2009.  
L. N. Trefethen and K. Embree. The (unfinished) pde coffee table book, 2001. URL https://people.maths.ox.ac.uk/trefethen/pdectb.html.  
Yinhao Zhu, Nicholas Zabaras, Phaedon-Stelios Koutsourelakis, and Paris Perdikaris. Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data. Journal of Computational Physics, 394:56-81, 2019.