# LEARNING TO SOLVE NONLINEAR PARTIAL DIFFERENTIAL EQUATION SYSTEMS TO ACCELERATE MOSFET SIMULATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Semiconductor device simulation uses numerical analysis, where a set of coupled nonlinear partial differential equations is solved with the iterative Newton-Raphson method. Since an appropriate initial guess to start the Newton-Raphson method is not available, a solution of practical importance with desired boundary conditions cannot be trivially achieved. Instead, several solutions with intermediate boundary conditions should be calculated to address the nonlinearity and introducing intermediate boundary conditions significantly increases the computation time. In order to accelerate the semiconductor device simulation, we propose to use a neural network to learn an approximate solution for desired boundary conditions. With an initial solution sufficiently close to the final one by a convolutional neural network, computational cost to calculate several unnecessary solutions is significantly reduced. Specifically, a convolutional neural network for MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor), the most widely used semiconductor device, is trained in a supervised manner to compute the initial solution. Particularly, we propose to consider device grids with varying size and spacing and derive a compact expression of the solution based upon the electrostatic potential. Finally, we empirically show that the proposed method accelerates the simulation by more than 12 times.

# 1 INTRODUCTION

Nonlinear partial differential equations (PDEs) appear frequently in many science and engineering problems including transport equations for certain quantities like heat, mass, momentum, and energy (Fischetti & Vandenberghe, 2016). The Maxwell equations for the electromagnetic fields (Jackson, 1999), which govern one of the fundamental forces in the physical world, is one of the examples. By calculating the solution of those equations, the status of system-under-consideration can be characterized.

Among many nonlinear partial differential equations, we consider the semiconductor device simulation (Grasser et al., 2003). The simulation is a pivotal application to foster next-generation semiconductor device technology at scale. Since the technology development heavily relies on the device simulation results, if the simulation time reduces, the turnaround time also significantly reduce. In order to reduce the simulation time, acceleration techniques based upon the multi-core computing have been successfully applied (Rupp et al., 2011; Sho & Odanaka, 2017). However, the number of cores cannot be exponentially increased and the cost also increases drastically as the number of cores involved increases. Moreover, as engineers submit many simulation jobs for a group of semiconductor devices, computing resources available to each simulation job is limited. As an alternative, we propose to improve the efficiency of the simulation per se.

In the semiconductor device simulation, a solution of a system of partial differential equations is numerically calculated with a certain boundary condition. Those differential equations are coupled together and the overall system is highly nonlinear. The Newton-Raphson method (Stoer & Bulirsch, 2002) is known to be one of the most robust methods to solve a set of coupled nonlinear equations. When the method converges to the solution, the error decreases rapidly as the Newton iterations proceed. To achieve a rapid convergence, it is crucial that initial guess for the solution needs to

be close enough to the real solution; otherwise, the method converges very slowly or may even diverge. Although we are interested in obtaining a solution at a specific boundary condition which is determined by an applied voltage, even obtaining an approximated solution to initiate the Newton-Raphson method successfully is challenging. In literature, in order to prepare an initial guess for the target boundary condition, several intermediate boundary conditions are introduced and solutions with those boundary conditions are computed sequentially (Synopsys, 2014). It, however, increases the overall computation time significantly. If the initial solution that is sufficiently close to the final one is provided by any means, we can save huge computational cost of calculating several unnecessary solutions.

Instead, we propose to learn an approximate initial solution of a set of coupled PDE for a target boundary condition by a convolutional neural network. Specifically, when a set of labeled images is available, a neural network can be trained to generate a similar image for a given label. The trained model generates a numerical solution for a target boundary condition. We show that the proposed initial solution by our method can speed up the device simulation significantly by providing a better initial guess.

We summarize our contributions as follows:

- We derive a compact solution for PDE systems based on the electrostatic potential. As a result, the network size is reduced by a factor of three. Since the electrostatic potential is well bounded, the normalization issue can be avoided.  
- For addressing various semiconductor devices, we propose a device template that can address various device structures with a set of hyper-parameters. Since the electrical characteristics of semiconductor devices are largely determined by the physical sizes of internal components, handling grids with varying size and spacing is particularly important.  
- We propose a convolutional neural network which generates the electrostatic potential from the device parameters. It can be used to accelerate the device simulation.  
- Compared with the conventional method, the simulation time is significantly reduced (at least 12 times) with the proposed method while the numerical stability is not hampered.

# 2 RELATED WORK

# 2.1 NEURAL NETWORKS FOR SOLVING DIFFERENTIAL EQUATIONS

Recently, there have been many attempts to build a neural network to solve a differential equation (Han et al., 2018; Long et al., 2018; Piscopo et al., 2019; Raissi et al., 2019; Zhu et al., 2019; Xiao et al., 2020). Among them, the Poisson equation is particularly of importance in the semiconductor device simulation. The Poisson equation plays a fundamental role in the semiconductor device simulation, by connecting the electrostatic potential and other physical quantities. In Magill et al. (2018), the Laplace equation (the Poisson equation with a vanishing source term) is considered for a nanofluidic device. A fully-connected neural network is trained to minimize the loss function, which combines the residue vector and the boundary condition. The authors assume a specific two-dimensional structure and the mixed boundary condition is applied. Another attempt to solve the Poisson equation is suggested in Özbay et al. (2019). The Poisson equation with the Dirichlet boundary condition is considered. It is decomposed into two equations. One is the Poisson equation with the homogeneous Neumann boundary condition. The other one is the Laplace equation with the Dirichlet boundary condition. A convolutional neural network architecture is adopted and the entire source term and the grid spacing are used as the input parameters. The network is trained with randomly generated source terms and a predicted solution shows a good agreement with the numerical solution.

# 2.2 NEURAL NETWORKS IN SEMICONDUCTOR DEVICE SIMULATION

Researchers are much interested with application of machine learning technique to the semiconductor device simulation (Carrillo-Nunez et al., 2019; Bankapalli & Wong, 2019; Han & Hong, 2019; Souma & Ogawa, 2020). However, the actual research activities are quite diverse. In many cases, the neural networks are merely used as an efficient descriptor of the input-output relation without

![](images/518cec98856052f4c191da52d1118f1113033049fc58a61eeda0586fa9aa43d2.jpg)  
(a)

![](images/adb5ddc91722cb44c2d41ce09f1e811a5d1b9285f2d18b0bd6052444aa619e8a.jpg)  
Figure 1: Components of the semiconductor device simulation. (a) Schematic diagram of a device with multiple terminals (yellow bars). (Input) (b) Current-voltage characteristics. (Output) (c) Internal quantity called the electrostatic potential. In order to calculate the terminal current, those internal quantities are needed.  
(b)

![](images/40da6f49516e219b24c49f772b558c7bc89509e1dea98cf63d08ff5b566a2987.jpg)  
(c)

considering the internal physical quantities (Carrillo-Nunez et al., 2019; Bankapalli & Wong, 2019). On the other hand, in their quantum transport simulation, Souma & Ogawa (2020) try to predict the electron density profile from the electrostatic potential. Although application of neural networks on the semiconductor device simulation is in its infancy, it has a huge potential.

# 3 PRELIMINARY: SEMICONDUCTOR DEVICE SIMULATION

We briefly introduce the semiconductor device simulation as a preliminary. As shown in Figure 1a, a multi-dimensional device structure has multiple terminals (yellow bars in the figure) whose voltages are controllable. The terminal currents (in Ampere) under the determined terminal voltages (in Volt) are the output of the simulator, as shown in Figure 1b. As the motion of electrons yields the current, the terminal current itself can be only calculated from the internal physical quantities. (See Figure 1c.) Since the electrons are charged particles, the electric field due to the net charge density is also affected by the electronic motion. Therefore, we need to consider both effects of electronic motion and electric field by solving two equations, to calculate the terminal currents.

The first one is the continuity equation for electrons:

$$
\nabla \cdot \mathbf {J} _ {n} = 0, \tag {1}
$$

where the electron current density vector,  $\mathbf{J}_n$ , is given by

$$
\mathbf {J} _ {n} = q \mu_ {n} n \mathbf {E} + q D _ {n} \nabla n, \tag {2}
$$

where the elementary charge  $(q)$ , the electron mobility  $(\mu_{n})$ , and the electron diffusion constant  $(D_{n})$  are scalar parameters and the electron density  $(n)$  is a position-dependent unknown variable. Note that the current density  $(\mathbf{J}_n)$  also depends on the electric field vector  $(\mathbf{E})$ . Similar relations hold for holes with minor modification.

The second equation is the Gauss law, also called as the Poisson equation:

$$
\nabla \cdot (\epsilon \mathbf {E}) = q (p - n + N _ {d o p} ^ {+}), \tag {3}
$$

where  $p$  is the hole density,  $\epsilon$  is the permittivity (Jackson, 1999) and  $N_{dop}^{+}$  is the positively charged impurity density. Under the electrostatic approximation, the electric field vector can be expressed in terms of the electrostatic potential  $(\phi)$  as

$$
\mathbf {E} = - \nabla \phi . \tag {4}
$$

With the electrostatic potential, the above set of equations has two unknown variables;  $n$  and  $\phi$ . Note that nonlinearity is originated from (2). The first term in the right-hand-side of (2) is proportional to  $n\mathbf{E}$ , which is nonlinear with respect to unknown variables of  $n$  and  $\phi$ . Therefore, when  $\phi$  is fixed, the continuity equation becomes linear.

For numerical analysis, we need to discretize the equations. In general, the  $i$ -th component of the residual vector can be written as

$$
r _ {i} = f _ {i} \left(\phi_ {1}, \dots \phi_ {N}, n _ {1}, \dots , n _ {N}, p _ {1}, \dots , p _ {N}\right) - s _ {i} = 0, \tag {5}
$$

where  $N$  is the number of grid points,  $f_{i}(\cdot)$  is a nonlinear function, and  $s_i$  is a constant.  $s_i$  becomes nonzero only at the boundary nodes and depends on the voltages applied to the device terminals. Since the system is nonlinear, the solution at each boundary condition must be computed. Detailed discussion can be found in Appendix A.

As discussed in introduction (and will be in the Appendix B), computing several intermediate boundary values is the main source of increase of computation. Let us denote the number of boundary values simulated during the solution procedure as  $N_{step}$ . To evaluate the acceleration quantitatively, we define the reduction factor of the simulation time,  $\alpha$ , and it is well approximated as

$$
\alpha \equiv \frac {\tau_ {c o n v}}{\tau_ {n n}} = \frac {N _ {s t e p} \times N _ {n e w t o n}}{N _ {n e w t o n} ^ {d i r e c t}} \approx N _ {s t e p}, \tag {6}
$$

where  $\tau_{conv}$  is the simulation time with the conventional method while  $\tau_{nn}$  is the one with the proposed method. Moreover,  $N_{newton}$  is the average number of the Newton iterations per a bias condition and  $N_{newton}^{direct}$  is the number of the Newton iterations at the target bias condition. Since  $N_{step}$  is typically larger than 10, we expect that the reduction factor can be larger than 10.

# 4 APPROACH

It is now clear that an approximate solution for a given boundary condition is a key to accelerate the device simulation. Instead of inventing yet another method to calculate the approximate solution efficiently, we propose a data driven approach by using a neural network to learn the numerical solution for desired boundary conditions.

# 4.1 COMPACT FORM OF A SOLUTION

In (5), the residual vector has  $3N$  components, because we consider three internal quantities,  $\phi$ ,  $n$ , and  $p$ , as solution variables. Typically, the number of grid points,  $N$ , ranges from thousands to tens of thousands. Therefore, it is desirable to introduce a compact form of a solution. Among various physical quantities in the semiconductor device, the electrostatic potential,  $\phi$ , is a key quantity. As discussed earlier, the nonlinearity arises from  $n\mathbf{E} = -n\nabla \phi$  in (2). Under a fixed electrostatic potential profile, the electron and hole continuity equations become linear. The continuity equation decoupled from (3) is easy to solve. In other words, the electron and hole densities,  $n$  and  $p$ , can be readily obtained as long as a reasonably good potential profile is provided. Thus, we propose to use a neural network to generate electrostatic potential profiles.

By the compact form of the solution, the number of output components reduces by a factor of three. Moreover, the electrostatic potential is well bounded in a small voltage range and varies smoothly over the device structure; no normalization of the input data is required.

# 4.2 DEVICE TEMPLATE

The initial device design usually exhibits a sub-optimal performance as the device parameters are not optimal at the early stage of the technology development. During the technology development cycle, numerous structures are simulated for engineers to achieve better performance. Those devices are different in terms of the physical sizes of internal components and the doping densities. To enable our method to address various device types, we propose a device template of MOSFETs. By determining each hyper-parameter of the template, we simulate various types of MOSFETs. We illustrate a device template that we use in Figure 2.

Device parameters, such as the gate length  $(L_{G})$ , the oxide thickness  $(t_{ox})$ , and the doping concentrations  $(N_{sd}$  for the source/drain regions and  $N_{sub}$  for the substrate region) are varied within predefined ranges. For that purpose, the  $x$ -directional grid spacing under the gate terminal is adjusted and the proposed method can be applied to a set of various device structures whose grids have different spacing. Since the electrical characteristics of semiconductor devices are largely determined by the physical sizes of internal components, handling grids with varying size and spacing is particularly important.

![](images/2106e025de380919a8316eac47263282d84a100c6c097cd001387a4e9b5e5c0c.jpg)  
Figure 2: A proposed device template. By changing the hyper-parameters, one can readily simulate various types of MOSFETs.

# 4.3 NETWORK ARCHITECTURE

We propose a convolutional neural network (CNN) to generate the two-dimensional potential profile for a given boundary condition. We design the architecture by adopting the generator part of the DCGAN (Radford et al., 2015) and illustrate it in Figure 3. It takes the device parameters such as  $L_{G}$ ,  $t_{ox}$ ,  $N_{sd}$ , and  $N_{sub}$  and the applied terminal voltages such as  $V_{G}$  and  $V_{D}$  as input. The output layer generates a 64-by-64 matrix as input for the simulation.

![](images/ad07b02897cb1b4e6dadebdb71bee7bd5ed36897a9fd8f53ab8d5934e9b03992.jpg)  
Figure 3: Layer structure of the CNN adopted in the two-dimensional MOSFET.

# 4.3.1 OBJECTIVE FUNCTION

To learn such network, we use and minimize a simple mean squared error objective function as

$$
\mathcal {L} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\tilde {\phi} _ {i} - \phi_ {i}\right) ^ {2}, \tag {7}
$$

where  $\tilde{\phi}_i$  is the electrostatic potential at the  $i$ -th node, predicted by the neural network. After 150 epochs, the mean square error of the electrostatic potential is sufficiently reduced.

# 5 EXPERIMENTS

# 5.1 EXPERIMENTAL SETUP

We conduct a set of experiments of MOSFET simulation (Taur & Ning, 1998). Device structures are shown in Figures 1a and 2. A two-dimensional grid with a size of 64-by-64 is used for all devices and the grid spacing is adjusted. Since  $N$  is  $64 \times 64 = 4,096$ , the solution vector has  $3N = 12,288$  unknown variables. We apply voltages to the gate terminal  $(V_{G})$  and the drain terminal  $(V_{D})$  to draw

a current through the drain terminal  $(I_{D})$ . The room temperature is assumed to be  $300\mathrm{K}$  throughout the experiments. We train our network in supervised fashion with the backpropagation algorithm on Pytorch 1.4.0 library.

# 5.1.1 DATASET

We use a simulator by Han & Hong (2019) as it is free from the license and the source code is publicly available. Detailed description on the simulator can be found in Appendix C. Each data point is specified with parameters of  $L_{G}$ ,  $t_{ox}$ ,  $N_{sd}$ ,  $N_{sub}$ ,  $V_{G}$ , and  $V_{D}$  and their ranges are summarized in Appendix D. We curate the dataset with 10,112 instances that are randomly selected. We split the dataset by training, validation and test set by  $70\%$  of the selected devices are the training set and the others in the validation  $(20\%)$  and test  $(10\%)$  sets. We will publicly release our splits for the future research in this avenue. The training and validation errors can be found in Appendix D.

# 5.1.2 COMPARISON WITH THE CONVENTIONAL METHOD

As our primary goal is to accelerate the semiconductor device simulation with help of the initial solution obtained by the neural network, we evaluate the speed up of the device simulation algorithm compared to the conventional method. Specifically, we report the number of the Newton iterations for the converged solution at the target bias condition  $(V_{G} = V_{D} = 1.1\mathrm{~V})$ . When the maximum potential update is smaller than  $10^{-10}\mathrm{~V}$ , the convergence criterion holds.

By the conventional method, starting from the equilibrium condition of  $V_{G} = V_{D} = 0.0\mathrm{V}$ , we first increase the drain voltage up to  $1.1\mathrm{V}$ . After that, the gate voltage is raised up to  $1.1\mathrm{V}$ , which is called bias ramping. During the bias ramping, a uniform voltage step is applied and the solution at the previous boundary condition is used as the initial guess. When the voltage step is small enough, every simulation is successfully finished to get a final solution. However, with a large voltage step, some simulation runs may fail. In other words, the simulation is numerically unstable with a large voltage step.

We also report the ratio of the failed runs to the total simulation runs to check the numerical stability. In contrast, by our method, when the neural network provides the approximate solution, we start the simulation directly at the target bias condition. We compare the number of the Newton iterations for the converged solution with the one by the conventional bias ramping method with a uniform voltage step.

# 5.2 RESULTS

After the training phase is finished, the trained neural network can be used to generate an approximate potential in the inference phase. Figure 4a shows the electrostatic potential profile generated by the trained neural network. The parameters of the MOSFET are  $L_{G} = 96 \mathrm{~nm}$ ,  $t_{ox} = 3.5 \mathrm{~nm}$ ,  $N_{sd} = 4.8 \times 10^{20} \mathrm{~cm}^{-3}$  and  $N_{sub} = 1.8 \times 10^{18} \mathrm{~cm}^{-3}$ . The bias condition is  $V_{G} = V_{D} = 0.24 \mathrm{~V}$ . The potential difference between the predicted potential profile and the numerical solution, shown in Figure 4b, reveals that the maximum difference is around  $0.1 \mathrm{~V}$ . The device in Figure 4a has a longer gate,  $L_{G} = 164 \mathrm{~nm}$ . Other parameters are also changed to  $t_{ox} = 3.1 \mathrm{~nm}$ ,  $N_{sd} = 2.1 \times 10^{20} \mathrm{~cm}^{-3}$  and  $N_{sub} = 2.0 \times 10^{18} \mathrm{~cm}^{-3}$ . Regardless  $L_{G}$ , the neural network can predict the electrostatic potential at a given bias condition ( $V_{G} = 0.05 \mathrm{~V}$  and  $V_{D} = 0.66 \mathrm{~V}$ ) accurately.

We demonstrate that the trained neural network generates the electrostatic potential profile close to the numerical solution in Figures 4 and 5. For 1,012 samples in the test set, average and variance of the maximum absolute potential error is  $92\mathrm{mV}$  and  $0.0016\mathrm{V}^2$ , respectively. The convergence behavior of the Newton-Raphson method is shown as a red curve in Figure 6a. With only five Newton iterations, the maximum potential update becomes smaller than  $10^{-10}\mathrm{V}$ . Note that as the initial update is quite small, about  $0.1\mathrm{V}$ , the fast convergence is achieved. On the other hand, the conventional method with a uniform voltage step of  $0.275\mathrm{V}$  (green curves) takes 40 iterations to reach the same result of our method. For each voltage condition, it takes only five iterations. However, after an intermediate boundary condition is solved, the next boundary condition should be solved again. Therefore, the green curves have peak values at every five iteration. Here, we achieve significant computation reduction factor of 8.

![](images/6b907f7c526215898438f9c61993d4c2669b63035b18759bf12b87157f722b2e.jpg)  
(a)

![](images/bc641be3a5266f8739b1fee817ae73c475ea43a4c659aafc1209ec9d3726261f.jpg)  
(b)

![](images/e3d336777265e72dae271f82fa22dc00b37b77e6cfdf5849542b94226e34b42d.jpg)  
Figure 4: (a) Electrostatic potential profile predicted by the neural network at  $V_{G} = V_{D} = 0.24 \mathrm{~V}$ . (b) Difference between (a) and the numerical solution. The maximum potential error is about  $0.1 \mathrm{~V}$ , which is reasonably small.  
(a)

![](images/927dbc425b2ffd8f410e0dc14ebf32fe2b3548b789e2ac274be6864df15a3e95.jpg)  
(b)

![](images/03df4e5ef4d699c024743feba00402fe202ae7b3c2d69171140931156d774fe9.jpg)  
(a)

![](images/97cfd55922804c0f326de339c840c4ab838eb4c23780b9b604deeca9d2d8a1a7.jpg)  
Figure 6: (a) Convergence behavior. (b) Failure rate for a drain voltage outside the training range.  $V_{D} = 1.1 \mathrm{~V}$ . (c) Failure rate for a gate voltage outside the training range.  $V_{G} = 1.1 \mathrm{~V}$ .  
(b)

![](images/41e1ee9859326bd1c509d47b3ea81101a10101b413a9486acf74539a0a632b47.jpg)  
Figure 5: Same quantities with Figure 4 for a different MOSFET with  $L_{G} = 164$  nm.  $V_{G} = 0.05$  V and  $V_{D} = 0.66$  V.  
(c)

# 5.2.1 NUMERICAL STABILITY

In order to demonstrate the numerical stability of our method, the voltages beyond the training range are tested in Figures 6b and 6c. In Figure 6b, the drain voltage larger than  $1.1\mathrm{V}$  is directly solved. For each drain voltage, 100 devices are generated and tested. Even for a high drain voltage of  $2.2\mathrm{V}$ , the failure rate is just  $4\%$ . We perform a similar test for the gate voltage larger than  $1.1\mathrm{V}$  in Figure 6c. Although the failure rate increases sharply above  $1.9\mathrm{V}$ , it remains quite small up to  $1.8\mathrm{V}$ . These examples clearly demonstrate that the trained neural network can provide a sufficiently good initial potential profile.

# 5.2.2 APPLICATIONS TO MANY DEVICES

We now investigate the applicability of our method to many devices. We evaluate our method on 1,000 different devices that are randomly generated. The simulation starts at  $V_{G} = V_{D} = 1.1$  V. We show the distribution of the Newton iterations in Figure 7a. Among 1,000 test instances, only two samples fail to converge. For most cases (more than  $99.7\%$ ), four or five iterations are required for the converged solution. On the other hand, when the conventional method is applied, the number of the Newton iteration increases significantly as shown in Figure 7b.

![](images/7806ccb9abb9a9810716b13c7f54973cd25149d5974613aa591dcdcd38021b72.jpg)  
(a)

![](images/e1c60287116f5a55e5a30e41c235e048e297aee43bd9ba10501b6cbb40896b7c.jpg)  
Figure 7: Numerical stability and speed of the proposed approach. (a) Distribution of the Newton iterations out of 1000 test cases. (b) Distribution of the Newton iterations with the conventional method. (c) Trade-off between the numerical stability and the simulation speed.  
(b)

![](images/068212d99aca556df912c628628b8c0d061564b3003b68cb128b534f032ad4b7.jpg)  
(c)

In the conventional method, the voltage step is an important parameter to control the numerical stability and the speed. When we adopt a small voltage step of  $0.1\mathrm{V}$ , there occurs no failure case. In this case, about 90 Newton iterations are needed to have the converged solution and the simulation is almost 20 times slower than the proposed approach. We may try to increase the voltage step for numerical efficiency. When the voltage step is  $0.22\mathrm{V}$ , the number of the Newton iterations is reduced almost a factor of 2. However, the failure rate is larger than  $10\%$ , which is certainly not tolerable. Even with a very large voltage step of  $0.367\mathrm{V}$ , the number is 21, which is much larger than that of the proposed approach.

These observations are summarized in Figure 7c and Table 1. For the conventional method, there exists a trade-off between the numerical stability (related with the failure rate) and the simulation speed (related with the number of iterations). Our proposed approach exhibits both superior numerical stability and short simulation time. When the failure rate is limited up to  $1\%$ , the reduction factor of the simulation time is larger than 12.

Table 1: Failure rate and speed of the conventional method and ours.  $\alpha$  is the reduction factor defined in (6). The conventional method is tested with various voltage steps. For example, in a case of 0.1 V, 22 boundary conditions (11 for the drain sweep and 11 for the gate sweep) are solved. When the failure rate is too high, the second simulation round should start to solve the failed cases with a smaller voltage step, taking overall simulation time even longer. It implies that a high failure rate is not tolerable.  

<table><tr><td></td><td>0.1 V</td><td>0.157 V</td><td>CNN (Ours)</td><td>0.22 V</td><td>0.367 V</td></tr><tr><td>Failure rate (%)</td><td>0</td><td>1.7</td><td>0.2</td><td>10.4</td><td>14.7</td></tr><tr><td>Number of iterations</td><td>91.6</td><td>65.0</td><td>5.4</td><td>46.6</td><td>21.5</td></tr><tr><td>α (Reduction factor (6))</td><td>17.0</td><td>12.0</td><td>1.0</td><td>8.6</td><td>4.0</td></tr></table>

# 6 CONCLUSION

We train a convolutional neural network to generate the electrostatic potential required for the semiconductor device simulation. By using the generated electrostatic potential as an initial guess, the target bias conditions can be simulated directly without a time-consuming ramping procedure. Our proposed approach can address a set of various device structures whose grids have different spacing. Moreover, we suggest a compact expression for the solution based upon the electrostatic potential to reduce the computational complexity.

In the empirical validations with two-dimensional MOSFETs, the simulation time has been significantly reduced (more than 12 times) with the proposed method while the numerical stability is not hampered. As deep neural networks perform well when a sufficient amount of data points are used in training, it is expected that the proposed approach would be accelerate the development cycle of semiconductors since massive simulation results are routinely generated during the technology optimization and are ready to be used for training such networks.

# REFERENCES

Y. S. Bankapalli and H. Y. Wong. Tcad augmented machine learning for semiconductor device failure troubleshooting and reverse engineering. In 2019 International Conference on Simulation of Semiconductor Processes and Devices (SISPAD), pp. 1-4, 2019. 2, 3  
H. Carrillo-Nunez, N. Dimitrova, A. Asenov, and V. Georgiev. Machine learning approach for predicting the effect of statistical variability in si junctionless nanowire transistors. IEEE Electron Device Letters, 40(9):1366-1369, 2019. 2, 3  
Timothy A. Davis. Algorithm 832: Umfpack v4.3—an unsymmetric-pattern multifrontal method. ACM Trans. Math. Softw., 30(2):196-199, June 2004. ISSN 0098-3500. doi: 10.1145/992200.992206. URL https://doi.org/10.1145/992200.992206.12  
M. V. Fischetti and W. G. Vandenberghe. Advanced physics of electron transport in semiconductors and nanostructures. Springer International Publishing Switzerland, 2016. 1  
Tibor Grasser, Ting-Wei Tang, Hans Kosina, and Siegfried Selberherr. A review of hydrodynamic and energy-transport models for semiconductor device simulation. Proceedings of the IEEE, 91 (2):251-274, February 2003. 1  
Jiequn Han, Arnulf Jentzen, and Weinan E. Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34):8505-8510, 2018. ISSN 0027-8424. doi: 10.1073/pnas.1718942115. URL https://www.pnas.org/content/115/34/8505.2  
S. Han and S. Hong. Deep neural network for generation of the initial electrostatic potential profile. In 2019 International Conference on Simulation of Semiconductor Processes and Devices (SISPAD), 2019. 2, 6, 12  
J. Jackson. Classical Electrodynamics. John Wiley and Sons, third edition, 1999. 1, 3, 11  
C. Jungemann and R. Meinerzhagen. Hierarchical Device Simulation: The Monte-Carlo Perspective. Springer-Verlag Wien, 2003. 12  
S. E. Laux and B. M. Grossman. A general control-volume formulation for modeling impact ionization in semiconductor transport. IEEE Transactions on Electron Devices, 32(10):2076-2082, October 1985. 11  
Zichao Long, Yiping Lu, Xianzhong Ma, and Bin Dong. PDE-net: Learning PDEs from data. volume 80 of Proceedings of Machine Learning Research, pp. 3208-3216, Stockholm mssan, Stockholm Sweden, 10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/long18a.html.2  
M. Magill, F. Z. Qureshi, and H. W. de Haan. Compact neural network solutions to Laplace's equation in a nanofluidic device. In Neural Information Processing Systems Workshop CDNNRIA, 2018. 2  
B. Meinerzhagen, K. H. Bach, I. Bork, and W. L. Engl. A new highly efficient nonlinear relaxation scheme for hydrodynamic MOS simulations. In NUPAD IV. Workshop on Numerical Modeling of Processes and Devices for Integrated Circuits, pp. 91-96, 1991. 11  
Maria Laura Piscopo, Michael Spannowsky, and Philip Waite. Solving differential equations with neural networks: Applications to the calculation of cosmological phase transitions. Phys. Rev. D, 100:016002, Jul 2019. doi: 10.1103/PhysRevD.100.016002. URL https://link.aps.org/doi/10.1103/PhysRevD.100.016002. 2  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks, 2015. arXiv preprint, arXiv:1511.06434. 5  
M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686 - 707, 2019. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp.2018.10.045. URL http://www.sciencedirect.com/science/article/pii/S0021999118307125.2

K. Rupp, T. Grasser, and A. Jüngel. Parallel preconditioning for spherical harmonics expansions of the boltzmann transport equation. In 2011 International Conference on Simulation of Semiconductor Processes and Devices, pp. 147-150, 2011. 1  
S. Sho and S. Odanaka. A hybrid mpi/openmp parallelization method for a quantum drift-diffusion model. In 2017 International Conference on Simulation of Semiconductor Processes and Devices (SISPAD), pp. 33-36, 2017. 1  
Satofumi Souma and Matsuto Ogawa. Acceleration of nonequilibrium greenaos; function simulation for nanoscale fets by applying convolutional neural network model. *IEICE Electronics Express*, advpub, 2020. doi: 10.1587/elex.17.20190739. 2, 3  
J. Stoer and R. Bulirsch. Introduction to Numerical Analysis. Springer-Verlag New York, third edition, 2002. 1  
Synopsys. "Sentaurus Device User Guide". 2014. 2  
Y. Taur and T. H. Ning. Fundamentals of Modern VLSI Devices. Cambridge University Press, 1998. 5  
X. Xiao, Y. Zhou, H. Wang, and X. Yang. A novel cnn-basedoisson solver for fluid simulation. IEEE Transactions on Visualization and Computer Graphics, 26(3):1454-1465, 2020. 2  
Yinhao Zhu, Nicholas Zabaras, Phaedon-Stelios Koutsourelakis, and Paris Perdikaris. Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data. Journal of Computational Physics, 394:56 - 81, 2019. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp.2019.05.024. URL http://www.sciencedirect.com/science/article/pii/S0021999119303559.2  
Ali Girayhan Özbay, Sylvain Laizet, Panagiotis Tzirakis, Georgios Rizos, and Björn Schuller. Poisson cnn: Convolutional neural networks for the solution of the poisson equation with varying meshes and dirichlet boundary conditions, 2019. 2