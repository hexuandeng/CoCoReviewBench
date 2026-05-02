# A DIFFERENTIABLE PHYSICS ENGINE FOR DEEP LEARNING IN ROBOTICS

Jonas Degrave, Michiel Hermans,* Joni Dambre & Francis wyffels

Department of Electronics and Information Systems (ELIS)

Ghent University - iMinds, IDLab

Technologiepark-Zwijnaarde 15, B-9052 Ghent, Belgium

{Jonas.Degrave, Joni.Dambre, Francis.wyffels}@UGent.be

# ABSTRACT

One of the most important fields in robotics is the optimization of controllers. Currently, robots are often treated as a black box in this optimization process, which is the reason why derivative-free optimization methods such as evolutionary algorithms or reinforcement learning are omnipresent. When gradient-based methods are used, models are kept small or rely on finite difference approximations for the Jacobian. This method quickly grows expensive with increasing numbers of parameters, such as found in deep learning. We propose an implementation of a modern physics engine, which can differentiate control parameters. This engine is implemented for both CPU and GPU. Firstly, this paper shows how such an engine speeds up the optimization process, even for small problems. Furthermore, it explains why this is an alternative approach to deep Q-learning, for using deep learning in robotics. Finally, we argue that this is a big step for deep learning in robotics, as it opens up new possibilities to optimize robots, both in hardware and software.

# 1 INTRODUCTION

To solve tasks efficiently, robots require an optimization of their control system. This optimization process can be done in automated testbeds (Degrave et al., 2015), but typically these controllers are optimized in simulation. Standard methods to optimize these controllers include particle swarms, reinforcement learning, genetic algorithms and evolutionary strategies. These are all derivative-free methods.

A recently popular alternative approach is to use deep Q-learning, a reinforcement learning algorithm. This method requires a lot of evaluations in order to train the many parameters (Levine et al., 2016). However, deep learning experience has taught us that optimizing with a gradient is often faster and more efficient. This fact is especially true when there are a lot of parameters, as is common in deep learning. However, in the optimization processes for control systems, the robot is almost exclusively treated as a non-differentiable black box. The reason for this is that the robot in hardware is not differentiable, nor are current physics engines able to provide the gradient of the robot models. The resulting need for derivative-free optimization approaches limits both the optimization speed and the number of parameters in the controllers.

Recent physics engines, such as mujoco (Todorov et al., 2012), can derive gradients through the model of a robot but rely on a finite difference method to approximate the gradient. Evaluating finite difference approximations, however, requires the same number of model evaluations as the number of parameters with respect to which is differentiated.

This finite difference method is powerful enough for the use of model predictive control in robotics (Tassa et al., 2012). It allows for optimizing trajectories, which in turn can be used to optimize controllers for global policy function approximation (Levine and Koltun, 2013; Mordatch et al., 2015). However, optimizing this way quickly becomes intractable (Mordatch and Todorov,

2014). In order to get around the difficulty of optimizing a lot of parameters in few enough model evaluations, a 2-step approximation process is used.

In this paper, we suggest an alternative approach, by introducing a differentiable physics engine. This idea is not novel. It has been done before with spring-damper models in 2D and 3D (Hermans et al., 2014). This technique is also similar to adjoint optimization, a method widely used in various applications such as thermodynamics (Jarny et al., 1991) and fluid dynamics (Iollo et al., 2001). However, modern engines to model robotics are not based on spring-damper systems. The most commonly used ones are 3D rigid body engines, which rely on impulse-based velocity stepping methods (Erez et al., 2015). In this paper, we test whether these engines are also differentiable and whether this gradient is computationally tractable. We will show how this method does speed up the optimization process tremendously, and give some examples where we optimize deep learned neural network controllers with millions of parameters.

# 2 A 3D RIGID BODY ENGINE

The goal is to implement a modern 3D Rigid body engine, in which parameters can be differentiated with respect to the fitness a robot achieves in a simulation, such that these parameters can be optimized with methods based on gradient descent.

The most frequently used simulation tools for model-based robotics, such as PhysX, Bullet, Havok and ODE, go back to MathEngine (Erez et al., 2015). These tools are all 3D rigid body engines, where bodies have 6 degrees of freedom, and the relations between them are defined as constraints. These bodies exert impulses on each other, but their positions are constrained, e.g. to prevent the bodies from penetrating each other. The velocities, positions and constraints of the rigid bodies define a linear complementarity problem (LCP) (Chappuis, 2013), which is then solved using a Gauss-Seidel projection (GSP) method (Jourdan et al., 1998). The solution of this problem are the new velocities of the bodies, which are then integrated by semi-implicit Euler integration to get the new positions (Stewart and Trinkle, 2000). This system is not always numerically stable. Therefore the constraints are usually softened (Catto, 2009).

The recent growth of automatic differentiation libraries, such as Theano (Al-Rfou et al., 2016), Caffe (Jia et al., 2014) and Tensorflow (Abadi et al., 2015), has allowed for efficient differentiation of remarkably complex functions before (Degrave et al., 2016). Therefore, we implemented such a physics engine as a mathematical expression in Theano (Al-Rfou et al., 2016), a software library which does automatic evaluation and differentiation of expressions with a focus on deep learning. The resulting computational graph to evaluate this expression is then compiled for both CPU and GPU. To be able to compile for GPU however, we had to limit our implementation to a restricted set of elementary operations. The range of implementable functions is therefore severely capped. However, since the gradient is determined automatically, the complexity of correctly implementing the differentiation is removed entirely.

One of these limitations with this restricted set of operations, is the limited support for conditionals. Therefore we needed to implement our physics engine without branching, as this is not yet available in Theano for GPU. Therefore some sacrifices had to be made. For instance, our system only allows for contact constraints between different spheres or between spheres and the ground plane. Collision detection algorithms for cubes typically have a lot of branching (Mirtich, 1998). However, this sphere based approach can in principle be extended to any other shape (Hubbard, 1996). On the other hand, we did implement a rather accurate model of servo motors, with gain, maximal torque, and maximal velocity parameters.

Another design choice was to use rotation matrices rather than the more common quaternions for representing rotations. Consequently, the states of the bodies are larger, but the operations required are matrix multiplications. This design reduced the complexity of the graph. However, cumulative operations on a rotation matrix might move the rotation matrix away from orthogonality. To correct for this, we renormalize our matrix with the update equation (Premerlani and Bizard, 2009):

$$
A ^ {\prime} = \frac {3 A - A \circ (A \cdot A)}{2} \tag {1}
$$

where  $A^{\prime}$  is the renormalized version of the rotation matrix  $A$

These design decisions are the most important aspects of difference with the frequently used simulation tools. In the following section, we will evaluate our physics simulator on some different problems. We take a look at the speed of computation and the number of evaluations required before the parameters of are optimized.

# 2.1 THROWING A BALL

To test our engine, we implemented the model of a giant soccer ball in the physics engine, as shown in Fig. 3a. The ball has a  $1\mathrm{m}$  diameter, a friction of  $\mu = 1.0$  and restitution  $e = 0.5$ . The ball starts off at position  $(0,0)$ . After  $5\mathrm{s}$  it should be at position  $(10,0)$  with zero velocity  $v$  and zero angular velocity  $\omega$ . We optimized the initial velocity  $v_{0}$  and angular velocity  $\omega_{0}$  at time  $t = 0\mathrm{s}$  until the errors at  $t = 5\mathrm{s}$  are less than  $0.01\mathrm{m}$  and  $0.01\mathrm{m/s}$  respectively.

Since the quantity we optimize is only known at the end of the simulation, but we need to optimize the parameters at the beginning of the simulation, we need to backpropagate our error through time (BPTT) (Sutskever, 2013). This approach is similar to the backpropagation through time method used for optimizing recurrent neural networks (RNN). In our case, every time step in the simulation can be seen as one pass through a neural network, which transforms the inputs from this timestep to inputs for the next time step. For finding the gradient, this RNN is unfolded completely, and the gradient can be obtained by differentiating this unfolded structure. This differentiation is done automatically by the Theano library.

Optimizing the six parameters in  $v_{0}$  and  $\omega_0$  took only 88 iterations with gradient descent and backpropagation through time. Optimizing this problem with CMA-ES (Hansen, 2006), a state of the art derivative-free optimization method, took 2422 iterations. Even when taking the time to compute the gradient into account, the optimization with gradient descent takes  $16.3\mathrm{s}$ , compared to  $59.9\mathrm{s}$  with CMA-ES. This result shows that gradient-based optimization of kinematic systems can in some cases already outperform gradient-free optimization algorithms from as little as six parameters.

# 3 POLICY SEARCH

To evaluate the relevance of our differentiable physics engine, we model the use of a neural network as a general controller for a robot, as shown in Figure 1. We consider a general robot model in a discrete-time dynamical system  $\mathbf{x}^{t + 1} = f_{\mathrm{ph}}(\mathbf{x}^t,\mathbf{u}^t)$  with a task cost function of  $l(\mathbf{x}^t,\mathbf{p})$ , where  $\mathbf{x}^t$  is the state of the system at time  $t$  and  $\mathbf{u}^t$  is the input of the system at time  $t$ .  $\mathbf{p}$  provides some freedom in parameterizing the loss. If  $X^t$  is the trajectory of the state up to time  $t - 1$ , the goal is to find a policy  $u^t = \pi (X^t)$  such that we minimize the loss  $\mathcal{L}_{\pi}$ .

$$
\mathcal {L} _ {\pi} = \sum_ {t = 0} ^ {T} l \left(\mathbf {x} ^ {t}, \mathbf {p}\right) \tag {2}
$$

$$
\begin{array}{l l} \text {s . t .} & \mathbf {x} ^ {t + 1} = f _ {\mathrm {p h}} (\mathbf {x} ^ {t}, \pi (X ^ {t})) \quad \text {a n d} \quad \mathbf {x} ^ {0} = x ^ {\text {i n i t}} \end{array}
$$

In previous research, finding a gradient for this objective has been described as presenting challenges (Mordatch and Todorov, 2014), an approximation to tackle these issues has been discussed in Levine and Koltun (2013).

In this paper, we completely overhaul this approach, since we were unaware of these results. We basically jam this entire equation into an automatic differentiation library, ignoring these challenges in finding the analytic gradient altogether.

We define our controller as a deep learning neural network  $g_{\mathrm{deep}}$  with weights  $\mathbf{W}$ . We do not pass all information  $X^t$  to this neural network, but rather only a vector of values  $\mathbf{s}^t$  observed by the modeled sensors  $s(\mathbf{x}^t)$ . We also provide our network with (some of the) task-specific parameters  $\mathbf{p}'$ . Finally, we add a recurrent connection to the controller in the previous timestep  $\mathbf{h}^t$ . Therefore, our policy is the following:

$$
\begin{array}{r l} & \pi (X ^ {t}) = g _ {\text {d e e p}} (s (\mathbf {x} ^ {t}), \mathbf {h} ^ {t}, \mathbf {p} ^ {\prime} \mid \mathbf {W}) \\ & \text {s . t .} \quad \mathbf {h} ^ {t} = h _ {\text {d e e p}} (s (\mathbf {x} ^ {t - 1}), \mathbf {h} ^ {t - 1}, \mathbf {p} ^ {\prime} \mid \mathbf {W}) \quad \text {a n d} \quad \mathbf {h} ^ {0} = 0 \end{array} \tag {3}
$$

Notice the similarity between equations 2 and 3. Indeed, the equations for recurrent neural networks (RNN) in equation 3 are very similar to the ones of the loss of a physical model in equation 2. Therefore, we optimize this entire system as one big RNN, where we unfold over time, as illustrated in Figure 2. The weights  $\mathbf{W}$  are optimized with stochastic gradient descent. The gradient required for that is the Jacobian  $d\mathcal{L} / d\mathbf{W}$ , which is found with automatic differentiation software.

We have now reduced the problem to a standard deep learning problem. We need to train our network  $g_{\mathrm{deep}}$  on a sufficient amount of samples  $x^{\mathrm{init}}$  and for a sufficient amount of sampled tasks  $\mathbf{p}$  in order to get adequate generalization. Standard RNN regularization approaches could also improve this generalization. We reckon that generalization of  $g_{\mathrm{deep}}$  to more models  $f_{\mathrm{ph}}$ , in order to ease the transfer of the controller from the model to the real system, is also possible (Hermans et al., 2014), but it is outside the scope of this paper.

![](images/7fefef512ee99f593eb59e0bae8da267b63673bc13ce5660c02c9819e4b4cceb.jpg)  
Figure 1: Illustration of how a closed loop neural network controller would be used to actuate a robot. The neural network receives sensor signals from the sensors on the robot and uses these to generate motor signals which are sent to the servo motors. The neural network can also generate a signal which it can use at the next timestep to control the robot.

![](images/77cd03882bf765e21bda401fae1d6e25c16ba56326f99aa553e9030c5be41dab.jpg)  
Figure 2: Illustration of the dynamic system with the robot and controller, after unrolling over time. The neural networks  $g_{\mathrm{deep}}$  and  $h_{\mathrm{deep}}$  with weights  $\mathbf{W}$  receive sensor signals  $\mathbf{s}^t$  from the sensors on the robot and use these to generate motor signals  $\mathbf{u}^t$  which are used by the physics engine  $f_{\mathrm{ph}}$  to find the next state of the robot in the physical system. These neural networks also have a memory, implemented with recurrent connections  $\mathbf{h}^t$ . From the state  $\mathbf{x}^t$  of these robots, the loss  $\mathcal{L}$  can be found. In order to find  $d\mathcal{L} / d\mathbf{W}$ , every block in this chart needs to be differentiable. The contribution of this paper, is to implement a differentiable  $f_{\mathrm{ph}}$ , which allows us to optimize  $\mathbf{W}$  to minimize  $\mathcal{L}$  more efficiently than was possible before.

# 3.1 QUADRUPEDAL ROBOT - COMPUTING SPEED

To verify the speed of our engine, we also implemented a small quadrupedal robot model, as illustrated in Fig. 3b. This model has a total of 81 sensors, e.g. encoders and an inertial measurement unit (IMU). The servo motors are controlled in a closed loop by a small neural network  $g_{\mathrm{deep}}$  with a number of parameters, as shown previously in Fig. 2. The gradient is the Jacobian of  $\mathcal{L}$ , the total traveled distance of the robot in  $10\mathrm{s}$ , differentiated with respect to all the parameters of the controller  $\mathbf{W}$ . This Jacobian is found by using BPTT and propagating all  $10\mathrm{s}$  back. The time it takes to compute this traveled distance and the accompanying Jacobian is shown in Table 1. We include both the computation time with and without the gradient, i.e. both the forward and backward pass and the forward pass alone. This way, the numbers can be compared to other physics engines, as those only calculate without gradient. Our implementation and our model can probably be made more efficient, and evaluating the gradient can probably be made faster a similar factor.

![](images/c66a559abe058b68c8c9fec89f03d6d31a70035bebf0f8aa3bf08f81378452ef.jpg)  
(a) Ball model

![](images/c2b5a25c0f8eec9dc872a8ff01e1bdf16f4f9f03a340902d878fda261e6de427.jpg)  
(b) quadruped model  
Figure 3: (a) Illustration of the ball model used in the first task. (b) Illustration of the quadruped robot model with 8 actuated degrees of freedom, 1 in each shoulder, 1 in each elbow. The spine of the robot can collide with the ground, through 4 spheres in the inside of the cuboid. (c) Illustration of the robot arm model with 4 actuated degrees of freedom.

![](images/22e033f61258cfa7c2b338deb1c983b57329d35d1d8395c1fd87421486f9d9ae.jpg)  
(c) robot arm model

When only a single controller is optimized, our engine runs more slowly on GPU than on CPU. To tackle this issue, we implemented batch gradient descent, which is commonly used in complex optimization problems. In this case, by batching our robot models, we achieve significant acceleration on GPU. Although backpropagating the gradient through physics slows down the computations by roughly a factor 10, this factor only barely increases with the number of parameters in our controller, unlike finite difference methods.

Combining this with our previous observation that fewer iterations are needed when using gradient descent, our approach can enable the use of gradient descent through physics for highly complex deep learning controllers with millions of parameters. Also note that by using a batch method, a single GPU can simulate about 864000 model seconds per day, or 8640000 model states. This should be plenty for deep learning. It also means that a single simulation step of a single robot, which includes collision detection, solving the LCP problem, integrating the velocities and backpropagating the gradient through it all, takes about 1 ms on average. Without the backpropagation, this process is only about seven times faster.

# 3.2 4 DEGREE OF FREEDOM ROBOT ARM

As a first test of optimizing robot controllers, we implemented a four degree of freedom robotic arm, as depicted in Fig. 3c. The bottom of the robot has a 2 degrees of freedom actuated universal joint; the elbow has a 2 degree of freedom actuated joint as well. The arm is  $1\mathrm{m}$  long, and has a total mass of  $32\mathrm{kg}$ . The servos have a gain of  $30\mathrm{s}^{-1}$ , a torque of  $30\mathrm{Nm}$  and a velocity of  $45^{\circ}\mathrm{s}^{-1}$ .

For this robot arm, we train controllers for a task with a gradually increasing amount of difficulty. To be able to train our parameters, we have to use a couple of tricks often used in the training of recurrent neural networks.

- We choose an objective which is evaluated at every time step and then averaged, rather than at specific points of the simulation. This approach vastly increases the number of samples over which the gradient is averaged, which in turn makes the gradient direction more reliable (Sjoberg et al., 1995).

Table 1: Evaluation of the computing speed of our engine on a robot model controlled by a closed loop controller with a variable number of parameters. We evaluated both on CPU (i7 5930K) and GPU (GTX 1080), both for a single robot optimization and for batches of multiple robots in parallel. The numbers are the time required in seconds for simulating the quadruped robot(s) for  $10\mathrm{s}$ , with and without calculating a gradient. The gradient calculated here is the Jacobian of the total traveled distance of the robot in  $10\mathrm{s}$ , differentiated with respect to all the parameters of the controller.  
Seconds of computing time required to simulate a batch of robots for 10 seconds  

<table><tr><td></td><td></td><td colspan="2">with gradient</td><td colspan="2">without gradient</td></tr><tr><td></td><td></td><td>CPU</td><td>GPU</td><td>CPU</td><td>GPU</td></tr><tr><td rowspan="2">1 robot</td><td>1 296 parameters</td><td>8.17</td><td>69.6</td><td>1.06</td><td>9.69</td></tr><tr><td>1 147 904 parameters</td><td>13.2</td><td>75.0</td><td>2.04</td><td>9.69</td></tr><tr><td rowspan="2">128 robots</td><td>1 296 parameters</td><td>263</td><td>128</td><td>47.7</td><td>17.8</td></tr><tr><td>1 147 904 parameters</td><td>311</td><td>129</td><td>50.4</td><td>18.3</td></tr></table>

- The value of the gradient is decreased by a factor  $\alpha < 1$  at every time step. This trick has the effect of a prior. Namely, events further in the past are less important for influencing current events, because intermediate events might diminish their influence altogether. It also improves robustness against exploding gradients (Hermans et al., 2014).  
- We initialize the controller intelligently. We do not want the controller to shake the actuators violently and explore outside the accurate domain of our simulation model. Therefore our controllers are initialized such that they only output zeros at the start of the simulation. The initial policy is the zero policy.  
- We constraint the size of the gradient to an L2-norm of 1. This makes sure that gradients close to discontinuities in the fitness landscape do not push the parameter values too far away, such that everything which was learned is forgotten (Sutskever, 2013).

# 3.2.1 REACHING A FIXED POINT

A first simple task, is to have a small neural net controller learn to move the controller to a certain fixed point in space, at coordinates  $(0.5\mathrm{m};0.5\mathrm{m};0.5\mathrm{m})$ . The objective we minimize for this task, is the distance between the end effector and the target point, averaged over the 8 seconds we simulate our model.

We provide the controller with a single sensor input, namely the current distance between the end effector and the target point. Input is not required for this task, as there are solutions for which the motor signals are constant in time. However, this would not necessarily be the optimal approach for minimizing the average distance over time, it only solves the distance at the end of the simulation, but does not minimize the distance during the trajectory to get at the final position.

As a controller, we use a dense neural network with 1 input, 2 hidden layers of 128 units with a rectifier activation function, and 4 outputs with an identity activation function. This controller has 17 284 parameters in total. We disabled the recurrent connections  $\mathbf{h}^t$ .

We use gradient descent with a batch size of 1 robot for optimization, as the problem is not stochastic in nature. The parameters are optimized with Adam's rule (Kingma and Ba, 2014) with a learning rate of 0.001. Every update step with this method takes about 5 seconds on CPU. We find that the controller comes within  $4\mathrm{cm}$  of the target in 100 model evaluations, and within  $1\mathrm{cm}$  in 150 model evaluations, which is small compared to the  $1\mathrm{m}$  arm of the robot. Moreover, the controller does find a more optimal trajectory which takes into account the sensor information.

Solving problems like these in fewer iteration steps than the number of parameters, is unfeasible with derivative free methods (Sjoberg et al., 1995). Despite that, we did try to optimize the same problem with CMA-ES. After a week of computing and 60 000 model evaluations, CMA-ES did not show any sign of convergence, as it cannot handle the sheer amount of parameters.

# 3.2.2 REACHING A RANDOM POINT

As a second task, we sample a random target point in the reachable space of the end effector. We give this point as input  $v'$  to the controller, and the task is to again minimize the average distance between the end effector and the target point  $v$ . Our objective  $\mathcal{L}$  is this distance averaged over all timesteps.

As a controller, we use a dense neural network comparable to the previous section, but this time with 3 inputs. We used 3 hidden layers with 1024 units each, so the controller has 2 107 396 parameters in total. This is not necessary for this task, but we do it like this to demonstrate the power of this approach. In order to train for this task, we use a batch size of 128 robots, such that every update step takes  $58\mathrm{s}$  on GPU. Each simulation takes  $8\mathrm{s}$  with a simulation step of  $0.01\mathrm{s}$ . Therefore, the gradient on the parameters of the controllers has been averaged over 51 200 timesteps at every update step. We update the parameters with Adam's rule, where we scale the learning rate with the average error achieved in the previous step.

We find that it takes 576 update steps before the millions of parameters are optimized, such that the end effector of the robot is on average less than  $10\mathrm{cm}$  of target, 2563 update steps before the error is less than  $5\mathrm{cm}$ .

# 3.3 A QUADRUPEDAL ROBOT - REVISITED

Optimizing a gait for a quadrupedal robot is a problem of a different order, something the authors have extensive experience with (Sproewitz et al., 2013; Degrave et al., 2013; 2015). The problem is way more challenging and allows for a broad range of possible solutions. In nature, we find a wide variety of gaits, from hopping over trotting, walking and galloping. With hand tuning on the robot model shown in Figure 3b, we were able to obtain a trotting motion with an average forward speed of  $0.7\mathrm{m / s}$ . We found it tricky to find a gait where the robot did not end up like an upside down turtle, as  $75\%$  of the mass of the robot is located in its torso.

As a controller for our quadrupedal robot, we use a neural network with 2 input signals  $\mathbf{s}^t$ , namely a sine and a cosine signal with a frequency of  $1.5\mathrm{Hz}$ . On top of this, we added 2 hidden layers of 128 units and a rectifier activation function. As output layer, we have a dense layer with 8 units and a linear activation function, which has as input both the input layer and the top layer of the hidden layers. In total, this controller has 17952 parameters. Since the problem is not stochastic in nature, we use a batch size of 1 robot. We initialize the output layer with zero weights, so the robot starts the optimization in a stand still position.

We optimize these parameters to maximize the average velocity of the spine over the course of 10 s of time in simulation. This way, the gradient used in the update step is effectively an average of the 1000 time steps after unrolling the recurrent connections. This objective does not take into account energy use, or other metrics typically employed in robotic problems.

In only 500 model evaluations or about 1 hour of optimizing on CPU, the optimization with BPTT comes up with a solution with a speed of  $1.17\mathrm{m / s}$ . This solution is a hopping gait, with a summersault every 3 steps<sup>1</sup>, despite limiting the torque of the servos to  $4\mathrm{Nm}$  on this  $28.7\mathrm{kg}$  robot. For more life-like gaits, energy efficiency could be used as a regularization method. Evaluating these improvements are however outside the scope of this paper.

# 4 DISCUSSION

Our results show the first prototype of a differentiable physics engine based on similar algorithms as those that are commonly used in modern robotics simulators. When initially addressing the problem, we had no idea whether finding the gradient would be computationally tractable, let alone whether evaluating it would be fast enough to be beneficial for optimization. In this paper, we have demonstrated that evaluating the gradient is tractable enough to even speed up optimization on problems with as little as six parameters. The speed of this evaluation mainly depends on the complexity of the physics model and only slightly on the number of parameters to optimize. Therefore, our results suggest that this cost is dominated by the gain achieved by the combination of using batch gradient

descent and GPU acceleration. This statement is particularly the case when optimizing controllers with very high numbers of parameters, where we suspect this approach is asymptotical of a lower order in the number of parameters, as each gradient step also contains information proportional to the number of parameters.

Optimizing the controller of a robot model with gradient-based optimization is equivalent to optimizing an RNN. After all, the gradient passes through each parameter at every time step. The parameter space is therefore very noisy. Consequently, training the parameters of this controller is a highly non-trivial problem, as it corresponds to training the parameters of an RNN. On top of that, exploding and vanishing signals and gradients cause far more challenging problems compared to feed forward networks.

In section 3.2, we already discussed some of the tricks used for optimizing RNNs. Earlier research shows that these methods can be extended to even more complicated tasks than the ones discussed here (Hermans et al., 2014; Sutskever, 2013). Hence, we believe that this approach towards learning controllers for robotics applies to far more complex problems than the simple examples tackled in this paper.

All of the results in this paper will of course largely depend on showing how these controllers will work on the physical counterparts of our models. Nonetheless, we would like to conjecture that to a certain extent, this gradient of a model is close to the gradient of the physical system. The gradient of the model is even more susceptible to high-frequency noise introduced by modeling the system, than the imaginary gradient of the system itself. Nonetheless, it contains information which might be indicative, even if it is not perfect. We would theorize that using this noisy gradient is still better than optimizing in the blind and that the transferability to real robots can be improved by evaluating the gradients on batches of (slightly) different robots in (slightly) different situations and averaging the results. This technique has already been applied in (Hermans et al., 2014) as a regularization method to avoid bifurcations during online learning. If the previous proves to be correct, our approach can offer an alternative to deep Q-learning for deep learning controllers in robotics.

We can see the use of this extended approach for a broad range of applications in robotics. Not only do we think there are multiple ways where recent advances in deep learning could be applied to robotics more efficiently with a differentiable physics engine, we also see various ways in which this engine could improve existing angles at which robotics are currently approached:

- In this paper, we added memory by introducing recurrent connections in the neural network controller. We reckon that advanced, recurrent connections such as ones with a memory made out of LSTM cells (Hochreiter and Schmidhuber, 1997) can allow for more powerful controllers than the controllers described in this paper.

- Using a differentiable physics engine, we reckon that knowledge of a model can be transferred more efficiently into a forward or backward model in the form of a neural network, similar to methods such as used in Johnson et al. (2016) and Dumoulin et al. (2016). By pulling the gradient through an exact model and defining a relevant error on the model, information should readily be able to transfer from a forward or backward model in the differentiable physics engine to a forward or backward neural network model. We reckon that neural network models trained this way might be more robust than the ones learned from generated trajectories (Christiano et al., 2016). In turn, this neural model can then be used for faster but approximate evaluation of the model.

- Although we did not address this in this paper, there is no reason why only control parameters could be differentiated. Hardware parameters of the robot have been optimized the same way before (Jarny et al., 1991; Iollo et al., 2001; Hermans et al., 2014). The authors reckon that the reverse process is also true. A physics engine can provide a strong prior, which can be used for robots to learn (or adjust) their robot models based on their hardware measurements faster than today. You could optimize the model parameters with gradient descent through physics, to have the model better mimic the actual observations.

- There is no reason why a camera model would not be differentiable either. We think that currently the only thing in the way to effectively learn deep robot controllers from camera information is someone implementing a differentiable camera model. As deep learning is especially effective in computer vision, and cameras are the bread and butter of sensors, this would probably form a good synergy.

- Where adversarial networks are already showing their use in generating image models, we believe adversarial robotics training (ART) will create some inventive ways to design and control robots. Like in generative adversarial nets (GAN) (Goodfellow et al., 2014), where the gradient is pulled through two competing neural networks, the gradient could be pulled through multiple competing robots as well. It would form an interesting approach for swarm robotics, similar to previous results in evolutionary robotics (Sims, 1994; Pfeifer and Bongard, 2006; Cheney et al., 2014), but possibly faster.

# 5 CONCLUSION

In this paper, we show it is possible to build a differentiable physics engine. We implemented a modern engine which can run a 3D rigid body model, using the same algorithm as other engines commonly used to simulate robots, but we can additionally differentiate control parameters with BPTT. Our implementation also runs on GPU, and we show that using GPUs to simulate the physics can speed up the process for large batches of robots.

We find that these gradients can be computed surprisingly fast. We also show that using gradient descent with BPTT speeds up optimization processes often found in robotics, even for rather small problems, due to the reduced number of model evaluations required. We show that this improvement in speed scales to problems with a lot of parameters. We also show that using this engine, finding policies for robot models can be done faster and in a more straightforward way. This method should allow for a new approach to apply deep learning techniques in robotics.

# ACKNOWLEDGMENTS

Special thanks to David Pfau for pointing out relevant prior art we were previously unaware of, and Iryna Korshunova for proofreading the paper. The research leading to these results has received funding from the Agency for Innovation by Science and Technology in Flanders (IWT). The NVIDIA Corporation donated the GTX 1080 used for this research.

# REFERENCES

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., and Zheng, X. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.org.  
Al-Rfou, R., Alain, G., Almahairi, A., Angermueller, C., Bahdanau, D., Ballas, N., Bastien, F., Bayer, J., Belikov, A., Belopolsky, A., Bengio, Y., Bergeron, A., Bergstra, J., Bisson, V., Bleecher Snyder, J., Bouchard, N., Boulanger-Lewandowski, N., Bouthillier, X., de Brebisson, A., Breuleux, O., Carrier, P.-L., Cho, K., Chorowski, J., Christiano, P., Coolijmans, T., Côté, M.-A., Côté, M., Courville, A., Dauphin, Y. N., Delal-leau, O., Demouth, J., Desjardins, G., Dieleman, S., Dinh, L., Ducoffe, M., Dumoulin, V., Ebrahimi Kahou, S., Erhan, D., Fan, Z., First, O., Germain, M., Glorot, X., Goodfellow, I., Graham, M., Gulcehre, C., Hamel, P., Harlouchet, I., Heng, J.-P., Hidasi, B., Honari, S., Jain, A., Jean, S., Jia, K., Korobov, M., Kulkarni, V., Lamb, A., Lamblin, P., Larsen, E., Laurent, C., Lee, S., Lefrancois, S., Lemieux, S., Leonard, N., Lin, Z., Livezey, J. A., Lorenz, C., Lowin, J., Ma, Q., Manzagol, P.-A., Mastropietro, O., McGibbon, R. T., Memisevic, R., van Merrienboer, B., Michalski, V., Mirza, M., Orlandi, A., Pal, C., Pascanu, R., Pezeshki, M., Raffel, C., Renshaw, D., Rocklin, M., Romero, A., Roth, M., Sadowski, P., Salvatier, J., Savard, F., Schluter, J., Schulman, J., Schwartz, G., Serban, I. V., Serdyuk, D., Shabanian, S., Simon, E., Spieckermann, S., Subramanyam, S. R., Sygnowski, J., Tanguay, J., van Tulder, G., Turian, J., Urban, S., Vincent, P., Visin, F., de Vries, H., Warde-Farley, D., Webb, D. J., Willson, M., Xu, K., Xue, L., Yao, L., Zhang, S., and Zhang, Y. (2016). Theano: A Python framework for fast computation of mathematical expressions. arXiv e-prints abs/1605.02688.  
Catto, E. (2009). Modeling and solving constraints. In Game Developers Conference.  
Chappuis, D. (2013). Constraints derivation for rigid body simulation in 3D.  
Cheney, N., Clune, J., and Lipson, H. (2014). Evolved electrophysiological soft robots. In ALIFE, volume 14, pages 222-229.

Christiano, P., Shah, Z., Mordatch, I., Schneider, J., Blackwell, T., Tobin, J., Abbeel, P., and Zaremba, W. (2016). Transfer from simulation to real world through learning deep inverse dynamics model. arXiv preprint arXiv:1610.03518.  
Degrave, J., Burm, M., Kindermans, P.-J., Dambre, J., and wyffels, F. (2015). Transfer learning of gaits on a quadrupedal robot. Adaptive Behavior, page 1059712314563620.  
Degrave, J., Burm, M., Waegeman, T., Wyffels, F., and Schrauwen, B. (2013). Comparing trotting and turning strategies on the quadrupedal oncilla robot. In Robotics and Biomimetics (ROBIO), 2013 IEEE International Conference on, pages 228-233. IEEE.  
Degrave, J., Dieleman, S., Dambre, J., and wyffels, F. (2016). Spatial chirp-Z transformer networks. In European Symposium on Artificial Neural Networks (ESANN).  
Dumoulin, V., Shlens, J., and Kudlur, M. (2016). A learned representation for artistic style. CoRR, abs/1610.07629.  
Erez, T., Tassa, Y., and Todorov, E. (2015). Simulation tools for model-based robotics: Comparison of bullet, havok, Mujoco, ode and physx. In International Conference on Robotics and Automation (ICRA), pages 4397-4404. IEEE.  
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672-2680.  
Hansen, N. (2006). The cma evolution strategy: a comparing review. In Towards a new evolutionary computation, pages 75-102. Springer Berlin Heidelberg.  
Hermans, M., Schrauwen, B., Bienstman, P., and Dambre, J. (2014). Automated design of complex dynamic systems. *PloS one*, 9(1):e86696.  
Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8):1735-1780.  
Hubbard, P. M. (1996). Approximating polyhedra with spheres for time-critical collision detection. ACM Transactions on Graphics (TOG), 15(3):179-210.  
Iollo, A., Ferlauto, M., and Zannetti, L. (2001). An aerodynamic optimization method based on the inverse problem adjoint equations. Journal of Computational Physics, 173(1):87-115.  
Jarny, Y., Ozisik, M., and Bardon, J. (1991). A general optimization method using adjoint equation for solving multidimensional inverse heat conduction. International journal of heat and mass transfer, 34(11):2911-2919.  
Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., Guadarrama, S., and Darrell, T. (2014). Caffe: Convolutional architecture for fast feature embedding. arXiv preprint arXiv:1408.5093.  
Johnson, J., Alahi, A., and Fei-Fei, L. (2016). Perceptual losses for real-time style transfer and super-resolution. arXiv preprint arXiv:1603.08155.  
Jourdan, F., Alart, P., and Jean, M. (1998). A gauss-seidel like algorithm to solve frictional contact problems. Computer methods in applied mechanics and engineering, 155(1):31-47.  
Kingma, D. P. and Ba, J. (2014). Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations (ICLR).  
Levine, S. and Koltun, V. (2013). Variational policy search via trajectory optimization. In Advances in Neural Information Processing Systems, pages 207-215.  
Levine, S., Pastor, P., Krizhevsky, A., and Quillen, D. (2016). Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. arXiv preprint arXiv:1603.02199.  
Mirtich, B. (1998). V-clip: Fast and robust polyhedral collision detection. ACM Transactions On Graphics (TOG), 17(3):177-208.  
Mordatch, I., Lowrey, K., Andrew, G., Popovic, Z., and Todorov, E. V. (2015). Interactive control of diverse complex characters with neural networks. In Advances in Neural Information Processing Systems, pages 3132-3140.  
Mordatch, I. and Todorov, E. (2014). Combining the benefits of function approximation and trajectory optimization. In Robotics: Science and Systems (RSS).

Pfeifer, R. and Bongard, J. (2006). How the body shapes the way we think: a new view of intelligence. MIT press.  
Premerlani, W. and Bizard, P. (2009). Direction cosine matrix IMU: Theory. DIY DRONE: USA, pages 13-15.  
Sims, K. (1994). Evolving 3d morphology and behavior by competition. Artificial life, 1(4):353-372.  
Sjoberg, J., Zhang, Q., Ljung, L., Benveniste, A., Delyon, B., Glorennec, P.-Y., Hjalmarsson, H., and Juditsky, A. (1995). Nonlinear black-box modeling in system identification: a unified overview. Automatica, 31(12):1691-1724.  
Sproewitz, A., Tuleu, A., D'Haene, M., Möckel, R., Degrave, J., Vespignani, M., Gay, S., Ajallooeian, M., Schrauwen, B., and Ijspeert, A. J. (2013). Towards dynamically running quadruped robots: performance, scaling, and comparison. In Adaptive Motion of Animals and Machines, pages 133-135.  
Stewart, D. and Trinkle, J. C. (2000). An implicit time-stepping scheme for rigid body dynamics with coulomb friction. In International Conference on Robotics and Automation (ICRA), volume 1, pages 162-169. IEEE.  
Sutskever, I. (2013). Training recurrent neural networks. PhD thesis, University of Toronto.  
Tassa, Y., Erez, T., and Todorov, E. (2012). Synthesis and stabilization of complex behaviors through online trajectory optimization. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 4906-4913. IEEE.  
Todorov, E., Erez, T., and Tassa, Y. (2012). Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026-5033. IEEE.