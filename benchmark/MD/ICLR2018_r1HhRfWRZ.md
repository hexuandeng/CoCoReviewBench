# LEARNING AWARENESS MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We consider the setting of an agent with a fixed body interacting with an unknown and uncertain external world. We show that by maximizing the entropy of predictions about the body—touch sensors, proprioception and vestibular information—we are able to learn dynamic models of the body that can be used for control. In spite of being trained with only internally available signals, these dynamic body models come to represent external objects through the necessity of predicting their effects on the agent's own body. Our dynamics model is able to successfully predict distributions over 132 sensor readings over 100 steps into the future. We demonstrate that even when the body is no longer in contact with an object, the latent variables of the dynamics model continue to represent its shape. That is, the model learns holistic persistent representations of objects in the world, even though the only training signals are body signals. We also collect data from a real robotic hand and show that the same models can be used to answer questions about properties of objects in the real world.

# 1 INTRODUCTION

Situation awareness is the perception of the elements in the environment within a volume of time and space, and the comprehension of their meaning, and the projection of their status in the near future. — Endsley (1987)

As artificial intelligence moves off of the server and out into the world at large; be this the simulated world, in the form of simulated walkers, climbers and other creatures (Heess et al., 2017), or the real world in the form of virtual assistants, self driving vehicles (Bojarski et al., 2016), and household robots (Jain et al., 2013); we are increasingly faced with the need to build systems that understand and reason about the world around them, and have access to this world through a bespoke interface.

When building systems like this it is natural to think of the physical world as breaking into two parts. The first part is the platform, the part we design and build, and therefore know quite a lot about; and the second part is everything else, which comprises all the strange and exciting situations that the platform might find itself in. As designers we have very little control over this part of the world, and the variety of situations that might arise are too numerous to anticipate in advance. Additionally, while the state of the platform is readily accessible (e.g. through deployment of integrated sensors), the state of the external world is generally not available to the system.

The platform hosts any sensors and actuators that are part of the system, and importantly it can be relied on to be the same across the wide variety situations where the system might be deployed. A virtual assistant can rely on having access to the camera and microphone on your smart phone, and the control system for a self driving car can assume it is controlling a specific make and model of vehicle, and that it has access to any specialized hardware installed by the manufacturer. These consistency assumptions hold regardless of what is happening in the external world.

This same partitioning of the world, into the self and the other, occurs naturally for living creatures as well. As a human being your platform is your body; it maintains a constant size and shape throughout your life (or at least these change vastly slower than the world around you), and you can hopefully rely on the fact that no matter what demands tomorrow might make of you, you will face them with the same number of fingers and toes.

The relationship between your body and the world is interesting because this interface is the critical point where information flows from the chaotic external world to the familiar internal world. Contact

with an object is where we transition from perceiving it merely through vision to also having access to its temperature, texture, mass and inertial properties through our tactile and proprioceptive senses.

This story of partitioning the world into the consistent body and the chaotic exterior parts that exchange information at the boundary suggests an approach to building models for reasoning about the world. If the body is a consistent vehicle through which an agent interacts with the world and proprioceptive and tactile senses live at the boundary of the body, then predictive models of these senses should result in models that represent external objects, in order to accurately predict their future effects on the body. Moreover, these models should be reusable in different situations because the body doesn't change.

Despite the fact that we are only modeling body signals, we show that the latent variables of the body models learn holistic persistent representations of objects in the world. Consequently, the agent is aware of the properties of objects in the world even when not touching them.

In our experiments we consider two bodies. The first is a simulated model of the hand of the Johns Hopkins Modular Prosthetic Limb (Johannes et al., 2011), realized in MuJoCo. The model is actuated by 13 motors each capable of exerting a bidirectional force on a single joint. The module is also instrumented with a series of sensors measuring angles and torques of the joints as well as pressure sensors measuring contact forces at several locations across the hand model. There are also inertial measurement units located at the end of each finger which measure translational and rotational accelerations. In total there are 132 sensor measurements whose values we predict using our dynamics model.

The second body we consider allows us to show that our ideas apply not only in simulation, but succeed in the real world as well. Shadow Dexterous Hand, which is real robotic hand with 20 degree of freedom control. The instrumentation of the Shadow Hand is similar to the MPL model we use in simulation. It is instrumented with sensors measuring the tension of the tendons driving the fingers, and also has pressure sensors on the pad of each fingertip that measure contact forces with objects in the world. We apply the same techniques used on the simulated model to data collected from this real platform and use the resulting model to make predictions about states of external objects in the real world.

In summary, this paper shows that models trained entirely to predict body measurements can be used to answer questions about objects in the world their bodies inhabit. The information in these models can also be leveraged for control, by choosing actions to drive the world toward states where the models make uncertain predictions we gather more informative data which in turn leads to more accurate predictive models.

# 2 RELATED WORK

Given our goal to gather information about the world and, and in particular to actively seek out information about external objects, our work is naturally related to work on intrinsic motivation. The literature on intrinsic motivation is vast and rich, and we do not attempt to review it fully here. Some representative works include Oudeyer & Kaplan (2008; 2009); Sequeira et al. (2011); Bellemare et al. (2016) and Pathak et al. (2017).

Several authors have implemented intrinsic motivation, or curiosity based objectives in visual space, through predicting interactions with objects (Pinto et al., 2016), or through predicting summary statistics of the future (Venkatraman et al., 2017; Downey et al., 2017). Other authors have also investigated using learned future predictions directly for control (Dosovitskiy & Koltun, 2016).

Humans use their hands to gather information in structured task driven ways (Lederman & Klatzky, 1987); and it will become clear from the experiments why hands are relevant to our work. Our interest in hands and touch brings us into contact with a vast literature on haptics (Zheng et al., 2016; Gao et al., 2016; Cao et al., 2016; Loeb, 2013; Edmonds et al., 2017; Su et al., 2015; Navarro et al., 2012; Aggarwal et al., 2015; Liu et al.; Sung et al., 2017; Ciobanu et al., 2013; Karl et al., 2016; Su et al., 2012).

There is also work in robotics on using the anticipation of sensation to guide actions (Indranil Sur, 2017), and on showing how touch sensing can improve the performance of grasping tasks (Calan-

dra et al., 2017). Model based planning has been very successful in these domains (Deisenroth & Rasmussen, 2011).

# 3 MODELS AND DIAGNOSTICS

Our goal in this work is to build predictive models of proprioception and to use these models to reason about the properties of external objects. In this section we formalize this notion and discuss how we can measure whether or not we have achieved this goal.

We consider some agent operating in a discrete-time setting where there is a stochastic unobservable global state  $s_t \in S$  at each timestep  $t$  and the agent obtains a stochastic observation  $o_t \in \mathcal{O}$  where  $\mathcal{O} \subseteq S$  and takes some action  $u_t \in \mathcal{U}$ . Our goal is to learn a predictive model of the agent's action-conditional future observations  $p(o_{t+1:t+k}|u_{1:t+k}, o_{1:t})$  for  $k$  time steps into the future given all of the previous observations and actions it has taken. We will then use these models to reason about the global state  $s_t$  even though no information about this state is available during training.

Demonstrating success in this setting involves showing two things, first that we succeed in training accurate predictive models and second that they can be used to reason about the world. Assessing the first part is straightforward and can be done simply by computing the likelihood our models assign to a reference set of trajectories over its observations  $o_{t}$ .

Showing the second point requires more care, since our claim is about the information content of the states of the predictive models. We do not claim that the states of the models will be interpretable, only that the information required for reasoning about external objects is present.

To show that information required for reasoning is present in the states of our predictive models we use auxiliary models, which we call "diagnostic" models. A diagnostic model looks at the states of a proprioceptive model (which we refer to as the "base" model in this context) and uses them to predict an interpretable quantity in the world,  $x_{t} \in \mathcal{X}$ , where  $\mathcal{X} \subseteq S$  and in most cases  $\mathcal{X} \cap \mathcal{O} = \emptyset$ . When training a diagnostic model we allow ourselves to use privileged information  $x_{t}$  to define the loss, but we do not allow the diagnostic loss to influence the representations of the base model.

The diagnostic models are a post-hoc analysis strategy. The base models are trained using only the proprioceptive information, and then frozen. After the base models are trained we train diagnostic models on their states, and the claim is that if we can successfully predict properties of external objects using diagnostic models trained in this way then information about the external objects is available in the states of the base model.

We consider two types of diagnostic model in this work. The first is a property-based diagnostic where the task is to predict a physical property of some external object (e.g. its shape or spatial extent). In order to define the objective for these diagnostic tasks in simulation we can read the desired ground truth directly from the simulator state. To build a property based diagnostic in the real world we build an experimental apparatus that records the desired diagnostic quantity in addition to the proprioceptive measurements.

# 4 AWARENESS MODELS

We treat predicting proprioception as a sequence-to-sequence problem, where the input sequence is a prefix of an episode and the encoder observes both the proprioceptive observations  $o_{t}$  and the actions  $u_{t}$  taken at each timestep. Probabilistic graphic modeling provides many ways to model the observation space  $p(o_{t + 1:t + k}|u_{1:t + k},o_{1:t})$ . Our approach shown in Figure 1 that allows us to easily obtain global state knowledge is to use deterministic hidden states and divide the model into an encoder part that encodes the past into some hidden state  $h_t$  and can decode into the future with hidden states  $z_{t}$ .

The encoder is trained to encode the past trajectory  $\{u_{1:t}, o_{1:t}\}$  into  $h_t$ . It does this by recurrently updating itself based on the current observation and action:  $h_{t+1} = \operatorname{Encoder}_{\theta}(o_t, u_t, h_t)$ . The decoder then receives this updated hidden state.

The decoder in our model is also recurrent and is conditioned on future actions, but it is intentionally not autoregressive in time:  $z_{t} = \mathrm{Decorder}_{\theta}(u_{t},z_{t - 1})$ . Conditioned on its hidden state, predictions

![](images/0d21dd629a631a72f33d816392d72f1a030673f5f1085976c26c011a66e3f7cc.jpg)

![](images/5b3c90b0fdd875abcaf0e9bf6fb07ecca0c4ea46519bfa298811ed1747f87d9a.jpg)  
Figure 1: Left: Diagram of the full model. Right: Top down view of the training graph showing only the cores and adapters at the interface between the encoder and decoder components.

made by the decoder at each future timestep are independent, meaning the model makes open loop predictions. The decoder is trained to predict the continuation of the proprioceptive signal for  $k$  steps into the future, conditioned on the actions that were taken,

$$
p \left(o _ {t + 1: t + k} \mid u _ {t + 1: t + k}, z _ {t + 1: t + k}\right) = \prod_ {\kappa = 1} ^ {k} p \left(o _ {t + \kappa} \mid u _ {t + \kappa}, z _ {t + 1: t + \kappa}\right).
$$

Between the encoder and decoder we have an adapter module that maps the final state of the encoder  $h_t$  to an initial state for the decoder  $z_{t+1}$ .

For training, there is no natural point within an episode to switch from encoding to prediction, and we use this opportunity to make forward predictions from every point, using a shared encoder to condition forward predictions at every step. The structure of the resulting graph is shown on the right of Figure 1. We call this technique overshooting, and we call the number of steps predicted forward by the decoder the overshooting length.

The effect of making open-loop (rather than closed-loop) predictions is two fold. First, it forces decisions about possible states of the external world to be made in the encoder rather than the decoder. An autoregressive model could make these decisions based on sampled future observations, but without the autoregressive connections the predicted trajectory is fixed given the encoder state and the future actions.

Second, forcing open loop predictions ensures that any true uncertainty in the state of the external world will manifest as high entropy in the decoder likelihoods. An autoregressive model can resolve this type of uncertainty through sampling, but our model is forced to make predictions that cover all possibilities. We show later sections that we can use high entropy in the predictions as an objective for control that allows us to plan actions to gather information about the environment.

All the recurrent parts of the models use single layer LSTM cores. All observations are embedded with an observation-type-specific encoder MLP (e.g. controls and sensors are each embedded with their own network in each part of the model where they are consumed). When a module takes more than one input (e.g. the encoder consumes both controls and proprioception) we embed each signal separately and concatenate the embeddings.

For making predictions we use independent mixtures of Gaussians at every step. Each dimension of each prediction is an independent mixture. We use separate MLPs to produce the means, standard deviations and mixture weights from the output of the decoder LSTM.

We run a parameter search over several model hyperparameters for each experiment separately. The particular choices of parameters we use for each experiment are listed in Appendix C.

# 5 ENVIRONMENT

The environment consists of a simulated model of the hand part of the Johns Hopkins Modular Prosthetic Limb (Johannes et al., 2011) which we refer to as the "MPL hand", or simply "the hand".

This model is distributed with the MuJoCo HAPTIX software and is available for download from the MuJoCo website. The hand is actuated by 13 motors and has sensors that provide a 132 dimensional observation. See Appendix B for a detailed description.

In each episode the hand starts suspended above the table with its palm facing downwards. A random geometric object is placed underneath the hand, which is free to move to grasp or manipulate the object. The shapes of the object are randomly chosen to be a box, cylinder or ellipsoid and the size and orientation of each object are randomly chosen from reasonable ranges.

# 6 PASSIVE AWARENESS

We begin by exploring awareness in the passive setting, as a pure supervised learning problem. In this setting we still build action-dependent models, but where actions are needed they are generated from a pre-programmed behavior that follows a sensible trajectory for the environment (we consider actively controlling actions in the next section). We hand-design a simple grasping motion that closes the hand about the object beneath it and then releases it. We generate data from the environment by running this hand designed grasp-and-release cycle three times for each episode.

We train one of the models described in Section 4. The full set of hyperparameters can be found in Appendix C.

We show that information about the target object can be recovered from the encoder's hidden states  $h_t$  using diagnostic model approach described in Section 3. We look at two different diagnostic tasks:

1. Classification: Predicting the class of the object. Each object is either a box, a cylinder or an ellipsoid.  
2. Regression: Predicting the spatial extents of the target object. We use a separate diagnostic network for predicting the extents of each type of object. During training this is implemented by training each diagnostic model on only episodes containing the corresponding object type, and at test time this is implemented by first applying the classification diagnostic to identify the object type and then applying the appropriate shape diagnostic.

Results of this experiment are shown in Figure 2. The results show that we can reliably recover the object type and parameters from the learned dynamics models, even though this information is not available to the model directly either in the input or in the training loss.

# 7 ACTIVE AWARENESS

In the previous section we showed that our models come to represent properties of the external world based solely on modelling the dynamics of proprioception. By training the models to make open loop predictions about future trajectories of proprioception we arrive at models that encode information about objects in the external world that they do not directly observe. We showed that this is the case by training diagnostic models, which look at the trained states and accurately predict properties of the external objects.

In this section we go a step further and show that not only do the models represent information about the external world, but this information is accessible, in the sense that we can use the models to reason about external objects without explicitly knowing what they are. In particular, we show how the models from the previous section can be used to choose actions to gather information about the external objects. We show that this active exploration leads to better predictive models and causes them to form more accurate representations of the external objects.

![](images/7152d7239f72077a808b8c190f9aa202294fd4446c4a4b5717635f727dd1323e.jpg)

![](images/d5d0f9df9c99d7a4c010fec17336cc827fc4770cd29e2d3b46d947ec352cc97a.jpg)

![](images/ecbccf596ed0fdd8009164a8aebebe0f96f2844f8abe08a70a3f431f0cbe820f.jpg)  
Figure 2: Top: Passive diagnostic curves on the Blocks dataset. The left plot shows the performance of the classification diagnostic and the right plot shows performance of the regression diagnostic. In both cases predictions based on model features (in blue) are compared with performance of the same diagnostic looking only at the sensor observations. Solid lines show the median error at each timestep, averaged over trajectories from the test set. The shaded regions enclose the 25th-75th percentile predictions. Black lines mark critical points in the trajectories: dashed lines show when the hand is fully open and solid lines show where it is fully closed. Bottom: Bootstrap estimates of the probability that the model features give a better estimate of the the corresponding diagnostic quantity than the sensor readings alone, again as a function of time.

![](images/9eb52104d2f5f788991adffb8e47b0d72dbc0dbda1d2870a40fe77efe09f83f2.jpg)

# 7.1 USING AWARENESS MODELS FOR CONTROL

We use Model Predictive Control (MPC) to achieve objectives with our awareness models. We express objectives as a cost function  $C(z_{t},f_{t})$  where  $z_{t}$  is the decoder's hidden state at some timestep  $t$  and  $f_{t}$  is the predicted PDF over the observations. Once the agent has taken some action  $u_{t}$ , obtained an observaiton  $o_{t}$ , and updated the encoder's hidden state  $h_{t}$ , the next action to take is found by solving the MPC problem

$$
u _ {1: T} ^ {*} = \operatorname * {a r g m i n} _ {u _ {1: T}, z _ {1: T}, f _ {1: T}} \sum_ {t} C (z _ {t}, f _ {t})
$$

s.t.  $f_{t} = \mathrm{DecodePDF}_{\theta}(z_{t})$

$$
z _ {t} = \operatorname {D e c o d e r} _ {\theta} \left(u _ {t}, z _ {t - 1}\right) \tag {1}
$$

$$
z _ {0} = \operatorname {A d a p t o r} _ {\theta} \left(h _ {t}\right)
$$

$$
u _ {1: T} \in \mathcal {U} _ {1: T}
$$

where the nominal timesteps are normalized so the first predicted timestep is at index 1 and  $\mathrm{DecodePDF}_{\theta}(z_t)$  returns a PDF over  $o_t$ . In most of our experiments, we parameterize  $\mathrm{DecodePDF}_{\theta}$  as a mixture of Gaussian distributions, ensure actions lie in a box  $||u_t||_{\infty} \leq 1$ , and enforce slew rate constraints  $||u_{t+1} - u_t||_{\infty} \leq 0.1$ .

# 7.2 GATHERING INFORMATION THROUGH MAXIMIZING UNCERTAINTY

The only source of randomness in our environment is over the properties of the target objects. The hand is always initialized in the same position and the physics simulation is deterministic. This means when our models make uncertain predictions, the source of that uncertainty comes from one of two places: (1) the model is poor, which comes either from too little data or from too small capacity, or (2) some property of the external objects are not yet resolved by the observations seen so far.

We can exploit this fact by choosing actions to maximize the uncertainty in the rollout predictions. An agent using this uncertainty maximizing policy attempts to seek actions for which the outcome is not yet known. This uncertainty can is then resolved by executing these actions and observing their outcome, and the resulting trajectory of observations, actions, and sensations can be used to refine the model.

To choose actions to gather information we use Model Predictive Control (MPC) over an objective that maximizes the uncertainty in the predictions. We use the Rényi entropy of our model predictions as our measure of uncertainty, since it can be easily computed in closed form for the Mixture of Gaussian predictions that we make for each sensor. Concretely, for a single Mixture of Gaussians prediction  $f(x)$  we can write

$$
H (f) = - \log \left[ \int f (x) ^ {2} \mathrm {d} x \right] = - \log \left[ \sum_ {i j} \alpha_ {i} \alpha_ {j} \frac {\exp \left\{- \frac {(\mu_ {i} - \mu_ {j}) ^ {2}}{2 (\sigma_ {i} ^ {2} + \sigma_ {j} ^ {2})} \right\}}{\sqrt {2 \pi} \sqrt {\sigma_ {i} ^ {2} + \sigma_ {j} ^ {2}}} \right]
$$

where  $i$  and  $j$  index the mixture components in the likelihood. We obtain an information seeking objective by summing the entropy of the predictions across sensors and across time.

We implement this information gathering policy to collect training data for the model in which it is planning. In our implementation these are two processes running in parallel: we have several actors each with a copy of the current model weights. These use MPC as described in Section 7.1 to plan and execute a trajectory of actions that maximizes the model's predicted uncertainty over a fixed horizon trajectory into the future. The observations and actions generated by the actors are collected into a large shared buffer and stored for the learner.

While the actors are collecting data, a single learner process samples batches of the collected trajectories from the buffer being written to by the actors. The learner trains the model by maximum likelihood, using the same objective as in Section 6, and the updated model propagates back to the actors who continue to plan using the updated model (Anonymous ICLR Submission, 2017).

We train a large number of models in this way, since each one tends to generate qualitatively different behavior, even when optimizing the same information seeking objective. Different behavior between models is a result of different random initialization leading them to each collect and train on different data, and therefore each model makes different errors.

After training many models in this way, we take all of the trained models and generate new trajectories from all of the models using the same planning objective we used for training. We collect these new trajectories into a new data set that contains trajectories generated from many different models across many different instances of the environment. This data set contains substantially more diversity in behavior than the data from fixed grasps that we used in Section 6.

# 7.3 ACHIEVING NEW OBJECTIVES

We can use the trained models from the previous section to execute new behaviors, provided that they are expressible in terms of predictions made by the model. We do this with MPC as described in Section 7.1.

1. Maximizing entropy of the predictions, as we did during training, leads to exploratory behavior. In Figure 3 we show a typical frame from an entropy maximizing trajectory, as well as typical frames from controlling for two different objectives.  
2. Optimizing for fingertip pressure tends to lead to grasping behavior, since the easiest way to achieve pressure on the fingertips is to push them against the target block. There is an alternative solution which is often found where the hand makes a tight fist, pushing its fingertips into its own palm.  
3. Controlling to minimize the prediction entropy is also quite interesting. This is the negation of the information gathering objective, and it attempts to make future observations as uninformative as possible. Optimizing for this objective results in behavior where the hand consistently pulls away from the target object.

![](images/2e2bd3f433a10b4c8a1b322133d196cd28a64db2cd145c59e04719c74838e21e.jpg)  
(a) Maximize predicted entropy

![](images/73be1810da4c2bb4a9e455ca915da99bc2b4e8ebbfb9fb6b5ca368a32d2f295e.jpg)  
(b) Maximize fingertip pressure

![](images/ebc72b549ea2c13cdc2b6ee57dfe36704b66df50df49472da8ecebf9c832fd21.jpg)  
(c) Minimize predicted entropy

![](images/b60b43951ebca5a11a2aba9c657f8c5c688b5c16a1d36e54fe8c583be6efa2a8.jpg)  
Figure 3: Acting to optimize different objectives at test time leads to different behaviors.  
Figure 4: A visualization of the model planning to maximize predicted entropy. Each plot shows one of the 132 sensors. The red line in each plot shows the actual sensor readings achieved in this 500 step episode. The blue shaded regions show the predicted distributions unrolled over a 100 step horizon.

Figure 4 shows a snapshot of the model's internal state during planning. Each panel shows the trajectory of a different sensor reading across a 500 step episode. The figure also shows the predicted sensor readings, unrolled for 100 steps from a point mid-episode. These predictions often do not match the red lines because they show predictions along the current planned trajectory, and the plan is iteratively refined at each timestep.

# 8 PASSIVE AWARENESS IN THE REAL WORLD ON THE SHADOW HAND

We have shown that our models work well in simulation. We now turn to demonstrating that they are effective in reality as well.

We use the 24-joint Shadow Dexterous Hand<sup>3</sup> with 20-DOF tendon position control and set up a real life analog of our simulated environment, as shown in Figure 5. Since varying the spatial extents of an object in real life would be very labor intensive we instead use a single object fixed to a turntable

that can rotate to any one of 255 orientations, and our diagnostic task in this environment is to recover the orientation of the grasped object.

We built a turntable mechanism for orienting the object beneath the hand, and design some randomized grasp trajectories for the hand to close around the block. The object is a soft foam wedge (the shape is chosen to have an unambiguous orientation) and fixed to the turntable. At each episode we turn the table to a randomly chosen orientation and execute two grasp release cycles with the hand robot.

Over the course of two days we collected 1140 grasp trajectories in three sessions of 47, 393 and 700 trajectories. We use the 47 trajectories from the initial session as test data, and use the remaining 1093 trajectories for training. Each trajectory is 81 frames long and consists of two grasp-release cycles with the target object at a fixed orientation. At each timestep we measure four different proprioceptive features from the robot:

1. The actions which is a set of 20 desired joint positions sent to the robot for the current timestep.  
2. The angles which is a set of 24 measured joint positions reported by the robot at the current timestep. There are more angles than actions because not all joints of the hand are separately actuated, and the measured angles may not match the intended actions due to force limits imposed by the low level controller.  
3. The efforts, which provide 20 distinct torque readings. Each effort measurement is the signed difference in tension between tendons on the inside and outside of one of the actuated joints.  
4. The pressures are five scalar measurements that indicate the pressure experienced by the pads on the end of each finger.

Joint ranges of the hand are limited to prevent fingers pushing each other, and the actuator strengths are limited for the safety of the robot and the apparatus. At each grasp-release cycle final grasped and released positions are sampled from handcrafted distributions. Position targets sent to the robot are calculated by interpolating between these two positions in 20 steps.

There are multiple complexities the sensor model needs to deal with. First of all once a finger touches the object actual positions and target positions do not match, and the foam object bends and deforms. Also the hand can occasionally overcome the resistance in the turntable motor causing the target object to rotate during the episode (for about 10-20 degrees and rarely more). This creates extra unrecorded source of error in the data.

We train a forward model on the collected data, and then treat prediction of the orientation of the block as a diagnostic task. Figure 5 shows that we can successfully predict the orientation of the block from the dynamics model state.

# 9 CONCLUSION

In this paper we showed that learning a forward predictive model of proprioception we obtain models that can be used to answer questions and reason about objects in the external world. We demonstrated this in simulation with a series of diagnostic tasks where we use the model features to identify properties of external objects, and also with a control task where we show that we can plan in the model to achieve objectives that were not seen during training.

We also showed that the same principles we applied to our simulated models are also successful in reality. We collected data from a real robotic platform and used the same modelling techniques to predict the orientation of a grasped block.

![](images/8d85f7fe74aa0f5cc7c8b98bb03e1159d4653e72b77b39b2eef304dd4830b6ce.jpg)  
Figure 5: Left: The robotic hand setup. Center: Results on predicting block orientation with sensor data recorded from the shadow hand. The upper plot shows the median error as a function of time and the bottom plot shows a bootstrap estimate of the probability that using the model features improves on using sensor measurements directly. Right: Predicted angles on test trajectories at step 40 using only sensor readings (top) and model features (bottom). Green lines show predicted angles for individual samples (rotated so ground truth is vertical). The solid and dashed red lines show 50 and 75 percentile error cones, respectively.

![](images/878ff2b6a24919617576484b2f7b3e58e6b05a152194bb1e6c2b19217218d0fe.jpg)

![](images/0f48e52cedaa609dc433123b6d1c5f829e5cb802a7b10bb3f306d01117996db4.jpg)

![](images/bf2fb7b550f19cbb6c43b36fc59a5d1d91e0f7124d9a72f142d45758a9fe050a.jpg)

![](images/c3ee82234dc49009bb8efa2f30873116bc1e813d495802c0b0d2667719935080.jpg)

# REFERENCES

Achint Aggarwal, Peter Kampmann, Johannes Lemburg, and Frank Kirchner. Haptic object recognition in underwater and deep-sea environments. Journal of field robotics, 32(1):167-185, 2015.  
Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. In Advances in Neural Information Processing Systems, pp. 1471-1479, 2016.  
Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316, 2016.  
Roberto Calandra, Andrew Owens, Manu Upadhyaya, Wenzhen Yuan, Justin Lin, Edward H. Adelson, and Sergey Levine. The feeling of success: Does touch sensing help predict grasp outcomes? arXiv preprint arXiv:1710.05512, 2017.  
Lele Cao, Ramamohanarao Kotagiri, Fuchun Sun, Hongbo Li, Wenbing Huang, and Zay Maung Maung Aye. Efficient spatio-temporal tactile object recognition with randomized tiling convolutional networks in a hierarchical fusion strategy. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, pp. 3337-3345. AAAI Press, 2016.  
Vlad Ciobanu, Adrian Petrescu, Norman Hendrich, and Jianwei Zhang. Tactile sensor value preprocessing pipeline. In System Theory, Control and Computing (ICSTCC), 2013 17th International Conference, pp. 674-680. IEEE, 2013.  
Marc Deisenroth and Carl E Rasmussen. Pilco: A model-based and data-efficient approach to policy search. In Proceedings of the 28th International Conference on machine learning (ICML-11), pp. 465-472, 2011.  
Alexey Dosovitskiy and Vladlen Koltun. Learning to act by predicting the future. arXiv preprint arXiv:1611.01779, 2016.  
Carlton Downey, Ahmed Hefny, Boyue Li, Byron Boots, and Geoffrey Gordon. Predictive state recurrent neural networks. In Advances in Neural Information Processing Systems, 2017.  
Mark Edmonds, Feng Gao, Xu Xie, Hangxin Liu, Siyuan Qi, Yixin Zhu, Brandon Rothrock, and Song-Chun Zhu. Feeling the force: Integrating force and pose for fluent discovery through imitation learning to open medicine bottles. In International Conference on Intelligent Robots and Systems (IROS), IEEE, 2017.  
Mica R Endsley. Sagat: A methodology for the measurement of situation awareness (nor doc 87-83). Hawthorne, CA: Northrop Corporation, 1987.

Yang Gao, Lisa Anne Hendricks, Katherine J Kuchenbecker, and Trevor Darrell. Deep learning for tactile understanding from visual and haptic data. In Robotics and Automation (ICRA), 2016 IEEE International Conference on, pp. 536-543. IEEE, 2016.  
Nicolas Heess, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne, Yuval Tassa, Tom Erez, Ziyu Wang, Ali Eslami, Martin Riedmiller, et al. Emergence of locomotion behaviours in rich environments. arXiv preprint arXiv:1707.02286, 2017.  
Heni Ben Amor Indranil Sur. Robots that anticipate pain: Anticipating physical perturbations from visual cues through deep predictive models. In IROS, 2017.  
Ashesh Jain, Brian Wojcik, Thorsten Joachims, and Ashutosh Saxena. Learning trajectory preferences for manipulators via iterative improvement. In Advances in neural information processing systems, pp. 575-583, 2013.  
Matthew S Johannes, John D Bigelow, James M Burck, Stuart D Harshbarger, Matthew V Kozlowski, and Thomas Van Doren. An overview of the developmental process for the modular prosthetic limb. Johns Hopkins APL Technical Digest, 30(3):207-216, 2011.  
Maximilian Karl, Justin Bayer, and Patrick van der Smagt. Unsupervised preprocessing for tactile data. arXiv preprint arXiv:1606.07312, 2016.  
Susan J Lederman and Roberta L Klatzky. Hand movements: A window into haptic object recognition. Cognitive psychology, 19(3):342-368, 1987.  
Chang Liu, Fuchun Sun, and Alan Yuille. Haptic object recognition: A recurrent approach.  
Gerald E Loeb. Estimating point of contact, force and torque in a biomimetic tactile sensor with deformable skin. 2013.  
Stefan Escaida Navarro, Nicolas Gorges, Heinz Worn, Julian Schill, Tamim Asfour, and Rüdiger Dillmann. Haptic object recognition for multi-fingered robot hands. In Haptics Symposium (HAPTICS), 2012 IEEE, pp. 497-502. IEEE, 2012.  
Pierre-Yves Oudeyer and Frederic Kaplan. How can we define intrinsic motivation? In Proceedings of the 8th International Conference on Epigenetic Robotics: Modeling Cognitive Development in Robotic Systems, Lund University Cognitive Studies, Lund: LUCS, Brighton. Lund University Cognitive Studies, Lund: LUCS, Brighton, 2008.  
Pierre-Yves Oudeyer and Frederic Kaplan. What is intrinsic motivation? a typology of computational approaches. Frontiers in neurorobotics, 1:6, 2009.  
Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International Conference on Machine Learning (ICML) 2017, 2017.  
Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, and Abhinav Gupta. The curious robot: Learning visual representations via physical interactions. In European Conference on Computer Vision, pp. 3-18. Springer, 2016.  
Pedro Sequeira, Francisco S Melo, and Ana Paiva. Emotion-based intrinsic motivation for reinforcement learning agents. In International Conference on Affective Computing and Intelligent Interaction, pp. 326-336. Springer, 2011.  
Zhe Su, Jeremy A Fishel, Tomonori Yamamoto, and Gerald E Loeb. Use of tactile feedback to control exploratory movements to characterize object compliance. Frontiers in neurorobotics, 6, 2012.  
Zhe Su, Karol Hausman, Yevgen Chebotar, Artem Molchanov, Gerald E Loeb, Gaurav S Sukhatme, and Stefan Schaal. Force estimation and slip detection/classification for grip control using a biomimetic tactile sensor. In Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on, pp. 297-303. IEEE, 2015.  
Jaeyong Sung, J Kenneth Salisbury, and Ashutosh Saxena. Learning to represent haptic feedback for partially-observable tasks. arXiv preprint arXiv:1705.06243, 2017.  
Arun Venkatraman, Nicholas Rhinehart, Wen Sun, Lerrel Pinto, Martial Hebert, Byron Boots, Kris M Kitani, and J Andrew Bagnell. Predictive-state decoders: Encoding the future into recurrent networks. In Advances in Neural Information Processing Systems, 2017.  
Haitian Zheng, Lu Fang, Mengqi Ji, Matti Strese, Yigitcan Ozer, and Eckehard Steinbach. Deep learning for surface material classification using haptic and visual information. IEEE Transactions on Multimedia, 18 (12):2407-2416, 2016.

![](images/aabb3580639febd404e90cbd59f4ebf61832c54b15813fbd09dcfcb6d5d28e51.jpg)  
Figure 6: Each plot shows the validation performance as evaluated on predictions with different horizons. Within each plot we see the curves for models trained using different horizons. These curves show the importance of training for long horizons if we want to make long horizon predictions.

![](images/692e5d5fc74e0f831851a4a6d42decdbd02832d46ed57eb43b5445bed46f36e0.jpg)

![](images/457d53f1698a2a0c4852638d95a727fdb4e3555d5289b0f53acbcc45e82eac8f.jpg)

![](images/fe42571fe0edc8cac87db65b42974c5e598f99d79dda14d83cd37e798e9ad831.jpg)  
Figure 7: A visualization of the model planning to maximize predicted fingertip pressure.
