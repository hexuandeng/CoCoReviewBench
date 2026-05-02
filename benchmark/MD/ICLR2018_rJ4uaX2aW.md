# LARGE BATCH TRAINING OF CONVOLUTIONAL NETWORKS WITH LAYER-WISE ADAPTIVE RATE SCALING

Anonymous authors

Paper under double-blind review

# ABSTRACT

A common way to speed up training of deep convolutional networks is to add computational units. Training is then performed using data-parallel synchronous Stochastic Gradient Descent (SGD) with a mini-batch divided between computational units. With an increase in the number of nodes, the batch size grows. However, training with a large batch often results in lower model accuracy. We argue that the current recipe for large batch training (linear learning rate scaling with warm-up) does not work for many networks, e.g. for Alexnet, GoGLenet,... We propose a more general training algorithm based on Layer-wise Adaptive Rate Scaling (LARS). The key idea of LARS is to stabilize training by keeping the magnitude of update proportional to the norm of weights for each layer. This is done through gradient rescaling per layer. Using LARS, we successfully trained AlexNet and ResNet-50 to a batch size of 16K.

# 1 INTRODUCTION

Training of large Convolutional Neural Networks (CNN) takes a lot of time. The brute-force way to speed up CNN training is to add more computational power (e.g. more GPU nodes) and train network using data-parallel Stochastic Gradient Descent, where each worker receives some chunk of global mini-batch (see e.g. Krizhevsky (2014) or Goyal et al. (2017)). The size of a chunk should be large enough to utilize the computational resources of the worker. So scaling up the number of workers results in the increase of batch size. But using large batch may negatively impact the model accuracy, as was observed in Krizhevsky (2014), Li et al. (2014), Keskar et al. (2016), Hoffer et al. (2017).

Increasing the global batch while keeping the same number of epochs means that you have fewer iterations to update weights. The straight-forward way to compensate for a smaller number of iterations is to do larger steps by increasing the learning rate (LR). For example, Krizhevsky (2014) suggests to linearly scale up LR with batch size. However using a larger LR makes optimization more difficult, and networks may diverge especially during the initial phase. To overcome this difficulty, Goyal et al. (2017) suggested a "learning rate warm-up": training starts with a small LR, which is slowly increased to the target "base" LR. With a LR warm-up and a linear scaling rule, Goyal et al. (2017) successfully trained ResNet-50 [He et al. (2016)] with batch  $\mathrm{B} = 8\mathrm{K}$ , see also [Cho et al. (2017)]. Linear scaling of LR with a warm-up is the "state-of-the-art" recipe for large batch training.

We tried to apply this linear scaling and warm-up scheme to train AlexNet [Krizhevsky et al. (2012)] on ImageNet [Deng et al. (2009)], but scaling stopped after  $\mathrm{B} = 2\mathrm{K}$  since training diverged for large LR-s. For  $\mathrm{B} = 4\mathrm{K}$  the accuracy dropped from the baseline  $57.6\%$  ( $\mathrm{B} = 512$ ) to  $53.1\%$ , and for  $\mathrm{B} = 8\mathrm{K}$  the accuracy decreased to  $44.8\%$ . To enable training with a large LR, we replaced Local Response Normalization layers in AlexNet with Batch Normalization (BN) [Ioffe & Szegedy (2015)]. We will refer to this models AlexNet-BN. BN improved model convergence for large LRs as well as accuracy: for  $\mathrm{B} = 8\mathrm{K}$  the accuracy gap decreased from  $14\%$  to  $2.2\%$ .

To analyze the training stability with large LRs we measured the ratio between the norm of the layer weights and norm of gradients update. We observed that if this ratio is too high, the training becomes unstable. On other hand, if the ratio is too small, then weights don't change fast enough. The layer with largest  $\frac{||\nabla W||}{||W||}$  defines the global limit on the learning rate. Since this ratio varies a lot between different layers, we can speed-up training by using a separate LR for each layer. Thus we propose a novel Layer-wise Adaptive Rate Scaling (LARS) algorithm.

There are two notable differences between LARS and other adaptive algorithms such as ADAM (Kingma & Ba (2014)) or RMSProp (Tieleman & Hinton (2012)): first, LARS uses a separate learning rate for each layer and not for each weight, which leads to better stability. And second, the magnitude of the update is defined with respect to the weight norm for better control of training speed. With LARS we trained AlexNet-BN and ResNet-50 with  $\mathrm{B} = 16\mathrm{K}$  without accuracy loss.

# 2 BACKGROUND

The training of CNN is done using Stochastic Gradient (SG) based methods. At each step  $t$  a minibatch of  $B$  samples  $x_{i}$  is selected from the training set. The gradients of loss function  $\nabla L(x_{i},w)$  are computed for this subset, and networks weights  $w$  are updated based on this stochastic gradient:

$$
w _ {t + 1} = w _ {t} - \lambda \frac {1}{B} \sum_ {i = 1} ^ {B} \nabla L \left(x _ {i}, w _ {t}\right) \tag {1}
$$

The computation of SG can be done in parallel by  $N$  units, where each unit processes a chunk of the mini-batch with  $\frac{B}{N}$  samples. Increasing the mini-batch permits scaling to more nodes without reducing the workload on each unit. However, it was observed that training with a large batch is difficult. To maintain the network accuracy, it is necessary to carefully adjust training hyper-parameters (learning rate, momentum etc).

Krizhevsky (2014) suggested the following rules for training with large batches: when you increase the batch  $B$  by  $k$  times, you should also increase LR by  $k$  times while keeping other hyper-parameters (momentum, weight decay, etc) unchanged. The logic behind linear LR scaling is straight-forward: if you increase  $B$  by  $k$  times while keeping the number of epochs unchanged, you will do  $k$  times fewer steps. So it seems natural to increase the step size by  $k$  times. For example, let's take  $k = 2$ . The weight updates for batch size  $B$  after 2 iterations would be:

$$
w _ {t + 2} = w _ {t} - \lambda * \frac {1}{B} \left(\sum_ {i = 1} ^ {B} \nabla L \left(x _ {i}, w _ {t}\right) + \sum_ {j = B + 1} ^ {2 B} \nabla L \left(x _ {j}, w _ {t + 1}\right)\right) \tag {2}
$$

The weight update for the batch  $B_{2} = 2*B$  with learning rate  $\lambda_{2}$ :

$$
w _ {t + 1} = w _ {t} - \lambda_ {2} * \frac {1}{2 * B} \sum_ {i = 1} ^ {2 B} \nabla L \left(x _ {i}, w _ {t}\right) \tag {3}
$$

will be similar if you take  $\lambda_{2} = 2*\lambda$ , assuming that  $\nabla L(x_{j},w_{t + 1})\approx \nabla L(x_{j},w_{t})$

Using the "linear LR scaling" Krizhevsky (2014) trained AlexNet with batch  $\mathrm{B} = 1\mathrm{K}$  with minor  $(\approx 1\%)$  accuracy loss. The scaling of AlexNet above  $2\mathrm{K}$  is difficult, since the training diverges for larger LRs. It was observed that linear scaling works much better for networks with Batch Normalization (e.g. Codreanu et al. (2017)). For example Chen et al. (2016) trained the Inception model with batch  $\mathrm{B} = 6400$ , and Li (2017) trained ResNet-152 for  $\mathrm{B} = 5\mathrm{K}$ .

The main obstacle for scaling up batch is the instability of training with high LR. Hoffer et al. (2017) tried to use less aggressive "square root scaling" of LR with special form of Batch Normalization ("Ghost Batch Normalization") to train AlexNet with  $\mathrm{B} = 8\mathrm{K}$ , but still the accuracy (53.93%) was much worse than baseline 58%. To overcome the instability during initial phase, Goyal et al. (2017) proposed to use LR warm-up: training starts with small LR, and then LR is gradually increased to the target. After the warm-up period (usually a few epochs), you switch to the regular LR policy ("multi-steps", polynomial decay etc). Using LR warm-up and linear scaling Goyal et al. (2017) trained ResNet-50 with batch  $\mathrm{B} = 8\mathrm{K}$  without loss in accuracy. These recipes constitute the current state-of-the-art for large batch training, and we used them as the starting point of our experiments.

Another problem related to large batch training is so called "generalization gap", observed by Keskar et al. (2016). They came to conclusion that "the lack of generalization ability is due to the fact that large-batch methods tend to converge to sharp minimizers of the training function." They tried a few methods to improve the generalization with data augmentation and warm-starting with small batch, but they did not find a working solution.

# 3 ANALYSIS OF ALEXNET TRAINING WITH LARGE BATCH

We used BVLC<sup>1</sup> AlexNet with batch B=512 as baseline. Model was trained using SGD with momentum 0.9 with initial LR=0.02 and the polynomial (power=2) decay LR policy for 100 epochs. The baseline accuracy is  $58.8\%$  (averaged over last 5 epochs). Next we tried to train AlexNet with B=4K by using larger LR. In our experiments we changed the base LR from 0.01 to 0.08, but training diverged with LR > 0.06 even with warm-up<sup>2</sup>. The best accuracy for B=4K is  $53.1\%$ , achieved for LR=0.05. For B=8K we couldn't scale-up LR either, and the best accuracy is  $44.8\%$ , achieved for LR=0.03 (see Table 1(a)).

To stabilize the initial training phase we replaced Local Response Normalization layers with Batch Normalization (BN). We will refer to this model as AlexNet-BN.  ${}^{3}$  . AlexNet-BN model was trained using SGD with momentum=0.9,weight decay=0.0005 for 128 epochs. We used polynomial (power 2) decay LR policy with base LR=0.02. The baseline accuracy for B=512 is 60.2%. With BN we could use large LR-s even without warm-up. For B=4K the best accuracy 58.9% was achieved for LR=0.18,and for B=8K the best accuracy 58% was achieved for LR=0.3. We also observed that BN significantly widens the range of LRs with good accuracy.

Table 1: AlexNet and AlexNet-BN trained for 100 epcohs:  $\mathrm{B} = 4\mathrm{\;K}$  and  $\mathrm{B} = 8\mathrm{\;K}$  . BatchNorm makes it possible to use larger learning rates, but training with large batch still results in lower accuracy.  
(a) AlexNet (warm-up 2.5 epochs)  

<table><tr><td>Batch</td><td>Base LR</td><td>accuracy,%</td></tr><tr><td>512</td><td>0.02</td><td>58.8</td></tr><tr><td>4096</td><td>0.04</td><td>53.0</td></tr><tr><td>4096</td><td>0.05</td><td>53.1</td></tr><tr><td>4096</td><td>0.06</td><td>51.6</td></tr><tr><td>4096</td><td>0.07</td><td>0.1</td></tr><tr><td>8192</td><td>0.02</td><td>29.8</td></tr><tr><td>8192</td><td>0.03</td><td>44.8</td></tr><tr><td>8192</td><td>0.04</td><td>43.1</td></tr><tr><td>8192</td><td>0.05</td><td>0.1</td></tr></table>

(b) AlexNet-BN (no warm-up)  

<table><tr><td>Batch</td><td>Base LR</td><td>accuracy,%</td></tr><tr><td>512</td><td>0.02</td><td>60.2</td></tr><tr><td>4096</td><td>0.16</td><td>58.1</td></tr><tr><td>4096</td><td>0.18</td><td>58.9</td></tr><tr><td>4096</td><td>0.21</td><td>58.5</td></tr><tr><td>4096</td><td>0.30</td><td>57.1</td></tr><tr><td>8192</td><td>0.23</td><td>57.6</td></tr><tr><td>8192</td><td>0.30</td><td>58.0</td></tr><tr><td>8192</td><td>0.32</td><td>57.7</td></tr><tr><td>8192</td><td>0.41</td><td>56.5</td></tr></table>

Still there is a  $2.2\%$  accuracy loss for  $\mathrm{B} = 8\mathrm{K}$ . To check if it is related to the "generalization gap" (Keskar et al. (2016)), we looked at the loss gap between training and testing (see Fig. 1). We did not find the significant difference in the loss gap between  $\mathrm{B} = 512$  and  $\mathrm{B} = 8\mathrm{K}$ . We conclude that in this case the accuracy loss was mostly caused by the slow training and was not related to a generalization gap.

![](images/45cc349522664f4c8d1f2823fedf114003c5dde16b179a01b2fabee7c6498a01.jpg)  
Figure 1: AlexNet-BN: the generalization gap between training and testing loss is practically the same for small  $(\mathrm{B} = 256)$  and large  $(\mathrm{B} = 8\mathrm{K})$  batches.

# 4 LAYER-WISE ADAPTIVE RATE SCALING (LARS)

The standard SGD uses the same LR  $\lambda$  for all layers:  $w_{t + 1} = w_t - \lambda \nabla L(w_t)$ . When  $\lambda$  is large, the update  $||\lambda*\nabla L(w_t)||$  can become larger than  $||w||$ , and this can cause the divergence. This makes the initial phase of training highly sensitive to the weight initialization and to initial LR. We found that the ratio of the L2-norm of weights and gradients  $||w|| / ||\nabla L(w_t)||$  varies significantly between weights and biases, and between different layers. For example, let's take AlexNet after one iteration (Table 2, "\*.w" means layer weights, and "\*.b" - biases). The ratio  $||w|| / ||\nabla L(w)||$  for the 1st convolutional layer ("conv1.w") is 5.76, and for the last fully connected layer ("fc6.w") - 1345. The ratio is high during the initial phase, and it is rapidly decreasing after few epochs (see Figure 2).

Table 2: AlexNet: The ratio of norm of weights to norm of gradients for different layers at 1st iteration. The maximum learning rate is limited by the layer with smallest ratio.  

<table><tr><td>Layer</td><td>conv1.b</td><td>conv1.w</td><td>conv2.b</td><td>conv2.w</td><td>conv3.b</td><td>conv3.w</td><td>conv4.b</td><td>conv4.w</td></tr><tr><td>||w||</td><td>1.86</td><td>0.098</td><td>5.546</td><td>0.16</td><td>9.40</td><td>0.196</td><td>8.15</td><td>0.196</td></tr><tr><td>||∇L(w)||</td><td>0.22</td><td>0.017</td><td>0.165</td><td>0.002</td><td>0.135</td><td>0.0015</td><td>0.109</td><td>0.0013</td></tr><tr><td>||w||/||∇L(w)||</td><td>8.48</td><td>5.76</td><td>33.6</td><td>83.5</td><td>69.9</td><td>127</td><td>74.6</td><td>148</td></tr><tr><td>Layer</td><td>conv5.b</td><td>conv5.w</td><td>fc6.b</td><td>fc6.w</td><td>fc7.b</td><td>fc7.w</td><td>fc8.b</td><td>fc8.w</td></tr><tr><td>||w||</td><td>6.65</td><td>0.16</td><td>30.7</td><td>6.4</td><td>20.5</td><td>6.4</td><td>20.2</td><td>0.316</td></tr><tr><td>||∇L(w)||</td><td>0.09</td><td>0.0002</td><td>0.26</td><td>0.005</td><td>0.30</td><td>0.013</td><td>0.22</td><td>0.016</td></tr><tr><td>||w||/||∇L(w)||</td><td>73.6</td><td>69</td><td>117</td><td>1345</td><td>68</td><td>489</td><td>93</td><td>19</td></tr></table>

If LR is large comparing to the ratio for some layer, then training may become unstable. The LR "warm-up" attempts to overcome this difficulty by starting from small LR, which can be safely used for all layers, and then slowly increasing it until weights will grow up enough to use larger LRs.

We would like to use different approach. We want to make sure that weights update is small comparing to the norm of weights to stabilize training

$$
\left| \left| \triangle w _ {t} ^ {l} \right| \right| <   \eta * \left| \left| w _ {t} ^ {l} \right| \right| \tag {4}
$$

where  $\eta < 1$  control the magnitude of update with respect to weights. The coefficient  $\eta$  defines how much we "trust" that the value of stochastic gradient  $\nabla L(w_t^l)$  is close to true gradient. The  $\eta$  depends on the batch size. "Trust"  $\eta$  is monotonically increasing with batch size: for example for Alexnet for batch  $B = 1K$  the optimal  $\eta = 0.0002$ , for batch  $B = 4K - \eta = 0.005$ , and for  $B = 8K - \eta = 0.008$ . We implemented this idea through defining local LR  $\lambda^l$  for each layer  $l$ :

$$
\triangle w _ {t} ^ {l} = \gamma * \lambda^ {l} * \nabla L \left(w _ {t} ^ {l}\right) \tag {5}
$$

where  $\gamma$  defines a global LR policy (e.g. steps, or exponential decay), and local LR  $\lambda^l$  is defined for each layer through "trust" coefficient  $\eta < 1^4$ :

$$
\lambda^ {l} = \eta \times \frac {\left| \left| w ^ {l} \right| \right|}{\left| \left| \nabla L \left(w ^ {l}\right) \right| \right|} \tag {6}
$$

Note that now the magnitude of the update for each layer doesn't depend on the magnitude of the gradient anymore, so it helps to partially eliminate vanishing and exploding gradient problems. The network training for SGD with LARS are summarized in the Algorithm  $1^5$ .

LARS was designed to solve the optimization difficulties, and it does not replace standard regularization methods (weight decay, batch norm, or data augmentation). But we found that with LARS we can use larger weight decay, since LARS automatically controls the norm of layer weights:

$$
\left| \left| w _ {t} \right| \right| <   \left| \left| w _ {0} \right| \right| * e ^ {\eta * \int_ {0} ^ {t} \gamma (\tau) d \tau} <   \left| \left| w _ {0} \right| \right| * e ^ {\frac {\eta * N * S}{B}} \tag {7}
$$

where  $B$  is min-batch size,  $N$  - number of training epochs, and  $S$  - number of samples in the training set. Here we assumed that global rate policy starts from 1 and decrease during training over training interval  $[0, N * S / B]$ .

Algorithm 1 SGD with LARS. Example with weight decay, momentum and polynomial LR decay.  
Parameters: base LR  $\gamma_0$  momentum  $m$  weight decay  $\beta$  LARS coefficient  $\eta$  number of steps  $T$  Init:  $t = 0,v = 0$  . Init weight  $w_{0}^{l}$  for each layer  $l$    
while  $t <   T$  for each layer  $l$  do  $g_{t}^{l}\gets \nabla L(w_{t}^{l})$  (obtain a stochastic gradient for the current mini-batch)  $\gamma_t\gets \gamma_0*\left(1 - \frac{t}{T}\right)^2$  (compute the global learning rate)  $\lambda^l\gets \eta *\frac{||w_t^l||}{||g_t^l|| + \beta||w_t^l||}$  (compute the local LR  $\lambda^l$ $v_{t + 1}^{l}\gets mv_{t}^{l} + \gamma_{t}*\lambda^{l}*(g_{t}^{l} + \beta w_{t}^{l})$  (update the momentum)  $w_{t + 1}^{l}\gets w_{t}^{l} - v_{t + 1}^{l}$  (update the weights)   
end while

# 5 TRAINING WITH LARS

We re-trained AlexNet and AlexNet-BN with LARS for batches up to  $32\mathrm{K}^6$ . To emulate large batches  $(\mathrm{B} = 16\mathrm{K}$  and  $32\mathrm{K}$ ) we used caffe parameter iter_size<sup>7</sup> to partition batch into smaller chunks. Both Alexnet and Alexnet-BN have been trained for 100 epochs using SGD with momentum=0.9, weight decay=0.0005. We used global learning rate with polynomial decay  $(\mathrm{p} = 2)$  policy and with warm-up (for Alexnet warm-up was 2 epochs, and for Alexnet-BN 5 epochs). We fixed LARS trust  $\eta = 0.001$  for all batch sizes, and scaled up initial LR as shown in the Table. 3. But one can instead fix global base LR to 1, and scale up the trust coefficient. For  $\mathrm{B} = 8\mathrm{K}$  the accuracy of both networks matched the baseline  $\mathrm{B} = 512$  (see Figure 2). AlexNet-BN trained with  $\mathrm{B} = 16\mathrm{K}$  lost  $0.9\%$  in accuracy, and trained with  $\mathrm{B} = 32\mathrm{K}$  lost  $2.6\%$ .

![](images/115d84701f9a609729b05b409d6ed03b7a2dd276e94a9271edb7ac29f88dfe27.jpg)  
(a) Training without LARS  
Figure 2: Training with LARS: AlexNet-BN with  $\mathrm{B} = 8\mathrm{K}$

![](images/1d0bb48434720472c8c2ab8a3f8d54b2157d0ad487a98bdf0a12c00d835332d7.jpg)  
(b) Training with LARS

We observed that the optimal LR do incresae with batch size, but not in linear or square root proportion as was suggested in theory: there is a relatively wide interval of base LRs which gives the "best" accuracy. For AlexNet-BN with  $\mathrm{B} = 16\mathrm{K}$  for example, all LRs from [13;22] interval give almost the same accuracy  $\approx 59.3$ .

Next we trained ResNet-50, v.1 [He et al. (2016)] with LARS. First we used minimal data augmentation: during training images are scaled to 256x256, and then random 224x224 crop with horizontal flip is taken. All training was done with SGD with momentum 0.9 and weight decay=0.0005 for 100 epochs. We used polynomial decay (power=2) LR policy with LARS and warm-up (5-12 epochs).

Table 3: Alexnet and Alexnet-BN training with LARS: the top-1 accuracy as function of batch size  
(a) AlexNet with LARS  

<table><tr><td>Batch</td><td>LR</td><td>accuracy,%</td></tr><tr><td>512</td><td>2</td><td>58.7</td></tr><tr><td>4K</td><td>10</td><td>58.5</td></tr><tr><td>8K</td><td>10</td><td>58.2</td></tr><tr><td>16K</td><td>14</td><td>55.0</td></tr><tr><td>32K</td><td>14</td><td>46.9</td></tr></table>

(b) AlexNet-BN with LARS  

<table><tr><td>Batch</td><td>LR</td><td>accuracy,%</td></tr><tr><td>512</td><td>2</td><td>60.2</td></tr><tr><td>4K</td><td>10</td><td>60.4</td></tr><tr><td>8K</td><td>14</td><td>60.1</td></tr><tr><td>16K</td><td>23</td><td>59.3</td></tr><tr><td>32K</td><td>22</td><td>57.8</td></tr></table>

![](images/3e8ee8d371e20dd0d64c8707937b1a0313cc23a951645cb8ac0eacbf335583cd.jpg)  
Figure 3: AlexNet-BN,  $\mathrm{B} = 16\mathrm{K}$  and  $32\mathrm{k}$ : Accuracy as function of LR

During testing we used one model and 1 central crop. The baseline  $(\mathrm{B} = 256)$  accuracy is  $73.8\%$  for minimal augmentation. To match the state-of-the-art accuracy from Goyal et al. (2017) and Cho et al. (2017) we used the second setup with an extended augmentation with variable image scale and aspect ratio similar to [Szegedy et al. (2015)]. The baseline top-1 accuracy for this setup is  $75.4\%$ .

Table 4: ResNet50 with LARS: top-1 accuracy as function of batch size for training with minimal and extended data augmentation  

<table><tr><td>Batch</td><td>γ</td><td>warm-up</td><td>min aug, accuracy,%</td><td>max aug, accuracy,%</td></tr><tr><td>256</td><td>4</td><td>N/A</td><td>73.8</td><td>75.8</td></tr><tr><td>1K</td><td>9</td><td>5</td><td>73.3</td><td>75.4</td></tr><tr><td>8K</td><td>30</td><td>5</td><td>73.5</td><td>75.2</td></tr><tr><td>16K</td><td>33</td><td>12</td><td>72.9</td><td>74.4</td></tr><tr><td>32K</td><td>40</td><td>12</td><td>72.5</td><td>72.5</td></tr></table>

The accuracy with  $\mathrm{B} = 16\mathrm{K}$  is  $0.7 - 1.4\%$  less than baseline. This gap is related to smaller number of steps. We will show in the next section that one can recover the accuracy by training for more epochs.

# 6 LARGE BATCH TRAINING: ACCURACY VS NUMBER OF STEPS

When batch becomes large (32K), even models trained with LARS and large LR don't reach the baseline accuracy. One way to recover the lost accuracy is to train longer (see [Hoffer et al. (2017)]). Note that when batch becomes large, the number of iteration decreases. So one way to try to improve the accuracy, would be train longer. For example for Alexnet and Alexnet-BN with  $\mathrm{B} = 16\mathrm{K}$ , when we double the number of iterations from 7800 (100 epochs) to 15600 (200 epochs) the accuracy improved by  $2 - 3\%$  (see Table 5). The same effect we observed for Resnet-50: training for additional 100 epochs recovered the top-1 accuracy to the baseline  $75.5\%$ .

![](images/53e91155aa47261fbf61d32c9878563e1def7921a0488c034c839630c24a2419.jpg)  
Figure 4: Scaling ResNet-50 (no data augmentation) up to  $\mathrm{B} = 32\mathrm{K}$  with LARS.

![](images/a20982a391cdeaadee00196eb4f2a0821a62b0e4cb0ee148e809ab82a66c2b58.jpg)

Table 5: Accuracy vs Training duration

(a) AlexNet,  $\mathrm{B} = 16\mathrm{k}$  

<table><tr><td>Epochs</td><td>accuracy,%</td></tr><tr><td>100</td><td>55.0</td></tr><tr><td>125</td><td>55.9</td></tr><tr><td>150</td><td>56.7</td></tr><tr><td>175</td><td>57.3</td></tr><tr><td>200</td><td>58.2</td></tr></table>

(b) AlexNet-BN, B=32K  

<table><tr><td>Epochs</td><td>accuracy,%</td></tr><tr><td>100</td><td>57.8</td></tr><tr><td>125</td><td>59.2</td></tr><tr><td>150</td><td>59.5</td></tr><tr><td>175</td><td>59.5</td></tr><tr><td>200</td><td>59.9</td></tr></table>

(c) ResNet-50, B=16K  

<table><tr><td>Epochs</td><td>accuracy,%</td></tr><tr><td>100</td><td>74.4</td></tr><tr><td>125</td><td>74.1</td></tr><tr><td>150</td><td>74.6</td></tr><tr><td>175</td><td>74.8</td></tr><tr><td>200</td><td>75.5</td></tr></table>

In general we found that we have to increase the training duration to keep the accuracy. Consider for example Googlenet [Szegedy et al. (2015)]. As a baseline we trained BVLC googlenet with batch=256 for 100 epoch. The top-1 accuracy of this model is  $69.2\%$ . Googlenet is deep, so in original paper authors used auxiliary losses to accelerate SGD. We used LARS to solve optimization difficulties so we don't need these auxiliary losses. The original model also has no Batch Normalization, so we used data augmentation for better regularization. The baseline accuracy for B=256 is  $70.3\%$  with extended augmentation and LARS. We found that Googlenet is very difficult to train with large batch even with LARS: we needed both large number of epoch and longer ramp-up to scale learning rate up (see Table 6).

Table 6: Googlenet training with LARS  

<table><tr><td>Batch</td><td>γ</td><td>epochs</td><td>warm-up</td><td>accuracy,%</td></tr><tr><td>256 (no LARS)</td><td>0.02</td><td>100</td><td>-</td><td>69.2</td></tr><tr><td>256 (LARS)</td><td>4</td><td>100</td><td>5</td><td>70.3</td></tr><tr><td>1K</td><td>6</td><td>100</td><td>10</td><td>69.7</td></tr><tr><td>2K</td><td>7</td><td>150</td><td>25</td><td>71.1</td></tr><tr><td>4K</td><td>10</td><td>200</td><td>30</td><td>69.9</td></tr><tr><td>8K</td><td>10</td><td>250</td><td>60</td><td>69.0</td></tr><tr><td>16K</td><td>10</td><td>350</td><td>110</td><td>67.2</td></tr></table>

# 7 CONCLUSION

Large batch is a key for scaling up training of convolutional networks. The existing approach for large-batch training, based on using large learning rates, leads to divergence, especially during the

initial phase, even with learning rate warm-up. To solve these difficulties we proposed the new optimization algorithm, which adapts the learning rate for each layer (LARS) proportional to the ratio between the norm of weights and norm of gradients. With LARS the magnitude of the update for each layer doesn't depend on the magnitude of the gradient anymore, so it helps with vanishing and exploding gradients. But even with LARS and warm-up we couldn't increase LR farther for very large batches, and to keep the accuracy we have to increase the number of epochs and use extensive data augmentation to prevent over-fitting.

# REFERENCES

Jianmin Chen, Rajat Monga, Samy Bengio, and Rafal Jozefowicz. Revisiting distributed synchronous sgd. arXiv preprint arXiv:1604.00981, 2016.  
Minsik Cho, Ulrich Finkler, Sameer Kumar, David Kung, Vaibhav Saxena, and Dheeraj Sreedhar. Powerai ddl. arXiv preprint arXiv:1708.02188, 2017.  
Valeriu Codreanu, Damian Podareanu, and Vikram Saletore. Blog: Achieving deep learning training in less than 40 minutes on imagenet-1k with scale-out intel® xeon™/xeon phi™ architectures. blog https://blog.surf.nl/en/imagenet-1k-training-on-intel-xeon-phi-in-less-than-40-minutes/, 2017.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pp. 248-255. IEEE, 2009.  
Priya Goyal, Piotr Dár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778, 2016.  
Elad Hoffer, Itay Hubara, and Daniel Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. arXiv preprint arXiv:1705.08741, 2017.  
Sergey Ioffe and Christian Szegedy. Batch normalization: accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint arXiv:1609.04836, 2016.  
Diederik Kingma and Jimmy Ba. Adam: a method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Alex Krizhevsky. One weird trick for parallelizing convolutional neural networks. arXiv preprint arXiv:1404.5997, 2014.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems NIPS-25, pp. 1097-1105, 2012.  
Jean Lafond, Nicolas Vasilache, and Léon Bottou. Diagonal rescaling for neural networks. arXiv preprint arXiv:1705.09319v1, 2017.  
Mu Li. Scaling Distributed Machine Learning with System and Algorithm Co-design. PhD thesis, CMU, 2017.  
Mu Li, Tong Zhang, Yuqiang Chen, and Alexander J Smola. Efficient mini-batch training for stochastic optimization. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 661-670. ACM, 2014.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Computer Vision and Pattern Recognition (CVPR), 2015. URL http://arxiv.org/abs/1409.4842.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop, course: Neural networks for machine learning. University of Toronto, Tech. Rep, 2012.