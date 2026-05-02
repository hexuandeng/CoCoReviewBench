# TOPICRNN: A RECURRENT NEURAL NETWORK WITH LONG-RANGE SEMANTIC DEPENDENCY

Adji B. Dieng *

Columbia University

abd2141@columbia.edu

Chong Wang

Deep Learning Technology Center

Microsoft Research

chowang@microsoft.com

Jianfeng Gao

Deep Learning Technology Center

Microsoft Research

jfgao@microsoft.com

John Paisley

Columbia University

jpaisley@columbia.edu

# ABSTRACT

In this paper, we propose TopicRNN, a recurrent neural network (RNN)-based language model designed to directly capture the global semantic meaning relating words in a document via latent topics. Because of their sequential nature, RNNs are good at capturing the local structure of a word sequence – both semantic and syntactic – but might face difficulty remembering long-range dependencies. Intuitively, these long-range dependencies are of semantic nature. In contrast, latent topic models are able to capture the global semantic structure of a document but do not account for word ordering. The proposed TopicRNN model integrates the merits of RNNs and latent topic models: it captures local (syntactic) dependencies using an RNN and global (semantic) dependencies using latent topics. Unlike previous work on contextual RNN language modeling, our model is learned end-to-end. Empirical results on word prediction show that TopicRNN outperforms existing contextual RNN baselines. In addition, TopicRNN can be used as an unsupervised feature extractor for documents. We do this for sentiment analysis on the IMDB movie review dataset and report an error rate of  $6.28\%$ . This is comparable to the state-of-the-art  $5.91\%$  resulting from a semi-supervised approach. Finally, TopicRNN also yields sensible topics, making it a useful alternative to document models such as latent Dirichlet allocation.

# 1 INTRODUCTION

When reading a document, short or long, humans have a mechanism that somehow allows them to remember the gist of what they have read so far. Consider the following example:

"The U.S.presidential race isn't only drawing attention and controversy in the United States - it's being closely watched across the globe. But what does the rest of the world think about a campaign that has already thrown up one surprise after another? CNN asked 10 journalists for their take on the race so far, and what their country might be hoping for in America's next —"

The missing word in the text above is easily predicted by any human to be either President or Commander in Chief or their synonyms. There have been various language models – from simple  $n$ -grams to the most recent RNN-based language models – that aim to solve this problem of predicting correctly the subsequent word in an observed sequence of words.

A good language model should capture at least two important properties of natural language. The first one is correct syntax. In order to do prediction that enjoys this property, we often only need to consider a few preceding words. Therefore, correct syntax is more of a local property. Word order matters in this case. The second property is the semantic coherence of the prediction. To achieve

this, we often need to consider many preceding words to understand the global semantic meaning of the sentence or document. The ordering of the words usually matters much less in this case.

Because they only consider a fixed-size context window of preceding words, traditional  $n$ -gram and neural probabilistic language models (Bengio et al., 2003) have difficulties in capturing global semantic information. To overcome this, RNN-based language models (Mikolov et al., 2010; 2011) use hidden states to "remember" the history of a word sequence. However, none of these approaches explicitly model the two main properties of language mentioned above, correct syntax and semantic coherence. Previous work by Chelba and Jelinek (2000) and Gao et al. (2004) exploit syntactic or semantic parsers to capture long-range dependencies in language.

In this paper, we propose TopicRNN, a RNN-based language model that is designed to directly capture long-range semantic dependencies via latent topics. These topics provide context to the RNN. Contextual RNNs have received a lot of attention (Mikolov and Zweig, 2012; Mikolov et al., 2014; Ji et al., 2015; Lin et al., 2015; Ji et al., 2016; Ghosh et al., 2016). However, the models closest to ours are the contextual RNN model proposed by Mikolov and Zweig (2012) and its most recent extension to the long-short term memory (LSTM) architecture (Ghosh et al., 2016). These models use pre-trained topic model features as an additional input to the hidden states and/or the output of the RNN. In contrast, TopicRNN does not require pre-trained topic model features and can be learned in an end-to-end fashion. We introduce an automatic way for handling stop words that topic models usually have difficulty dealing with. Under a comparable model size set up, TopicRNN achieves better perplexity scores than the contextual RNN model of Mikolov and Zweig (2012) on the Penn TreeBank dataset<sup>1</sup>. Moreover, TopicRNN can be used as an unsupervised feature extractor for downstream applications. For example, we derive document features of the IMDB movie review dataset using TopicRNN for sentiment classification. We reported an error rate of  $6.28\%$ . This is close to the state-of-the-art  $5.91\%$  (Miyato et al., 2016) despite that we do not use the labels and adversarial training in the feature extraction stage.

The remainder of the paper is organized as follows: Section 2 provides background on RNN-based language models and probabilistic topic models. Section 3 describes the TopicRNN network architecture, its generative process and how to perform inference for it. Section 4 presents per-word perplexity results on the Penn TreeBank dataset and the classification error rate on the IMDB 100K dataset. Finally, we conclude and provide future research directions in Section 5.

# 2 BACKGROUND

We present the background necessary for building the TopicRNN model. We first review RNN-based language modeling, followed by a discussion on the construction of latent topic models.

# 2.1 RECURRENT NEURAL NETWORK-BASED LANGUAGE MODELS

Language modeling is fundamental to many applications. Examples include speech recognition and machine translation. A language model is a probability distribution over a sequence of words in a predefined vocabulary. More formally, let  $V$  be a vocabulary set and  $y_{1},\ldots ,y_{T}$  a sequence of  $T$  words with each  $y_{t}\in V$ . A language model measures the likelihood of a sequence through a joint probability distribution,

$$
p \left(y _ {1}, \dots , y _ {T}\right) = p \left(y _ {1}\right) \prod_ {t = 2} ^ {T} p \left(y _ {t} \mid y _ {1: t - 1}\right).
$$

Traditional  $n$ -gram and feed-forward neural network language models (Bengio et al., 2003) typically make Markov assumptions about the sequential dependencies between words, where the chain rule shown above limits conditioning to a fixed-size context window.

RNN-based language models (Mikolov et al., 2011) sidestep this Markov assumption by defining the conditional probability of each word  $y_{t}$  given all the previous words  $y_{1:t-1}$  through a hidden

state  $h_t$  (typically via a softmax function):

$$
p \left(y _ {t} \mid y _ {1: t - 1}\right) \triangleq p \left(y _ {t} \mid h _ {t}\right),
$$

$$
h _ {t} = f \left(h _ {t - 1}, x _ {t}\right).
$$

The function  $f(\cdot)$  can either be a standard RNN cell or a more complex cell such as GRU (Cho et al., 2014) or LSTM (Hochreiter and Schmidhuber, 1997). The input and target words are related via the relation  $x_{t} \equiv y_{t - 1}$ . These RNN-based language models have been quite successful (Mikolov et al., 2011; Chelba et al., 2013; Jozefowicz et al., 2016).

While in principle RNN-based models can "remember" arbitrarily long histories if provided enough capacity, in practice such large-scale neural networks can easily encounter difficulties during optimization (Bengio et al., 1994; Pascanu et al., 2013; Sutskever, 2013) or overfitting issues (Srivastava et al., 2014). Finding better ways to model long-range dependencies in language modeling is therefore an open research challenge. As motivated in the introduction, much of the long-range dependency in language comes from semantic coherence, not from syntactic structure which is more of a local phenomenon. Therefore, models that can capture long-range semantic dependencies in language are complementary to RNNs. In the following section, we describe a family of such models called probabilistic topic models.

# 2.2 PROBABILISTIC TOPIC MODELS

Probabilistic topic models are a family of models that can be used to capture global semantic coherency (Blei and Lafferty, 2009). They provide a powerful tool for summarizing, organizing, and navigating document collections. One basic goal of such models is to find groups of words that tend to co-occur together in the same document. These groups of words are called topics and represent a probability distribution that puts most of its mass on this subset of the vocabulary. Documents are then represented as mixtures over these latent topics. Through posterior inference, the learned topics capture the semantic coherence of the words they cluster together (Mimno et al., 2011).

The simplest topic model is latent Dirichlet allocation (LDA) (Blei et al., 2003). It assumes  $K$  underlying topics  $\beta = \{\beta_{1},\dots,\beta_{K}\}$ , each of which is a distribution over a fixed vocabulary. The generative process of LDA is as follows:

First generate the  $K$  topics,  $\beta_{k}\sim_{i id}$  Dirichlet(τ). Then for each document containing words  $y_{1:T}$ , independently generate document-level variables and data:

1. Draw a document-specific topic proportion vector  $\theta \sim$  Dirichlet  $(\alpha)$  
2. For the  $t$ th word in the document,

(a) Draw topic assignment  $z_{t} \sim \mathrm{Discrete}(\theta)$ .  
(b) Draw word  $y_{t}\sim \mathrm{Discrete}(\beta_{z_{t}})$

Marginalizing each  $z_{t}$ , we obtain the probability of  $y_{1:T}$  via a matrix factorization followed by an integration over the latent variable  $\theta$ ,

$$
p \left(y _ {1: T} \mid \beta\right) = \int p (\theta) \prod_ {t = 1} ^ {T} \sum_ {z _ {t}} p \left(z _ {t} \mid \theta\right) p \left(y _ {t} \mid z _ {t}, \beta\right) \mathrm {d} \theta = \int p (\theta) \prod_ {t = 1} ^ {T} \left(\beta \theta\right) _ {y _ {t}} \mathrm {d} \theta . \tag {1}
$$

In LDA the prior distribution on the topic proportions is a Dirichlet distribution; it can be replaced by many other distributions. For example, the correlated topic model (Blei and Lafferty, 2006) uses a log-normal distribution. Most topic models are "bag of words" models in that word order is ignored. This makes it easier for topic models to capture global semantic information. However, this is also one of the reasons why topic models do not perform well on general-purpose language modeling applications such as word prediction. While bi-gram topic models have been proposed (Wallach, 2006), higher order models quickly become intractable.

Another issue encountered by topic models is that they do not model stop words well. This is because stop words usually do not carry semantic meaning; their appearance is mainly to make the sentence more readable according to the grammar of the language. They also appear frequently in

![](images/a44403d95b8d88a4b02da21246c195821532165415405107eed42417ee4dc272.jpg)  
(a)

![](images/864a6cc3d0470b98a439f30db4d6aa13a4f3babaf1ceadc6c24f49bca3f33710.jpg)  
(b)  
Figure 1: (a) The unrolled TopicRNN architecture:  $x_{1}, \ldots, x_{6}$  are words in the document,  $h_{t}$  is the state of the RNN at time step  $t$ ,  $x_{i} \equiv y_{i-1}$ ,  $l_{1}, \ldots, l_{6}$  are stop word indicators, and  $\theta$  is the latent representation of the input document and is unshaded by convention. (b) The TopicRNN model architecture in its compact form:  $l$  is a binary vector that indicates whether each word in the input document is a stop word or not. Here red indicates stop words and blue indicates content words.

almost every document and can co-occur with almost any word $^2$ . In practice, these stop words are chosen using tfidf (Blei and Lafferty, 2009).

# 3 THE TOPICRNN MODEL

We next describe the proposed TopicRNN model. In TopicRNN, latent topic models are used to capture global semantic dependencies so that the RNN can focus its modeling capacity on the local dynamics of the sequences. With this joint modeling, we hope to achieve better overall performance on downstream applications.

The model. TopicRNN is a generative model. For a document containing the words  $y_{1:T}$

1. Draw a topic vector $^3$ $\theta \sim N(0, I)$ .  
2. Given word  $y_{1:t-1}$ , for the  $t$ th word  $y_t$  in the document,

(a) Compute hidden state  $h_t = f_W(x_t, h_{t-1})$ , where we let  $x_t \triangleq y_{t-1}$ .  
(b) Draw stop word indicator  $l_{t}\sim$  Bernoulli  $(\sigma (\Gamma^{\top}h_{t}))$  , with  $\sigma$  the sigmoid function.  
(c) Draw word  $y_{t}\sim p(y_{t}|h_{t},\theta ,l_{t},B)$  , where

$$
p (y _ {t} = i | h _ {t}, \theta , l _ {t}, B) \propto \exp \left(v _ {i} ^ {\top} h _ {t} + (1 - l _ {t}) b _ {i} ^ {\top} \theta\right).
$$

The stop word indicator  $l_{t}$  controls how the topic vector  $\theta$  affects the output. If  $l_{t} = 1$  (indicating  $y_{t}$  is a stop word), the topic vector  $\theta$  has no contribution to the output. Otherwise, we add a bias to favor those words that are more likely to appear when mixing with  $\theta$ , as measured by the dot product between  $\theta$  and the latent word vector  $b_{i}$  for the  $i$ th vocabulary word. As we can see, the long-range semantic information captured by  $\theta$  directly affects the output through an additive procedure. Unlike Mikolov and Zweig (2012), the contextual information is not passed to the hidden layer of the RNN. The main reason behind our choice of using the topic vector as bias instead of passing it into the hidden states of the RNN is because it enables us to have a clear separation of the contributions of global semantics and those of local dynamics. The global semantics come from the topics which are meaningful when stop words are excluded. However these stop words are needed for the local dynamics of the language model. We hence achieve this separation of global vs local via a binary decision model for the stop words. It is unclear how to achieve this if we pass the topics to the

hidden states of the RNN. This is because the hidden states of the RNN will account for all words (including stop words) whereas the topics exclude stop words.

We show the unrolled graphical representation of TopicRNN in Figure 1(a). We denote all model parameters as  $\Theta = \{\Gamma, V, B, W, W_c\}$  (see Appendix A.1 for more details). Parameter  $W_{c}$  is for the inference network, which we will introduce below. The observations are the word sequences  $y_{1:T}$  and stop word indicators  $l_{1:T}$ . The log marginal likelihood of the sequence  $y_{1:T}$  is

$$
\log p \left(y _ {1: T}, l _ {1: T} \mid h _ {t}\right) = \log \int p (\theta) \prod_ {t = 1} ^ {T} p \left(y _ {t} \mid h _ {t}, l _ {t}, \theta\right) p \left(l _ {t} \mid h _ {t}\right) \mathrm {d} \theta . \tag {2}
$$

Model inference. Direct optimization of Equation 2 is intractable so we use variational inference for approximating this marginal (Jordan et al., 1999). Let  $q(\theta)$  be the variational distribution on the marginalized variable  $\theta$ . We construct the variational objective function, also called the evidence lower bound (ELBO), as follows:

$$
\begin{array}{l} \mathcal {L} \left(y _ {1: T}, l _ {1: T} \mid q (\theta), \Theta\right) \triangleq \mathbb {E} _ {q (\theta)} \left[ \sum_ {t = 1} ^ {T} \log p \left(y _ {t} \mid h _ {t}, l _ {t}, \theta\right) + \log p \left(l _ {t} \mid h _ {t}\right) + \log p (\theta) - \log q (\theta) \right] \\ \leq \log p (y _ {1: T}, l _ {1: T} | h _ {t}, \Theta). \\ \end{array}
$$

Following the proposed variational autoencoder technique, we choose the form of  $q(\theta)$  to be an inference network using a feed-forward neural network (Kingma and Welling, 2013; Miao et al., 2015). Let  $X_{c} \in \mathcal{N}_{+}^{|V_{c}|}$  be the term-frequency representation of  $y_{1:T}$  excluding stop words (with  $V_{c}$  the vocabulary size without the stop words). The variational autoencoder inference network  $q(\theta |X_c,W_c)$  with parameter  $W_{c}$  is a feed-forward neural network with ReLU activation units that projects  $X_{c}$  into a  $K$ -dimensional latent space. Specifically, we have

$$
q (\theta | X _ {c}, W _ {c}) = N (\theta ; \mu (X _ {c}), \operatorname {d i a g} \left(\sigma^ {2} (X _ {c})\right)),
$$

$$
\mu \left(X _ {c}\right) = W _ {1} g \left(X _ {c}\right) + a _ {1},
$$

$$
\log \sigma (X _ {c}) = W _ {2} g (X _ {c}) + a _ {2},
$$

where  $g(\cdot)$  denotes the feed-forward neural network. The weight matrices  $W_{1}$ ,  $W_{2}$  and biases  $a_{1}$ ,  $a_{2}$  are shared across documents. Each document has its own  $\mu(X_{c})$  and  $\sigma(X_{c})$  resulting in a unique distribution  $q(\theta|X_{c})$  for each document. The output of the inference network is a distribution on  $\theta$ , which we regard as the summarization of the semantic information, similar to the topic proportions in latent topic models. We show the role of the inference network in Figure 1(b). During training, the parameters of the inference network and the model are jointly learned and updated via truncated backpropagation through time using the Adam algorithm (Kingma and Ba, 2014). We use stochastic samples from  $q(\theta|X_{c})$  and the reparameterization trick towards this end (Kingma and Welling, 2013; Rezende et al., 2014).

Generating sequential text and computing perplexity. Suppose we are given a word sequence  $y_{1:t-1}$ , from which we have an initial estimation of  $q(\theta | X_c)$ . To generate the next word  $y_t$ , we compute the probability distribution of  $y_t$  given  $y_{1:t-1}$  in an online fashion. We choose  $\theta$  to be a point estimate  $\hat{\theta}$ , the mean of its current distribution  $q(\theta | X_c)$ . Marginalizing over the stop word indicator  $l_t$  which is unknown prior to observing  $y_t$ , the approximate distribution of  $y_t$  is

$$
p \left(y _ {t} \mid y _ {1: t - 1}\right) \approx \sum_ {l _ {t}} p \left(y _ {t} \mid h _ {t}, \hat {\theta}, l _ {t}\right) p \left(l _ {t} \mid h _ {t}\right).
$$

The predicted word  $y_{t}$  is a sample from this predictive distribution. We update  $q(\theta | X_{c})$  by including  $y_{t}$  to  $X_{c}$  if  $y_{t}$  is not a stop word. However, updating  $q(\theta | X_{c})$  after each word prediction is expensive, so we use a sliding window as was done in Mikolov and Zweig (2012). To compute the perplexity, we use the approximate predictive distribution above.

Model Complexity. TopicRNN has a complexity of  $O(H \times H + H \times (C + K) + W_c)$ , where  $H$  is the size of the hidden layer of the RNN,  $C$  is the vocabulary size,  $K$  is the dimension of the topic vector, and  $W_c$  is the number of parameters of the inference network. The contextual RNN of Mikolov and Zweig (2012) accounts for  $O(H \times H + H \times (C + K))$ , not including the pre-training process, which might require more parameters than the additional  $W_c$  in our complexity.

# 4 EXPERIMENTS

We assess the performance of our proposed TopicRNN model on word prediction and sentiment analysis<sup>5</sup>. For word prediction we use the Penn TreeBank dataset, a standard benchmark for assessing new language models (Marcus et al., 1993). For sentiment analysis we use the IMDB 100k dataset (Maas et al., 2011), also a common benchmark dataset for this application<sup>6</sup>. We use RNN, LSTM, and GRU cells in our experiments leading to TopicRNN, TopicLSTM, and TopicGRU.

Table 1: Five Topics from the TopicRNN Model with 100 Neurons and 50 Topics on the PTB Data. (The word  $s \& p$  below shows as  $sp$  in the data.)  

<table><tr><td>Law</td><td>Company</td><td>Parties</td><td>Trading</td><td>Cars</td></tr><tr><td>law</td><td>spending</td><td>democratic</td><td>stock</td><td>gm</td></tr><tr><td>lawyers</td><td>sales</td><td>republicans</td><td>s&amp;p</td><td>auto</td></tr><tr><td>judge</td><td>advertising</td><td>GOP</td><td>price</td><td>ford</td></tr><tr><td>rights</td><td>employees</td><td>republican</td><td>investor</td><td>jaguar</td></tr><tr><td>attorney</td><td>state</td><td>senate</td><td>standard</td><td>car</td></tr><tr><td>court</td><td>taxes</td><td>oakland</td><td>chairman</td><td>cars</td></tr><tr><td>general</td><td>fiscal</td><td>highway</td><td>investors</td><td>headquarters</td></tr><tr><td>common</td><td>appropriation</td><td>democrats</td><td>retirement</td><td>british</td></tr><tr><td>mr</td><td>budget</td><td>bill</td><td>holders</td><td>executives</td></tr><tr><td>insurance</td><td>ad</td><td>district</td><td>merrill</td><td>model</td></tr></table>

![](images/7c3bbe83e2da5b805f834437e54cb0c80a604b799a9aa1ebd76dd61d34854630.jpg)  
Figure 2: Inferred distributions using TopicGRU on three different documents. The content of these documents is added on the appendix. This shows that some of the topics are being picked up depending on the input document.

![](images/510c98cf0067f28496feb40eaf8787b80837dc2af56467c2db55502e0a71588c.jpg)

![](images/af66ff62199fceb1b9c4130a3f17d0f6ea5708b90a75dfa7e2295d0a66215d48.jpg)

# 4.1 WORD PREDICTION

We first tested TopicRNN on the word prediction task using the Penn Treebank (PTB) portion of the Wall Street Journal. We use the standard split, where sections 0-20 (930K tokens) are used for training, sections 21-22 (74K tokens) for validation, and sections 23-24 (82K tokens) for testing (Mikolov et al., 2010). We use a vocabulary of size  $10K$  that includes the special token unk for rare words and  $eos$  that indicates the end of a sentence. TopicRNN takes documents as inputs. We split the PTB data into blocks of 10 sentences to constitute documents as done by (Mikolov and Zweig, 2012). The inference network takes as input the bag-of-words representation of the input document. For that reason, the vocabulary size of the inference network is reduced to 9551 after excluding 449 pre-defined stop words.

In order to compare with previous work on contextual RNNs we trained TopicRNN using different network sizes. We performed word prediction using a recurrent neural network with 10 neurons,

Table 2: TopicRNN and its counterparts exhibit lower perplexity scores across different network sizes than reported in Mikolov and Zweig (2012). Table 2a shows per-word perplexity scores for 10 neurons. Table 2b and Table 2c correspond to per-word perplexity scores for 100 and 300 neurons respectively. These results prove TopicRNN has more generalization capabilities: for example we only need a TopicGRU with 100 neurons to achieve a better perplexity than stacking 2 LSTMs with 200 neurons each: 112.4 vs 115.9)  

<table><tr><td colspan="3">(a)</td></tr><tr><td>10 Neurons</td><td>Valid</td><td>Test</td></tr><tr><td>RNN (no features)</td><td>239.2</td><td>225.0</td></tr><tr><td>RNN (LDA features)</td><td>197.3</td><td>187.4</td></tr><tr><td>TopicRNN</td><td>184.5</td><td>172.2</td></tr><tr><td>TopicLSTM</td><td>188.0</td><td>175.0</td></tr><tr><td>TopicGRU</td><td>178.3</td><td>166.7</td></tr></table>

<table><tr><td colspan="3">(b)</td></tr><tr><td>100 Neurons</td><td>Valid</td><td>Test</td></tr><tr><td>RNN (no features)</td><td>150.1</td><td>142.1</td></tr><tr><td>RNN (LDA features)</td><td>132.3</td><td>126.4</td></tr><tr><td>TopicRNN</td><td>128.5</td><td>122.3</td></tr><tr><td>TopicLSTM</td><td>126.0</td><td>118.1</td></tr><tr><td>TopicGRU</td><td>118.3</td><td>112.4</td></tr></table>

(c)  

<table><tr><td>300 Neurons</td><td>Valid</td><td>Test</td></tr><tr><td>RNN (no features)</td><td>-</td><td>124.7</td></tr><tr><td>RNN (LDA features)</td><td>-</td><td>113.7</td></tr><tr><td>TopicRNN</td><td>118.3</td><td>112.2</td></tr><tr><td>TopicLSTM</td><td>104.1</td><td>99.5</td></tr><tr><td>TopicGRU</td><td>99.6</td><td>97.3</td></tr></table>

100 neurons and 300 neurons. For these experiments, we used a multilayer perceptron with 2 hidden layers and 200 hidden units per layer for the inference network. The number of topics was tuned depending on the size of the RNN. For 10 neurons we used 18 topics. For 100 and 300 neurons we found 50 topics to be optimal. We used the validation set to tune the hyperparameters of the model. We used a maximum of 15 epochs for the experiments and performed early stopping using the validation set. For comparison purposes we did not apply dropout and used 1 layer for the RNN and its counterparts in all the word prediction experiments as reported in Table 2. One epoch for 10 neurons takes 2.5 minutes. For 100 neurons, one epoch is completed in less than 4 minutes. Finally, for 300 neurons one epoch takes less than 6 minutes. These experiments were ran on Microsoft Azure NC12 that has 12 cores, 2 Tesla K80 GPUs, and 112 GB memory. First, we show five randomly drawn topics in Table 1. These results correspond to a network with 100 neurons. We also illustrate some inferred topic distributions for several documents from TopicGRU in Figure 2. Similar to standard topic models, these distributions are also relatively peaky.

Next, we compare the performance of TopicRNN to our baseline contextual RNN using perplexity. Perplexity can be thought of as a measure of surprise for a language model. It is defined as the exponential of the average negative log likelihood. Table 2 summarizes the results for different network sizes. We learn three things from these tables. First, the perplexity is reduced the larger the network size. Second, RNNs with context features perform better than RNNs without context features. Third, we see that TopicRNN gives lower perplexity than the previous baseline result reported by Mikolov and Zweig (2012). Note that to compute these perplexity scores for word prediction we use a sliding window to compute  $\theta$  as we move along the sequences. The topic vector  $\theta$  that is used from the current batch of words is estimated from the previous batch of words. This enables fair comparison to previously reported results (Mikolov and Zweig, 2012).<sup>7</sup>

Another aspect of the TopicRNN model we studied is its capacity to generate coherent text. To do this, we randomly drew a document from the test set and used this document as seed input to the inference network to compute  $\theta$ . Our expectation is that the topics contained in this seed document are reflected in the generated text. Table 3 shows generated text from models learned on the PTB and IMDB datasets. See Appendix A.3 for more examples.

Table 3: Generated text using the TopicRNN model on the PTB (top) and IMDB (bottom).

they believe that they had senior damages to guarantee and frustration of unk stations eos the rush to minimum effect in composite trading the compound base inflated rate before the common charter 's report eos wells fargo inc. unk of state control funds without openly scheduling the university 's exchange rate has been downgraded it 's unk said eos the united cancer & began critical increasing rate of  $N N$  at  $N N$  to  $N N$  are less for the country to trade rate for more than three months  $\$ N$ workers were mixed eos

lee is head to be watched unk month she eos but the acting surprisingly nothing is very good eos i cant believe that he can unk to a role eos may appear of for the stupid killer really to help with unk unk unk if you wan na go to it fell to the plot clearly eos it gets clear of this movie 70 are so bad mexico direction regarding those films eos then go as unk 's walk and after unk to see him try to unk before that unk with this film

Table 4: Classification error rate on IMDB 100k dataset. TopicRNN provides the state of the art error rate on this dataset.  

<table><tr><td>Model</td><td>Reported Error rate</td></tr><tr><td>BoW (bnc) (Maas et al., 2011)</td><td>12.20%</td></tr><tr><td>BoW (bΔ tć) (Maas et al., 2011)</td><td>11.77%</td></tr><tr><td>LDA (Maas et al., 2011)</td><td>32.58%</td></tr><tr><td>Full + BoW (Maas et al., 2011)</td><td>11.67%</td></tr><tr><td>Full + Unlabelled + BoW (Maas et al., 2011)</td><td>11.11%</td></tr><tr><td>WRRBM (Dahl et al., 2012)</td><td>12.58%</td></tr><tr><td>WRRBM + BoW (bnc) (Dahl et al., 2012)</td><td>10.77%</td></tr><tr><td>MNB-uni (Wang &amp; Manning, 2012)</td><td>16.45%</td></tr><tr><td>MNB-bi (Wang &amp; Manning, 2012)</td><td>13.41%</td></tr><tr><td>SVM-uni (Wang &amp; Manning, 2012)</td><td>13.05%</td></tr><tr><td>SVM-bi (Wang &amp; Manning, 2012)</td><td>10.84%</td></tr><tr><td>NBSVM-uni (Wang &amp; Manning, 2012)</td><td>11.71%</td></tr><tr><td>seq2-bown-CNN (Johnson &amp; Zhang, 2014)</td><td>14.70%</td></tr><tr><td>NBSVM-bi (Wang &amp; Manning, 2012)</td><td>8.78%</td></tr><tr><td>Paragraph Vector (Le &amp; Mikolov, 2014)</td><td>7.42%</td></tr><tr><td>SA-LSTM with joint training (Dai &amp; Le, 2015)</td><td>14.70%</td></tr><tr><td>LSTM with tuning and dropout (Dai &amp; Le, 2015)</td><td>13.50%</td></tr><tr><td>LSTM initialized with word2vec embeddings (Dai &amp; Le, 2015)</td><td>10.00%</td></tr><tr><td>SA-LSTM with linear gain (Dai &amp; Le, 2015)</td><td>9.17%</td></tr><tr><td>LM-TM (Dai &amp; Le, 2015)</td><td>7.64%</td></tr><tr><td>SA-LSTM (Dai &amp; Le, 2015)</td><td>7.24%</td></tr><tr><td>Virtual Adversarial (Miyato et al. 2016)</td><td>5.91%</td></tr><tr><td>TopicRNN</td><td>6.28%</td></tr></table>

# 4.2 SENTIMENT ANALYSIS

We performed sentiment analysis using TopicRNN as a feature extractor on the IMDB 100K dataset. This data consists of 100,000 movie reviews from the Internet Movie Database (IMDB) website. The data is split into  $75\%$  for training and  $25\%$  for testing. Among the 75K training reviews, 50K are unlabelled and 25K are labelled as carrying either a positive or a negative sentiment. All 25K test reviews are labelled. We trained TopicRNN on 65K random training reviews and used the remaining 10K reviews for validation. To learn a classifier, we passed the 25K labelled training reviews through the learned TopicRNN model. We then concatenated the output of the inference network and the last state of the RNN for each of these 25K reviews to compute the feature vectors. We then used these feature vectors to train a neural network with one hidden layer, 50 hidden units, and a sigmoid activation function to predict sentiment, exactly as done in Le and Mikolov (2014).

To train the TopicRNN model, we used a vocabulary of size 5,000 and mapped all other words to the unk token. We took out 439 stop words to create the input of the inference network. We used 500 units and 2 layers for the inference network, and used 2 layers and 300 units per-layer for the

![](images/98d5b540f67e24101aed89bef2b013d9b6648df3f2ec9945c8f915552cfb7105.jpg)  
Figure 3: Clusters of a sample of 10000 movie reviews from the IMDB 100K dataset using TopicRNN as feature extractor. We used K-Means to cluster the feature vectors. We then used PCA to reduce the dimension to two for visualization purposes. red is a negative review and green is a positive review.

RNN. We chose a step size of 5 and defined 200 topics. We did not use any regularization such as dropout. We trained the model for 13 epochs and used the validation set to tune the hyperparameters of the model and track perplexity for early stopping. This experiment took close to 78 hours on a MacBook pro quad-core with 16GHz of RAM. See Appendix A.4 for the visualization of some of the topics learned from this data.

Table 4 summarizes sentiment classification results from TopicRNN and other methods. Our error rate is  $6.28\%$ . This is close to the state-of-the-art  $5.91\%$  (Miyato et al., 2016) despite that we do not use the labels and adversarial training in the feature extraction stage. Our approach is most similar to Le and Mikolov (2014), where the features were extracted in a unsupervised way and then a one-layer neural net was trained for classification.

Figure 3 shows the ability of TopicRNN to cluster documents using the feature vectors as created during the sentiment analysis task. Reviews with positive sentiment are coloured in green while reviews carrying negative sentiment are shown in red. This shows that TopicRNN can be used as an unsupervised feature extractor for downstream applications. Table 3 shows generated text from models learned on the PTB and IMDB datasets. See Appendix A.3 for more examples. The overall generated text from IMDB encodes a negative sentiment.

# 5 DISCUSSION AND FUTURE WORK

In this paper we introduced TopicRNN, a RNN-based language model that combines RNNs and latent topics to capture local (syntactic) and global (semantic) dependencies between words. The global dependencies as captured by the latent topics serve as contextual bias to an RNN-based language model. This contextual information is learned jointly with the RNN parameters by maximizing the evidence lower bound of variational inference. TopicRNN yields competitive per-word perplexity on the Penn Treebank dataset compared to previous contextual RNN models. We have reported a competitive classification error rate for sentiment analysis on the IMDB 100K dataset. We have also illustrated the capacity of TopicRNN to generate sensible topics and text.

In future work, we will study the performance of TopicRNN when stop words are dynamically discovered during training. We will also extend TopicRNN to other applications where capturing context is important such as in dialog modeling. If successful, this will allow us to have a model that performs well across different natural language processing applications.

# REFERENCES

Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE transactions on neural networks, 5(2):157-166, 1994.  
Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin. A neural probabilistic language model. *journal of machine learning research*, 3(Feb):1137-1155, 2003.  
D. Blei and J. Lafferty. Correlated topic models. Advances in neural information processing systems, 18:147, 2006.  
D. M. Blei and J. D. Lafferty. Topic models. Text mining: classification, clustering, and applications, 10(71):34, 2009.  
D. M. Blei, A. Y. Ng, and M. I. Jordan. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan):993-1022, 2003.  
C. Chelba and F. Jelinek. Structured language modeling. Computer Speech & Language, 14(4): 283-332, 2000.  
C. Chelba, T. Mikolov, M. Schuster, Q. Ge, T. Brants, P. Koehn, and T. Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, 2013.  
K. Cho, B. Van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using rn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
A. M. Dai and Q. V. Le. Semi-supervised sequence learning. In Advances in Neural Information Processing Systems, pages 3079-3087, 2015.  
J. Gao, J.-Y. Nie, G. Wu, and G. Cao. Dependence language model for information retrieval. In Proceedings of the 27th annual international ACM SIGIR conference on Research and development in information retrieval, pages 170-177. ACM, 2004.  
S. Ghosh, O. Vinyals, B. Strope, S. Roy, T. Dean, and L. Heck. Contextual LSTM (clstm) models for large scale nlp tasks. arXiv preprint arXiv:1602.06291, 2016.  
S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.  
Y. Ji, T. Cohn, L. Kong, C. Dyer, and J. Eisenstein. Document context language models. arXiv preprint arXiv:1511.03962, 2015.  
Y. Ji, G. Haffari, and J. Eisenstein. A latent variable recurrent neural network for discourse relation language models. arXiv preprint arXiv:1603.01913, 2016.  
M. I. Jordan, Z. Ghahramani, T. S. Jaakkola, and L. K. Saul. An introduction to variational methods for graphical models. Machine learning, 37(2):183-233, 1999.  
R. Jozefowicz, O. Vinyals, M. Schuster, N. Shazeer, and Y. Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.  
D. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Q. V. Le and T. Mikolov. Distributed representations of sentences and documents. In ICML, volume 14, pages 1188-1196, 2014.  
R. Lin, S. Liu, M. Yang, M. Li, M. Zhou, and S. Li. Hierarchical recurrent neural network for document modeling. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 899-907, 2015.

A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1, pages 142-150. Association for Computational Linguistics, 2011.  
M. P. Marcus, M. A. Marcinkiewicz, and B. Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313-330, 1993.  
Y. Miao, L. Yu, and P. Blunsom. Neural variational inference for text processing. arXiv preprint arXiv:1511.06038, 2015.  
T. Mikolov and G. Zweig. Context dependent recurrent neural network language model. In SLT, pages 234-239, 2012.  
T. Mikolov, M. Karafiát, L. Burget, J. Cernocký, and S. Khudanpur. Recurrent neural network based language model. In *Interspeech*, volume 2, page 3, 2010.  
T. Mikolov, S. Kombrink, L. Burget, J. Černocký, and S. Khudanpur. Extensions of recurrent neural network language model. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5528-5531. IEEE, 2011.  
T. Mikolov, A. Joulin, S. Chopra, M. Mathieu, and M. Ranzato. Learning longer memory in recurrent neural networks. arXiv preprint arXiv:1412.7753, 2014.  
D. Mimno, H. M. Wallach, E. Talley, M. Leenders, and A. McCallum. Optimizing semantic coherence in topic models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 262-272. Association for Computational Linguistics, 2011.  
T. Miyato, A. M. Dai, and I. Goodfellow. Adversarial training methods for semi-supervised text classification. stat, 1050:7, 2016.  
R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. ICML (3), 28:1310-1318, 2013.  
D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082, 2014.  
N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1): 1929-1958, 2014.  
I. Sutskever. Training recurrent neural networks. PhD thesis, University of Toronto, 2013.  
H. M. Wallach. Topic modeling: beyond bag-of-words. In Proceedings of the 23rd international conference on Machine learning, pages 977-984. ACM, 2006.  
H. M. Wallach, D. M. Mimno, and A. McCallum. Rethinking lda: Why priors matter. In Advances in neural information processing systems, pages 1973-1981, 2009.
