# KrwEmd: Revising the Imperfect Recall Abstraction from Forgetting Everything

Anonymous Author(s)

Affiliation

Address

email

# Abstract

A recent research has shown that an extreme interpretation of imperfect recall abstraction – completely forgetting all past information – has led to excessive abstraction issues. Currently, there are no hand abstraction algorithms that effectively integrate historical information. This paper aims to develop the first such algorithm. Initially, we introduce the KRWI abstraction for Texas Hold'em-style games, which categorizes hands based on K-recall winrate features that incorporate historical information. Statistical results indicate that, in terms of the number of distinct infosets identified, KRWI significantly outperforms POI, an abstraction that identifies the most abstracted infosets that forget all historical information. Following this, we introduce the KrwEmd algorithm, the first hand abstraction algorithm to effectively use historical information by combining K-recall win rate features and earth mover's distance for hand classification. Experimental studies conducted in the Numeral211 Hold'em environment show that under identical abstracted infoset sizes, KrwEmd not only surpasses POI but also outperforms state-of-the-art hand abstraction algorithms such as Ehs and PaEmd. These findings suggest that incorporating historical information can significantly enhance the performance of hand abstraction algorithms, positioning KrwEmd as a promising approach for advancing strategic computation in large-scale adversarial games.

# 1 Introduction

Imperfect recall abstraction has proven to be very important for solving large-scale computational games, significantly reducing computational complexity. Recently, AI using imperfect recall abstraction has developed better-than-human strategies for Texas Hold'em testbed—even when using limited computational resources [23, 7, 8].

The task of hand abstraction in Texas Hold'em aims to reduce computational overhead by applying the same strategy to similar hands. In an imperfect recall setting [29, 20], the hand abstraction in the later phase does not strict depend on the results of the hand abstraction in the earlier phase. However, the term imperfect recall is often interpreted in an extreme manner in practice. Researchers typically understand it as completely forgetting all past information—in other words, considering only future information—and design abstraction algorithms based on this understanding [16, 17, 19, 15, 14]. There are two major factors that mainly affect the results of abstraction for each phase: the number of clustering centers (i.e. centroids), which can be set man-

![](images/b27f52e9f4816616682674a4bafe19a80367f0aecab10b3231759414a4145961.jpg)  
Figure 1: In a 4-phase game hand abstraction task, the current goal is to classify hands A and B.

ually, and the number of distinct features that are used to categorize hands at each phase. Recent research [12] has found that constructing hand features solely based on future information can lead to excessive abstraction. For example, as shown in the Figure 1, two hands: A and B constructed with only future information can have the same hand features. As the game progresses, the rate of feature repetition among different hands gradually increases, while the distribution of distinct hand features assumes a spindle-shaped pattern. Additionally, constructing hand features with historical information in addition to the future may differentiate two hands sharing the same future information and hence makes more features available for clustering as well as enhances the performance of hand abstraction.

However, there still remain two unsolved issues. First, Fu et al. [12] have introduced a K-recall outcome feature, which incorporates historical information. This feature can only identify if elements are identical or not, but it lacks the capability to discern the extent of differences between features. Therefore, it is difficult to adjust the number of clusters appropriately, which makes it challenging to construct an effective hand abstraction algorithm that integrates historical information. Second, due to the inability to modify the number of clusters, Fu et al. [12] only compared the performance between the maximum clusters cases of integration of historical information (KROI) and no integration at all (POI). In this condition, although KROI significantly outperforms POI, the comparison is inconclusive because KROI recognizes more abstracted infosets than POI. Thus, it does not prove that the performance of abstraction algorithms that integrate historical information is necessarily superior under the condition of having the same number of abstracted infosets.

This paper introduces a framework for constructing hand features based on winrates, with the K-recall winrate feature being the most crucial one. Based on this, we developed the K-recall winrate isomorphism (KRWI), an abstraction that integrates historical information. Across the same game phases, KRWI identifies slightly fewer hand features than KROI but significantly more than POI. Importantly, the K-recall winrate feature is capable of discerning the extent of differences between features. Therefore, by combining the earth mover's distance with the K-recall winrate feature, we developed the first hand abstraction algorithm that integrates historical information, named KrwEmd, and designed an efficient computational method. We validated our approach in the Numeral211 game environment, where KrwEmd demonstrated superior performance to POI under the same效果图 conditions. Additionally, in clustering settings, KrwEmd also outperformed the Ehs and PaEmd algorithms, with PaEmd being the current state-of-the-art hand abstraction algorithm.

# 2 Background and Notation

Generally, Texas Hold'em-style poker games are modeled as imperfect information games. However, for the task of hand abstraction, games with ordered signals [18, 12] offer a better theoretical tool. The game with ordered signals is a subclass of imperfect information games in that they further subdivide the nodes (also called histories, states, or trajectories) in imperfect information games into mutually independent signals and public nodes. This allows for each aspect to be studied in isolation. Under this framework, the hand abstraction task in Texas Hold'em-style games is modeled as signal abstraction.

In a game with ordered signals  $\tilde{\Gamma} = \left\langle \tilde{\mathcal{N}}, \tilde{H}, \tilde{Z}, \tilde{\rho}, \tilde{A}, \tilde{\chi}, \tilde{\tau}, \gamma, \Theta, \varsigma, O, \omega, \succeq, \tilde{u} \right\rangle$ , there is a set of players  $\tilde{\mathcal{N}} = \mathcal{N} \cup \{c, pub\}$ , which includes not only the main participants  $\mathcal{N} = \{1, \dots, N\}$  but also a special nature player  $c$  who controls the randomness and an observer player  $pub$  who can see everything but doesn't take any actions. The game progresses through a series of public nodes  $\tilde{X} = \tilde{H} \cup \tilde{Z}$ . Some of these public nodes are terminal public nodes  $\tilde{Z}$  where the game ends and outcomes are determined, while the others are non-terminal public nodes  $\tilde{H}$ . Among the non-terminal public nodes, some are where players make decisions within the action space  $\tilde{A}$ , and the remaining are chance public nodes where the nature player reveals signals, with the special action  $Reveal$  within  $\tilde{A}$ .

At every non-terminal public node,  $\tilde{\rho}:\tilde{H}\mapsto \mathcal{N}c$  (i.e.,  $\mathcal{N}\cup \{c\}$ ) specifies which player is responsible for making an action, and  $\tilde{\chi}:\tilde{H}\mapsto 2^{\tilde{A}}$  confines the possible actions they can take. When the nature player makes a move, it reveals signals  $\theta \in \Theta$  that carry information relevant to the game. These signals are then observed by all players except  $c$ ,  $O(\theta) = (O_1(\theta),\dots,O_N(\theta),O_{pub}(\theta))$ , though what they can see might differ.

The progression from one public node to another is clearly defined  $\tilde{\tau}:\tilde{H}\times \tilde{A}\mapsto \tilde{X}$ , ensuring that the game's structure is sequential and predictable. Similarly, the signals are revealed according to a probability distribution  $\varsigma :\Theta \mapsto \Delta (\Theta)$ , which specifies the likelihood of the next signal given the current one. We use  $\tilde{h}\subseteq \tilde{h}^{\prime}$  to indicate that  $\tilde{h}$  is a predecessor of  $\tilde{h}^\prime$ , and  $\theta \subseteq \theta^{\prime}$  to indicate that  $\theta$  is a predecessor of  $\theta^{\prime}$ . Each phase of the game is the number of times nature player has revealed signals, denoted by  $\gamma :\tilde{X}\mapsto \mathbb{N}^{+}$ .  $\mathfrak{r} = \{\gamma (\tilde{x})\mid \tilde{x}\in \tilde{X}\}$  represents the phases that a game with ordered signals may go through. Since the root is a chance public node, we have min  $\mathfrak{r} = 1$ .

At the end of the game, players receive their payoffs based on the signals and the terminal public node, represented by  $\tilde{u} = (\tilde{u}_1,\dots ,\tilde{u}_N)$ , where  $\tilde{u}_i:\Theta \times \tilde{Z}\mapsto \mathbb{R}$ . Additionally, each player's survival status is determined at these terminal public nodes, denoted by  $\omega = (\omega_{1},\ldots ,\omega_{N})$ , where  $\omega_{i}:\tilde{Z}\mapsto \{\text{true,false}\}$ . The signals possess a partial order within their subset, terminal signals  $\tilde{\Theta}$ , indicated by  $\succeq$ $\tilde{\Theta}\times \mathcal{N}\times \mathcal{N}\mapsto \{\text{true,false}\}$ . It is required that for any terminal signal  $\theta \in \tilde{\Theta}$  and terminal public nodes  $\tilde{z}\in \{\tilde{z}^{\prime}\in \tilde{Z}\mid \omega_{i}(\tilde{z}^{\prime}) = \omega_{j}(\tilde{z}^{\prime}) = true\}$ , if  $\succeq (\theta ,i,j) = true$ , then  $\tilde{u}_i(\theta ,\tilde{z})\geq \tilde{u}_j(\theta ,\tilde{z})$ .

Players make decisions based on their observations of signals and the current non-terminal public node. A player may have the same observation for different signals, forming a signal infoset for signals they cannot distinguish. For a player  $i \in \mathcal{N}$ , the signal infoset for a signal  $\theta$  is denoted as  $\vartheta_i(\theta) = \{\theta' \in \Theta \mid O_i(\theta) = O_i(\theta') \land O_{pub}(\theta) = O_{pub}(\theta')\}$ . Specifically, for the nature player,  $\vartheta_c(\theta) = \{\theta' \in \Theta \mid O_{pub}(\theta') = O_{pub}(\theta)\}$ . We abuse the notation  $\vartheta \in \Theta_i$  to represent a signal infoset, where for any player  $i \in \mathcal{N}$ ,  $\Theta_i$  is a partition of  $\Theta$ , representing the collection of player  $i$ 's signal infosets.  $\Theta_i^{(1)}, \ldots, \Theta_i^{(|\mathfrak{r}|)}$  are the collections of player  $i$ 's signal infosets for each phase, and they form a partition of  $\Theta_i$ . In games with ordered signals, the signals describe all private information. The signal infoset, combined with public nodes, can be transformed into the infoset of an imperfect information game. Fu et al. [12] detailed this transformation process.

The game with ordered signals model allows us to study the issue of signal abstraction independently. For this purpose, we introduce a signal (infoset) abstraction profile,  $\alpha = (\alpha_{1},.,\alpha_{N})$ , where for each player  $i\in \mathcal{N}$ ,  $\alpha_{i}$  is a partition of  $\Theta$  called the signal (infoset) abstraction. Any  $\hat{\vartheta}\in \alpha_{i}$  then is said to be an abstracted signal infoset for player  $i$ , and it can be further divided into several signal infosets within  $\Theta_{i}$ . These finer signal infosets collectively form a partition of  $\hat{\vartheta}$ . In general, two signal abstractions cannot be directly compared in terms of performance, but in a few specific cases there does exist a special relationship between them, which is called refinement. Consider two abstractions  $\alpha_{i}$  and  $\beta_{i}$ . If  $\forall \hat{\vartheta}\in \beta_{i}$ , there exists one or more abstracted signal infosets in  $\alpha_{i}$  such that the union of these forms a partition of  $\hat{\vartheta}$ , then we said that  $\alpha_{i}$  refines  $\beta_{i}$ , symbolically  $\alpha_{i}\supseteq \beta_{i}$ . The signal abstracted game  $\tilde{\Gamma}^{\alpha}$  was derived by substituting  $\Theta_{i}$  with  $\alpha_{i}$  across all  $\tilde{x}\in \tilde{X}$ .

Perfect/imperfect recall originally describes a property of imperfect information games, indicating that players do not need to remember all the information they have observed throughout the game. Since games with ordered signals are a subset of imperfect information games, we derived the concept of signal perfect/imperfect recall from them. A player  $i$  in a game  $\tilde{\Gamma}$  is said to have signal perfect recall if, for any  $\theta_1', \theta_2' \in \vartheta'$ , any predecessor  $\theta_1$  of  $\theta_1'$  has a corresponding predecessor  $\theta_2$  of  $\theta_2'$  such that  $\theta_2 \in \vartheta(\theta_1)$ . If all players have signal perfect recall, the game  $\tilde{\Gamma}$  is said to have signal perfect recall. For a game  $\tilde{\Gamma}$  with signal perfect recall, if  $\alpha_i$  is the signal abstraction of player  $i \in \mathcal{N}$ , let  $(\alpha_i, \Theta_{-i})$  denote the signal abstraction profile where player  $i$  adopts the signal abstraction  $\alpha_i$  while other players do not do abstraction. If  $\tilde{\Gamma}^{(\alpha_i, \Theta_{-i})}$  retains signal perfect recall, then  $\alpha_i$  is considered a signal abstraction with perfect recall; otherwise, it is an signal abstraction with imperfect recall.

In games with ordered signals, the strategy  $\pi_{i}$  for player  $i$  maps from a non-terminal public node and a signal infographic to a probability distribution over actions, with the strategy profile denoted as  $\pi = (\pi_1,\dots ,\pi_N)$ . When all players adopt the strategy profile  $\pi$ , the expected sum of future rewards, also known as expected value, for player  $i$  at public node  $\tilde{x}$  and signal  $\theta$  is denoted as  $v_{i}^{\pi}(\theta ,\tilde{x})$  and the expected value for the entire game is denoted as  $v_{i}(\pi)$ . A Nash equilibrium is a strategy profile where no player can obtain a higher expected value by changing their strategy. Formally,  $\pi^{*}$  is a Nash equilibrium if for every player  $i$ ,  $v_{i}(\pi^{*}) = \max_{\pi_{i}}v_{i}(\pi_{i},\pi_{-i}^{*})$ , where  $\pi_{-i}$  denotes the strategies of all players except  $i$ . In two-player zero-sum scenarios, the exploitability of  $\pi$  is denoted as  $\epsilon (\pi) = \frac{\max_{\pi_1'}v_i(\pi_1',\pi_2) + \max_{\pi_2'}v_i(\pi_1,\pi_2')}{2}$ .

# 3 Related Work

Our research focuses on hand abstraction techniques in AI systems for Texas Hold'em-style games (i.e. the signal abstraction in games with ordered signals), building on the initial works of Shi and Littman [25] and Billings et al. [4]. These seminal works introduced the concept of game abstraction, which aims to simplify games while preserving essential characteristics. The researchers started by manually forming hand buckets as a result of their expertise with game-playing strategy. The first automated hand abstraction was that of Gilpin and Sandholm [16]. Later, a model of games with ordered signals was given for Texas Hold'em by Gilpin and Sandholm [18]; lossless isomorphism (LI) was developed with signal rotation. Despite the elegance of LI, its low compression rates hinder its application in large-scale games, whereas lossy abstraction shows potential for such application. An expectation-based clustering method was proposed by Gilpin and Sandholm [17] in their work, and a histogram-based clustering method was introduced by Gilpin et al. [19]. The former is known as Ehs, while the latter is referred to as the potential-aware method. Subsequent studies by Gilpin and Sandholm [15] and Johanson et al. [20] compared Ehs and potential-aware methods, concluding that the latter holds an advantage in large-scale games. Johanson et al. [20] also introduced the use of earth mover's distance<sup>1</sup> (EMD) in potential-aware methods. Ganzfried and Sandholm [14] introduced a more efficient approximation algorithm for earth mover's distance in potential-aware methods (PaEmd). Brown et al. [9] further applied PaEmd to distributed environments for solving large-scale imperfect-information games. This paradigm has found success in Texas Hold'em AI systems and is considered state-of-the-art in hand abstraction. Very recently, Fu et al. [12] proposed several novel tools, such as abstraction resolution and common refinement. They introduced two signal abstraction: one is the potential outcome isomorphism (POI), which identifies the maximum number of abstracted signal infosets considering future information only; The other is the K-recall outcome isomorphism (KROI), which identifies the maximum number of abstracted signal infosets considering historical information. They emphasized that current imperfect recall signal abstraction algorithms, which consider only future information, are prone to excessive abstraction. However, they did not provide practical signal abstraction algorithms.

Other abstraction techniques for decision-making problems include action abstraction [13, 6, 21] and general imperfect recall abstraction [10, 11] in extensive-form games, as well as state abstraction and action abstraction in reinforcement learning [1, 2].

# 4 Winrate Isomorphism

The first contribution of this paper is an isomorphism framework of winrate-based features, including the potential winrate isomorphism (PWI) and the k-recall winrate Isomorphism (KRWI). Compared with outcome-based features, winrate-based features offer a streamlined approach, focusing exclusively on the distribution of loss, draw, and win outcomes of signals emanating from a signal效果图 (and its predecessors) as it evolves towards the terminal signals. Winrate-based features are numerical vectors of consistent length. In this section, an identical Winrate-based feature uniquely determines an abstracted signal效果图. It is worth noting that the similarity of Winrate-based features reflects the similarity among signal效果图, allowing for clustering based on these features (see Section 5).

Both PWI and KRWI share the similar isomorphism construction process for player  $i$  in phase  $r$ , as illustrated in algorithm 1. The difference lies only in the construction operator for the winrate-based features, FEATURE, used in lines 5 and 12. The isomorphism construction process starts by iterating through all signal infosets of  $\Theta_i^{(r)}$  and collecting their features. Next, these features are deduplicated and stored in lexicographical order within set  $\mathcal{C}_i^{(r)}$ , which is implemented as a vector data structure. Within  $\mathcal{C}_i^{(r)}$ , the index of a feature serves as an identifier for an abstracted signal infoset. Then, by utilizing a hash table  $\mathcal{C}\mathcal{I}_i^{(r)}$ , we can identify an abstracted signal infoset's identifier based on its feature. In the final step, we traverse  $\Theta_i^{(r)}$  again, associating the identifier of a signal infoset with the identifier of its corresponding abstracted signal infoset, and this relationship is recorded in  $\mathcal{D}_i^{(r)}$ , an isomorphism map. The function Index  $i(r,\cdot)$  is a domain-specific mapping that assigns a unique identifier to each signal infoset at phase  $r$ , within the numeric range of 0 to  $|\Theta_i^{(r)}| - 1$ . In

Algorithm 1 Isomorphism Constructor  
Require:  $r = 1,\dots ,R$  . Phases.  $\Theta_i^{(r)}$  . Signal infoset space for player i. Indexi(r,):  $\Theta_{i}^{(r)}\mapsto \mathbb{N}$  . Signal infoset index function for player i.   
1: procedure ISOMORPHISMCONSTRUCTOR(r,  $\Theta_i^{(r)}$  , FEATURE(\cdot))   
2: Initialize  $\mathcal{C}_i^{(r)}$  vector as empty.   
3: Initialize  $\mathcal{D}_i^{(r)}$  array arbitrarily with length  $|\Theta_i^{(r)}|$    
4: for  $\vartheta \in \Theta_i^{(r)}$  do   
5: feature  $\leftarrow$  FEATURE(v).   
6: Append feature to  $\mathcal{C}_i^{(r)}$    
7: end for   
8: Eliminate duplicates from  $\mathcal{C}_i^{(r)}$    
9: Sort the elements of  $\mathcal{C}_i^{(r)}$  in lexicographical order.   
10: Construct hash table  $\mathcal{CI}_i^{(r)}$  from  $\mathcal{C}_i^{(r)}$  . Store the index lexid and value feature of  $\mathcal{C}_i^{(r)}$  in  $\mathcal{CI}_i^{(r)}$  as key-value pairs (feature,lexid).   
11: for  $\vartheta \in \Theta_i^{(r)}$  do   
12: feature  $\leftarrow$  FEATURE(v), idx  $\leftarrow$  Indexi(r,v).   
13: Update  $\mathcal{D}_i^{(r)}[idx]$  with  $\mathcal{CI}_i^{(r)}[feature]$    
14: end for   
15: return  $(\mathcal{C}_i^{(r)},\mathcal{D}_i^{(r)})$    
16: end procedure

Texas Hold'em-style games, one optional approach for implementing this function is through lossless isomorphism [18, 27].

# 4.1 Potential Winrate Isomorphism

Potential winrate isomorphism (PWI) is a signal abstraction that classify signal infosets based on its potential winrate features. These features focus on the distribution of a player's winrate over terminal signals after passing through a given signal infographic, without considering the history of how the player reached the signal infographic. Specifically, for player  $i$  in phase  $r$ , the potential winrate feature associated with  $\vartheta \in \Theta_i^{(r)}$  is defined as

$$
p f _ {i} ^ {(r)} (\vartheta) = \left(p f _ {i} ^ {(r), 0} (\vartheta), p f _ {i} ^ {(r), 1} (\vartheta), \dots , p f _ {i} ^ {(r), N} (\vartheta)\right), \tag {1}
$$

where

-  $pf_{i}^{(r),0}(\vartheta)$  denotes the probability that player  $i$  ranks lower than least one other player in the terminal signals, after passing through  $\vartheta$ .  
-  $pf_i^{(r),l}(\vartheta)$ , for  $l > 0$ , denotes the probability that player  $i$  ranks no lower than any other player and ranks higher than exactly  $l - 1$  other players in the terminal signals, after passing through  $\vartheta$ .

In the terminal phase, the winrate feature is calculated by directly statisticing the game outcomes for players in the given signal效果图. Moreover, in the non-terminal phases, we use a recursive approach to simplify the computation of the winrate feature, thereby avoiding the need to enumerate every signal效果图 down to the terminal phase. The recursive formula is

$$
p f _ {i} ^ {(r), l} (\vartheta) = \sum_ {\substack {\vartheta^ {(r + 1)} \in \Theta_ {i} ^ {(r + 1)} \\ \vartheta \sqsubseteq \vartheta^ {(r + 1)}}} p f _ {i} ^ {(r + 1), l} \left(\vartheta^ {(r + 1)}\right) P r \left\{\vartheta^ {(r + 1)} \mid \vartheta \right\} \tag{2}
$$

<table><tr><td></td><td colspan="2">Preflop</td><td colspan="2">Flop</td><td colspan="3">Turn</td><td colspan="4">River</td></tr><tr><td>Recall</td><td>0</td><td>0</td><td>1</td><td>0</td><td>1</td><td>2</td><td>0</td><td>1</td><td>2</td><td>3</td><td></td></tr><tr><td>KRWI</td><td>169</td><td>1028325</td><td>1123442</td><td>1850624</td><td>34845952</td><td>37659309</td><td>20687</td><td>33117469</td><td>529890863</td><td>577366243</td><td></td></tr><tr><td>KROI</td><td>100</td><td>1137132</td><td>1241210</td><td>2337912</td><td>38938975</td><td>42040233</td><td>20687</td><td>39792212</td><td>586622784</td><td>638585633</td><td></td></tr><tr><td>W/O (%)</td><td>100.0</td><td>90.43</td><td>90.51</td><td>79.16</td><td>89.49</td><td>89.58</td><td>100.0</td><td>83.23</td><td>90.33</td><td>90.41</td><td></td></tr></table>

Table 1: The number of abstracted signal infosets identified by KRWI, and KROI in each phase and  $k$  of HUNL&HUNLE, with W/O indicating the ratio identified by PWI and POI.

The PWI algorithm is derived from the POI algorithm [12], and the details of the PWI algorithm are elaborated in Appendix A.1. Both algorithms use the potential winrate feature to distinguish between different abstracted signalinfosets in the terminal phase. However, unlike POI, PWI also uses the potential winrate feature in non-terminal phases to identify different abstracted signal_infoset classes, while POI relies on the potential outcome

feature (which captures the distribution of the abstracted signal效果图 class for future signal效果图). In non-terminal phases, the potential winrate feature is a simplified version of the potential outcome feature. Unsurprisingly, PWI also results in excessive abstraction similar to POI. As shown in Table 2, in heads-up limit hold'em (HULHE) and heads-up no-limit hold'em (HUNL), the number of abstracted signal效果图 identifiable by lossless isomorphism increases with each phase, indicating that the game becomes increasingly complex. However, the number of abstracted signal效果图 identifiable by PWI and POI first increases and then decreases, showing a spindle-shaped pattern. And we observed that when only future information is considered, winrate-based features may lead to greater information loss compared to outcome-based features. For instance, in the River phase, the number of abstracted signal效果图 identified by PWI is only  $79.16\%$  of that identified by POI.

<table><tr><td></td><td>Preflop</td><td>Flop</td><td>Turn</td><td>River</td></tr><tr><td>LI</td><td>169</td><td>1286792</td><td>55190538</td><td>2428287420</td></tr><tr><td>PWI</td><td>169</td><td>1028325</td><td>1850624</td><td>20687</td></tr><tr><td>POI</td><td>169</td><td>1137132</td><td>2337912</td><td>20687</td></tr><tr><td>W/O (%)</td><td>100.0</td><td>90.43</td><td>79.16</td><td>100.0</td></tr></table>

Figure 2: The number of abstracted signalinfosets identified by LI, PWI, and POI in each phase of HUNL&HUNLE, with W/O indicating the ratio identified by PWI and POI.

# 4.2 K-Recall Winrate Isomorphism

As Fu et al. [12] mentioned, supplementing historical information can enhance the ability of signal abstraction to identify abstracted signal infosets. Inspired by KROI's construction approach, we developed the k-recall winrate isomorphism (KRWI). The key difference is that instead of using k-recall outcome features to distinguish between different signal infosets, KRWI utilizes k-recall winrate features.

In a game with signal perfect recall, all signals within the signal infographic  $\vartheta$  have their predecessors at phase  $r'$ , which belong to the identical signal infographic  $\vartheta'$ . For player  $i$  at phase  $r$ , the signal infographic  $\vartheta \in \Theta_i^{(r)}$  has a  $k$ -recall winrate feature  $(k < r)$  represented as a numerical array with a dimension of  $(k + 1)(N + 1)$ :

$$
r f _ {i} ^ {(r, k)} (\vartheta) = \left(p f _ {i} ^ {(r)} (\vartheta); p f _ {i} ^ {(r - 1)} (\vartheta); \dots ; p f _ {i} ^ {(r - k)} (\vartheta)\right) \tag {3}
$$

When  $r'$  is less than  $r$ ,  $pf_i^{(r')}(\vartheta)$  denotes the potential winrate feature for the predecessor signal infoset  $\vartheta'$  of  $\vartheta$  at phase  $r'$ . Since we have stored all the potential winrate features of  $\vartheta \in \Theta_i^{(r)}$  through  $\mathcal{PC}_i^{(r)}, \mathcal{PD}_i^{(r)}$  and assigned them unique identifiers in Algorithm A1. To save storage space and facilitate retrieval, what we actually store is

$$
r f i _ {i} ^ {(r, k)} (\vartheta) = \left(\mathcal {P D} _ {i} ^ {(r)} [ \vartheta ], \mathcal {P D} _ {i} ^ {(r - 1)} [ \vartheta ], \dots , \mathcal {P D} _ {i} ^ {(r - k)} [ \vartheta)\right) \tag {4}
$$

$\mathcal{PD}_i^{(r')}[\vartheta]$  is the identifier for the potential winrate feature of the predecessor  $\vartheta'$  of  $\vartheta$  in the  $r'$  phase,  $r' \leq r$ . For algorithm details, please refer to Appendix A.2.

Just as the potential winrate feature is a simplified version of the potential outcome feature, the k-recall winrate feature is a simplified version of the k-recall outcome feature. Table 1 shows the number of signal infosets that KRWI and KROI can identify and their ratio in HUNL&HULHE. We were pleasantly surprised to find that while the ratio of PWI to POI resolution can drop below  $80\%$ ,

when  $k$  is set to its maximum value, i.e.  $r - 1$ , the ratio of KRWI to KROI resolution can reach nearly  $90\%$  at a minimum, with most of the information preserved. Also, we can easily observe that the number of abstracted signal infosets identified by KRWI is much higher than that identified by POI.

# 5 K-Recall Winrate Abstraction with Earth Mover's Distance

Fu et al. [12] introduced potential and k-recall outcome features, referred to as outcome-based features, to distinguish different abstracted signal infosets. In the previous section, we developed potential and k-recall winrate features, termed winrate-based features, for the same purpose. In these two methods, Each unique feature corresponds to a single abstracted signal infoset. Intuitively, we can infer that feature similarity might reflect the similarity among abstracted signal infosets, enabling further abstraction and compression for application in large-scale games. However, assessing similarity with outcome-based features is challenging because the identification code indicates only the category, without reflecting the degree of similarity. In contrast, winrate-based features represent winrates, which are inherently comparable, allowing for an easy definition of distances between them.

For the signal information sets  $\vartheta, \vartheta'$  of player  $i$  at phase  $r$ , we can define the distance of their k-recall winrate feature as

$$
d \left(r f _ {i} ^ {(r, k)} (\vartheta), r f _ {i} ^ {(r, k)} \left(\vartheta^ {\prime}\right)\right) = \sum_ {j = 0} ^ {k} w _ {j} \cdot \operatorname {E m d} \left(p f _ {i} ^ {(r - j)} (\vartheta), p f _ {i} ^ {(r - j)} \left(\vartheta^ {\prime}\right)\right) \tag {5}
$$

Among Equation (5), Emd is the operator used to calculate the earth mover's distance (EMD) [24]. The EMD calculates the distance between two histograms using optimal transport theory. Since it requires solving linear programming equations, the computational complexity of the EMD is sensitive to the dimensionality of the histograms, and approximate algorithms are usually used for larger-scale problems. However, the dimensionality of winrate-based features is small, with a dimension of 3 in a two-player scenario, so we attempt to use a fast algorithm for accurately computing the EMD [5].  $w_{0}, \ldots, w_{k}$  are hyperparameters used to control the importance of EMD at each phase  $r, \ldots, r - k$ . We use the KMeans++ algorithm [3], combined with the distance of their k-recall winrate feature, to cluster the abstracted signal infosets of KRWI. We named this algorithm KrwEmd.

Although calculating EMD on small-dimensional histograms is already very fast, clustering actual Texas Hold'em still faces a significant computation. For example, for the River phase of HUNL&HULHE, the clustering input size of the KRWI abstracted signal info set is approximately  $5.8 \times 10^{8}$ . When we set the number of centroids to 20000, a single Kmeans++ iteration takes about 19000 core hours on a computer with a 2.40GHz clock frequency, which is a significant time cost. Therefore, we need to find ways to reduce this time cost. We have developed an accelerated algorithm, please refer to Appendix A.3 for details.

# 6 Experimental Setup

We conducted experiments on the Numeral211 Hold'em [12] testbed. Numeral211 is a two-player three-phase Taxes Hold'em-style game with more complex hand systems than the Leduc Hold'em [26] and Rhode Island Hold'em [25] test environments, making it suitable for studying hand abstraction issues. Detailed rules are included in Appendix B. Table 3 shows the number of abstracted signal infosets recognized by KRWI and KROI, along with lossless isomorphism, in Numeral211 Hold'em.

Figure 3: The number of abstracted signalinfosets identified by LI, PWI, and POI in each phase of HUNL&HUNLE, with W/O indicating the ratio identified by PWI and POI.  

<table><tr><td rowspan="2">LI</td><td colspan="2">Preflop</td><td colspan="2">Flop</td><td colspan="3">Turn</td></tr><tr><td colspan="2">100</td><td colspan="2">2260</td><td colspan="3">62020</td></tr><tr><td>Recall</td><td colspan="2">0</td><td>0</td><td>1</td><td>0</td><td>1</td><td>2</td></tr><tr><td>KRWI</td><td colspan="2">100</td><td>2234</td><td>2248</td><td>3957</td><td>51000</td><td>51070</td></tr><tr><td>KROI</td><td colspan="2">100</td><td>2250</td><td>2260</td><td>3957</td><td>51176</td><td>51228</td></tr><tr><td>W/O (%)</td><td colspan="2">100.0</td><td>99.29</td><td>99.47</td><td>100.0</td><td>99.67</td><td>99.69</td></tr></table>

Let  $\alpha = (\alpha_{1},\alpha_{2})$  be the signal abstraction we would like to assess. We will test the strength of the signal abstraction by measuring exploitability of the approximate equilibrium derived using the

CSMCCFR algorithm [30, 22] in different abstracted signalinfoset scales. We gauge the performance over exploitability. For doing that, we consider both symmetric and asymmetric abstraction scenarios.

In this symmetric abstraction setting, we measure the exploitability of approximate equilibrium that is yielded when both the players in the game employ signal abstraction in the original game. However, it may lead to the abstraction pathology [28]. To avoid such problems, we illustrate the theoretical performance of the signal abstraction under evaluation through asymmetric abstraction. The approximate equilibrium in the signal abstracted games  $\tilde{\Gamma}^{(\alpha_1,\Theta_2)}$  and  $\tilde{\Gamma}^{(\Theta_1,\alpha_2)}$  is obtained to obtain  $\pi^{*,1}$  and  $\pi^{*,2}$ , respectively. Finally, we concat the two strategies to get  $\pi' = (\pi_1^{*,1},\pi_2^{*,2})$  and check the exploitability of  $\pi'$ .

# 7 Experiment

![](images/4d310fc998c5295dbad6c5ecdda6418654c7291f0c96ba1d40597a0ca62e26e1.jpg)  
(a)  
Figure 4: Full abstraction setting experiment, trained for  $5.5 \times 10^{10}$  iterations.

![](images/d28af0f63f8ac2e48daf4e73a11098b155d7cda18590028c97be3c4e2914303e.jpg)  
(b)

Firstly, we provide an evaluation of the performance of KRWI (2-RWI) compared with KROI (2-ROI) and POI (0-ROI) approaches and lossless isomorphism. We keep the most abstracted signal infosets identified under the full abstraction setting. Note that POI is the common refinement of existing signal abstraction algorithms that only consider future information. And, since previous works cannot control the number of abstracted infoset, they cannot justify their performance in that considering historical information in signal abstraction was better than that in signal abstraction with the same number of abstracted infoset. To investigate this issue, we included KrwEmd and set the clustering scale to be consistent with POI. Note here, that 2-RWI and 2-ROI share the same capability of infoset recognition in Preflop and Flop, while POI is only a little bit worse than 2-RWI and 2-ROI in Flop. Thus, we can directly allow clustering of KrwEmd abstraction use the abstracted signal infosets identified by POI in Preflop and Flop, and only perform clustering in River. Here, we design four sets of hyper-parameters:  $(w_0, w_1, w_2)$ , i.e., exponentially decreasing:  $(16, 4, 1)$ , linearly decreasing:  $(7, 5, 3)$ , constant:  $(1, 1, 1)$ , and increasing:  $(3, 5, 7)$  in the importance of historical information. We only show the result of best- and worst-performing parameters (to make the figure neat). The full figures appear in the Appendix C. Figure 4a shows the result of symmetric abstraction, while Figure 4b shows the result of asymmetric abstraction. We observed that both symmetric and asymmetric abstractions maintained consistent abstraction performance without abstraction pathologies. As expected, overfitting was observed in the symmetric abstraction scenario while in the asymmetric scenario overfitting was significant only for POI. The performance difference between 2-RWI and 2-ROI is small, which means that under the full abstraction setting, using simple winrate-based features instead of complex outcome-based features can achieve nearly the same performance. Even with the worst parameter configuration (increasing importance), KrwEmd with the same number of abstracted signal infosets as POI still outperforms POI.

![](images/917ada1893ca358d5585dbe91578c40173a32b5f5c5ce7aee8dc7b1439bc824a.jpg)  
(a)

![](images/e8547dd4278a7c2b546c2fd1373cc7dad07d6d98caf9204d47a10847fb1d4e92.jpg)  
Figure 5: Performance comparison of KrwEmd versus other imperfect recall signal abstraction algorithms considering only future information, trained for  $3.7 \times 10^{10}$  iterations.  
(b)

Next, we compared the performance of KrwEmd with the currently applied signal abstraction algorithms Ehs and PaEmd. It should be noted that POI is the common refinement both for Ehs and PaEmd, meaning that the maximum number of abstracted signal infosets they can recognize will not exceed that of POI. Thus, we set a compression rate that is 10 times lower than that of POI, while not performing abstraction for Preflop. The final number of abstracted infosets is set to (100, 225, 396). To exclude the influence of random events on performance, we generated 3 sets of abstractions for Ehs and PaEmd each. KrwEmd used hyperparameters  $(w_{3,0}, w_{3,1}, w_{3,2}; w_{2,0}, w_{2,1})$  in Flop and River, which are exponentially decreasing (16, 4, 1; 4, 1), linearly decreasing (7, 5, 3; 5, 3), constant (1, 1, 1; 1, 1), and increasing (3, 5, 7; 5, 7) in the importance of historical information. Additionally, since PaEmd uses approximate EMD calculations, its approximate distance is asymmetric, making it difficult for the algorithm to converge. We truncated after 1000 iterations on a single core, with an average cost of 1427.7s, while Ehs and KrwEmd both achieved convergent clustering results, requiring an average of 12.3 and 96.7 iterations, with average time costs of 11.2s and 341.4s, respectively.

Figure 5a shows the results of symmetric abstraction experiments, while Figure 5b shows the results of asymmetric abstraction experiments. We observed that both symmetric and asymmetric abstractions maintained consistent abstraction performance, similar to the full abstraction scenario, without significant abstraction pathologies. The experimental results show that KrwEmd's performance is far superior to that of Ehs and PaEmd under all parameter settings. Our experiments also confirmed that, despite PaEmd's convergence issues, it is indeed a more effective abstraction algorithm than Ehs. Additionally, we further validated that the importance of historical information decreases progressively from bottom to top, although this time the best-performing parameter was exponentially decreasing rather than linearly decreasing as in the previous experiment.

These two experiments validate that considering historical information is indeed more effective than considering future information only in signal abstraction even in imperfect recall setting.

# 8 Conclusion

This research introduces the first imperfect recall signal abstraction algorithm that considers historical information. This algorithm has the ability to adjust the scale of the abstracted signal infosets. Based on this, we fully verified that the imperfect recall signal abstraction and abstraction algorithms considering historical information is superior to that only considering future information. Therefore, the KrwEmd algorithm has replaced the PaEmd algorithm and become the SOTA in this field. Based on the KrwEmd algorithm, we are expected to build a stronger Texas Hold'em AI.

# References

[1] David Abel. A theory of state abstraction for reinforcement learning. In AAAI conference on artificial intelligence, volume 33, pages 9876-9877, 2019.  
[2] David Abel, Nate Umbanhowar, Khimya Khetarpal, Dilip Arumugam, Doina Precup, and Michael Littman. Value preserving state-action abstractions. In International Conference on Artificial Intelligence and Statistics (AISTATS), pages 1639-1650, 2020.  
[3] David Arthur and Sergei Vassilvitskii. k-means++ the advantages of careful seeding. In ACM-SIAM symposium on Discrete algorithms (SODA), pages 1027-1035, 2007.  
[4] D Billings, N Burch, A Davidson, R Holte, J Schaeffer, T Schauenberg, and D Szafron. Approximating game-theoretic optimal strategies for full-scale poker. In International Joint Conference on Artificial Intelligence (IJCAI), volume 3, pages 661-668, 2003.  
[5] Nicolas Bonneel, Michiel van de Panne, Sylvain Paris, and Wolfgang Heidrich. Displacement Interpolation Using Lagrangian Mass Transport. ACM Transactions on Graphics (SIGGRAPH ASIA 2011), 30(6), 2011.  
[6] Noam Brown and Tuomas Sandholm. Regret transfer and parameter optimization. In AAAI Conference on Artificial Intelligence, volume 28, 2014.  
[7] Noam Brown and Tuomas Sandholm. Superhuman ai for heads-up no-limit poker: Libratus beats top professionals. Science, 359(6374):418-424, 2018.  
[8] Noam Brown and Tuomas Sandholm. Superhuman ai for multiplayer poker. Science, 365 (6456):885-890, 2019.  
[9] Noam Brown, Sam Ganzfried, and Tuomas Sandholm. Hierarchical abstraction, distributed equilibrium computation, and post-processing, with application to a champion no-limit texas hold'em agent. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 7-15, 2015.  
[10] Jiří Čermák, Branislav Bošansky, and Viliam Lisy. An algorithm for constructing and solving imperfect recall abstractions of large extensive-form games. In International Joint Conference on Artificial Intelligence (IJCAI), pages 936–942, 2017.  
[11] Jiri Čermák, Viliam Lisý, and Branislav Bošanský. Automated construction of bounded-loss imperfect-recall abstractions in extensive-form games. Artificial Intelligence, 282:103248, 2020.  
[12] Yanchang Fu, Junge Zhang, Dongdong Bai, Lingyun Zhao, Jialu Song, and Kaiqi Huang. Expanding the resolution boundary of outcome-based imperfect-recall abstraction in games with ordered signals. arXiv preprint arXiv:2403.11486, 2024.  
[13] Sam Ganzfried and Tuomas Sandholm. Action translation in extensive-form games with large action spaces: axioms, paradoxes, and the pseudo-harmonic mapping. In International Joint Conference on Artificial Intelligence (IJCAI), pages 120-128, 2013.  
[14] Sam Ganzfried and Tuomas Sandholm. Potential-aware imperfect-recall abstraction with earth mover's distance in imperfect-information games. In AAAI Conference on Artificial Intelligence, volume 28, 2014.  
[15] Andrew Gilpin and Thomas Sandholm. Expectation-based versus potential-aware automated abstraction in imperfect information games: An experimental comparison using poker. In National Conference on Artificial Intelligence (NCAI), volume 3, pages 1454-1457, 2008.  
[16] Andrew Gilpin and Tuomas Sandholm. A competitive texas hold'em poker player via automated abstraction and real-time equilibrium computation. In National Conference on Artificial Intelligence (NCAI), volume 21, page 1007. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2006.

[17] Andrew Gilpin and Tuomas Sandholm. Better automated abstraction techniques for imperfect information games, with application to texas hold'em poker. In International Joint Conference on Artificial Intelligence (IJCAI), pages 1-8, 2007.  
[18] Andrew Gilpin and Tuomas Sandholm. Lossless abstraction of imperfect information games. Journal of the ACM (JACM), 54(5):25-es, 2007.  
[19] Andrew Gilpin, Tuomas Sandholm, and Troels Bjerre Sorensen. Potential-aware automated abstraction of sequential games, and holistic equilibrium analysis of texas hold'em poker. In National Conference on Artificial Intelligence (NCAI), volume 22, page 50. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2007.  
[20] Michael Johanson, Neil Burch, Richard Valenzano, and Michael Bowling. Evaluating state-space abstractions in extensive-form games. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 271-278, 2013.  
[21] Christian Kroer and Tuomas Sandholm. Discretization of continuous action spaces in extensive-form games. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 47-56, 2015.  
[22] Marc Lanctot, Kevin Waugh, Martin Zinkevich, and Michael Bowling. Monte carlo sampling for regret minimization in extensive games. International Conference on Neural Information Processing Systems (NeurIPS), 22, 2009.  
[23] Matej Moravčík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, and Michael Bowling. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356(6337):508-513, 2017.  
[24] Yossi Rubner, Carlo Tomasi, and Leonidas J Guibas. The earth mover's distance as a metric for image retrieval. International journal of computer vision, 40:99-121, 2000.  
[25] Jiefu Shi and Michael L Littman. Abstraction methods for game theoretic poker. In Computers and Games: Second International Conference, CG 2000 Hamamatsu, Japan, October 26-28, 2000 Revised Papers 2, pages 333-345. Springer, 2001.  
[26] Finnegan Southey, Michael Bowling, Bryce Larson, Carmelo Piccione, Neil Burch, Darse Billings, and Chris Rayner. Bayes' bluff: opponent modelling in poker. In Proceedings of the Twenty-First Conference on Uncertainty in Artificial Intelligence, pages 550–558, 2005.  
[27] Kevin Waugh. A fast and optimal hand isomorphism algorithm. In AAAI Workshop on Computer Poker and Incomplete Information, 2013.  
[28] Kevin Waugh, David Schnizlein, Michael Bowling, and Duane Szafron. Abstraction pathologies in extensive games. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS), volume 2, pages 781-788, 2009.  
[29] Kevin Waugh, Martin Zinkevich, Michael Johanson, Morgan Kan, David Schnizlein, and Michael Bowling. A practical use of imperfect recall. In Symposium on Abstraction, Reformulation and Approximation (SARA), 01 2009.  
[30] Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. In International Conference on Neural Information Processing Systems (NeurIPS), pages 1729-1736, 2007.

Algorithm A1 Potential Winrate Isomorphism  
Require:  $r = 1,\dots ,R$  . Phases.  $\Theta_{i} = \bigcup_{r = 1}^{R}\Theta_{i}^{(r)}$  . Signal infos set space for player i. Index  $\boldsymbol { \mathfrak { r } } ( \boldsymbol { \mathfrak { r } } , \cdot ) : \boldsymbol { \mathfrak { O } } _ { i } ^ { ( r ) } \mapsto \mathbb { N }$  . Signal [+oset index function for player i.   
1: procedure POTENTIALWINRATEISOMORPHISM(0i)   
2: for  $r = R$  to 1 do   
3: if  $r = = R$  then   
4: FEATUREFUNC  $\leftarrow$  POTENTIALWINRATEFEATURELASTPHASE(·).   
5: else   
6: FEATUREFUNC  $\leftarrow$  POTENTIALWINRATEFEATURE(·,  $r,\mathcal{P}\mathcal{C}_i^{(r + 1)},\mathcal{P}\mathcal{D}_i^{(r + 1)}$    
7: end if   
8:  $(\mathcal{PC}_i^{(r)},\mathcal{PD}_i^{(r)})\gets$  ISOMORPHISMCONSTRUCTOR(r,  $\Theta_i^{(r)}$  , FEATUREFUNC).   
9: end for   
10: return  $(\mathcal{PC}_i^{(1)},\mathcal{PD}_i^{(1)}),\ldots ,(\mathcal{PC}_i^{(R)},\mathcal{PD}_i^{(R)})$    
11: end procedure   
12: procedure POTENTIALWINRATESFEATURELASTPHASE(v)   
13: return  $p f_{i}^{(R)}(\vartheta)$ $\triangleright$  compute according Equation (1)   
14: end procedure   
15: procedure POTENTIALWINRATEFEATURE(v,  $r,\mathcal{P}\mathcal{C}_i^{(r + 1)},\mathcal{P}\mathcal{D}_i^{(r + 1)})$    
16: feature  $\vartheta \gets$  zero array with length  $N + 1$    
17: for  $\vartheta^{\prime}\in \Theta_{i}^{(r + 1)}$  , such that  $\exists \theta^{\prime}\in \vartheta^{\prime},\exists \theta \in \vartheta :\varsigma (\theta^{\prime}|\theta) > 0$  do   
18: idx  $\leftarrow$  Indexi  $(r + 1,\vartheta ')$  abs  $\leftarrow$  PD  $(r + 1)[idx]$  ,featureov'  $\leftarrow$  PC(r+1)[abs].   
19: for  $j = 0$  to  $N$  do   
20: feature[  $j]\gets$  feature[  $j] +$  feature[  $j]\Pr \{\vartheta^{\prime}|\vartheta \}$    
21: end for   
22: end for   
23: end procedure
