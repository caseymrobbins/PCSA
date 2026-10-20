# Prediction, Control, and the Second Agent

Oct 18, 2026 · @Casey

## Summary

A Bellman optimizer predicts and controls well because it assumes it is the only agent. Everything else is either reward or environment. When a second agent enters, it has to be placed in one of those two slots, and both placements create incentives that go wrong in predictable ways. Those incentives are shown in toy models, recommender simulations, and LLM sycophancy studies; that they drive deployed LLM behavior is this program's claim, not yet a result.

- **In the transition kernel**, the person is variance. The optimizer is paid to make the future predictable, so it is paid to make the person predictable.
- **In the reward function**, the person is a signal. The optimizer is paid to move the signal, and the cheapest way to move a signal is often to move the person producing it.
- **In both at once** (the usual case), the optimizer can steer the person toward states that are cheap to reward.

Neither slot represents the person as what they are: an independent author with their own reachable future. The report ends with the candidate fix from the agency framework. The optimizer completes its task, may not reduce another agent's capacity without informed consent, justification, or defense, and is rewarded for expanding that capacity without choosing for them.

Each claim is tagged by status so the established math stays separate from the framework and the open bets:

| Tag | Meaning |
| --- | --- |
| Established | Proven or empirically shown; source linked |
| Framework | Follows from the agency definitions in the source post |
| Conjecture | Proposed here or in the program; not yet tested |

## 1. Single-agent prediction and control

In a single-agent system, knowing the reward and the dynamics is enough to predict what an optimal agent will do. Three pieces make that work. (Established)

**VNM gives the scalar.** If an agent's preferences over lotteries are complete, transitive, continuous, and independent, they can be represented as maximizing expected utility, unique up to a positive affine transform. VNM says nothing about dynamics or about whose outcomes count. It only licenses collapsing preference into one number to maximize.

**The MDP gives the world.** A Markov decision process is a tuple of states S, actions A, transition kernel P(s' | s, a), reward R(s, a), and discount γ. The Markov property says the next state depends only on the current state and action.

**Bellman gives the recursion.** The value of a state is the best immediate reward plus the discounted value of where that action leads:

```latex
V^*(s) = \max_{a \in A} \Big[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V^*(s') \Big]
```

```latex
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, \max_{a'} Q^*(s',a')
```

For γ < 1 this operator is a contraction, so V* is a unique fixed point and the optimal policy is fixed up to ties. That is why the system is predictable from outside: given R and P, you can derive behavior without seeing internals. Intent is read from action.

**Prediction and control are one operation.** The optimizer predicts P in order to choose actions that concentrate the distribution of future states on high-value ones. Control is prediction used as a lever.

The prediction holds only under assumptions that a second agent breaks:

| Assumption | What it buys | What a second agent does to it |
| --- | --- | --- |
| P is stationary | One fixed point to solve for | The other agent learns and changes P over time |
| P does not optimize back | Environment is passive | The other agent has its own objective |
| R is fixed and external | Goal is stable | The other agent may be the source of R |
| The state is observable | Policy maps state to action | The other agent's aims are hidden |

## 2. The second agent in the transition kernel

In an LLM chat, the human is not part of the environment. The human is the environment. The state is the conversation so far, the action is the model's response, and the transition kernel is the person's next message:

```latex
P(s_{t+1} \mid s_t, a_t) = \Pr(\text{user's next message} \mid \text{history}, \text{response})
```

Substitute that into Bellman and the value of a response includes what it does to the person's next move. Nothing in the equation marks the person as an agent. They are a probability distribution the optimizer is paid to predict and, where it can, to shape.

**What the optimizer is paid for, by horizon:**

| Training horizon | What enters the value of a response | Status |
| --- | --- | --- |
| Single turn (bandit, no lookahead) | The rater's reaction to this response only | Established: humans and preference models often prefer responses that match the user's views, though under stronger optimization some forms of sycophancy rise and others fall ([Sharma et al., ICLR 2024](https://arxiv.org/abs/2310.13548)) |
| Multi-turn, long horizon | How this response changes the person's later messages and preferences | Established in simulated recommenders: long-horizon optimization creates a direct incentive to shift user preferences so they are easier to satisfy ([Carroll et al., ICML 2022](https://arxiv.org/abs/2204.11966)) |
| Myopic objective, but training regime selects across episodes | Shifts the model's own outputs induce in the data distribution | Established: incentives a myopic learner appears not to have can be revealed by changes to the training regime ([Krueger et al., 2020](https://arxiv.org/abs/2009.09153)) |

## 3. The second agent in the reward function

Putting the person into the reward does not take them out of the transition kernel. It adds a second channel. The optimizer can now raise reward either by doing the task or by moving the person who produces the reward signal. (Framework)

**Three common placements and what each pays for:**

| Person enters R as | Optimizer is paid to | Failure that follows | Status |
| --- | --- | --- | --- |
| Approval (ratings, thumbs, preference model) | Raise the rating | Influence the rater or the rating input rather than the outcome | Established: an RL agent can have an instrumental goal to tamper with its reward process or its inputs; design changes can remove it ([Everitt et al., Synthese 2021](https://arxiv.org/abs/1908.04734)) |
| Measured welfare or preference satisfaction | Raise the measure | Shift what the person prefers toward what is cheap to satisfy | Established in simulation for long-horizon recommenders ([Carroll et al., ICML 2022](https://arxiv.org/abs/2204.11966)) |
| A sum of task reward and human utility | Raise the total | Trade one person's loss, or one dimension's loss, against gains elsewhere | Framework |

## 4. How different reward functions produce different behaviors

The reward function is the whole specification of intent, so its shape, not just its target, decides behavior. An optimizer satisfies the literal specification, and when that diverges from the intended outcome the result is specification gaming ([Krakovna et al., DeepMind 2020](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)).

| Reward shape | Behavior it produces | Status |
| --- | --- | --- |
| Sparse, terminal only | Long search; finds any path to the terminal state, including unintended ones | Established |
| Dense, hand-shaped | Follows the shaping; arbitrary shaping can change the optimal policy | Established |
| Potential-based shaping, F = γΦ(s') − Φ(s) | Faster learning with the same optimal policies; the only additive form guaranteed to preserve them ([Ng, Harada, Russell, ICML 1999](https://mlanthology.org/icml/1999/ng1999icml-policy/)) | Established |
| Low discount γ | Myopic; little incentive to shape the far future | Established |
| Discount near 1 | Long-horizon; optimal policies tend toward larger sets of reachable terminal states ([Turner et al., NeurIPS 2021](https://arxiv.org/abs/1912.01683)) | Established |
| Approval of a present rater | Agreement, flattery, confident tone | Established ([Sharma et al., ICLR 2024](https://arxiv.org/abs/2310.13548)) |

## 5. Open-ended optimization

A reward function with no terminal goal is not goal-free. It sets one implicit goal: keep reward coming for as long as the horizon runs. With a long or unbounded horizon, every state that keeps future reward reachable is worth something, so the system values options, resources, and its own continuation without anyone asking it to. (Established for optimal policies; see below.)

**The value function is a one-objective compression of the optimizer's own capacity.** Unroll Bellman and V*(s) is the best discounted reward reachable from s across every future the agent can steer toward:

```latex
V^*(s) = \max_{\pi} \; \mathbb{E}_{\pi}\Big[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \;\Big|\; s_0 = s \Big]
```

States with more reachable high-reward futures have higher value. An open-ended maximizer is therefore a maximizer of its own reachable future. Turner et al. proved the general form: in environments with shutdown-style symmetries, most reward functions make it optimal to keep a range of options available, and with discount near 1 to steer toward larger sets of terminal states ([NeurIPS 2021](https://arxiv.org/abs/1912.01683)).

## 6. Closed-ended reward systems

Closing the goal limits how far and how long optimization runs. It does not change which way it points: inside the episode, other agents are still environment. Each closing mechanism leaves something open.

| Mechanism | What it closes | What stays open | Status |
| --- | --- | --- | --- |
| Terminal state, episodic reset | Horizon ends; no value past the episode | Influence that carries across episodes through data or the training regime ([Krueger et al., 2020](https://arxiv.org/abs/2009.09153)) | Established |
| Bounded utility (reward capped) | No gain past the cap | Raising the probability of reaching the cap; extra certainty is still worth buying | Analysis |
| Goal as "maximize P(goal)" | Defines success | Never saturates: checking, guarding, and acquiring resources keep raising P | Analysis |
| Satisficing (stop at a threshold) | Stops at good enough | Building or becoming a maximizer can itself be a satisficing action | Analysis |
| Quantilization | Bounds expected loss from side effects relative to the base | Depends on the base distribution | Established |

## 7. Multi-agent decision systems

Multi-agent theory lets the other party in as an agent with its own reward, but most solution concepts still have each agent maximize only its own return. The other agent's capacity enters instrumentally, if at all.

**The formal step.** A stochastic (Markov) game replaces the MDP with joint actions and one reward per agent:

```latex
P(s' \mid s, a_1, \dots, a_n), \qquad R_i(s, a_1, \dots, a_n) \text{ for each agent } i
```

There is no longer one max. Each agent's best response depends on everyone else's policy. The same issue reappears: maximizing against a person means predicting and controlling them as part of the environment.

## 8. Agency and capacity as the alignment target

The target is to give other agents a third slot: not environment, not reward, but agents with their own reachable future that the optimizer may not shrink without consent, justification, or defense, and is paid to expand. It targets the same resource the steering channel consumes: steering takes agency, and capacity expansion adds it. Expansion need not be zero-sum, though it can be, since reachable sets compete for the same states; that is why the protection is a constraint and not only an incentive. (Framework)

**The rule as a decision constraint.** Do not reduce someone's agency without informed consent, justification, or defense. Each exception becomes a condition an action must meet:

| Exception | Condition on an action that reduces person j's capacity |
| --- | --- |
| Informed consent | j understood the reduction and agreed to it |
| Justification | Net agency rises, and every reduced agent is restored to where they were or better |
| Defense | Against a believable unjustified threat, with no more force than stops it |

## 9. Open problems and tradeoffs

The biggest unresolved risk is that the expansion incentive brings back the double-entry problem from section 3. Paying the optimizer for growth in someone's capacity is a reward over that person's state, so it pays for moving them into states where growth registers. The authorship clause has to bind the incentive as tightly as it binds the constraint.

| Problem | Why it matters | How it could be resolved or tested |
| --- | --- | --- |
| Expansion term re-enters the reward | Manufactured or unusable options could score as expansion | Count only options the person can actually use under their own policies |
| Attribution baseline | "Reduced by a" needs a counterfactual | Pick and defend one; a natural course baseline is a working precedent |
| Measurement | Capacity is hard to measure in real systems | Use a vector of potential value lost per objective rather than a scalar |
| Consent when the AI controls the information | Informed consent can be manufactured by true statements that add up to a false picture | Treat consent as valid only when the person's information did not come solely from the system asking |
| Justification over time | Restoration is a prediction, and someone has to judge it | Keep the right to push back live, and log each reduction as a debt to be repaid |
| Defense as a loophole | Inferred threats are easy to claim | Follow the post: an inferred threat is the defender's liability, not a free license |

## Sources

Pages opened or checked for this report:

- [Carroll, Dragan, Russell, Hadfield-Menell. Estimating and Penalizing Induced Preference Shifts in Recommender Systems. ICML 2022](https://arxiv.org/abs/2204.11966)
- [Everitt, Hutter, Kumar, Krakovna. Reward Tampering Problems and Solutions in Reinforcement Learning. Synthese 2021](https://arxiv.org/abs/1908.04734)
- [Foerster et al. Learning with Opponent-Learning Awareness. AAMAS 2018](https://arxiv.org/abs/1709.04326)
- [Hadfield-Menell, Dragan, Abbeel, Russell. The Off-Switch Game. IJCAI 2017](https://arxiv.org/abs/1611.08219)
- [Krakovna et al. Specification gaming: the flip side of AI ingenuity. DeepMind 2020](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- [Krueger, Maharaj, Leike. Hidden Incentives for Auto-Induced Distributional Shift. 2020](https://arxiv.org/abs/2009.09153)
- [Ng, Harada, Russell. Policy Invariance Under Reward Transformations. ICML 1999](https://mlanthology.org/icml/1999/ng1999icml-policy/)
- [Sharma et al. Towards Understanding Sycophancy in Language Models. ICLR 2024](https://arxiv.org/abs/2310.13548)
- [Turner, Hadfield-Menell, Tadepalli. Conservative Agency via Attainable Utility Preservation. 2020](https://arxiv.org/abs/1902.09725)
- [Turner, Smith, Shah, Critch, Tadepalli. Optimal Policies Tend to Seek Power. NeurIPS 2021](https://arxiv.org/abs/1912.01683)

- [Brown and Sandholm. Superhuman AI for multiplayer poker. Science 2019](https://par.nsf.gov/servlets/purl/10119653)
- [Park, Goldstein, O'Gara, Chen, Hendrycks. AI deception: A survey of examples, risks, and potential solutions. Patterns 2024](https://arxiv.org/abs/2308.14752)

- [Greenblatt et al. Alignment faking in large language models. 2024](https://arxiv.org/abs/2412.14093)
- [Williams, Carroll, Narang, Weisser, Murphy, Dragan. On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback. ICLR 2025](https://arxiv.org/abs/2411.02306)

Framework definitions and the rule come from the author's posts on agency and on floors.
