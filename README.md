
This is a project to verifying the robustness of DRL algorithm of DQN, D3QN and PER-D3QN for the dynamic traffic situation, like lane blockage and drasmatic flow increase.

#  Adaptive Traffic Signal Control Using DQN, D3QN and PER-D3QN  
*A Comparison of Deep Reinforcement Learning Algorithms Under Sudden Traffic Changes*

This repository contains the implementation and experiment framework for comparing three Deep Reinforcement Learning (DRL) algorithms in adaptive traffic signal control:

- **DQN (Deep Q-Network)**
- **D3QN (Double Dueling DQN)**
- **PER-D3QN (D3QN with Prioritized Experience Replay)**

The project evaluates their adaptability under sudden traffic condition changes such as:
-  Lane blockage
-  Sudden traffic-flow increase
-  Combined disturbances

Implementation is based on **Python + SUMO**, following the methodology of the related study.

---

##  Features

###  Single-Intersection SUMO Environment
- Four-way intersection with straight/left/right lanes
- 80-dimensional DTSE state representation

###  Discrete Action Space
Each action corresponds to a traffic signal phase:
| Action | Phase | Description |
|--------|--------|--------------|
| 0 | NS through/right | North–South straight + right |
| 1 | NS left | North–South left-turn |
| 2 | EW through/right | East–West straight + right |
| 3 | EW left | East–West left-turn |

###  Reward Mechanism
- Base reward: change in cumulative waiting time  
- D3QN & PER-D3QN also use:
  - Congestion penalty  
  - Throughput reward  

---

##  Algorithms

### **DQN**
- Fully connected neural network  
- Sensitive to Q-value overestimation  

### **D3QN**
- Double Q-Learning (reduces overestimation)  
- Dueling architecture (better state-action distinction)  

### **PER-D3QN**
- Prioritized Experience Replay (TD-error based)  
- Fastest convergence  

---


