# Mini AI Project: Search Algorithms

This project aims to develop a navigational search algorithm using a family of uninformed and informed algorithms to find the shortest path between two tourist destinations using a simplified real-world problem.   Key objectives of this project include formulating the search problem, testing the different algorithms for navigational search context, and recommending a suitable algorithm for a given problem. 

Using the data of Australia's top sixteen tourist destinations, this project implemented and compared breadth-first search, greedy best-first search, and A* (star) search algorithms. 

The results showed the advantages and limitations of these algorithms. First, the breadth-first search required significant memory and time to process. Secondly, the greedy best-first search is the fastest and most efficient algorithm regarding computation time and memory usage. However, it could not find the optimal solution. Finally, A* search combines the advantages of breadth-first and greedy best-first search. It was the only algorithm that found the optimal solution in our case. 

A* search can be used for a complex graph to find an optimal solution. However, we must consider choosing the correct heuristic design and memory issue. The proper heuristic function is crucial for greedy best-first and A* search.

## 1. Introduction
Classic search algorithms, such as breadth-first search (BFS), depth-first search (DFS), Dijkstra’s algorithm or uniform cost search, greedy best-first search and A* search algorithm, have been successfully applied in various domains. Uninformed and informed search techniques have been developed since 1957, when Newell and other scholars started working in this field. The breadth-first search was established by Moore (1959), Simon and Newell (1958) first developed the heuristic functions, and the A* technique was introduced by Hart et al. (1968). These papers are fundamental to modern search techniques. In this report, the uninformed and informed family of search algorithms, namely, breadth-first search, greedy-best first search and A* search algorithms, will be implemented to solve the navigational search problem in Australia's tourism industry.
##### Problem
Finding the shortest path between cities is complex and time-consuming for tourists without an intelligent application. The search algorithms help tourists to find an optimal route from the start destination to the goal destination. 
This report aims to harness the power of the search algorithms to solve the simplified real-world problem of efficient travel routes, considering various constraints and preferences.
The significance of this report lies in its potential to revolutionise how tourists plan and experience their trips. By leveraging classic search algorithms and modern data-driven techniques, implementing the algorithm will provide tourists with personalised, cost-efficient decision-making on their travel.
##### Data sources
To implement the solution, collected and prepared data using Australia’s top tourist destinations and sightseeing. Australia’s top sixteen tourist destinations were adapted from the Ytravel Blog. Then, the coordinates of each location were collected using Google Maps. A more detailed explanation of data pre-processing will be discussed in section 2. 
##### Navigational search problem formulation
The search problem is formulated in this report: The agent starts from Freycinet National Park and aims to reach Kakadu National Park in the northern territory. The search algorithm will find the shortest path between the start location and the goal destination. 
 
Components of the search problem (Russell & Norvig, 2021):
- 	**The state space.** State space is a set of possible states agents can travel in the environment. The states are shown below in Figure 1. Location X and Location Y define the Longitude and Latitude, respectively. 
- **Initial state.** In this problem, the initial or start state is Freycinet National Park in Tasmania.
S = FreycinetPark
- **Goal state.** The agent’s goal is to reach the Kakadu National Park in the Northern Territory. 
G = Kakadu
- **Actions.** Actions available to the agent can be seen in Figure 2. When provided with a state denoted as "s," the function ACTIONS(s) provides a set of actions that can be carried out within that state. For instance, actions can be executed in the state FreicynetPark is:
ACTION(s)= ACTIONS(FreycinetPark) = { ToMelbourne,ToHobart,ToSydney}
- **Transition model.** The transition model returns the state, which results from an action a.
RESULT(s,a)=RESULT(FreycinetPark,ToMelbourne) = Melbourne
- **Action cost function.** Action costs are the numeric costs of performing action a in state s to reach state s’. In this report, the action costs between the states are computed using Euclidean distance. Figure 2 shows the associated path costs between states in kilometres. 
ACTION-COST(s,a,s’) = ACTION-COST(FreycinetPark,ToMelbourne,Melbourne) = 532 

|№| Location Name Full	| Location Name	| Location X	| Location Y|
|---|-----------------------|---------------|---------------|-----------|
1| Whitsunday Islands, Queensland	| Whitsunday	|149.0	|-20.2|
2|Kakadu National Park, Northern Territory|	Kakadu	|132.5	|-11.1
3|	Margaret River, Western Australia	|MargaretRiver|	113.9	|-27.8
4	|Sydney, New South Wales	|Sydney|	151.3	|-32.2
5	|Broome, Western Australia|	Broome	|122.3	|-16.1
6	|Freycinet National Park, Tasmania|	FreycinetPark	|150.5|	-38.8
7	|Uluru, Northern Territory|	Uluru	|131.0	|-25.3
8	|Daintree Rainforest, Queensland	|Daintree	|145.4	|-16.2
9	|Fraser Island, Queensland	|FraserIsland|	153.4|	-25.0
10|Melbourne, Victoria|	Melbourne|	145.1|	-37.9
11|	Rottnest Island, Perth, Western Australia|	Perth	|116.0|	-25.3
12	|Hobart, Tasmania	|Hobart	|147.3	|-42.8
13	|Barossa Valley, Adelaide, South Australia|	Adelaide	|139.3|	-33.0
14	|Karijini National Park, Western Australia	|KarijiniPark|	118.3|	-22.7
15|	Gold Coast, Queensland|	GoldCoast|	153.4|	-28.0
16	|Kangaroo Island, South Australia|	KangarooIsland|	137.2|	-35.8

*Figure 1. State Space of Australia’s tourist destinations*

Graphical representation of states, actions, and action costs are shown in Figure 2.

![img](https://i.imgur.com/M7eF8bi.png)
*Figure 2. A simplified map of Australia’s top tourist destinations, with distances in kilometres (km).*

**A simplified formulation**

To avoid unnecessary complexity, simplified the problem as follows:
-	The given network is connected by road and air. 
-	The given network is undirected.
-	The distance between locations is computed using the Euclidean distance. The cost of travel between two locations equals to this distance. 
-	To simplify the problem, some actual links were removed from the network.
Using the above search problem and simplified formulation, this report will implement and compare one blind search – Breadth-first search, and the two informed search techniques, greedy best-first search and the A* search algorithm. 

In the subsequent sections of this report, we will delve deeper into AI techniques, data pre-processing, and algorithm implementations to demonstrate how uninformed and informed search algorithms solve the problem. 

## 2. Solution
### 2.1. Methods/AI techniques used
This section explains the AI techniques used to solve the search problem defined in the introduction section. 

#### 2.1.1. Uninformed (Blind) search
Uninformed or Blind search algorithms do not know how far a current state is from the goal state. In our example, our agent in Freycinet Park had no information about the Australian map or whether to go to Hobart, Melbourne, or Sydney. 

**Breadth-first search (BFS)**

Breadth-first search works systematically. This entails initially expanding the root node, then expanding all its immediate successor nodes, continuing this pattern for their respective successors, and so forth (Moore,1959). Figure 3 shows the progress of the breadth-first search tree in our Tourism problem.

![img](https://i.imgur.com/ekbfznR.png)

*Figure 3.  Progress of a Breadth-first search tree from start state Freycinet Park to goal state Kakadu.*

F – Freycinet Park, H – Hobart, M – Melbourne, S – Sydney, A – Adelaide, 
U – Uluru, G – Gold Coast, P – Perth, B – Broome, K – Kakadu, FI – Fraser Island.

#### 2.1.2.	Informed (Heuristic) search

Unlike an uninformed search strategy, an informed search relies on specific clues about the goal's location (Russell & Norvig, 2021). These clues are provided through a heuristic function of node n represented as h(n):

h(n) = estimated cost of the cheapest path from the state at node n to a goal state.

This cheapest path equals the straight-line distances (h_SLD) from the current state to the goal state. 

h(n)= h_SLD

In our problem, straight-line distances from the current state to the goal (Kakadu) are shown in Figure 4. 

|City|h_SLD |City|h_SLD|
|-----------|----|--------------|-----|
|Whitsunday	|2092|	FraserIsland|	2786|
Kakadu	|0	|Melbourne|	3287|
MargaretRiver|	2775	|Perth	|2416|
Sydney	|3173|	Hobart	|3883|
Broome	|1261	|Adelaide	|2545|
FreycinetPark|	3703|	KarijiniPark|	2035|
Uluru	|1585	|GoldCoast|	2983|
Daintree| 1540	|KangarooIsland|	2791|

*Figure 4. Values of h_SLD—straight-line distances to Kakadu in kilometres*

The greedy best-first search and A* search are two main informed search algorithms. The following sections will explain the idea and difference between these two algorithms. 