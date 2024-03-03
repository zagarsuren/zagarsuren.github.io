# Mini AI Project: Search Algorithms

This project aims to develop a navigational search algorithm using a family of uninformed and informed algorithms to find the shortest path between two tourist destinations using a simplified real-world problem.   Key objectives of this project include formulating the search problem, testing the different algorithms for navigational search context, and recommending a suitable algorithm for a given problem. 

Using the data of Australia's top sixteen tourist destinations, this project implemented and compared breadth-first search, greedy best-first search, and A* (star) search algorithms. 

The results showed the advantages and limitations of these algorithms. First, the breadth-first search required significant memory and time to process. Secondly, the greedy best-first search is the fastest and most efficient algorithm regarding computation time and memory usage. However, it could not find the optimal solution. Finally, A* search combines the advantages of breadth-first and greedy best-first search. It was the only algorithm that found the optimal solution in our case. 

A* search can be used for a complex graph to find an optimal solution. However, we must consider choosing the correct heuristic design and memory issue. The proper heuristic function is crucial for greedy best-first and A* search.

## 1. Introduction
Classic search algorithms, such as breadth-first search (BFS), depth-first search (DFS), Dijkstra’s algorithm or uniform cost search, greedy best-first search and A* search algorithm, have been successfully applied in various domains. Uninformed and informed search techniques have been developed since 1957, when Newell and other scholars started working in this field. The breadth-first search was established by Moore (1959), Simon and Newell (1958) first developed the heuristic functions, and the A* technique was introduced by Hart et al. (1968). These papers are fundamental to modern search techniques. In this report, the uninformed and informed family of search algorithms, namely, breadth-first search, greedy-best first search and A* search algorithms, will be implemented to solve the navigational search problem in Australia's tourism industry.

**A Problem.**
Finding the shortest path between cities is complex and time-consuming for tourists without an intelligent application. The search algorithms help tourists to find an optimal route from the start destination to the goal destination. 
This report aims to harness the power of the search algorithms to solve the simplified real-world problem of efficient travel routes, considering various constraints and preferences.
The significance of this report lies in its potential to revolutionise how tourists plan and experience their trips. By leveraging classic search algorithms and modern data-driven techniques, implementing the algorithm will provide tourists with personalised, cost-efficient decision-making on their travel.

**Data sources.**
To implement the solution, collected and prepared data using Australia’s top tourist destinations and sightseeing. Australia’s top sixteen tourist destinations were adapted from the Ytravel Blog. Then, the coordinates of each location were collected using Google Maps. A more detailed explanation of data pre-processing will be discussed in section 2. 

**Navigational search problem formulation.**
The search problem is formulated in this report: The agent starts from Freycinet National Park and aims to reach Kakadu National Park in the northern territory. The search algorithm will find the shortest path between the start location and the goal destination. 
 
Components of the search problem (Russell & Norvig, 2021):
- **The state space.** State space is a set of possible states agents can travel in the environment. The states are shown below in Figure 1. Location X and Location Y define the Longitude and Latitude, respectively. 
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

**Greedy best-first search (GBFS)**

A Greedy best-first search algorithm utilises the evaluation function of f(n) = h(n). This makes the algorithm quicker to reach the goal (Barr & Feigenbaum, 1981).

Key points of greedy best-first search algorithm (Adapted from Lecture 2):
	Expands first the node with the lowest h(n) value.
	The evaluation function f(n) = h(n)
	In our problem, h(n) is defined as the straight-line distance between the current node to a goal node, h(n)= h_SLD  

Figure 5 illustrates the stages of the greedy best-first search algorithm to find a path from FreycinetPark to Kakadu using the  h_SLD, as shown in Figure 5. According to the heuristic, the first node to be expanded from FreycinetPark is Sydney, because it is the closest node to the goal node compared to Melbourne and Hobart. h_SLD (Sydney)=3173.  The next node will be Uluru as its heuristic is the closest among other nodes h_SLD (Uluru)=1585. Then Uluru generates the goal node Kakadu. As seen the calculation and result from the greedy best-first search, even though it finds solution faster, the cost of the solution is not optimal.

![Imgur](https://i.imgur.com/oRyp2CL.png)
*Figure 5. Stages in a greedy best-first search for Kakadu with the straight-line distance heuristic hSLD.*

**A-Star search**

A* search is the most common informed search algorithm. The difference between the A* search and greedy best-first search algorithm is that it uses the evaluation function of f(n) = g(n) + h(n), where g(n) is the path cost from the start state to node n (Hart et al., 1968).

Key points of A* search algorithm (Adapted from Lecture 2):
- A form of best-first search with the evaluation function f(n) = g(n) + h(n)
- g(n) is the path cost from the start state to node n
- h(n) is the estimated cost of the shortest path from n to a goal state.
- f(n) = estimated cost of a path that begins from the start state to n and continues from n to a goal.

Figure 6 shows the stages of the A* search. Nodes are labelled with f(n) = g(n) + h(n). The h values are the straight-line distances to Kakadu taken from Figure 4. 

![Imgur](https://i.imgur.com/7xLBLHf.png)
*Figure 6. Stages in the A-Star search for Kakadu.*

The first node to be expanded from Freycinet Park is Melbourne because according to the evaluation function f(n)= g(n)+ h(n)=532+3287=3819, which is the lowest compared to Sydney (f =4040) and Hobart (f =4296). Similar to this step, A* search found the path – {FreycinetPark, Melbourne, Adelaide, Uluru, Kakadu}. The associated cost of this path equals 4214. In our problem, A* search found the solution with the optimal cost. 
The following section will explain the implementation of breadth-first, greedy best-first, and A* search algorithms. 


### 2.2. Implementation of AI techniques
To solve the defined problem and implement the search techniques, conducted data collection and pre-processing. After preparing the data, created an environment on Google Colab and prepared the required modules and functions using Lab 2 materials. Finally, run the search algorithms on the problem data and compare the results.

#### 2.2.1. Data collection and pre-processing
The top tourist destinations in Australia are adapted from the Ytravel Blog. To simplify the problem, the Great Barrier Reef, Byron Bay NSW, Wilsons Promontory VIC, and the Great Ocean Road were removed from the locations used. In total, 16 locations will be used to construct the search space. Then, each coordinate, LocationX and LocationY, are collected using Google Maps. The coordinates of the locations are shown in Figure 1.
The function `define_map` is adapted from Lab to calculate the Euclidean distance between nodes. The distance is based on the degree of latitude and longitude. To understand the real-world scenario, converted the degree of latitude and longitude to kilometres, multiplying by 111. According to the National Geographic Society, one degree of latitude, called an arc degree, covers about 111 kilometres.  In line with “Figure 2 – The simplified map of Australia’s top tourist destinations”, a new dataset called `destinations_correlation_matrix.csv` was created and displayed in the figure below. This data will be used to create the graph data and perform search algorithms. 

![Imgur](https://i.imgur.com/f1RYc8L.png)
*Figure 7. Correlation matrix between locations (destinations_correlation_matrix.csv)*

After finishing the data pre-processing, we have the following two datasets to perform the search algorithm. 
- data_coordinates.csv
- destinations_correlation_matrix.csv

#### 2.2.2.	Classes and methods
- BFS.py – Implementation of breadth-first search algorithm.
- BFS_for_graph.py – Visualisation of Breadth-first search algorithm.
- BestFirstSearch.py – Implementation of best first search algorithms.
- BestFirstSearch_for_graph.py – Visualisation of best-first search algorithms.
- graphSource.py – Define graph and display result.
- utils.py – Utility functions. 

Some of the classes and methods to implement the solutions are explained below. 

Function define_graph. 
- Input: This function gets the location and neighbour as input and creates an undirected graph, configures node colours, node positions, node label positions and edge weights. 
- Output: It creates graph data using these attributes and returns graph data and graph. The graph data and graph will be used to create a class ‘GraphProblem’ and display results. 

Class GraphProblem. This class defines the graph problem and performs a series of steps. 
- Inputs: Initial node, Goal node, and graph.
- Methods: actions, result, path_cost, find_min_edges and heuristic.

Class BFS_tree_algorithm
- A method ‘breadth_first_tree_search’. Gets the graph problem as an input start from the initial state and iterates until reaches the goal state. It uses a first-in, first-out (FIFO) queue in the frontier. If found the goal state, the function will return iteration times and nodes. 

Class GBFS_algorithm
- A method ‘best_first_search_graph’. Gets problem and evaluation function f as an input, and if reaches the goal, it returns the number of iterations and nodes. 
- A method ‘greedy_best_firts_search’. Gets problem and heuristic as an input and returns the number of iterations and nodes with evaluation function f(n) = h(n).

Class AStar_algorithm
- A method ‘best_first_search_graph’. Gets problem and evaluation function f as an input, and if reaches the goal, it returns the number of iterations and nodes. 
- A method ‘astar_search’. Gets problem and heuristic as an input and returns the number of iterations and nodes with evaluation function f(n)= g(n)+h(n).

After defining the graph, GraphProblem, and search algorithms, using the method ‘display_result’ get the result. The method ‘display_result’ gets defined problem and selected algorithm as an input, returns the iteration times and path of the solution. 

#### 2.2.3.	Results
After implementing the code, the results of breadth-first, greedy best-first search, and A* search are listed below when the agent’s initial state = ‘FreycinetPark’ and goal state = ‘Kakadu’. 


Algorithm: BFS
-	Iteration times: 254
-	Path: {FreycinetPark, Sydney, Uluru, Kakadu}

Algorithm: GBFS
-	Iteration times: 19
-	Path: {FreycinetPark, Sydney, Uluru, Kakadu}

Algorithm: A Star
-	Iteration times: 47  
-	Path: {FreycinetPark, Melbourne, Adelaide, Uluru, Kakadu}

The results show that the greedy best-first performs best regarding iteration times, followed by A* and breadth-first searches. The solution path is identical for the best-first and greedy best-first search. In addition, considering the optimality, the A* search performed better than other algorithms. A graphical display of the solutions is shown below.  

![Imgur](https://i.imgur.com/E9OJC9o.png)
*Figure 8. Graphical result of breath-first, greedy best-first and A* star search*

### 2.3. Comparisons of different AI techniques
The following figure shows the iteration times and associated costs from the initial state ‘FreycinetPark’ to the goal state ‘Kakadu’. 

|	 | Iteration times	| Cost (in km)|
|-----|---------------|-------------|
| BFS 	|254|	4825|
|GBFS	|19|	4825|
|A* search	|47	|4214|

*Figure 9. Comparison of different search techniques by iteration times and cost.*

The four main criteria of search algorithms are summarised in the table below. 

|Measure/Criteria|	BFS	| GBFS | 	A* |
|--------------|-----|-------|------|
|Complete?	|Yes|	Yes	|Yes|
|Time complexity	|O(bd)|	O(bm)	|O(bd )|
|Space complexity	|O(bd)	|O(bm)|	O(bd)|
|Cost optimal?|	No	|No	|Yes|

*Figure 10. Comparison of search evaluation metrics on the tourism problem. Search measures and criteria were taken from the textbook. (b – branching factor, d – depth, m - maximum depth)*

**Pros and cons**
The pros and cons of these different algorithms are summarised in Figure 11. 

|	|Pros	| Cons|
|---|-----|-------|
|BFS |Complete. The algorithm finds solution successfully as it explores all nodes at a given depth level before moving deeper into the graph. Simple. The algorithm simple and systematically explores the search space.| Exponential memory usage. It uses the FIFO queue to remember the previously explored nodes, which is the main cause of the memory usage. The iteration time is high (254) compared to the GBFS and A*. Lack of optimality. The solution is not cost-optimal compared to the A* search algorithm. Not suitable for when all actions have different costs.
|GBFS |Complete. The algorithm finds solution successfully.Memory efficient. GBFS is the efficient algorithm compared to the BFS and A*. Because it only uses the evaluation function: f(n) = h(n) | Lack of optimality. Even though the algorithm finds a solution efficiently, the solution is not cost-optimal. Sensitive to heuristic quality. The overall performance of the algorithm depends on the heuristic quality. |
|A*	| Complete. The algorithm finds solution successfully. Optimal. It finds optimal solutions compared to BFS and GBFS. Efficient. A* combines the advantages of BFS and GBFS.	| Memory usage. It consumes more memory compared to GBFS because it needs to store additional information for each node, including the estimated cost to the goal state.Complexity. Implementing A* require more effort compared to BFS or GBFS. Sensitive to heuristic quality. 

*Figure 11. Pros and cons of search algorithms in the case of Australia’s tourism problem.*

## 3. Conclusions and Recommendations

**Conclusions**

In this report, three classic search algorithms were implemented: breadth-first search, greedy best-first search, and A* search. Each algorithm has unique advantages and disadvantages in finding a solution to a problem. 

Breadth-first search is a simple and complete approach to finding the shortest path. However, from the tourism problem results, its memory usage and processing time were significant compared to other algorithms. Especially for large networks, it may take time and require significant memory. 

Greedy best-first search always tries to find a node closest to the goal state in each iteration, making the algorithm faster processing time and memory usage. In our problem, it has the lowest iteration time of 19. 
However, the algorithm did not guarantee the optimal solution.

A* search combines the advantages of breadth-first and greedy best-first search. It was the only algorithm that found the optimal solution in our case by using the evaluation function f(n) = g(n) + h(n). When an admissible heuristic is used, A* search guarantees finding the optimal path (Hart et al., 1968). However, the result shows that it required more processing time and memory usage compared to the greedy best-first search because of the additional information it uses. 

In conclusion, choosing the search algorithm depends on the problem and requirements. The three classic algorithms used in this report are powerful and fundamental to many different algorithms we use today. Each algorithm has strengths and constraints; therefore, understanding their difference and use cases is crucial for future AI specialists. 

**Recommendations**

The choice of search algorithm depends on what kind of problem we are trying to solve. When the action costs are identical or an unweighted small graph, the breadth-first search is a suitable solution for that problem. When the task requires memory and is time efficient, the greedy best-first search could solve the problem. A* search can be used for a complex graph to find an optimal solution. Nevertheless, we must consider choosing the correct heuristic design and memory issue. The proper heuristic function is crucial for greedy best-first and A* search. In future work, implementing and testing the extended versions of algorithms, such as Weighted A*, Bidirectional A*, and beam search, could be helpful. 

---
## List of References
Barr, A., Feigenbaum, E. A., & Cohen, P. R. (1981). The Handbook of artificial intelligence. HeirisTech Press.
Evers, J., Editing, E., & Editing, E. (20 December 2022). Latitude is the measurement of distance north or south of the Equator. National Geographic. https://education.nationalgeographic.org/resource/latitude/ 

Makepeace, C. (3 October 2022). Top 20 places in Australia for your bucket list. Y Travel. https://www.ytravelblog.com/places-in-australia-bucket-list/ 

Moore, E. F. (1959). The Shortest Path Through a Maze. United States: Bell Telephone System.

Newell, A., Shaw, J. C., & Simon, H. A. (1958). Elements of a theory of human problem-solving. Psychological Review, 65(3), 151–166. https://doi.org/10.1037/h0048495

Hart, P., Nilsson, N., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100–107. https://doi.org/10.1109/TSSC.1968.300136
Russell, S., & Norvig, P. (2021). Artificial intelligence: A modern approach, global edition. Pearson Education, Limited.
