include "alldifferent.mzn";
include "globals.mzn";

% -------------------------------------------------------------------------------------
% Parameters
% Number of couriers
int: couriers;

% Number of items
int: items;

% Load size of the courier
array[1..couriers] of int: max_load;

% Size of the item
array[1..items] of int: item_size;

% The distances between all pairs of items, as well as the distances from each item to a central depot (or origin)
array[1..items+1, 1..items+1] of int: distances;

% -------------------------------------------------------------------------------------
% Decision variables
% Assign each item to a courier
array[1..items] of var 1..couriers: courier_assignment;

% Next item to visit for each item (0 represents the origin/depot)
array[0..items] of var 0..items: next_item; 

% -------------------------------------------------------------------------------------
% Auxiliary variables (1, 2 & 3)
% 1. Indicate the starting point for each courier
array[1..couriers] of var 0..items: start_point;

% 2. Indicate the ending point for each courier
array[1..couriers] of var 0..items: end_point;

% 3. Boolean array to track if each item is visited
array[1..items] of var bool: item_visited;

% -------------------------------------------------------------------------------------
% Constraints 
% Each courier cannot exceed its maximum load
constraint forall(j in 1..couriers)(sum([item_size[i]| i in 1..items where courier_assignment[i] == j]) <= max_load[j]);

% Each item is assigned to exactly one courier
constraint forall(i in 1..items)(sum ([courier_assignment[i] == j | j in 1..couriers]) == 1);

% Each package must be delivered only once 
constraint forall(i in 1..items)(sum([next_item[j] == i | j in 0..items]) == 1);

% Always start at the depot/origin point
constraint forall(j in 1..couriers)(start_point[j] = 0 /\ next_item[0] = start_point[j]);

% Always end at the origin point o
constraint forall(j in 1..couriers)(end_point[j] = 0 /\ forall(i in 1..items)(next_item[i] != end_point[j] \/ courier_assignment[i] != j));

% No self-loops and next item must either be the depot or a valid item index
constraint forall(i in 0..items)(next_item[i] != i /\ (next_item[i] <= items));

% Every item must be delivered by setting item_visited to true if the item is visited
% Specifically, it sets item_visited[next_item[i]] to true whenever next_item[i] is not the depot (0).
constraint forall(i in 0..items)(if next_item[i] != 0 then item_visited[next_item[i]] = true endif);

% Ensure that every item is visited, mark it in the item_visited array
constraint forall(i in 1..items)(item_visited[i] == true);

% Ensure that every node (including the depot) is entered and left once by the same courier
constraint forall(i in 1..items)(
  % Ensure each item has exactly one predecessor
  sum([next_item[j] == i | j in 0..items]) == 1 /\
  % Ensure each item has exactly one successor
  sum([next_item[i] == k | k in 0..items]) == 1 /\
  % Ensure the courier entering is the same as the courier leaving
  forall(j in 1..couriers)(
  (sum([courier_assignment[j] == courier_assignment[i] | j in 0..items where next_item[j] == i]) == 1) /\
  (sum([courier_assignment[k] == courier_assignment[i] | k in 0..items where next_item[i] == k]) == 1)));

% Ensure the depot has exactly one successor per courier and is visited once by each courier
constraint forall(j in 1..couriers)(
  sum([next_item[0] == i | i in 1..items where courier_assignment[i] == j]) == 1 /\
  sum([next_item[i] == 0 | i in 1..items where courier_assignment[i] == j]) == 1
);

% No node is to be visited by more than one courier
% If the next node for node i is j, then the courier handling node i must be the same as the courier handling node j
constraint forall(i in 0..items)(forall(j in 0..items)(next_item[i] == j -> courier_assignment[i] == courier_assignment[j]));

% Each courier should deliver at least one item
constraint forall(j in 1..couriers)(sum([courier_assignment[i] == j | i in 1..items])>= 1);

% -------------------------------------------------------------------------------------
% Objective function
% Minimize maximum travel distance of all couriers
% If next_item[i] == j, then distances[i+1, j+1] * 1 contributes the distance to the sum. If next_item[i] != j, then distances[i+1, j+1] * 0 contributes 0 to the sum.
var int: total_distance = sum(i in 0..items, j in 0..items)(distances[i+1, j+1] * (next_item[i] == j));

solve minimize total_distance;

% -------------------------------------------------------------------------------------
% Output
output[
"Total distance: ", show(total_distance), "\n",
"Courier assignments: ", show(courier_assignment), "\n",
"Next items: ", show(next_item)
];

