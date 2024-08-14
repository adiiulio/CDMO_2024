include "alldifferent.mzn";
include "globals.mzn";

% MCP problem definition

% Input parameters
int: n_couriers; 
int: items_count; 
array[1..n_couriers] of int: max_load;
array[1..items_count] of int: size_item;

int: matrix_size;
array[1..matrix_size*matrix_size] of int: all_distances;

% Convert the 1D array into a 2D array within MiniZinc
array[1..matrix_size, 1..matrix_size] of int: dist_matrix = array2d(1..matrix_size, 1..matrix_size, all_distances);

set of int: courier_ID = 1..n_couriers;
set of int: ITEMS = 1..items_count;
int: steps = items_count+2;
set of int: STEPS = 1..steps; 
set of int: STEPS_NO_LAST = 1..steps-1;
set of int: STEPS_NO_FIRST_NO_LAST = 2..steps-1;

% int: ubound = sum(all_distances);
% float: ubound_f = sum(all_distances);

int: ubound = sum([dist_matrix[i,j] | i in 1..matrix_size, j in 1..matrix_size]);
float: ubound_f = sum([dist_matrix[i,j] | i in 1..matrix_size, j in 1..matrix_size]);

% Decision variables
array[1..n_couriers, 1..items_count] of var 0..1: delivery_assignment;
array[1..n_couriers, 1..items_count+1, 1..items_count+1] of var 0..1: travel_route;

array[STEPS, courier_ID] of var 0..items_count: delivery_order;
array[courier_ID] of var 0..sum(all_distances): courier_dist;

% Constraints

% Ensure each item is delivered exactly once
constraint forall(i in 1..items_count)(sum(c in 1..n_couriers)(delivery_assignment[c,i]) = 1);

% Ensure each courier does not exceed their maximum load
constraint forall(c in 1..n_couriers)(
    sum(i in 1..items_count)(delivery_assignment[c,i] * size_item[i]) <= max_load[c]
);

% Link delivery assignments to travel routes
constraint forall(c in 1..n_couriers)(
    forall(i in 1..items_count)(
        sum(j in 1..items_count+1)(travel_route[c,i,j]) = delivery_assignment[c,i]
    )
);

% Ensure each courier starts and ends at the origin (n+1)
constraint forall(c in 1..n_couriers)(
    sum(i in 1..items_count+1)(travel_route[c,items_count+1,i]) = 1 /\
    sum(i in 1..items_count+1)(travel_route[c,i,items_count+1]) = 1
);

% Ensure the routes form a valid tour
constraint forall(c in 1..n_couriers)(
    forall(j in 1..items_count)(
        sum(i in 1..items_count+1)(travel_route[c,i,j]) = sum(k in 1..items_count+1)(travel_route[c,j,k])
    )
);

% All couriers depart and return to the origin
constraint forall(c in courier_ID)
                  (delivery_order[1,c] == 0 /\ delivery_order[items_count+2,c] == 0);

% Each package must be delivered only once
constraint alldifferent_except_0(delivery_order);

% Each package must be delivered
constraint count(delivery_order, 0, (steps*n_couriers)-items_count);

% Each courier must not exceed its maximum load
constraint forall(c in courier_ID) (
  sum([size_item[delivery_order[i,c]] | i in STEPS where delivery_order[i,c] != 0]) <= max_load[c]);

% Removing solutions with 'reload' from couriers
constraint forall(c in courier_ID, i in STEPS_NO_FIRST_NO_LAST)      
                 (delivery_order[i,c] != 0 \/ sum(delivery_order[i..,c]) == 0);

% Each courier must deliver at least one item
constraint forall(c in courier_ID)(delivery_order[2,c] != 0 /\
              sum(delivery_order[steps-n_couriers+1..,c]) == 0);

% Ensure correct distance calculation
constraint forall(c in courier_ID)
                  (courier_dist[c] = sum(s in STEPS_NO_LAST)
                  (dist_matrix[delivery_order[s,c] + 1, delivery_order[s + 1,c] + 1] ));

% Objective: minimize the maximum distance travelled by any courier
var 0..ubound: max_dist = max(courier_dist);

solve minimize max_dist;

% Output
output [
    "Maximum distance travelled by any courier: ", show(max_dist), "\n",
    "Item assignments:\n",
    show(delivery_assignment), "\n",
    "Travel routes:\n",
    show(travel_route), "\n",
    "Delivery order:\n",
    show(delivery_order), "\n",
    "Courier distances:\n",
    show(courier_dist), "\n"
];
