% Multi-Courier Optimization Problem
include "globals.mzn";

% Number of couriers
int: m_couriers_count;

% Number of items to be delivered
int: n_items_count;

% Maximum load each courier can carry
array[1..m_couriers_count] of int: l_max_loads;

% Weight of each item/package
array[1..n_items_count] of int: s_load_sizes;

% Distance matrix between locations
array[1..n_items_count+1, 1..n_items_count+1] of int: D_distances_matrix;

% The sequence in which each courier delivers the items
array[1..m_couriers_count, 1..n_items_count+2] of var int: order_of_delivery;

% Maximum distance covered by any courier
var int: maximum_distance;

% The total load each courier is transporting
array[1..m_couriers_count] of var int: courier_weight = 
  [sum(j in 2..n_items_count+1 where order_of_delivery[i,j] != n_items_count+1)(s_load_sizes[order_of_delivery[i,j]]) | i in 1..m_couriers_count];

% The total distance each courier travels
array[1..m_couriers_count] of var int: courier_distances = 
  [sum(j in 1..n_items_count+1)(D_distances_matrix[order_of_delivery[i,j], order_of_delivery[i,j+1]]) | i in 1..m_couriers_count];

% Ensure each item is delivered by exactly one courier
constraint
all_different(i in 1..m_couriers_count, j in 1..n_items_count+2 where order_of_delivery[i,j] != n_items_count+1)(order_of_delivery[i,j]);

% Ensure every item is assigned to a courier
constraint forall(p in 1..n_items_count)(
    exists(i in 1..m_couriers_count, j in 1..n_items_count+2)(p == order_of_delivery[i,j])
);

% Make sure the total weight carried by each courier does not exceed their maximum load capacity
constraint forall(i in 1..m_couriers_count)(
    courier_weight[i] <= l_max_loads[i]
);

% Ensure every item is delivered, and each courier starts and ends at the base location
constraint forall(i in 1..m_couriers_count)(
    order_of_delivery[i,1] = n_items_count+1 /\ order_of_delivery[i, n_items_count+2] = n_items_count+1
);

% Couriers should start with a delivery right after leaving the base
constraint forall(i in 1..m_couriers_count)(
    exists(j in 1..n_items_count+1)(order_of_delivery[i,j] != n_items_count+1) -> (order_of_delivery[i, 2] != n_items_count+1)
);

% Remove this constraint if couriers are allowed to return to the base before taking additional deliveries
constraint forall(i in 1..m_couriers_count)(
    forall(j in 2..n_items_count+1)((order_of_delivery[i,j] == n_items_count+1) -> not exists(k in j..n_items_count+1)(order_of_delivery[i,k] != n_items_count+1))
);

% If a courier has enough capacity, ensure they carry at least one item
constraint forall(c1 in 1..m_couriers_count)(
  (l_max_loads[c1] >= max(s_load_sizes)) -> exists(k in 1..n_items_count+1)(order_of_delivery[c1,k] != n_items_count+1)
);

% Symmetry breaking 
% If two couriers have the same capacity, they are symmetric;
% we impose an order on the items they pick up to break this symmetry.
constraint symmetry_breaking_constraint(
    forall(c1 in 1..m_couriers_count, c2 in 1..m_couriers_count where c1 < c2)
     ((l_max_loads[c1] == l_max_loads[c2]) -> (lex_less(row(order_of_delivery, c1), row(order_of_delivery, c2))))
);

% If the maximum weight carried by two couriers is less than or equal to the smallest load capacity,
% their routes are interchangeable; we enforce an order to avoid this.
constraint symmetry_breaking_constraint(
    forall(c1 in 1..m_couriers_count, c2 in 1..m_couriers_count where c1 < c2)
    (
      (max(courier_weight[c1], courier_weight[c2]) <= min(l_max_loads[c1], l_max_loads[c2])) ->
      (lex_less(row(order_of_delivery, c1), row(order_of_delivery, c2))))
);

% The goal is to minimize the maximum distance traveled by any courier
maximum_distance = max(i in 1..m_couriers_count)(courier_distances[i]);

solve minimize maximum_distance;  % This is a simple and direct approach

% Other potential strategies to reduce solving time, though the direct approach is nearly optimal
% solve 
% :: int_search(
%     [order_of_delivery[i,j] | i in 1..m_couriers_count, j in 1..n_items_count+2],  % Variables for decision-making - Focus first on the first courier's path, then the second, etc.
%     input_order,      % Variable selection strategy: input order. Alternatives include first_fail or input_order
%     indomain_min,     % Value selection strategy: minimum value first. Alternatives include indomain_random or indomain_median.
%     complete          % Apply this strategy across the entire search process
% ) 
% minimize maximum_distance;

output 
[
  "maximum_distance = ", show(maximum_distance), ";\n",
  "order_of_delivery = [\n",
  concat([ show([order_of_delivery[courier,j] | j in 1..n_items_count+2]) ++ if courier != m_couriers_count then ",\n" else "\n" endif | courier in 1..m_couriers_count ]),
  "];\n",
  "courier_distances = ", show(courier_distances), ";\n"
];
% output 
%     [show(maximum_distance)] ++ [ "\n"] ++
%     [show(order_of_delivery)] ++ [ "\n"] ++
%     [show(courier_distances)] ++ [ "\n"]
% ;
