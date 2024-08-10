% MCP problem definition

% Input parameters
int: m; % number of couriers
int: n; % number of items
array[1..m] of int: l; % maximum load of each courier
array[1..n] of int: s; % size of each item
array[1..n+1, 1..n+1] of int: D; % distance matrix

% Decision variables
array[1..m, 1..n] of var 0..1: x; % x[i,j] = 1 if courier i delivers item j, 0 otherwise
array[1..m, 1..n+1, 1..n+1] of var 0..1: y; % y[i,j,k] = 1 if courier i travels from j to k, 0 otherwise

% Constraints
constraint
    forall(i in 1..m)(
        sum(j in 1..n)(x[i,j] * s[j]) <= l[i]
    );

constraint
    forall(j in 1..n)(
        sum(i in 1..m)(x[i,j]) = 1
    );

constraint
    forall(i in 1..m)(
        forall(j in 1..n)(
            sum(k in 1..n+1)(y[i,j,k]) = x[i,j]
        )
    );

constraint
    forall(i in 1..m)(
        sum(j in 1..n+1)(y[i,n+1,j]) = 1 /\
        sum(j in 1..n+1)(y[i,j,n+1]) = 1
    );

% Objective: minimize the maximum distance travelled by any courier
array[1..m] of var int: dist;
constraint
    forall(i in 1..m)(
        dist[i] = sum(j in 1..n+1, k in 1..n+1)(y[i,j,k] * D[j,k])
    );

var int: max_dist = max(dist);

solve minimize max_dist;

% Output
output [
    "Maximum distance travelled by any courier: ", show(max_dist), "\n",
    "Item assignments:\n",
    show(x), "\n",
    "Travel routes:\n",
    show(y), "\n"
];
