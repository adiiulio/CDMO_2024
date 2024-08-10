from minizinc import Instance, Model, Solver

def read_instance(filename):
    with open(filename, "r") as file:
        m = int(file.readline().strip())
        n = int(file.readline().strip())
        l = list(map(int, file.readline().strip().split()))
        s = list(map(int, file.readline().strip().split()))
        D = []
        for _ in range(n+1):
            D.append(list(map(int, file.readline().strip().split())))
    return m, n, l, s, D

def solve_mcp(m, n, l, s, D):
    # Load MiniZinc model
    mcp_model = Model("/Users/kristinabaycheva/Downloads/MCP Probelm.mzn")
    solver = Solver.lookup("gecode")

    # Create an instance of the model
    instance = Instance(solver, mcp_model)

    # Assign data to model parameters
    instance["m"] = m
    instance["n"] = n
    instance["l"] = l
    instance["s"] = s
    instance["D"] = D

    # Solve the model
    result = instance.solve()

    return result

def main():
    filename = "/Users/kristinabaycheva/Downloads/Multiple-Vehicle-Routing-Problem-main/solution_checker/input/inst03.dat"
    m, n, l, s, D = read_instance(filename)
    result = solve_mcp(m, n, l, s, D)
    print(result)

if __name__ == "__main__":
    main()
