from problem import Problem
from multiprocessing import Pool
import os

s = Problem.load_problem("data/data_scenarios_a_example.in")
s.Scol[0] = 12
s.Srow[0] = 3
s.Scol[1] = 7
s.Srow[1] = 6
s.Scol[2] = 11
s.Srow[2] = 7
s.Scol[3] = 2
s.Srow[3] = 4


def solve(filename):
    s = Problem.from_file(filename)
    print(f"problem loaded {filename}")
    random_sol(s)
    s.dump()


if __name__ == "__main__":
    with Pool(6) as p:
        p.map(solve, [f"data/{f}" for f in os.listdir("data") if f.endswith(".txt")])


