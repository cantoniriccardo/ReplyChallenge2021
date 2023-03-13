import numpy as np


class Problem:

    def __init__(self):
        self.name = ""
        self.W = 0
        self.H = 0
        self.N = 0
        self.M = 0
        self.R = 0
        self.Nrow = None
        self.Ncol = None
        self.Nl = None
        self.Nc = None
        self.Mr = None
        self.Mc = None
        self.b_grid = None  # building
        self.Srow = None
        self.Scol = None
        self.b_score = None

    def init_solution(self):
        self.S = set()  # antenna used
        self.Srow = np.zeros(self.M, dtype=np.int32)
        self.Scol = np.zeros(self.M, dtype=np.int32)
        for j in range(self.M):
            self.Srow[j] = 0
            self.Scol[j] = 0

        self.b_grid = np.ndarray((self.H, self.W), dtype=np.int32)
        self.b_grid[:, :] = -1
        for i in range(self.N):
            self.b_grid[self.Nrow[i], self.Ncol[i]] = i

        self.a_grid = np.ndarray((self.H, self.W), dtype=np.int32)
        self.a_grid[:, :] = -1

        # max score for each building
        self.b_score = np.zeros(self.N, dtype=np.int32)
        self.b_score[:] = -1

        self.tot_connected = 0
        self.tot_score = 0

        self.new_connected = 0

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            p = cls()
            p.name = filename
            p.W, p.H = [int(e) for e in f.readline().split()]
            p.N, p.M, p.R = [int(e) for e in f.readline().split()]

            p.Ncol = np.zeros(p.N, dtype=np.int32)
            p.Nrow = np.zeros(p.N, dtype=np.int32)
            p.Nl = np.zeros(p.N, dtype=np.int32)
            p.Nc = np.zeros(p.N, dtype=np.int32)
            for i in range(p.N):
                x, y, l, c = f.readline().split(" ")
                p.Nrow[i] = y
                p.Ncol[i] = x
                p.Nl[i] = l
                p.Nc[i] = c

            p.Mr = np.zeros(p.M, dtype=np.int32)
            p.Mc = np.zeros(p.M, dtype=np.int32)

            for j in range(p.M):
                r, c = f.readline().split(" ")
                p.Mr[j] = r
                p.Mc[j] = c

            p.init_solution()
            return p

    def dump(self):
        score = self.score()
        with open(f"{self.name}-{score}", "w") as f:
            f.write(f"{self.M}\n")
            for i in range(self.M):
                f.write(f"{i} {self.Scol[i]} {self.Srow[i]}\n")

    def score(self):
        self.tot_score = 0
        self.tot_connected = 0
        self.b_score[:] = -1
        for j in self.S:
            r = self.Srow[j]
            c = self.Scol[j]
            delta = self.d_score(j, r, c, update_scores=True)
            self.tot_score += delta
            self.tot_connected += self.new_connected

        return self.tot_score

    def d_score(self, j, r, c, update_scores=False):
        deltax = np.array([-1, 1, 1, -1])
        deltay = np.array([1, 1, -1, -1])

        delta = 0
        self.new_connected = 0
        for rng in range(0, self.Mr[j] + 1):
            rr = r
            cc = c - rng
            # scan the cells around the cel (Mx[j], My[j]) that are at distance r
            for d in range(4 if rng > 0 else 1):
                for _ in range(0, rng if rng > 0 else 1):
                    if 0 <= rr < self.H and 0 <= cc < self.W:
                        if self.b_grid[rr, cc] != -1:
                            if self.b_score[self.b_grid[rr, cc]] == -1:
                                self.new_connected += 1
                            b = self.b_grid[rr, cc]
                            score = self.Nc[b] * self.Mc[j] - self.Nl[b] * rng
                            if score > self.b_score[b]:
                                delta += score - self.b_score[b]
                                if update_scores:
                                    self.b_score[b] = score
                    if rng != 0:
                        rr += deltax[d]
                        cc += deltay[d]
        delta += (self.R if (self.new_connected > 0 and self.tot_connected + self.new_connected == self.N) else 0)
        return delta

    def solve(self, n=100):
        self.random_sol()

        for _ in range(n):
            # pick random antenna
            m = np.random.randint(0, self.M - 1)

            # pick random coordinates or the coordinate of one building or move the antenna by random delta in random
            # direction
            rnd = np.random.rand()
            if rnd < 0.1:
                # Pick random position
                r = np.random.randint(0, self.H - 1)
                c = np.random.randint(0, self.W - 1)
            elif rnd < 0.1:
                # Pick random building
                b = np.random.randint(0, self.N - 1)
                r = self.Nrow[b]
                c = self.Ncol[b]
            else:
                # Move antenna by random delta in a random direction
                r = self.Srow[m] + np.random.randint(-10, 10)
                c = self.Scol[m] + np.random.randint(-10, 10)

                # Check if the new position is valid
                if r < 0 or r >= self.H or c < 0 or c >= self.W:
                    continue

            if self.a_grid[r, c] == -1:
                prev_score = self.score()
                old_r = self.Srow[m]
                old_c = self.Scol[m]

                self.Srow[m] = r
                self.Scol[m] = c
                self.a_grid[old_r, old_c] = -1
                self.a_grid[r, c] = m

                new_score = self.score()

                if new_score > prev_score:
                    print(f"New score: ", new_score, "prev score ", prev_score, " Connected: ", self.tot_connected)
                else:
                    self.a_grid[r, c] = -1
                    self.Srow[m] = old_r
                    self.Scol[m] = old_c
                    self.a_grid[old_r, old_c] = m

    def solve2(self, n=100, step=10):
        self.random_sol()

        for _ in range(n):
            # pick random antenna
            m = np.random.randint(0, self.M - 1)

            # find best position
            # remove from solution
            self.S.remove(m)
            self.a_grid[self.Srow[m], self.Scol[m]] = -1

            # Score the solution
            self.score()
            best_r = 0
            best_c = 0
            best_delta = 0
            best_new_connected = 0

            for r in range(0, self.H, step):
                for c in range(0, self.W, step):
                    if self.a_grid[r, c] == -1:
                        delta = self.d_score(m, r, c)
                        if delta > best_delta:
                            best_delta = delta
                            best_new_connected = self.new_connected
                            best_r = r
                            best_c = c

            # add to the solution
            self.Srow[m] = best_r
            self.Scol[m] = best_c
            self.S.add(m)
            self.a_grid[self.Srow[m], self.Scol[m]] = m
            self.tot_score += self.d_score(m, best_r, best_c, update_scores=True)
            self.tot_connected += best_new_connected

            print(f"Score: ", self.tot_score, " Connected: ", self.tot_connected, " Antennas: ", len(self.S))


    def solve3(self, n=100, range_size=50):
        self.random_sol()

        for _ in range(n):
            # pick random antenna
            m = np.random.randint(0, self.M - 1)

            # find best position
            # remove from solution
            self.S.remove(m)
            self.a_grid[self.Srow[m], self.Scol[m]] = -1

            # Score the solution
            self.score()
            best_r = 0
            best_c = 0
            best_delta = 0
            best_new_connected = 0

            for r in range(self.Srow[m] - int(range_size/2), self.Srow[m] + range_size):
                for c in range(self.Scol[m] - int(range_size/2), self.Scol[m] + range_size):
                    if r < 0 or r >= self.H or c < 0 or c >= self.W:
                        continue
                    if self.a_grid[r, c] == -1:
                        delta = self.d_score(m, r, c)
                        if delta > best_delta:
                            best_delta = delta
                            best_new_connected = self.new_connected
                            best_r = r
                            best_c = c

            # add to the solution
            self.Srow[m] = best_r
            self.Scol[m] = best_c
            self.S.add(m)
            self.a_grid[self.Srow[m], self.Scol[m]] = m
            self.tot_score += self.d_score(m, best_r, best_c, update_scores=True)
            self.tot_connected += best_new_connected

            print(f"Score: ", self.tot_score, " Connected: ", self.tot_connected, " Antennas: ", len(self.S))

    def random_sol(self):
        self.init_solution()
        generated_pairs = set()
        for i in range(0, self.M):
            x = np.random.randint(0, self.W)
            y = np.random.randint(0, self.H)
            while (x, y) in generated_pairs:
                x = np.random.randint(0, self.W)
                y = np.random.randint(0, self.H)
            generated_pairs.add((x, y))
            self.Scol[i] = x
            self.Srow[i] = y
            self.a_grid[y, x] = i

        self.S.update(range(0, self.M))
