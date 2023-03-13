import numpy as np
import array
import time

# define an array with 4 elements
cdef int[:] deltax = array.array('i', [-1, 1, 1, -1])
cdef int[:] deltay = array.array('i', [1, 1, -1, -1])

# define Row and col
cdef int ROW = 0
cdef int COL = 1


cdef class CProblem:
    cdef public str name
    cdef public int W, H, N, M, R
    cdef public int[:] Nrow, Ncol, Nl, Nc, Mr, Mc
    cdef public int[:, :] b_grid

    cdef public set S
    cdef public int[:, :] Spos
    cdef public int[:, :] a_grid
    cdef public int[:] b_score
    cdef public long tot_connected, tot_score, new_connected

    def init_solution(self):
        self.S = set()  # antenna used
        self.Spos = np.zeros((self.M, 2), dtype=np.int32)

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
                p.Nrow[i] = int(y)
                p.Ncol[i] = int(x)
                p.Nl[i] = int(l)
                p.Nc[i] = int(c)

            p.Mr = np.zeros(p.M, dtype=np.int32)
            p.Mc = np.zeros(p.M, dtype=np.int32)

            for j in range(p.M):
                r, c = f.readline().split(" ")
                p.Mr[j] = int(r)
                p.Mc[j] = int(c)

            # build the grid
            p.b_grid = np.ndarray((p.H, p.W), dtype=np.int32)
            p.b_grid[:, :] = -1
            for i in range(p.N):
                p.b_grid[p.Nrow[i], p.Ncol[i]] = i

            p.init_solution()
            return p

    def dump(self):
        score = self.score()
        with open(f"{self.name}-{score}", "w") as f:
            f.write(f"{self.M}\n")
            for i in range(self.M):
                f.write(f"{i} {self.Spos[i, COL]} {self.Spos[i, ROW]}\n")

    cpdef long score(self) except -1:
        self.tot_score = 0
        self.tot_connected = 0
        self.b_score[:] = -1
        cdef int j, r, c
        for j in self.S:
            r = self.Spos[j, ROW]
            c = self.Spos[j, COL]
            delta = self.d_score(j, r, c, update_scores=True)
            self.tot_score += delta
            self.tot_connected += self.new_connected

        return self.tot_score

    cdef inline long d_score(self, int j, int r, int c, bint update_scores=False) except -1:
        cdef int rr, cc, b, rng, d
        cdef long delta = 0, score = 0

        self.new_connected = 0
        for rng in range(0, self.Mr[j] + 1):
            rr = r
            cc = c - rng
            # scan the cells around the cel (Mx[j], My[j]) that are at distance r
            for d in range(4 if rng > 0 else 1):
                for _ in range(0, rng if rng > 0 else 1):
                    if 0 <= rr < self.H and 0 <= cc < self.W:
                        if self.b_grid[rr, cc] != -1:
                            b = self.b_grid[rr, cc]
                            if self.b_score[b] == -1:
                                self.new_connected += 1
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

    cpdef long simutaed_anealling(self, float temp_i=100, float temp_f=.1, float alpha=0.1, bint verbose=False):
        # initial time
        start = time.time()
        last_time = start

        self.random_sol()

        # Needed local variables
        cdef int i = 0 , j, r, c, rr, cc, dist, delta_r, abs_delta_r, b, rng, d, old_r, old_c, m
        cdef long delta = 0, current_score, best_score = 0
        cdef float rnd

        cdef long accepted_score = self.score()

        # Best solution
        best_score = self.score()
        cdef int[:, :] best_Spos = np.zeros_like(self.Spos)
        best_Spos[:, :] = self.Spos[:, :]
        cdef int[:, :] best_a_grid = np.zeros_like(self.a_grid)
        best_a_grid[:, :] = self.a_grid[:, :]

        cdef float current_temp = temp_i
        while current_temp > temp_f:
            i += 1

            # pick random antenna
            m = np.random.randint(0, self.M - 1)

            # Pick random building
            b = np.random.randint(0, self.N - 1)
            r = self.Nrow[b]
            c = self.Ncol[b]

            # Move antenna by random delta in a random direction of at most the radius of the antenna
            if self.Mr[m] != 0:
                dist = np.random.randint(0, self.Mr[m] + 1)
                delta_r = np.random.randint(-dist, dist + 1)
                abs_delta_r = -delta_r if delta_r < 0 else delta_r
                r += delta_r
                c += np.random.randint(-(dist-abs_delta_r), (dist-abs_delta_r) + 1)

            # Check if the new position is valid
            if r < 0 or r >= self.H or c < 0 or c >= self.W:
                continue

            if self.a_grid[r, c] != -1:
                # TODO swap antennas otherwise if a position is taken by an antenna is difficult to move it away
                continue

            old_r = self.Spos[m, ROW]
            old_c = self.Spos[m, COL]

            self.Spos[m, ROW] = r
            self.Spos[m, COL] = c
            self.a_grid[old_r, old_c] = -1
            self.a_grid[r, c] = m

            current_score = self.score()

            # Check if the new solution is better than the best solution
            if current_score > best_score:
                best_Spos[:,:] = self.Spos[:, :]
                best_a_grid[:,:] = self.a_grid[:, :]
                best_score = current_score
                if verbose:
                    print(f"{current_temp}/{temp_f} found new best solution {best_score}")
                    print(f"best_score {best_score}, accepted score {accepted_score} score {current_score}")

            # if the new solution is better, accept it. If equal promote diversity
            if current_score >= accepted_score:
                accepted_score = current_score
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            elif np.random.uniform(0, 1) < np.exp(-(accepted_score - current_score) / current_temp):
                accepted_score = current_score
                if verbose:
                    print(f"{current_temp}/{temp_f} accepted new solution with lower score {current_score}")
                    print(f"best_score {best_score}, accepted score {accepted_score} score {current_score}")
            else:
                self.a_grid[r, c] = -1
                self.Spos[m, ROW] = old_r
                self.Spos[m, COL] = old_c
                self.a_grid[old_r, old_c] = m

            # report every 10 seconds
            if verbose or (i % 100 == 0 and time.time() - last_time > 5):
                last_time = time.time()
                print(
                    f"{self.name}: {current_temp}/{temp_f} best_score {best_score}, accepted score {current_score}")

            # decrement the temperature
            current_temp -= alpha

        # Restore best solution
        self.Spos[:,:] = best_Spos[:, :]
        self.a_grid[:,:] = best_a_grid[:, :]
        return self.score()

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
                r = self.Spos[m, ROW] + np.random.randint(-10, 10)
                c = self.Spos[m, COL] + np.random.randint(-10, 10)

                # Check if the new position is valid
                if r < 0 or r >= self.H or c < 0 or c >= self.W:
                    continue

            if self.a_grid[r, c] == -1:
                prev_score = self.score()
                old_r = self.Spos[m, ROW]
                old_c = self.Spos[m, COL]

                self.Spos[m, ROW] = r
                self.Spos[m, COL] = c
                self.a_grid[old_r, old_c] = -1
                self.a_grid[r, c] = m

                new_score = self.score()

                if new_score > prev_score:
                    print(f"New score: ", new_score, "prev score ", prev_score, " Connected: ", self.tot_connected)
                else:
                    self.a_grid[r, c] = -1
                    self.Spos[m, ROW] = old_r
                    self.Spos[m, COL] = old_c
                    self.a_grid[old_r, old_c] = m

    def solve2(self, n=100, step=10):
        self.random_sol()

        for _ in range(n):
            # pick random antenna
            m = np.random.randint(0, self.M - 1)

            # find best position
            # remove from solution
            self.S.remove(m)
            self.a_grid[self.Spos[m, ROW], self.Spos[m, COL]] = -1

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
            self.Spos[m, ROW] = best_r
            self.Spos[m, COL] = best_c
            self.S.add(m)
            self.a_grid[self.Spos[m, ROW], self.Spos[m, COL]] = m
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
            self.a_grid[self.Spos[m, ROW], self.Spos[m, COL]] = -1

            # Score the solution
            self.score()
            best_r = 0
            best_c = 0
            best_delta = 0
            best_new_connected = 0

            for r in range(self.Spos[m, ROW] - int(range_size/2), self.Spos[m, ROW] + range_size):
                for c in range(self.Spos[m, COL] - int(range_size/2), self.Spos[m, COL] + range_size):
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
            self.Spos[m, ROW] = best_r
            self.Spos[m, COL] = best_c
            self.S.add(m)
            self.a_grid[self.Spos[m, ROW], self.Spos[m, COL]] = m
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
            self.Spos[i, COL] = x
            self.Spos[i, ROW] = y
            self.a_grid[y, x] = i

        self.S.update(range(0, self.M))

    def greedy_solution(self):
        sorted_buildings = sorted(range(self.N), key=lambda x: self.Nc[x], reverse=True)
        sorted_antenna = sorted(range(self.M), key=lambda x: self.Mc[x], reverse=True)

        for i in range(len(sorted_antenna)):
            self.Spos[sorted_antenna[i], ROW] = self.Nrow[sorted_buildings[i]]
            self.Spos[sorted_antenna[i], COL] = self.Ncol[sorted_buildings[i]]

            self.a_grid[self.Spos[sorted_antenna[i], ROW], self.Spos[sorted_antenna[i], COL]] = sorted_antenna[i]

        self.S.update(range(0, self.M))
        return self.score()

    cpdef test(self):
        return deltax[0], deltax[1], deltax[2], deltax[3]
