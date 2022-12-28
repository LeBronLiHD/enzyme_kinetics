# Haodong LI
# 8.2 of Enzyme Kinetics

import matplotlib.pyplot as plt
from numpy import diff

class Data:

    def __init__(self, K, es_0, s_0, p_0, e_0, unit="second") -> None:
        self.K = [k/60 for k in K] if unit == "second" else K
        self.es = [es_0]
        self.s = [s_0]
        self.p = [p_0]
        self.e = [e_0]

    def f(self, t, es, s):
        # calculate dES/dt from S and ES
        return self.K[0]*(self.e[0] - es)*s - (self.K[1] + self.K[2])*es
    
    def g(self, t, es, s):
        # calculate dS/dt from S and ES
        return -self.K[0]*(self.e[0] - es)*s + self.K[2]*es

class RK(Data):

    def __init__(self, K, es_0, s_0, p_0, e_0, lim_l, lim_r, n, unit="second") -> None:
        self.n = n
        self.h = (lim_r - lim_l)/n
        super().__init__(K, es_0, s_0, p_0, e_0, unit)
    
    def step(self, t):
        k, l = [0], [0]
        param = [0, 1/2, 1/2, 1]
        # calculate K and L
        for i in range(4):
            k.append(self.h * self.f((t + param[i])*self.h, self.es[t] + param[i]*k[-1], self.s[t] + param[i]*l[-1]))
            l.append(self.h * self.g((t + param[i])*self.h, self.es[t] + param[i]*k[-1], self.s[t] + param[i]*l[-1]))
        # update ES and S
        self.es.append(self.es[t] + 1/6*(k[1] + 2*k[2] + 2*k[3] + k[4]))
        self.s.append(self.s[t] + 1/6*(l[1] + 2*l[2] + 2*l[3] + l[4]))

    def run(self):
        for i in range(self.n):
            self.step(i)
    
    def update(self):
        # update P and E
        for i in range(1, self.n + 1):
            self.p.append(self.s[0] - (self.es[i] + self.s[i]))
            self.e.append(self.e[0] - self.es[i])

    def plot(self):
        names = ["ES", "S", "P", "E"]
        y_s = [self.es, self.s, self.p, self.e]
        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["m", "m"],
                ["dm", "v"],
            ],
        )
        x = [i*self.h for i in range(self.n + 1)]
        for (fig_i, key) in enumerate(ax_dict):
            if fig_i == 2:
                dy = diff(y_s[2])/diff(x)
                ax_dict[key].plot(y_s[1][:-1], dy)
                ax_dict[key].set_ylabel("Rate of change of P (/min)")
                ax_dict[key].set_xlabel("S (µM)")
                for i in range(self.n):
                    if max(dy) == dy[self.n - i - 1]:
                        idx = self.n - i - 1
                        break
                ax_dict[key].text(y_s[1][idx],  dy[idx], \
                    "max value: " + str(dy[idx]) + "\nat: " + str(y_s[1][idx]), \
                        horizontalalignment="right", verticalalignment="top", fontsize=17)
                ax_dict[key].plot(y_s[1][idx], dy[idx], 'o', markersize=20, alpha=0.5)
                continue
            for i in range(len(y_s)):
                ax_dict[key].plot((x if fig_i == 0 else x[:-1]), \
                    (y_s[i] if fig_i == 0 else diff(y_s[i])/diff(x)), \
                        label=(names[i] + ("(µM)" if fig_i == 0 else " (µM/sec)")))
            ax_dict[key].set_ylabel("Runge-Kutta result " + ("(µM)" if fig_i == 0 else " (µM/sec)"))
            ax_dict[key].set_xlabel("Time (sec)")
            ax_dict[key].legend()
            ax_dict[key].grid(True)
        plt.show()

def main():
    rk = RK([100, 600, 150], es_0=0, s_0=10, p_0=0, e_0=1, lim_l=0, lim_r=10, n=1000)
    rk.run()
    rk.update()
    rk.plot()

if __name__ == "__main__":
    main()
