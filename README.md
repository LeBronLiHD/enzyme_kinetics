# Question 2: Enzyme Kinetics

Haodong LI, Dec. 27, 2022

## 8.1

- Firstly, we can divide the whole chemical reaction into three basic reactions:

$$
\begin{aligned}
& E+S \ \ \underset{k_2}{\stackrel{k_1}{\rightleftharpoons}} \ \ E S\ \  \stackrel{k_3}{\rightarrow} \ \ E+P \\ \\
\Rightarrow\quad & E+S \stackrel{k_1}{\rightarrow} E S
\\ \\
\Rightarrow\quad &E S \stackrel{k_2}{\rightarrow} E+S \\\\
\Rightarrow\quad &E S \stackrel{k_3}{\rightarrow} E+P
\end{aligned}
$$

- According to the law of mass action, we get the reaction rate equation system:

$$
\begin{aligned}
\frac{d P}{d t}&=k_3 E S
\\\\\frac{d E S}{d t}&=k_1 ( E\cdot S )-(k_2+k_3) E S
\\\\
\frac{d S}{d t}&=-k_1 ( E\cdot S )+k_2 E S
\\\\ \frac{d E}{d t}&=-k_1 ( E\cdot S )+(k_2+k_3) E S
\end{aligned}
$$

# 8.2

- Since there are only substrates and enzymes in the initial system, we give the following initial values:

$$
E(0)=E_0 =  1 \mu M ,\quad S(0)=S_0=10 \mu M, \quad ES(0)=ES_0=0, \quad P(0)=P_0=0
$$

- And the rate constants are: 

$$
k_1=100/\mu M/\min, \quad k_2=600 /\min, \quad k_3=150 /\min.
$$

- It is not difficult to see that our current system is redundant, so the number of differential equations can be reduced according to the conservation law. We notice that:

$$
\begin{aligned}
&\frac{d}{d t}(ES+E)=0 \\\\
\Rightarrow \quad & ES(t)+E(t)=E_0
\\\\
\Rightarrow \quad & E(t)=E_0-ES(t)
\end{aligned}
$$

- Therefore, we can discard the rate-of-change equation that $E$ satisfies, and replace $E$ in other equations with $E_0 −ES$:

$$
\begin{aligned}
\frac{d P}{d t}&=k_3 E S
\\\\\frac{d E S}{d t}&=k_1 ( (E_0 −ES)\cdot S )-(k_2+k_3) E S
\\\\
\frac{d S}{d t}&=-k_1 ( (E_0 −ES)\cdot S )+k_2 E S
\end{aligned}
$$

- Continue to deduce, we notice that:

$$
\begin{aligned}
&\frac{d P}{d t}=k_3 E S = -\frac{d E S}{d t} - \frac{d S}{d t}\\\\
\Rightarrow \quad & P(t) - P_0 = -(ES(t) -ES_0) - (S(t)- S_0)\\\\
\Rightarrow \quad &
P(t) = S_0 - (ES(t) + S(t))\\\\
\text{and}\quad & S(t) = S_0 - (ES(t) + P(t))
\end{aligned}
$$

- Therefore we can get:

$$
\begin{aligned}
\frac{d E S}{d t}&= k_1( (E_0 −ES)\cdot S) - (k_2 + k_3)ES
\\\\
\frac{d S}{d t}&=-k_1( (E_0 −ES)\cdot S) + k_2ES
\end{aligned}
$$

- Now we know that for two general 1st order ODE's:

$$
\begin{aligned}
& \frac{des}{d t}=f(t,es,s) =k_1(e_0 - es)\cdot s - (k_2 + k_3)es \\\\
& \frac{ds}{d t}=g(t,es,s) = -k_1(e_0 - es)\cdot s + k_2es
\end{aligned}
$$

- The 4th order Runge-Kutta formula's for a system of 2 ODE's are:

$$
\begin{aligned}
es_{i+1} & =es_i+\frac{1}{6}\left(k_0+2 k_1+2 k_2+k_3\right) \\\\
s_{i+1} & =s_i+\frac{1}{6}\left(l_0+2 l_1+2 l_2+l_3\right)
\end{aligned}
$$

- We must do the calculations in a certain order as there are dependencies between the numerical calculations. This order is:

$$
\begin{aligned}
& k_0=h f\left(t_i, es_i, s_i\right) \\\\
& l_0=h g\left(t_i, es_i, s_i\right) \\\\
& k_1=h f\left(t_i+\frac{1}{2} h, es_i+\frac{1}{2} k_0, s_i+\frac{1}{2} l_0\right) \\\\
& l_1=h g\left(t_i+\frac{1}{2} h, es_i+\frac{1}{2} k_0, s_i+\frac{1}{2} l_0\right) \\\\
& k_2=h f\left(t_i+\frac{1}{2} h, es_i+\frac{1}{2} k_1, s_i+\frac{1}{2} l_1\right) \\\\
& l_2=h g\left(t_i+\frac{1}{2} h, es_i+\frac{1}{2} k_1, s_i+\frac{1}{2} l_1\right) \\\\
& k_3=h f\left(t_i+h, es_i+k_2, s_i+l_2\right) \\\\
& l_3=h g\left(t_i+h, es_i+k_2, s_i+l_2\right) \\\\
& es_{i+1}=es_i+\frac{1}{6}\left(k_0+2 k_1+2 k_2+k_3\right) \\\\
& s_{i+1}=s_i+\frac{1}{6}\left(l_0+2 l_1+2 l_2+l_3\right)
\end{aligned}
$$

- Here is the python program:

```python
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
            self.p.append(
                self.s[0] - (self.es[i] + self.s[i])
            )
            self.e.append(
                self.e[0] - self.es[i]
            )

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
```

- And the Runge-Kutta result:

![img](https://github.com/LeBronLiHD/NTU_Enzyme_Kinetics/blob/main/result.png)

## 8.3

- We assume that the substrate is in instantaneous equilibrium with the complex, and thus:

$$
\begin{aligned}
k_2 E S &= k_1 ( E\cdot S ) \\\\
\Rightarrow\quad  ES &= \frac{k_1(E\cdot S)}{k_2}
\end{aligned}
$$

- Since $E = E_0 −ES$ , we find that:

$$
\begin{aligned}
ES &= \frac{k_1(E\cdot S)}{k_2}\\\\
&= \frac{k_1(S\cdot (E_0-ES))}{k_2}\\\\
\Rightarrow\quad  ES &= \frac{k_1(S\cdot E_0)}{k_1\cdot S + k_2}\\\\
&=\frac{S\cdot E_0}{S + K_1}
\end{aligned}
$$

- where $K_1=k_2/ k_1$. Hence, the velocity, $V$, of the reaction, i.e the product is formed, is given by

$$
\begin{aligned}
V&=\frac{d P}{d t}\\\\&=k_2 ES\\\\&=\frac{k_2 E_0\cdot S}{K_1+S}\\\\&=\frac{V_{\max } S}{K_1+S}\\\\
\Rightarrow\quad V_{max} &= k_2 E_0
\end{aligned}
$$

- When the concentrations of $S$ are small, the velocity $V$ increases approximately linearly. At large concentrations of $S$, however, the velocity $V$ saturates to a maximum value, $V_{max}$
- And from the above image, the $V_{max} = 5.38 (\mu M/\min)$
