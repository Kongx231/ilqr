from symbolic_dynamics import symbolic_cart_pole
import numpy as np
import matplotlib.pyplot as plt

class DynamicsSymbolic:
  """
  system dynamics: https://underactuated.mit.edu/acrobot.html
  states: x_dot, x, theta_dot, theta
  """
  par = np.array([0.5, 0.2, 9.8, 0.3])
  Ad = None
  Bd = None
  fd = None

  def __init__(self):
    self.fd,self.Ad,self.Bd = symbolic_cart_pole()

  def DiscreteTimeLinearize(self, z_ref:np.array, u_ref: np.array, dT:float) -> tuple:
    """ Linear system dynamics in discrete time. """
    return self.Ad(z_ref, u_ref, dT, self.par), self.Bd(z_ref, u_ref, dT, self.par)

  def CalcXNext(self, z:np.array, u:np.array, dT:float) -> np.array:
    """ This is the continuous time nonlinear dynamics. """
    return self.fd(z, u, dT, self.par)

class ILQR:
  """
  iLQR steps:
    backward:
      Calculate list[Ad], list[Bd] from initial trajectory x_ref, u_ref.
      Given Ad, Bd, list[Qd], list[Rd], Qf, calculate list[k1], list[k2], such that du = k1 + k2 * dx
    forward:
      rollout along time sequence, given du = k1 + k2 * dx, rollout the new x_traj and use it as x_ref in the next iteration.

  Consider problem:
  min_u sum( l(xn, un) ) + lf(xN, uN)
  lf(xN, uN) = Q1 * x_dot**2 + Q2 * x**2 + Q3 * theta_dot**2 + Q4 * (theta-pi)**2
  l(xn, un) = Q1 * x_dot**2 + Q2 * x**2 + Q3 * theta_dot**2 + Q4 * (theta-pi)**2 + R * u**2
  Vx[n] below means delta_V / delta_x at location x_ref[n], this is coefficient for delta_x[n]
  """
  x_traj: list # length N
  u_traj: list # length N+1
  x_init: np.array
  Nx: int
  Nu: int
  dyn: DynamicsSymbolic
  Q: list
  R: float
  N: int # horizon of the LQR
  dT: float

  target_x = np.array([0.0,0.0,0.0,np.pi])

  expected_cost_reduction: float
  expected_cost_reduction_: float
  expected_cost_reduction_hess_: float

  debug_on = False

  def __init__(self, x_init, dT, u_traj, Q=[1.0, 0.0, 1.0, 1.0], R=1, Qf=[1.0, 0.0, 1.0, 1.0]):
    self.dyn = DynamicsSymbolic()
    self.Nx = x_init.shape[0]
    self.Nu = u_traj[0].shape[0]
    self.dT = dT
    # Q, Qf, R are tunned for discrete system dynamics with dT
    self.Q = Q
    self.R = np.array(R)
    self.Qf = Qf
    self.N = len(u_traj)
    self.u_traj = u_traj
    self.x_traj = [x_init]
    for i in range(self.N):
      x_next = self.dyn.CalcXNext(self.x_traj[i], self.u_traj[i], self.dT).flatten()
      self.x_traj.append(x_next)

  def __get_l_derivatives(self, x_ref, u_ref):
    """
    Derivative terms up to 2nd order for running cost l(xn, un)
    """
    lx = 2*np.diag(self.Q) @ (x_ref - self.target_x)
    lu = np.zeros(self.Nu)
    lu[0] = 2 * self.R * u_ref[0]
    lux = np.zeros((self.Nu,self.Nx))
    lxx = 2 * np.diag(self.Q)
    luu = np.zeros((self.Nu, self.Nu))
    luu[0,0] = 2 * self.R
    if self.debug_on:
      print("__get_l_derivatives", lx, lu, lux, lxx, luu)
    return lx, lu, lux, lxx, luu

  def __get_V_terminal_derivatives(self, xN):
    Vx_N = np.zeros(xN.shape)
    Vx_N[0] = 2 * self.Qf[0] * xN[0]
    Vx_N[1] = 2 * self.Qf[1] * xN[1]
    Vx_N[2] = 2 * self.Qf[2] * xN[2]
    Vx_N[3] = 2 * self.Qf[3] * (xN[3] - np.pi)
    Vxx_N = 2 * np.diag(self.Qf)
    return Vx_N, Vxx_N


  def __get_q_derivatives(self, x_ref, u_ref, Vx_nplus1, Vxx_nplus1):
    lx, lu, lux, lxx, luu = self.__get_l_derivatives(x_ref, u_ref)
    Ad, Bd = self.dyn.DiscreteTimeLinearize(x_ref, u_ref, self.dT)
    # Bd = Bd.flatten
    if self.debug_on:
      print("Ad", Ad)
      print("Bd", Bd)

    Qx = lx + Ad.T @ (Vx_nplus1)
    Qu = lu + (Bd.T @ (Vx_nplus1)).flatten()
    Qxx = lxx + Ad.T @ (Vxx_nplus1) @ Ad
    Qux = lux + Bd.T @ (Vxx_nplus1) @ Ad
    Quu = luu + Bd.T @ Vxx_nplus1 @ Bd
    if self.debug_on:
      print("__get_q_derivatives", Qx, Qu, Qxx, Qux, Quu)
    return Qx, Qu, Qxx, Qux, Quu


  def __solveGains(self, Qx, Qu, Qxx, Qux, Quu):
    """
    Perform optimization to find k1, k2 such that du = k1 + k2 * dx
    """
    if self.debug_on:
      print("Quu", Quu)
    Quu_inv = np.linalg.inv(Quu)
    k1 = -Quu_inv @ (Qu)
    k2 = -Quu_inv @ (Qux)
    if self.debug_on:
      print("__solveGains", k1, k2)
    return (k1, k2)


  def __update_v_derivatives(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, k1, k2):
    V_x = np.zeros(Q_x.shape)
    V_xx = np.zeros(Q_xx.shape)
    V_x = Q_x + k2.T.dot(Q_u) + Q_ux.T.dot(k1) + k2.T @ Q_uu @ k1
    V_xx = Q_xx + 2*k2.T.dot(Q_ux) + (k2.T.dot(Q_uu)).dot(k2)
    return V_x, V_xx


  def forward(self, k1_list, k2_list, learning_rate) -> None:
    # update x_traj and u_traj
    delta_x = np.zeros((self.Nx, 1))
    x_traj_new = [x for x in self.x_traj]
    u_traj_new = [u for u in self.u_traj]
    for i in range(self.N):
      if self.debug_on:
        print("k1_list[i]",k1_list[i])
        print("k2_list[i]",k2_list[i])
        print("delta_x",delta_x)
        print("self.u_traj[i]",self.u_traj[i])

      delta_u = k1_list[i] * learning_rate + k2_list[i].flatten() @ delta_x.flatten()
      if self.debug_on:
        print("delta_u",delta_u)
      u_traj_new[i] = delta_u + self.u_traj[i]
      x_next = self.dyn.CalcXNext(x_traj_new[i], u_traj_new[i], self.dT).flatten()
      delta_x = x_next - x_traj_new[i+1]
      x_traj_new[i+1] = x_next
    return x_traj_new, u_traj_new


  def backward(self) -> tuple:
    # create containers:
    k1_list,k2_list = [np.zeros((self.Nu,self.Nu)) for _ in range(self.N)],[np.zeros((self.Nu,self.Nx)) for _ in range(self.N)]
    Vx_list = [np.zeros((self.Nx,1)) for _ in range(self.N+1)]
    Vxx_list = [np.zeros((self.Nx,self.Nx)) for _ in range(self.N+1)]

    # Initialize Vx[N] and Vxx[N]:
    Vx_list[self.N], Vxx_list[self.N] = self.__get_V_terminal_derivatives(self.x_traj[self.N])

    # Initialize cost reduction
    expected_cost_reduction = 0.0
    expected_cost_reduction_grad = 0.0
    expected_cost_reduction_hess = 0.0

    if self.debug_on:
      print("Vx_list[self.N]",Vx_list[self.N])
      print("Vxx_list[self.N]",Vxx_list[self.N])

    for n in reversed(range(self.N)):
      Qx, Qu, Qxx, Qux, Quu = self.__get_q_derivatives(self.x_traj[n], self.u_traj[n], Vx_list[n+1], Vxx_list[n+1])
      k1, k2 = self.__solveGains(Qx, Qu, Qxx, Qux, Quu)
      Vx_list[n], Vxx_list[n] = self.__update_v_derivatives(Qx, Qu, Qxx, Qux, Quu, k1, k2)

      # Update the expected reduction
      current_cost_reduction_grad = -Qu.T@k1
      current_cost_reduction_hess = 0.5 * k1.T @ Quu @ (k1)
      current_cost_reduction = current_cost_reduction_grad + current_cost_reduction_hess

      expected_cost_reduction_grad +=  current_cost_reduction_grad
      expected_cost_reduction_hess +=  current_cost_reduction_hess
      expected_cost_reduction += + current_cost_reduction

      # logging
      k1_list[n] = k1
      k2_list[n] = k2

    # Store expected cost reductions
    self.expected_cost_reduction_grad_ = expected_cost_reduction_grad
    self.expected_cost_reduction_hess_ = expected_cost_reduction_hess
    self.expected_cost_reduction_ = expected_cost_reduction
    return k1_list, k2_list


  def compute_cost(self,x_traj,u_traj):
    # Initialize cost
    total_cost = 0.0
    for ii in range(self.N):
        current_x = x_traj[ii] # Not being used currently
        current_u = u_traj[ii].flatten()
        current_cost = current_x.T @ np.diag(self.Q) @ current_x + current_u.T @ np.diag(self.R) @ current_u
        total_cost = total_cost+current_cost
    # Compute terminal cost
    terminal_difference = (self.target_x - x_traj[-1]).flatten()
    terminal_cost = terminal_difference.T@np.diag(self.Qf)@terminal_difference
    total_cost = total_cost+terminal_cost
    return total_cost


  def run(self, n_iter = 50):
    armijo_threshold = 0.1
    
    current_cost = self.compute_cost(self.x_traj, self.u_traj)
    for iter in range(n_iter):
      learning_rate = 1
      k1_list, k2_list = self.backward()
      print('Initial expected cost reduction: ',self.expected_cost_reduction_)

      if (self.expected_cost_reduction_ < 0.001):
        print("Stopping optimization, optimal trajectory")
        break

      print('Starting iteration: ',iter,', Current cost: ',current_cost)
      while learning_rate > 0.05:
        x_traj, u_traj = self.forward(k1_list, k2_list, learning_rate)
        new_cost = self.compute_cost(x_traj, u_traj)
        expected_cost_redu = learning_rate*self.expected_cost_reduction_grad_ + learning_rate*learning_rate*self.expected_cost_reduction_hess_
        # print("expected_cost_redu, ", expected_cost_redu, "new_cost", new_cost, "current_cost", current_cost)
        if (current_cost - new_cost)/expected_cost_redu > armijo_threshold:
          current_cost = new_cost
          self.x_traj = x_traj
          self.u_traj = u_traj
          break
        else:
          learning_rate = 0.95 * learning_rate
      if (learning_rate < 0.05):
        print("Stopping optimization, low learning rate")
        break
      self.expected_cost_reduction_ = expected_cost_redu
      print("expected_cost_redu, ", expected_cost_redu, "new_cost", new_cost, "current_cost", current_cost)
    return


  def plot(self):
    times = np.arange(self.N) * self.dT
    fig, ax = plt.subplots(2,1)
    for x_id in range(self.Nx-1):
      ax[0].plot(times, [x[x_id] for x in self.x_traj[:self.N]], '-', label=x_id)

    ax[0].plot(times, [x[3] - np.pi for x in self.x_traj[:self.N]], '-', label=3)
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(times, [u[0] for u in self.u_traj[:self.N]], '-', label='u')
    ax[1].legend()
    ax[1].grid()
    fig.show()
