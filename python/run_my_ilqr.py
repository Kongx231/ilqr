import numpy as np
import matplotlib.animation as animation
from my_ilqr import ILQR
# Import animator
from animate import animate_cart_pole

x_init = np.zeros(4) # pole is at lowest location
dt = 0.005
total_n = 1000
u_traj_init = 0.1*np.ones((total_n, 1))

ilqr = ILQR(x_init, dt, u_traj_init, Q=[0.0, 0.0, 0.0, 0.0], R=[0.001], Qf=[10.0, 0.0, 100.0, 50.0])
ilqr.plot()

# # # Run ilqr to find trajectory
total_iteration = 50
ilqr.run(total_iteration)
ilqr.plot()

print("final states:{}".format(ilqr.x_traj[-1]) )

anim = animate_cart_pole(np.array(ilqr.x_traj),np.array(ilqr.u_traj),dt,ilqr.dyn.par)
anim