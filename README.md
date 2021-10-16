# ilqr
Matlab iLQR class which will solve for the optimal set of inputs and gains to get to a desired state

# ilqr_mpc
Matlab iLQR mpc class which will take a trajectory as an input, and solve for the optimal tracking input for a given horizon.

# Instructions
To run pendulum example, go to example_pendulum.m to create a pendulum swing up trajectory.
To run car parking example, go to example_parking.m to create a car parking trajectory.

To run mpc example, go to example_pendulum_ilqr_mpc.m to track the swingup trajectory produced from example_pendulum.m

For your own system, it is advised to create a similar file as the "symbolic_dynamics".


