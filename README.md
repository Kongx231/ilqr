I am actively developing these classes and trying to add more examples. Please always pull the most recent changes.
# Languages
## Matlab
Original code is written in matlab. No installation is needed, the matlab files are located in the matlab directory.

## Python
To make this code more open source, I've also started porting it over to python as well. However, python is currently significantly slower due to sympy's lambdify function.
### Installation
This package makes use of numpy, sympy, and matplotlib. The python files are located in the python directory.
```
pip install numpy
pip install matplotlib
pip install sympy
```

# Main classes and functions
## ilqr
iLQR class which will solve for the optimal set of inputs and gains to get to a desired state

## ilqr_mpc
iLQR mpc class which will take a trajectory as an input, and solve for the optimal tracking input for a given horizon.
The mpc does not need to be ran on every timestep, and instead the feedback law and trajectory produced by the current solve can be used as a stabilizing controller for the rest of the horizon.

## ltvlqr
Linear time varying linear quadratic regulator (ltvlqr) function solves for the optimal gain schedule given a trajectory. Use this function, if you already have a trajectory that you would like to track.

## h_ilqr
Hybrid iLQR class which will solve the optimal set of inputs and gains for a hybrid system.

## h_ltvlqr
Coming soon: Schedule the piecewise smooth gain schedule given a hybrid trajectory.

# Systems
## Acrobot
Double pendulum model without actuation on the first link.

### Creaing optimal swing up for acrobot
[![Optimal swing up for acrobot](https://img.youtube.com/vi/4xQUNSdZmvo/0.jpg)](https://www.youtube.com/watch?v=4xQUNSdZmvo&ab_channel=NathanKong "Optimal swing up for acrobot")

## Car
A kinematic bicycle model where the inputs are steering velocity and linear acceleration.

### Tracking car parking behavior using iterative Linear Quadratic Regulator as MPC
[![Car parking using iLQR MPC](https://img.youtube.com/vi/wN9ARncBKoo/0.jpg)](https://www.youtube.com/watch?v=wN9ARncBKoo&ab_channel=NathanKong "Car parking using iLQR MPC")

## Cartpole
A cart pole model where the input is a thruster on the cart.

### Creating optimal swing up for cartpole 
[![Creating optimal swing up for cartpole](https://img.youtube.com/vi/aqu8GfT1iwU/0.jpg)](https://www.youtube.com/watch?v=aqu8GfT1iwU "Creating optimal swing up for cartpole")

## Pendulum
Single pendulum with torque on the joint as input.

### Visualizing optimal pendulum swing up for iLQR
[![Visualizing optimal pendulum swing up for iLQR](https://img.youtube.com/vi/h998mOwAlrI/0.jpg)](https://www.youtube.com/watch?v=h998mOwAlrI "Visualizing optimal pendulum swing up for iLQR")

### Tracking pendulum swing up trajectory with iLQR MPC
[![Tracking pendulum swing up trajectory with iLQR MPC](https://img.youtube.com/vi/RiQ6XPwgSgM/0.jpg)](https://www.youtube.com/watch?v=RiQ6XPwgSgM "Tracking pendulum swing up trajectory with iLQR MPC")
## Quadcopter
Quadcopter in full 3D space where the inputs are the 4 thrusters.
