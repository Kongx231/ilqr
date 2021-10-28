import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_pendulum(states,inputs,dt,parameters):
    # Animation follows https://matplotlib.org/2.0.2/examples/animation/double_pendulum_animated.html
    # Animate
    pendulum_length = parameters[2]
    x_pos = pendulum_length*np.sin(states[:, 0])
    y_pos = -pendulum_length*np.cos(states[:, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    def animate(i):
        thisx = [0, x_pos[i]]
        thisy = [0, y_pos[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)),
                                  interval=25, blit=True, init_func=init)

    # ani.save('double_pendulum.mp4', fps=15)
    plt.show()

def animate_pendulum_tracking(states,states_new,inputs,dt,parameters):
    # Animation follows https://matplotlib.org/2.0.2/examples/animation/double_pendulum_animated.html
    # Animate
    pendulum_length = parameters[2]
    x_pos = pendulum_length*np.sin(states[:, 0])
    y_pos = -pendulum_length*np.cos(states[:, 0])

    x_pos_new = pendulum_length*np.sin(states_new[:, 0])
    y_pos_new = -pendulum_length*np.cos(states_new[:, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    line_new, = ax.plot([], [], 'o-', lw=2,linestyle='dashed',color='red')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        line.set_data([], [])
        line_new.set_data([], [])
        time_text.set_text('')
        return line,line_new, time_text


    def animate(i):
        thisx = [0, x_pos[i]]
        thisy = [0, y_pos[i]]

        thisx_new = [0, x_pos_new[i]]
        thisy_new = [0, y_pos_new[i]]

        line.set_data(thisx, thisy)
        line_new.set_data(thisx_new, thisy_new)
        time_text.set_text(time_template % (i*dt))
        return line,line_new, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)),
                                  interval=25, blit=True, init_func=init)

    # ani.save('double_pendulum.mp4', fps=15)
    plt.show()