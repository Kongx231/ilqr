import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_cart_pole(states,inputs,dt,parameters):
    pendulum_length = parameters[3]
    x_cart = states[:,1]
    y_cart = 0.0 * x_cart
    x_pos = pendulum_length*np.sin(states[:, 3]) + x_cart
    y_pos = -pendulum_length*np.cos(states[:, 3]) + y_cart

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(states[:,1].min()-pendulum_length, states[:,1].max()+pendulum_length), ylim=(-pendulum_length*2, pendulum_length*2))
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    def animate(i):
        thisx = [x_cart[i], x_pos[i]]
        thisy = [y_cart[i], y_pos[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)),
                                  interval=2, blit=True, init_func=init)

    return ani

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

    # ani.save('pendulum_swingup.mp4', fps=15)
    # plt.show()
    return ani

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

    # ani.save('pendulum_swingup_tracking.mp4', fps=15)
    plt.show()

def animate_acrobot(states,inputs,dt,parameters):
    # Animation follows https://matplotlib.org/2.0.2/examples/animation/double_pendulum_animated.html
    # Animate
    pendulum_length = parameters[2]
    x1_pos = pendulum_length*np.cos(states[:, 0])
    y1_pos = pendulum_length*np.sin(states[:, 0])

    x2_pos = pendulum_length*np.cos(states[:, 0]+states[:, 1]) + x1_pos
    y2_pos = pendulum_length*np.sin(states[:, 0]+states[:, 1]) + y1_pos

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    def animate(i):
        thisx = [0, x1_pos[i],x2_pos[i]]
        thisy = [0, y1_pos[i],y2_pos[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)),
                                  interval=1, blit=True, init_func=init)

    # ani.save('acrobot_swingup.mp4', fps=15)
    plt.show()

def animate_acrobot_tracking(states,states_new,inputs,dt,parameters):
    # Animation follows https://matplotlib.org/2.0.2/examples/animation/double_pendulum_animated.html
    # Animate
    pendulum_length = parameters[2]
    x1_pos = pendulum_length*np.cos(states[:, 0])
    y1_pos = pendulum_length*np.sin(states[:, 0])

    x2_pos = pendulum_length*np.cos(states[:, 0]+states[:, 1]) + x1_pos
    y2_pos = pendulum_length*np.sin(states[:, 0]+states[:, 1]) + y1_pos

    x1_pos_new = pendulum_length*np.cos(states_new[:, 0])
    y1_pos_new = pendulum_length*np.sin(states_new[:, 0])

    x2_pos_new = pendulum_length*np.cos(states_new[:, 0]+states_new[:, 1]) + x1_pos_new
    y2_pos_new = pendulum_length*np.sin(states_new[:, 0]+states_new[:, 1]) + y1_pos_new

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
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
        thisx = [0, x1_pos[i],x2_pos[i]]
        thisy = [0, y1_pos[i],y2_pos[i]]

        thisx_new = [0, x1_pos_new[i], x2_pos_new[i]]
        thisy_new = [0, y1_pos_new[i], y2_pos_new[i]]

        line.set_data(thisx, thisy)
        line_new.set_data(thisx_new, thisy_new)
        time_text.set_text(time_template % (i*dt))
        return line,line_new, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)),
                                  interval=5, blit=True, init_func=init)

    # ani.save('acrobot_swingup.mp4', fps=15)
    plt.show()
