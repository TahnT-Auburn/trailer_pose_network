#%%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation


est_array = np.random.randn(100)
truth_array = np.random.randn(100)

fig, ax = plt.subplots(2,1)
est, = ax[0].plot(np.rad2deg(est_array[0]))
truth, = ax[0].plot(np.rad2deg(truth_array[0]), '--')
# ax[0].xlabel('Frames')
# ax[0].ylabel('Hitch [deg]')
ax[0].legend(['Estimated', 'Truth'], loc='upper left')
ax[0].set(xlim=[0,len(est_array)],
          ylim=[-15,30],
          ylabel = 'Hitch [deg]',
          xlabel = 'Frames')

# plt.subplot(2,1,2)
error, = ax[1].plot(np.rad2deg(truth_array[0]) - np.rad2deg(est_array[0]))
# ax[1].xlabel('Frames')
# ax[1].ylabel('Error [deg]')
# ax[1].title('Trailer Articulation Angle')
ax[1].set(xlim=[0,len(est_array)],
          ylim=[-2.5,1.5],
          ylabel = 'Error [deg]',
          xlabel = 'Frames')

fig.tight_layout()
# plt.show()

def animate(i):
    est.set_data(np.arange(i), np.rad2deg(est_array[:i]))
    truth.set_data(np.arange(i), np.rad2deg(truth_array[:i]))
    error.set_data(np.arange(i), np.rad2deg(truth_array[:i]) - np.rad2deg(est_array[:i]))
    return est,truth,error

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              interval=40,
                              frames=len(est_array))
                            #   frames=len(est_array))
plt.show()