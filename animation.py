import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def collatz_sequence(n):
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq

seq = collatz_sequence(27)
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', color='blue')
ax.set_xlim(0, len(seq))
ax.set_ylim(0, max(seq) * 1.1)
ax.set_xlabel("Step")
ax.set_ylabel("Value")
ax.set_title("Collatz Sequence Animation (n=27)")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x = np.arange(frame + 1)
    y = seq[:frame + 1]
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(seq), init_func=init,
                              blit=True, interval=50, repeat=False)

# Save the animation as GIF
ani.save("collatz_sequence.gif", writer='pillow', fps=30)

plt.close(fig)  # Close the plot window after saving









