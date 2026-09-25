import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MEAN_HOURS = 2.5
STD_DEV = 0.8
NUM_DAYS = 30

np.random.seed(42)
daily_gaming_hours = np.random.normal(loc=MEAN_HOURS, scale=STD_DEV, size=NUM_DAYS)
daily_gaming_hours = np.clip(daily_gaming_hours, 0, None)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 5))

x = np.linspace(0, 4 * np.pi, 500)
line, = ax.plot([], [], lw=2.5, color='#00FFAA', label='Gaming Activity Wave')
mean_line = ax.axhline(MEAN_HOURS, color='#FF5555', linestyle='--', alpha=0.7, label=f'Mean ({MEAN_HOURS} hrs)')

ax.axhspan(MEAN_HOURS - STD_DEV, MEAN_HOURS + STD_DEV, color='#FF5555', alpha=0.15, label=f'±1 SD Band ({STD_DEV} hrs)')

ax.set_xlim(0, 4 * np.pi)
ax.set_ylim(-1, MEAN_HOURS + 2.5 * STD_DEV)
ax.set_title("Gaming Activity: Statistical Sine Wave", fontsize=14, pad=12)
ax.set_xlabel("Cycle Phase (t)", fontsize=11)
ax.set_ylabel("Gaming Intensity / Hours", fontsize=11)
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.4)

def animate(frame):
    # The frame number will now just increment indefinitely
    day_idx = (frame // 10) % NUM_DAYS # Cycle through daily_gaming_hours based on frame
    current_intensity = daily_gaming_hours[day_idx]

    phase_shift = frame * 0.08
    frequency = 1.0 + (current_intensity / (MEAN_HOURS + 1e-5)) * 0.3
    amplitude = current_intensity * 0.5

    y = MEAN_HOURS + amplitude * np.sin(frequency * x - phase_shift)
    line.set_data(x, y)
    return line,

ani = FuncAnimation(
    fig,
    animate,
    interval=30,
    blit=True,
    repeat=True, # Set repeat to True for continuous loop
    cache_frame_data=False # Suppress UserWarning for unbounded cache
)

plt.show()
