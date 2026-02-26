import matplotlib.pyplot as plt
import numpy as np

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.figure(figsize = (8,8))
plt.plot(x, y)

plt.axis((0, 125, 0, 330))

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.legend(["Sports Watch Data", "Calorie Burnage"])
plt.grid(linewidth = 0.3)

plt.savefig('plots/foo.jpg', bbox_inches='tight')
plt.close()

def generic_plot(*data_sets):
