import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

state = np.random.rand([16, 16])
plt.imshow(state)
plt.show()