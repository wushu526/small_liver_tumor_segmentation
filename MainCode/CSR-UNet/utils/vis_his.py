import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

d = np.array([0.893,0.913,0.883,0.843,0.765,0.742,0.638,0.843,0.712,0.888,0.388,0.806,0.662,0.764,0.836,0.913,
              0.760,0.933,0.934,0.955,0.949,0.963,0.880,0.939,0.947,0.580,0.943,0.924,0.974,0.923,0.925,0.883,
              0.804,0.939,0.871,0.844,0.975,0.949,0.958,0.814,0.727,0.917,0.889,0.888,0.960,0.644,0.845,0.642,
              0.933,0.878,0.858,0.749,0.899,0.820])
n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# 设置y轴的上限
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

