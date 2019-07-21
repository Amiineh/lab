import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

obs_data = np.load('trajs/obs_data.npy')
pos_data = np.load('trajs/pos_data.npy')
pos_data = pos_data[:,:2]

plt.plot(pos_data[:, 0], pos_data[:, 1])
plt.show()


save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images_rat/')
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(obs_data.shape[0]):
    im = Image.fromarray(obs_data[i])
    im.save(save_path + 'image_' + str(i) + '.jpeg')
