# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import chainer
from DCGAN import Generator

gen = Generator()

# if you want to generate a pic from epoch i, below
# chainer.serializers.load_npz("Generator_epochi.npz", gen)
chainer.serializers.load_npz("Generator_final.npz", gen)

z = np.random.uniform(-1.0, 1.0,
                      (128, 100)).astype(dtype=np.float32)
# change 0th index to see other pics
plt.imshow(gen(z).data[10][0], cmap='gray')
plt.show()
