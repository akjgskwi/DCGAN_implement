# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import chainer
from DCGAN import Generator

gen = Generator()
chainer.serializers.load_npz("Generator_model.npz", gen)

z = np.random.uniform(-1.0, 1.0,
                      (128, 100)).astype(dtype=np.float32)
plt.imshow(gen(z).data[0][0], cmap='gray')
plt.show()
