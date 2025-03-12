import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

couleurs = np.array([[1,255,112],[46,204,64],[61,153,112],[17,17,17],[255,255,255],[170,170,170],[240,18,190],[255,0,0],[255,220,0],[221,221,221],[255,133,27],[127,219,255],[0,116,217],[57,204,204],[117,13,201]])


img = mpimg.imread('nico.png')
new_image = np.zeros_like(img)
img *= 255
for nx,x in enumerate(img):
    for ny,y in enumerate(x):
        c_max = np.array([np.inf,np.inf,np.inf])
        for c in couleurs:
            dist = np.linalg.norm(np.array(c)-y)
            if dist < np.linalg.norm(np.array(c_max)-y):
                c_max = c
        new_image[nx,ny] = c_max

plt.imsave('nico_transfo.png', new_image.astype(np.uint8))