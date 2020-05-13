from src.util_images import *
import matplotlib.pyplot as plt
import numpy as np

J=np.array([[1,2,3],[2,3,4],[1,1,1]])
plt.imshow(J)
G=np.array([[5,1,6],[7,1,8],[1,1,1]])
H=np.stack((G,J))
print(H.shape)
T=H.reshape((2,3,3,1))
print(T.shape)
print(T[0][:,:,0])
X,Y = get_annotated_data(1)



F=X[0][:,:,0]
plt.imshow(F)
print(F.shape)
F1=Y[0][:,:,0]
plot_image(image_with_mask(F,F1))
plot_image(X[0][:,:,0])
X, Y = get_annotated_data(10, new_size=(572, 572), show_images=False)
print(X[0][:,:,0].shape)
plot_image(X[0][:,:,0])