from src.Matthieu import *
from src.util_images import *

Unet=U_net((544,544,1))
img_dim = (572,572,1)
train_test_split = 0.4
X, Y = get_annotated_data(4, new_size=(544,544))
X_train, Y_train = X[:2], Y[:2]
X_test, Y_test = X[2:], Y[2:]

model = Unet.construct_network()
model.fit(X_train,Y_train, validation_split=0.1, batch_size=32, epochs=2, shuffle=True)
model.evaluate(X_train,Y_train)
model.evaluate(X_test, Y_test)

