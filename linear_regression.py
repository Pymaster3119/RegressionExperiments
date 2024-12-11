import environment
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import os

#Train
sample_num = 10000
data = []
for i in range(sample_num):
    data.append(environment.run_simulation())

X = np.array([[point[0],point[1]] for point in data])
y = np.array([point[2] for point in data])

linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

#Test
test_data = [environment.run_simulation() for i in range(sample_num)]
X_test = np.array([[point[0], point[1]] for point in test_data])
y_test = np.array([point[2] for point in test_data])

y_pred = linear_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#Create a plot of regressor data
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x = []
y = []
z = []

for i in range(sample_num):
    x.append(X_test[i][0])
    y.append(X_test[i][1])
    z.append(y_pred[i])

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax.scatter(x, y, z, c="b", marker = 'o')
ax.set_xlabel('X Value')
ax.set_ylabel('Y Value')
ax.set_zlabel('Signed Distance')
ax.set_zlim(z.min(),z.max())

for angle in tqdm.tqdm(range(0,360)):
    ax.view_init(elev = 30, azim = angle)
    plt.savefig(f"linearregressorpred/frame_{angle}.png")
os.system("ffmpeg -framerate 30 -i linearregressorpred/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p linear_regressor_pred.mp4")

#Create the residual plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x = []
y = []
z = []

for i in range(sample_num):
    x.append(X_test[i][0])
    y.append(X_test[i][1])
    z.append(environment.run_simulation(X_test[i][0],X_test[i][1])[2] - y_pred[i])

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax.scatter(x, y, z, c="b", marker = 'o')
ax.set_xlabel('X Value')
ax.set_ylabel('Y Value')
ax.set_zlabel('Residual')
ax.set_zlim(z.min(),z.max())

for angle in tqdm.tqdm(range(0,360)):
    ax.view_init(elev = 30, azim = angle)
    plt.savefig(f"linearregressorresiduals/frame_{angle}.png")
os.system("ffmpeg -framerate 30 -i linearregressorresiduals/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p linear_regressor_residuals.mp4")
