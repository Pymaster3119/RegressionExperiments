import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import os
line_slope = 1
line_y_int = 1
def run_simulation(x = None, y = None):
    if x == None:
        x = random.random() * 100
    if y == None:
        y = random.random() * 100

    signed_distance = (abs(line_slope * x - y + line_y_int))/math.sqrt(line_slope ** 2 + 1)
    
    #Return tuple: (x, y, ground truth)
    return (x,y,signed_distance)



if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    num_samples = 10000
    x = []
    y = []
    z = []
    for i in range(num_samples):
        result = run_simulation()
        x.append(result[0])
        y.append(result[1])
        z.append(result[2])
    ax.scatter(x, y, z, c="b", marker = 'o')
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.set_zlabel('Signed Distance')
    ax.set_zlim(-100,100)

    for angle in tqdm.tqdm(range(0,360)):
        ax.view_init(elev = 30, azim = angle)
        plt.savefig(f"simgroundtruth/frame_{angle}.png")

    os.system("ffmpeg -framerate 30 -i simgroundtruth/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation_ground_truth.mp4")
    plt.show()