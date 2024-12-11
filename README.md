# RegressionExperiments
## Experiment 1:
I made a simulation (environment.py) that randomly generates a 2-D point (x,y) and calculates its distance away from the line y=x+1 (run_simulation function). This program then plots the (x,y) coordinates of the points and its distance in a 3-D plot.

https://github.com/user-attachments/assets/aa9a3c05-cc90-4ecb-94a6-ad69bc2815ae

I then trained a linear regressor (linear_regression.py) to predict the distance given the (x,y) coordinates. It's mean squared error was rather high (275.3209416214334) and its predictions did not reflect the ground truth:

https://github.com/user-attachments/assets/3da1ae6c-07b6-4b84-b2ea-3624d2c120ce

I then created its' residual plot, and it shows a clear pattern:

https://github.com/user-attachments/assets/c3889249-96c8-451a-b3b3-11f1985dcb2b


I then trained a similar SVM model (SVM.py) to predict the distance given the (x,y) coordinates. It's mean squared error was much lower (0.012274280029574867), and its predictions seem to reflect the ground truth:

https://github.com/user-attachments/assets/382ca31e-2713-4f91-bf12-e9878373e1e3

I then created its' residual plot, which is mostly random:

https://github.com/user-attachments/assets/a31fae2b-2857-438b-9b54-4ffd1135d328

This implies that the SVM is the better fit for this dataset.
