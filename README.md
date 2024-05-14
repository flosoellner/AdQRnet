# AdQRnet

* Framework for Adaptive Sampling with LQR-augmented Neural Network Controls

* The interactive notebook experiments.ipynb guides through an exemplary experimental setup, allowing the reproduction of the results in Section 5, as well as testing other parameter settings and configurations. The default configuration is displayed in Table 1 [1] which is chosen close to [2] and [3] to ensure comparability.

* The algorithm.py file contains the train() method, our own implementation of Algorithm 1. The controller architecture can be set to either ’GradientNN’ for a plain neural network approximation of the optimal control and ’GradientQRnet’ for the λ-QRnet controller. The training routine is based on Tensorflow 1.

* The problem.py script provides numerical implementation of the optimal control problem aimed at stabilizing Burgers’ equation laid out in the next subsection and is a streamlined version of the version provided in /Tenavi/QRnet.

* Following a class-based approach, the controller.py routine builds-up each, the LQR-, neural network-, and λ-QRnet controllers. While most parts have been readily available from /Tenavi/QRnet, the dynamic sampling heuristic has been implemented from scratch.

* generate.py contains the eponymous method implementing Algorithm 1, which solves the PMP-BVP for a given number of trajectories starting at specified initial conditions using controller warm-start and sequential extension of the time-horizon. For each trajectory it returns a data set of the form and size 150 − 200.

* The file simulate contains the routines integrate.py for solving the IVP and openloop.py for solving the OCP. The script closedloop.py simulates the controlled system and montecarlo.py is a wrapper function performing several closed-loop simulations for the optimality analysis.
