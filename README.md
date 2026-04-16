Stochastic Gradient Descent Project
Overview

This project implements Projected Stochastic Gradient Descent (SGD) for logistic regression and evaluates its performance under varying training set sizes and noise levels.

The goal is to study how SGD behaves in practice and compare empirical results with theoretical expectations from convex learning.

Problem Setup
Feature dimension: d = 4
Parameter dimension: 5 (includes bias)
Feature space: unit ball in R
4
Parameter space: unit ball in R
5
Loss Function

Logistic loss:

ℓ(w,(x,y))=log(1+exp(−y⟨w,
x
~
⟩))

where 
x
~
=(x,1)

Algorithm

We implement Projected Stochastic Gradient Descent:

Initialize w
1
	​

=0
For each iteration t:
Compute stochastic gradient using one example

Update:

w
t+1/2
	​

=w
t
	​

−η
t
	​

∇ℓ(w
t
	​

)

Project:

w
t+1
	​

=Π
C
	​

(w
t+1/2
	​

)
Step size:
η
t
	​

=
t
	​

1
	​


The final predictor is the average of iterates.

Data Generation

Each example is generated as:

y∈{−1,+1} uniformly
If y=−1:
u∼N(μ
0
	​

,σ
2
I), where μ
0
	​

=(−1/4,...,−1/4)
If y=+1:
u∼N(μ
1
	​

,σ
2
I), where μ
1
	​

=(1/4,...,1/4)

Then project:

x=Π
X
	​

(u)
Experiments

Parameters used:

Noise levels:
σ∈{0.2,0.4}
Training sizes:
n∈{50,100,500,1000}
Test set size:
N=400
Trials per setting:
30

Metrics evaluated:

Logistic loss
Classification error
Excess risk = mean − min
Results
Key Observations
Increasing n reduces excess risk and classification error
Higher noise (σ=0.4) leads to worse performance
Variance across trials decreases with larger datasets
Plots
Excess Risk vs Training Size

Classification Error vs Training Size

Project Structure
sgd-project/
│
├── sgd_project.py        # main implementation
├── results/              # plots
│   ├── excess_risk_plot.png
│   ├── classification_error_plot.png
├── README.md
How to Run

Install dependencies:

pip install numpy matplotlib

Run the project:

python3 sgd_project.py

Outputs:

printed results table
plots saved in results/
Authors
Shayan Manoharan
Nihal Patil
Notes

This project demonstrates how SGD:

scales to large datasets
converges with noisy gradients
is affected by data distribution noise