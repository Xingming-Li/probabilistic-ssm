import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ssm import LinearSSM

# Load and preprocess data
data_path = "../data/hr_data.npz"
data = np.load(data_path)
state_array = data["state"]
hr_array = data["hr"]
array_length = len(state_array)

# Set hyperparameters
n_week = 40
prop_factor = 15.3
age = 38
hr_max = 220 - age
day1mins = 24 * 60
np.random.seed(42)

# Set inputs u. u_t shape is (3,) (one-hot vector)
u = np.zeros((array_length, 3))
u[state_array == 0, 0] = 1   # sleep
u[state_array == 1, 1] = 1   # awake
u[state_array == 2, 2] = 1   # exercise

# Set initial state x_init. x_init shape is (2,)
day1hr_sleep = hr_array[:day1mins][state_array[:day1mins] == 0]
day1hr_rest = day1hr_sleep.mean()
vo2_max_init = prop_factor * hr_max / day1hr_rest  # x_init_1: estimated VO₂max in day 1
# x_init_2: sleeping, office work or running V0₂ from assignment table 2
vo2_init = 3 if state_array[0] == 0 else 6 if state_array[0] == 1 else 31
x_init = np.array([vo2_max_init, vo2_init])

# Load model and parameter matrices
model = LinearSSM()   
A = model.A
B = model.B
C = model.C
D = model.D
std_Q1 = model.std_Q1
std_Q2 = model.std_Q2
std_R = model.std_R

# Prepare for Kalman filter
Q = np.diag([std_Q1 ** 2, std_Q2 ** 2])  # Process covariance
R = std_R ** 2  # Measurement covariance

x = np.zeros((array_length, 2))  # Posteriori estimate
p = np.zeros((array_length, 2, 2))  # Error covariance
x_current = x_init  # Initial priori estimate
p_current = np.zeros((2, 2))  # Initial error covariance

# Kalman filter loop
for t in range(array_length):
    if t % 50000 == 0:
        print(f"Updating {t}th step ...")
    # Predict
    x_pred = A @ x_current + B @ u[t]
    p_pred = A @ p_current @ A.T + Q
    
    # Update
    k_gain = (p_pred @ C.T) / (C @ p_pred @ C.T + R)  # Shape: (2, 1)
    y_pred = C @ x_pred + D @ u[t]
    x_post = x_pred + k_gain.flatten() * (hr_array[t] - y_pred)
    p_post = (np.eye(2) - k_gain @ C) @ p_pred
    
    # Store and reassign
    x[t] = x_post
    p[t] = p_post
    x_current = x_post
    p_current = p_post

    if t == array_length - 1:
        print("Kalman filter done!")

# Compute weekly averages
vo2_max_ssm = x[:, 0]
vo2_max = np.zeros(n_week)
vo2_max_chunks = np.array_split(vo2_max_ssm, n_week)
for i, w in enumerate(vo2_max_chunks):
    vo2_max[i] = w.mean()
    
# Save arrays
np.save("../results/est_week_ssm.npy", vo2_max)
np.save("../results/est_minute_ssm.npy", x)

# Plot
vo2_max_uth = np.load("../results/est_week_formula.npy")

plt.figure(figsize=(4 * 1.62, 4))
sns.set_theme(style="whitegrid")
sns.lineplot(x=[i+1 for i in range(n_week)], y=vo2_max_uth, marker="o", label="Formula Estimation", alpha=0.8)
sns.lineplot(x=[i+1 for i in range(n_week)], y=vo2_max, marker="o", label="SSM Estimation", alpha=0.8)
plt.legend()

plt.xlim(0, n_week+1)
plt.ylim(45, 52)

label_weeks = [1, 10, 20, 30, 40]
for w in label_weeks:
    plt.text(w, vo2_max[w-1]+0.1, f"{vo2_max[w-1]:.2f}", ha="center", va="bottom", fontsize=9)
    plt.text(w, vo2_max_uth[w-1]+0.1, f"{vo2_max_uth[w-1]:.2f}", ha="center", va="bottom", fontsize=9)

plt.xlabel("Week")
plt.ylabel("Estimated $VO_{2}max$ (ml/kg/min)")
plt.title("Weekly $VO_{2}max$ Estimation - SSM vs. Formula")

plt.tight_layout()
plt.show()
