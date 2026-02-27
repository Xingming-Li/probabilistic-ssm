import numpy as np
import matplotlib.pyplot as plt
from ssm import LinearSSM

# Load and preprocess data
data_path = "../data/hr_data.npz"
data = np.load(data_path)  
hr_array = data["hr"]
state_array = data["state"]
array_length = len(state_array)

# Set hyperparameters
states = {"sleep": 0, "awake": 1, "exercise": 2}
prop_factor = 15.3
age = 38
hr_max = 220 - age
day1mins = 24 * 60
np.random.seed(42)

# Set y
y = hr_array

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

# Load model and run generation
model = LinearSSM()
x_pred, y_pred = model.generate(u, x_init, data_length=array_length)

# RMSE, mean and std for measured and generated HR
rmse = np.sqrt(np.mean((y_pred - hr_array)**2))
print(f"Task 2 RMSE = {rmse:.3f}")

print("\nHeart rate statistics per state:")
for name, s in states.items():
    idx = (state_array == s)
    hr_mean = hr_array[idx].mean()
    hr_std = hr_array[idx].std()

    y_pred_mean = y_pred[idx].mean()
    y_pred_std = y_pred[idx].std()

    print(f"\nState {name}:")
    print(f"Measured HR : mean = {hr_mean:.2f}, std = {hr_std:.2f}")
    print(f"Generated HR: mean = {y_pred_mean:.2f}, std = {y_pred_std:.2f}")

# Plot HR per minute
plt.figure(figsize=(12,4))
plt.plot(y, label="Measured HR", alpha=0.8)
plt.plot(y_pred, label="Generated HR", alpha=0.8)
plt.legend()
plt.title("Measured vs. Generated Heart Rate")
plt.xlabel("Minute")
plt.ylabel("Heart rate (bpm)")
plt.tight_layout()
plt.show()
