import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = "../data/hr_data.npz"
data = np.load(data_path)  

# Set hyperparameters
n_week = 40
prop_factor = 15.3
age = 38
hr_max = 220 - age

# Prepare a VO₂max (zeros) array, a weekly state array and a weekly HR array
vo2_max = np.zeros(n_week)
state_chunks = np.array_split(data["state"], n_week)
hr_chunks = np.array_split(data["hr"], n_week)

# Compute weekly VO₂max: VO₂max = prop_factor * HR_max / HR_rest
for i, (state_array, hr_array) in enumerate(zip(state_chunks, hr_chunks)):
    hr_sleep = hr_array[state_array == 0]  
    hr_rest = hr_sleep.mean()    
    vo2_max[i] = prop_factor * hr_max / hr_rest   
    
# Save weekly VO₂max estimation
np.save("../results/est_week_formula.npy", vo2_max)

# Plot
plt.figure(figsize=(6 * 1.62, 6))
sns.set_theme(style="whitegrid")
sns.lineplot(x=[i+1 for i in range(n_week)], y=vo2_max, marker="o")

plt.xlim(0, n_week+1)
plt.ylim(45, 50)

label_weeks = [1, 10, 20, 30, 40]
for w in label_weeks:
    plt.text(w, vo2_max[w-1]+0.1, f"{vo2_max[w-1]:.2f}", ha="center", va="bottom", fontsize=9)

plt.xlabel("Week")
plt.ylabel("Estimated $VO_{2}max$ (ml/kg/min)")
plt.title("Weekly $VO_{2}max$ Estimation - Uth Formula")

plt.tight_layout()
plt.show()
