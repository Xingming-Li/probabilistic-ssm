# VO₂max Estimation with a Linear State-Space Model

## Overview
This repo implements a two-state linear Gaussian state-space model (SSM) to estimate latent VO₂max from minute-level heart rate data collected over 40 weeks.

The model captures:
- Slowly varying **VO₂max**
- Activity-dependent **VO₂** consumption
- Noisy heart-rate observations
- Sequential inference via the **Kalman filter**

---

## Model

State transition:
$x_{t+1} = A x_t + B u_t + w_t$

Observation:
$y_t = C x_t + D u_t + v_t$

- **State:** $x_t = [VO₂max, VO₂]$
- **Input:** one-hot activity (sleep / awake / exercise)
- **Observation:** heart rate (bpm)

---

## Task 1 – Heart Rate Simulation

- Generates synthetic heart rate from the SSM
- Compares generated vs measured data (RMSE, per-state statistics)

Run:
```
cd scr/
python generate_hr.py
```
---

## Task 2 – VO₂max Inference with Kalman Filtering

- Estimates latent states from heart rate
- Computes weekly VO₂max trajectory
- Compares against a baseline from [Uth's formula](https://pubmed.ncbi.nlm.nih.gov/14624296/).

Run:
```
cd scr/
python uth_formula.py
python kalman_filter.py
```
Outputs (in `results/` folder):
- est_week_formula.npy
- est_week_ssm.npy
- est_minute_ssm.npy

---

## Dependencies

`pip install numpy matplotlib seaborn`

---

## Summary

Demonstrates probabilistic modeling of physiological time-series using:
- Linear dynamical systems
- Activity-conditioned transitions
- Kalman filtering for latent state inference