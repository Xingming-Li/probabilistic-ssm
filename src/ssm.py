import numpy as np

class LinearSSM:
    """
    Two-state Linear-Gaussian SSM:
        x_{t+1} = A * x_t + B * u_t + w_t, w_t ~ N(0, Q) 
        y_t     = C * x_t + D * u_t + v_t, v_t ~ N(0, R)
    x_t: [VO₂max VO₂]
    u_t: activity one-hot vector
    y_t: heart rate
    """
    def __init__(self):
        # A (state transition): how states affect next states & how state variables connect
        # Assume exponential decay: x_{t+1} depends on alpha * x_t, where alpha = e ^ (-dt / tau) and dt = 1
        tau = 200  # Physiological recovery time constant is ~200 minutes
        alpha = np.exp(-1 / tau)
        self.A = np.array([[1, 0], [0, alpha]])
        # B: how inputs affect states
        self.B = np.array([[0,  0,  0.001], [-0.001,  0.05, 1.2]])
        # C: how states affect outputs
        self.C = np.array([[-0.25, 1.2]])          
        # D: how inputs affect outputs 
        self.D = np.array([[70, 90, 125]]) 
        
        # Standard deviations for noise
        self.std_Q1 = 0.0001
        self.std_Q2 = 0.5
        self.std_R = 2.5
    
    # Noise helper
    def noise(self, data_length):    
        self.w1 = np.random.normal(0, self.std_Q1, data_length)  # Small VO₂max noise
        self.w2 = np.random.normal(0, self.std_Q2, data_length)  # Larger VO₂ noise
        self.v = np.random.normal(0, self.std_R, data_length)    # Measurement noise
        
    # Generate both hidden states and outputs
    def generate(self, u, x_init, data_length):
        self.noise(data_length)
        x = np.zeros((data_length, 2))
        y = np.zeros(data_length)        
        x_current = x_init
        
        for t in range(data_length):            
            w_t = np.array([self.w1[t], self.w2[t]])
            # State equation            
            x_next = self.A @ x_current + self.B @ u[t] + w_t
            x[t] = x_next
            # Observation equation
            y[t] = self.C @ x_next + self.D @ u[t] + self.v[t]     
            x_current = x_next
        return x, y
