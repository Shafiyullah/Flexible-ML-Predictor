import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class DataFrameEnv(gym.Env):
    """
    A custom Gymnasium Environment that turns a DataFrame into a 
    sequential decision-making task.
    
    Task: Look at the current row's features and predict if the 
    Target Value in the NEXT row will increase or decrease/stay same.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, target_col):
        super(DataFrameEnv, self).__init__()

        self.df = df
        self.target_col = target_col
        
        # Separate features and target
        self.features = df.drop(columns=[target_col]).values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        
        self.n_steps = len(self.df)
        self.current_step = 0

        # Action Space: 0 (Predict Decrease/Same), 1 (Predict Increase)
        self.action_space = spaces.Discrete(2)

        # Observation Space: The feature values of the current row
        # We assume standardized data (roughly -5 to 5)
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning of the dataset."""
        super().reset(seed=seed)
        self.current_step = 0
        
        observation = self.features[self.current_step]
        return observation, {}

    def step(self, action):
        """
        The agent takes an action (prediction).
        We check if it was correct based on the *next* row's target.
        """
        truncated = False
        terminated = False
        
        # If we are at the last row, we can't predict the next one.
        if self.current_step >= self.n_steps - 1:
            terminated = True
            return self.features[self.current_step], 0, terminated, truncated, {}

        # Get current and next target values
        current_val = self.targets[self.current_step]
        next_val = self.targets[self.current_step + 1]
        
        # Determine the "Truth"
        # 1 if increased, 0 if decreased/same
        actual_trend = 1 if next_val > current_val else 0
        
        # Calculate Reward
        if action == actual_trend:
            reward = 1.0  # Correct prediction
        else:
            reward = -0.1 # Wrong prediction (small penalty)

        # Move to next step
        self.current_step += 1
        
        # Get new observation (state)
        observation = self.features[self.current_step]
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        pass # Visualization can be added later