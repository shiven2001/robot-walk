import gymnasium as gym
from stable_baselines3 import A2C
import numpy as np

# Create the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Create the model
model = A2C("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10_000)

# Reset the environment
obs = env.reset()

# Run the trained model and display frames using OpenCV
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Capture the frame
    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Display the frame
    cv2.imshow("CartPole", frame)

    # Wait for a short period to allow display
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

    if done:
        obs = env.reset()

# Close the environment and OpenCV windows
env.close()
cv2.destroyAllWindows()