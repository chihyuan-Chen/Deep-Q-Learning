# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym

import os
import cv2
import datetime


def create_environment():
    seed = 42
    # Use the Baseline Atari environment because of Deepmind helper functions
    env = gym.make("ALE/Pong-v5", render_mode='human')
    
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 10)
    env.seed(seed)
    return env

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(80, 80, 10,))                            

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(filters = 10, kernel_size = 3, padding='same',activation="relu")(inputs)
    layer2 = layers.MaxPool2D(pool_size=(2, 2))(layer1)
    
    layer3 = layers.Conv2D(filters = 20, kernel_size = 3, padding='same',activation="relu")(layer2)
    layer4 = layers.Conv2D(filters = 20, kernel_size = 3, padding='same',activation="relu")(layer3)
    layer5 = layers.MaxPool2D(pool_size=(2, 2))(layer4)
    
    layer6 = layers.Conv2D(filters = 40, kernel_size = 3, padding='same', activation="relu")(layer5)
    layer7 = layers.Conv2D(filters = 40, kernel_size = 3, padding='same', activation='relu')(layer6)
    layer8 = layers.Conv2D(filters = 40, kernel_size = 3, padding='same', activation='relu')(layer7)
    layer9 = layers.MaxPool2D(pool_size=(2, 2))(layer8)
    
    layer10 = layers.Conv2D(filters = 80, kernel_size = 3, padding='same', activation="relu")(layer9)
    layer11 = layers.Conv2D(filters = 80, kernel_size = 3, padding='same', activation='relu')(layer10)
    layer12 = layers.Conv2D(filters = 80, kernel_size = 3, padding='same', activation='relu')(layer11)
    layer13 = layers.MaxPool2D(pool_size=(2, 2))(layer12)
    
    layer14 = layers.Conv2D(filters = 80, kernel_size = 3, padding='same', activation="relu")(layer13)
    layer15 = layers.Conv2D(filters = 80, kernel_size = 3, padding='same', activation='relu')(layer14)
    layer16 = layers.Conv2D(filters = 80, kernel_size = 3, padding='same', activation='relu')(layer15)
    layer17 = layers.MaxPool2D(pool_size=(2, 2))(layer16)
    
    layer18 = layers.Flatten()(layer17)
    layer19 = layers.Dense(160, activation="relu")(layer18)
    layer20 = layers.Dense(160, activation="relu")(layer19)
    action = layers.Dense(len(num_actions), activation="softmax")(layer20)

    return keras.Model(inputs=inputs, outputs=action)

# Experience replay buffers
class ExperienceReplay():
    def __init__(self, buffer_size):
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.buffer_size = buffer_size

    # Add a transition to the memory 
    def store_transition(self, state, state_next, action, reward, done):
        # Save actions and states in replay buffer
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)
    
    def store_reward(self, episode_reward):
        self.episode_reward_history.append(episode_reward)
        if len(self.episode_reward_history) > 100:
            del self.episode_reward_history[:1]
        return np.mean(self.episode_reward_history)
    
    # Limit the state and reward history
    def limit(self):
        del self.rewards_history[:1]
        del self.state_history[:1]
        del self.state_next_history[:1]
        del self.action_history[:1]
        del self.done_history[:1]

    # Sampling that the memory has been stored.
    def sample(self, batch_size):
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.done_history)), size=batch_size)

        # Using list comprehension to sample from replay buffer
        state_sample = np.array([self.state_history[i] for i in indices])
        # Preprocess state next sample
        state_next_sample = np.array([preprocess(self.state_next_history[i]) for i in indices])
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])
        return (state_sample, state_next_sample, rewards_sample, action_sample, done_sample)     
    
class Agent():
    def __init__(self, max_memory_length):
        self.gamma = 0.95
        
        # Size of batch taken from replay buffer
        self.batch_size = 128
        
        # Train the model after 10 actions
        self.update_after_actions = 10
        
        # How often to update the target network
        self.update_target_network = 10000
        
        # Experience replay buffers
        self.memory = ExperienceReplay(max_memory_length)
        
        # The first model makes the predictions for Q-values which are used to
        # make a actio
        self.model = create_q_model()
        
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        self.model_target = create_q_model()
        
        # In the Deepmind paper they use RMSProp however then Adam optimizer
        # improves training time
        self.optimizer = keras.optimizers.Adam(learning_rate=0.0002, clipnorm=1.0)         
        
        # Using huber loss for stability
        self.loss_function = keras.losses.Huber()
        
    def step(self, state, state_next, action, reward, done, episode_reward, frame_count, num_actions, timestep):
        # Save actions and states in replay buffer
        self.memory.store_transition(state, state_next, action, reward, done)
        
        # Update every fourth frame and once batch size is over 32
        if frame_count % self.update_after_actions == 0 and len(self.memory.done_history) > self.batch_size:
            self.learn(episode_reward, frame_count, num_actions, timestep)
        
    def learn(self, episode_reward, frame_count, num_actions, timestep):
        # Sample random minibatch of transitions from Experience Replay
        state_sample, state_next_sample, rewards_sample, action_sample, done_sample = self.memory.sample(self.batch_size)
        
        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = self.model_target.predict(state_next_sample)
        
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
        
        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, len(num_actions))
        
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            #q_values = model(state_sample, training=True)
            q_values = self.model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Show the training trend
        TensorBoard(episode_reward, loss, timestep)
        
    def choose_action(self, state, frame_count, epsilon_random_frames, epsilon, num_actions):
        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
            
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
        return action
        
    # Loading best pretraining weights
    def load(self, path, best_episode, best_reward):
        if os.path.exists(path) == True:
            self.model, self.model_target = load_model(self.model, self.model_target, best_episode, best_reward)
            
    def save(self, episode_count, running_reward):
        save_model(self.model, self.model_target, episode_count, running_reward)
    
    # Update the target network
    def update_weights(self, frame_count, running_reward, episode_count):
        if frame_count % self.update_target_network == 0:
            # update the the target network with new weights
            self.model_target.set_weights(self.model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))
          
    # Limit the state and reward history
    def limit(self):
        if len(self.memory.rewards_history) > self.memory.buffer_size:
            self.memory.limit()
            
    def running_reward(self, episode_reward):
        return self.memory.store_reward(episode_reward)  
        
# Preprocess frames
def preprocess(state):
    # Remove backgound colors
    state[(state==87)] = 1
    
    # Crop frames (change size to 160*160)
    state = state[34:194]
    
    # Down sampling (change size to 80*80)
    state = state[::2,::2]
    return state

# Save best training weights
def save_model(model, model_target, episode_count, running_reward):
    # Get model weights
    model_target.set_weights(model.get_weights())
    
    # Save to the specified path 
    model_target.save("./weights/DQN_"+str(episode_count)+"_reward_"+str(running_reward)+".h5")

# Load model
def load_model(model, model_target, best_episode, best_reward):
    model = keras.models.load_model(path)
    model_target = keras.models.load_model(path)
    model.summary()
    return model, model_target

# TensorBoard
def TensorBoard(episode_reward, loss, timestep):
    # Define metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_reward = tf.keras.metrics.Mean('train_reward', dtype=tf.float32)
    
    train_loss(loss)
    print("Reward:", episode_reward)
    loss = loss.numpy()
    print("Loss:", loss)
    train_reward(episode_reward) 
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    with train_summary_writer.as_default():
        tf.summary.scalar('Loss', loss, step=timestep)
        tf.summary.scalar('Reward', episode_reward, step=timestep)
    

if __name__ == "__main__":
    #TensorBoard initial
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    
    # Configuration paramaters for the whole setup
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.12  # Minimum epsilon greedy parameter         
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    # Rate at which to reduce chance of random action being taken
    epsilon_interval = (epsilon_max - epsilon_min)
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    epsilon_greedy_frames = 1000000.0
    max_steps_per_episode = 10000
    
    running_reward = 0
    episode_count = 0
    frame_count = 0
    
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 1000000
    
    # Three actions
    num_actions = [0, 2, 3]
    action_list = {'0':'NOOP', '2':'RIGHT', '3':'LEFT'}
    
    # Record best episode and rewards
    best_reward = -20
    best_episode = 0

    # Use flag to check the loading state
    check = 0
    
    env = create_environment()
    agent = Agent(max_memory_length)
    
    while True:  # Run until solved
        state = np.array(env.reset())
        state = state.transpose(1,2,0)
        episode_reward = 0
        
        # Loading best pretraining weights
        path = "./weights/DQN_"+str(best_episode)+"_reward_"+str(best_reward)+".h5"
        if check == 1:
            agent.load(path, best_episode, best_reward)
            check = 0
                
        for timestep in range(1, max_steps_per_episode):
            # Preprocess frames
            state = preprocess(state)
            
            cv2.imshow('show0',state.transpose(2,0,1)[0])
            cv2.imshow('show1',state.transpose(2,0,1)[1])
            cv2.imshow('show2',state.transpose(2,0,1)[2])
            cv2.imshow('show3',state.transpose(2,0,1)[3])
            cv2.imshow('show4',state.transpose(2,0,1)[4])
            cv2.imshow('show5',state.transpose(2,0,1)[5])
            cv2.imshow('show6',state.transpose(2,0,1)[6])
            cv2.imshow('show7',state.transpose(2,0,1)[7])
            cv2.imshow('show8',state.transpose(2,0,1)[8])
            cv2.imshow('show9',state.transpose(2,0,1)[9])
            
            # of the agent in a pop up window.
            frame_count += 1
             
            # Depending on probability of epsilon, either select a random action
            action = agent.choose_action(state, frame_count, epsilon_random_frames, epsilon, num_actions)
            
            # Show action
            print("Action:", action_list[str(action)])
                
            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
    
            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            
            state_next = np.array(state_next)
            state_next = state_next.transpose(1,2,0)
            
            agent.step(state, state_next, action, reward, done, episode_reward, frame_count, num_actions, timestep)
    
            episode_reward += reward
            
            state = state_next
            
            # update the the target network with new weights
            agent.update_weights(frame_count, running_reward, episode_count)
            
            # Limit the state and reward history
            agent.limit()
    
            if done:
                break
            
        # Update running reward to check condition for solving  
        running_reward = agent.running_reward(episode_reward)
        
        episode_count += 1
        
        if running_reward > 10:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break
        
        
        # Update best training results
        if running_reward > best_reward:
            check = 1
            best_episode = episode_count
            best_reward = running_reward
            agent.save(episode_count, running_reward)
        