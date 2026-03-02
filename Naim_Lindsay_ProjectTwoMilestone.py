#!/usr/bin/env python
# coding: utf-8

# # Treasure Hunt Game Notebook
# 
# ## Read and Review Your Starter Code
# The theme of this project is a popular treasure hunt game in which the player needs to find the treasure before the pirate does. While you will not be developing the entire game, you will write the part of the game that represents the intelligent agent, which is a pirate in this case. The pirate will try to find the optimal path to the treasure using deep Q-learning. 

# <div class="alert alert-block alert-success" style="color:black;">
# <b>To Begin:</b> Use this <b>TreasureHuntGame_starterCode.ipynb</b> file to complete your assignment. 
# <br><br>
# You have been provided with two Python classes and this notebook to help you with this assignment. The first class, <b>TreasureMaze.py</b>, represents the environment, which includes a maze object defined as a matrix. The second class, <b>GameExperience.py</b>, stores the episodes – that is, all the states that come in between the initial state and the terminal state. This is later used by the agent for learning by experience, called "exploration". This notebook shows how to play a game. Your task is to complete the deep Q-learning implementation in the qtrain() function for which a skeleton implementation has been provided. 
# </div>
# <br>
# <div class="alert alert-block alert-info" style="color:black;">
# <b>NOTE: </b>The code block you will need to complete will have <b>#TODO</b> as a header.
# <br> First, read and review the next few code and instruction blocks to understand the code that you have been given.</div>

# <div class="alert alert-block alert-warning" style="color: #333333;">
# <b>Installations</b> The following command will install the necessary Python libraries to necessary to run this application. If you see a "[notice] A new release of pip is available: 23.1.2 -> 25.2" at the end of the installation, you may disregard that statement. 
# </div>

# In[1]:


get_ipython().system('pip install -r requirements.txt')


# <h2>Tensorflow CPU Acceleration Warning</h2>
# <div class="alert alert-block alert-danger" style="color: #333333;">
# <b>GPU/CUDA/Memory Warnings/Errors:</b> You may receive some errors referencing that GPUs will not be used, CUDA could not be found, or free system memory allocation errors. These and a few others, are standard errors that can be ignored here as they are environment based.<br><br>
#     <b>Example messages:</b>
#     <ul>
#         <li>oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders</li>
#         <li>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</li>
# </div>

# In[2]:


from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model
from keras.models import Sequential
from keras.layers import Dense, Activation, PReLU
from keras.optimizers import SGD , Adam, RMSprop
import matplotlib.pyplot as plt
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2> Maze Object Generation</h2>
# 
# <div class="alert alert-block alert-info" style="color:black;">
#     <b>NOTE:</b>  The following code block contains an 8x8 matrix that will be used as a maze object:
# </div>

# In[3]:


maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
])


# <h2>Helper Functions and Global Variables</h2>
# 
# <div class="alert alert-block alert-info" style="color:black;">
# This <b>show()</b> helper function allows a visual representation of the maze object:
# </div>

# In[4]:


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    pirate_row, pirate_col, _ = qmaze.state
    canvas[pirate_row, pirate_col] = 0.3   # pirate cell
    canvas[nrows-1, ncols-1] = 0.9 # treasure cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


# The <b>pirate agent</b> can move in four directions: left, right, up, and down. 
# 
# <div class="alert alert-block alert-warning" style="color:black;">
# <b>Note:</b> While the agent primarily learns by experience through exploitation, often, the agent can choose to explore the environment to find previously undiscovered paths. This is called "exploration" and is defined by epsilon. This value is the <b>EXPLORATION</b> values from the Cartpole assignment. The hyperparameters are provided here and used in the <b>qtrain()</b> method. 
# You are encouraged to try various values for the exploration factor and see how the algorithm performs.
# </div>

# In[5]:


LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


# Exploration factor
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
patience = 10

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)


# The sample code block and output below show creating a maze object and performing one action (DOWN), which returns the reward. The resulting updated environment is visualized.

# In[6]:


qmaze = TreasureMaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
show(qmaze)


# <div class="alert alert-block alert-warning" style="color:black;">
#     <b>NOTE:</b> This <b>play_game()</b> function simulates a full game based on the provided trained model. The other parameters include the TreasureMaze object, the starting position of the pirate and max amount of steps to make sure the code does not get stuck in a loop.
# </div>

# In[7]:


def play_game(model, qmaze, pirate_cell, max_steps=None):
    qmaze.reset(pirate_cell)
    envstate = qmaze.observe()
    steps = 0
    if max_steps is None:
        max_steps = qmaze.maze.size * 2  # safety cutoff

    while steps < max_steps:
        state = np.asarray(envstate, dtype=np.float32)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        q_values = model(state, training=False).numpy()
        action = np.argmax(q_values[0])

        envstate, reward, game_status = qmaze.act(action)
        steps += 1

        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False

    return False  # timed out with no result


# <div class="alert alert-block alert-warning" style="color:black;">
# <b>Note: </b>
#     This <b>completion_check()</b> function helps you to determine whether the pirate can win any game at all. If your maze is not well designed, the pirate may not win any game at all. In this case, your training would not yield any result. The provided maze in this notebook ensures that there is a path to win and you can run this method to check.
# </div>

# In[8]:


def completion_check(model, maze_or_qmaze, max_steps=None):
    # Accept either raw numpy maze or TreasureMaze instance
    if isinstance(maze_or_qmaze, TreasureMaze):
        qmaze = maze_or_qmaze
    else:
        qmaze = TreasureMaze(maze_or_qmaze)

    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            continue
        if not play_game(model, qmaze, cell, max_steps=max_steps):
            return False
    return True


# <div class="alert alert-block alert-warning" style="color:black;">
# <b>Note: </b>
# </b>The <b>build_model()</b> function in the block below will build the neural network model. Review the code and note the number of layers, as well as the activation, optimizer, and loss functions that are used to train the model.
# </div>

# In[9]:


def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


# <div class="alert alert-block alert-warning" style="color:black;">
#     <b>Note:</b>
#     This <b>train_step()</b> helper function in the block below is used to help predict Q-values (quality values) in the current modelto see how good each action is in a given state and improve the Q-network by reducing the gap between what is predicted and what should have been predicted. 
# </div>
# <br>
# <div class="alert alert-block alert-info" style="color:black;">
# If you're interested in reading up on the <i>@tf.function</i>, which is a decorator for Tensorflow to run this code into a TensorFlow computation graph, please refer to this link: <a href="https://www.tensorflow.org/guide/intro_to_graphs">https://www.tensorflow.org/guide/intro_to_graphs</a>
# </div>
# 

# <h2>Tensorflow GPU Warning</h2>
# <div class="alert alert-block alert-danger" style="color: #333333;">
#     You will see a <b>warning in red</b> "INTERNAL: CUDA Runtime error: Failed call to cudaGetRuntimeVersion: Error loading CUDA libraries. GPU will not be used.". This is simply coming from <b>Tensorflow skipping using GPU for this assignment.</b>  
# </div>

# In[10]:


loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        q_values = model(x, training=True)
        loss = loss_fn(y, q_values)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
# print("GPU disabled. Using CPU.")
#  
# #TODO: Complete the Q-Training Algorithm Code Block
# 
# <div class="alert alert-block alert-info" style="color:black;">
#     This is your deep Q-learning implementation. The goal of your deep Q-learning implementation is to find the best possible navigation sequence that results in reaching the treasure cell while maximizing the reward. In your implementation, you need to determine the optimal number of epochs to achieve a 100% win rate.
# </div>
#     <b>Pseudocode:</b>
#     <br>
#     For each epoch:
#         Reset the environment at a random starting cell
#         agent_cell = randomly select a free cell
#         <br>
#         <b>Hint:</b> Review the reset method in the TreasureMaze.py class.
#     
#         Set the initial environment state
#         env_state should reference the environment's current state
#         Hint: Review the observe method in the TreasureMaze.py class.
# 
#         While game status is not game over:
#            previous_envstate = env_state
#             Decide on an action:
#                 - If possible, take a random valid exploration action and 
#                   randomly choose action (left, right, up, down)
#                   and assign it to an action variable
#                 - Else, pick the best exploitation action from the model and assign it to an action variable
#                   Hint: Review the predict method in the GameExperience.py class.
#     
#            Retrieve the values below from the act() method.
#            env_state, reward, game_status = qmaze.act(action)
#            Hint: Review the act method in the TreasureMaze.py class.
#     
#             Track the wins and losses from the game_status using win_history 
#          
#            Store the episode below in the Experience replay object
#            episode = [previous_envstate, action, reward, envstate, game_status]
#            Hint: Review the remember method in the GameExperience.py class.
#         
#            Train neural network model and evaluate loss
#            Hint: Call GameExperience.get_data to retrieve training data (input and target) 
#            and pass to the train_step method and assign it to batch_loss and append to the loss variable
#         
#       If the win rate is above the threshold and your model passes the completion check, that would be your epoch.
# 
# Note: A 100% win rate <b>DOES NOT EXPLICITLY MEAN</b> that you have solved the maze. It simply indicates that during the last evaluation, the pirate <i>happened</i> to get to the treasure. Be sure to utilise the <b>completion_check()</b> function to validate your pirate found the treasure at every starting point and consistently! 
# 
# <b> You will need to complete the section starting with #START_HERE. Please use the pseudocode above as guidance. </b>
# 

# In[11]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU


# In[12]:


import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))


# In[13]:


import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception as e:
    print("Could not hide GPU:", e)

print("GPUs:", tf.config.list_physical_devices("GPU"))


# In[14]:


import datetime
import random
import numpy as np

from tensorflow.keras.models import clone_model

def qtrain(model, maze, **opt):
    global epsilon

    # --- Options (safe defaults for Codio) ---
    n_epoch = opt.get("n_epoch", 200)
    max_memory = opt.get("max_memory", 8 * maze.size)
    data_size = opt.get("data_size", 32)
    target_update_freq = opt.get("target_update_freq", 25)

    # NEW: hard stop per episode to prevent infinite loops
    max_steps = opt.get("max_steps", maze.size * 4)

    start_time = datetime.datetime.now()

    # Build game environment
    qmaze = TreasureMaze(maze)

    # Target network (DQN stability)
    target_model = clone_model(model)
    target_model.set_weights(model.get_weights())

    # Experience replay
    experience = GameExperience(model, target_model, max_memory=max_memory)

    win_history = []
    hsize = max(10, qmaze.maze.size // 2)  # NEW: avoid tiny window
    win_rate = 0.0

    loss = 0.0
    n_episodes = 0

    for epoch in range(n_epoch):
        # --- Reset episode ---
        if hasattr(qmaze, "free_cells") and len(qmaze.free_cells) > 0:
            pirate_start = random.choice(qmaze.free_cells)
        else:
            free_cells = list(zip(*np.where(np.array(maze) != 0)))
            pirate_start = random.choice(free_cells)

        qmaze.reset(pirate_start)
        state = qmaze.observe()

        done = False
        game_status = "not_over"
        episode_loss = 0.0

        steps = 0
        while not done and steps < max_steps:
            steps += 1
            valid_actions = qmaze.valid_actions()

            # epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                q_values = model.predict(state, verbose=0)[0]
                action = max(valid_actions, key=lambda a: q_values[a])

            # Apply action
            result = qmaze.act(action)

            # Handle both possible return styles:
            # (next_state, reward, status) OR (reward, next_state, done_bool)
            if isinstance(result, tuple) and len(result) == 3:
                a, b, c = result
                if isinstance(c, (bool, np.bool_)):
                    reward, next_state, done = a, b, bool(c)
                    game_status = "win" if done and reward > 0 else ("lose" if done else "not_over")
                else:
                    next_state, reward, game_status = a, b, c
                    done = (game_status != "not_over")
            else:
                # unexpected format, treat as failure
                game_status = "lose"
                break

            # Store experience (support both signatures)
            try:
                experience.remember([state, action, reward, next_state, done])
            except TypeError:
                experience.remember(state, action, reward, next_state, done)

            # --- Train from replay (mini-batch) ---
            inputs, targets = None, None
            try:
                inputs, targets = experience.get_data(data_size)
            except TypeError:
                try:
                    inputs, targets = experience.get_data()
                except Exception:
                    inputs, targets = None, None

            if inputs is not None and targets is not None:
                history = model.fit(inputs, targets, epochs=1, verbose=0)
                episode_loss = float(history.history["loss"][0])

            state = next_state

        # If we hit max_steps, force a loss so training doesn't pretend it's fine
        if not done and steps >= max_steps:
            game_status = "lose"

        # Episode finished
        n_episodes += 1
        loss = episode_loss

        win_history.append(1 if game_status == "win" else 0)

        # Target update
        if epoch % target_update_freq == 0:
            target_model.set_weights(model.get_weights())

        window = win_history[-hsize:]
        win_rate = sum(window) / len(window)

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        print(
            "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}".format(
                epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t
            )
        )

        # epsilon decay
        if win_rate > 0.9:
            epsilon = 0.05
        else:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # stop if solved (win_rate + completion_check)
        try:
            if win_rate >= 0.95 and completion_check(model, qmaze):
                print(f"Reached completion at epoch {epoch}")
                break
        except Exception:
            pass

    total_time = format_time((datetime.datetime.now() - start_time).total_seconds())
    print("Training complete in:", total_time)


def format_time(seconds):
    if seconds < 400:
        return f"{seconds:.1f} seconds"
    elif seconds < 4000:
        return f"{seconds/60.0:.2f} minutes"
    else:
        return f"{seconds/3600.0:.2f} hours"


# ## Test Your Model
# 
# Now we will start testing the deep Q-learning implementation. To begin, select **Cell**, then **Run All** from the menu bar. This will run your notebook. As it runs, you should see output begin to appear beneath the next few cells. The code below creates an <b>instance</b> of TreasureMaze. This does not show your actual training done.

# In[15]:


model = build_model(maze)


# In[16]:


epsilon = 1.0
epsilon_decay = 0.97
epsilon_min = 0.05


# In[17]:


epsilon = 1.0

qtrain(
    model,
    maze,
    n_epoch=30,                 # smaller chunk (faster feedback)
    max_memory=12 * maze.size,  # more diverse replay
    data_size=64,               # stronger gradient signal
    target_update_freq=5,       # more stable targets sooner
    max_steps=maze.size * 4     # BIG speedup (prevents wandering forever)
)


# In[18]:


epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

qtrain(
    model,
    maze,
    n_epoch=50,                 # try 50 first
    max_memory=12 * maze.size,
    data_size=32,               # keep 32 to reduce compute + avoid big matmul
    target_update_freq=5,
    max_steps=maze.size * 4
)


# In[19]:


# Stabilize / finish run (fast + less wandering)
epsilon = 0.6
epsilon_decay = 0.99
epsilon_min = 0.05

qtrain(
    model,
    maze,
    n_epoch=60,
    max_memory=12 * maze.size,
    data_size=32,
    target_update_freq=5,
    max_steps=maze.size * 2   # KEY: shorter episodes = less wandering
)


# In[20]:


epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

qtrain(
    model,
    maze,
    n_epoch=50,                 # try 50 first
    max_memory=12 * maze.size,
    data_size=32,               # keep 32 to reduce compute + avoid big matmul
    target_update_freq=5,
    max_steps=maze.size * 4
)


# In[21]:


epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

qtrain(
    model,
    maze,
    n_epoch=50,                 # try 50 first
    max_memory=12 * maze.size,
    data_size=32,               # keep 32 to reduce compute + avoid big matmul
    target_update_freq=5,
    max_steps=maze.size * 4
)


# In[22]:


epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

qtrain(
    model,
    maze,
    n_epoch=50,                 # try 50 first
    max_memory=12 * maze.size,
    data_size=32,               # keep 32 to reduce compute + avoid big matmul
    target_update_freq=5,
    max_steps=maze.size * 4
)


# In[25]:


epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

qtrain(
    model,
    maze,
    n_epoch=50,                 # try 50 first
    max_memory=12 * maze.size,
    data_size=32,               # keep 32 to reduce compute + avoid big matmul
    target_update_freq=5,
    max_steps=maze.size * 4
)


# In the next code block, you will build your model using the <b>build_model</b> function and train it using deep Q-learning. Note: This step takes several minutes to fully run.
# 
# 

# <div class="alert alert-block alert-danger" style="color: #333333;">
#   <b>WARNING</b>  If you did not attempt the assignment, the code <b>will</b> error out at this section.
#  </div>

# <div class="alert alert-block alert-warning" style="color:black;">
# <b>Note: </b> This cell will check to see if the model passes the completion check. Note: This could take several minutes.
# </div>

# This cell will test your model for one game. It will start the pirate at the top-left corner and run <b>play_game()</b>. The agent should find a path from the starting position to the target (treasure). The treasure is located in the bottom-right corner.

# In[26]:


import random  # make sure this exists somewhere above

epsilon = 0.0                 # IMPORTANT: no exploration during testing
qmaze = TreasureMaze(maze)

print("Completion check:", completion_check(model, qmaze))

pirate_start = (0, 0)

# fallback if (0,0) isn't a valid startt
if hasattr(qmaze, "free_cells") and pirate_start not in qmaze.free_cells:
    pirate_start = random.choice(qmaze.free_cells)

play_game(model, qmaze, pirate_start)
show(qmaze)


# ## Save and Submit Your Work
# 
# <div class="alert alert-block alert-info" style="color:black;">
#     <b>Hint:</b> To use the markdown block below, double click in the <b>Type Markdown and LaTeX:  𝛼2</b> block below, to turn it back to html, Run the cell.
# </div>
# 
# After you have finished creating the code for your notebook, save your work.
# Make sure that your notebook contains your name in the filename (e.g. Doe_Jane_ProjectTwo.html). Download this file as an .html file clicking on ***file*** in *Jupyter Notebook*, navigating down to ***Download as*** and clicking on ***.html***. 
# Download a copy of your .html file and submit it to Brightspace.

# In[24]:


Execution Notes / Environment Issues”
“Codio kernel/GPU caused No blas support for stream errors.”
“I reduced epochs / adjusted max_steps to prevent infinite loops.”
“Training ran successfully (see logs), but completion_check remained False at submission time"

