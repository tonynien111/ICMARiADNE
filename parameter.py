# saving path
FOLDER_NAME = 'ariadne1'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 32  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100  # how many episodes before saving a gif

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 16  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 2  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 16  # how many threads does your CPU have

# network parameters
NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value, need it for batch training

# ICM parameters
USE_ICM = True  # enable intrinsic curiosity module
ICM_FEATURE_DIM = 256  # dimension of ICM feature representations
ICM_LR = 1e-4  # learning rate for ICM
ICM_BETA = 0.2  # weight for forward loss in ICM (1-beta for inverse loss)
ICM_ETA = 0.01  # base weight for intrinsic reward in total reward
ICM_ADAPTIVE = True  # enable adaptive ICM weighting based on exploration progress
ICM_ETA_MIN = 0.001  # minimum ICM weight when fully explored
ICM_ETA_MAX = 0.05   # maximum ICM weight when unexplored
ICM_EXPLORATION_THRESHOLD = 0.8  # exploration rate threshold for adaptive scaling
ICM_ACTION_DIM = K_SIZE  # action dimension for ICM (same as K_SIZE)

# Data split parameters
DATA_SPLIT_MODE = 'train'  # 'train', 'val', 'test' - which dataset to use
TRAIN_SPLIT = 0.6  # 60% for training
VAL_SPLIT = 0.2    # 20% for validation  
TEST_SPLIT = 0.2   # 20% for testing
RANDOM_SEED = 42   # for reproducible splits

# Behavior Cloning parameters
USE_BEHAVIOR_CLONING = True  # enable behavior cloning pretraining
EXPERT_EPISODES = 50  # number of expert episodes to collect
BC_EPOCHS = 100  # number of behavior cloning epochs
BC_BATCH_SIZE = 64  # batch size for behavior cloning
BC_LR = 1e-4  # learning rate for behavior cloning
BC_VALIDATION_SPLIT = 0.2  # validation split for early stopping
BC_PATIENCE = 10  # early stopping patience
BC_DEMONSTRATIONS_PATH = 'expert_demonstrations.pkl'  # path to save/load demonstrations
BC_COLLECT_DATA_ON_START = True  # collect expert data at start if not found
BC_MAX_STEPS_PER_EPISODE = 50  # maximum steps per expert episode

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = False  # do you want to train the network using GPUs - set to False for CPU-only
NUM_GPU = 0  # 0 unless you want to collect data using GPUs

