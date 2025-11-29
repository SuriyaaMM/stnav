import torch
from terrain import Terrain
from network import ActorCriticNetwork, train, train_lbfgs

terrain = Terrain(seed=145)
# terrain.render()
print(terrain.get_state_shape())
terrain.render()

TRAIN_ITERATIONS = 25
REPLAY_ITERATIONS = 50
MAX_REPLAY_ITERATIONS = 2500 
PPO_EPOCHS = 6
GAMMA = 0.99
LAMBDA = 0.98
CLIP_COEFF = 0.2
VALUE_LOSS_COEFF = 0.25
ENTROPY_INITIAL = 0.15
ENTROPY_MIN = 0.01

total_updates = TRAIN_ITERATIONS * PPO_EPOCHS
def lr_lambda(update_step):
    return 1.0 - (update_step / total_updates)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = ActorCriticNetwork(terrain.get_state_shape(), num_actions=len(terrain.actions)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
replay_df, ppo_df = train(
    terrain,
    model,
    optimizer,
    scheduler,
    device,
    TRAIN_ITERATIONS,
    REPLAY_ITERATIONS,
    MAX_REPLAY_ITERATIONS,
    PPO_EPOCHS,
    GAMMA, 
    LAMBDA,
    CLIP_COEFF,
    VALUE_LOSS_COEFF,
    ENTROPY_INITIAL,
    ENTROPY_MIN
)

print(replay_df)
print(ppo_df)