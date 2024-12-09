### SAC with HER:
- bad performance for training up to 10,000,000 steps
- problem: agent quickly learns to move into corner and stay there (possible bug)
- problem: automatic environment reset
    - false input to compute_reward()
    - current solution: delay done by one time step (might not be the right way, try obtaining terminal observation from info)
    - current solution: delay done signal b
- bad goals: ball pos/vel as goal might not be effective as player-ball interactions are too rare, not moving the ball at all in an episode results in false learning signal
    - possible solution: initial policy of moving towards the ball and kicking it (no improvement in performance so far)

### TD-MPC2:
- implementation only supports single environment
- vectorized environment adaptation:
    - rollout collection, replay buffer for vectorized data: done
    - vectorized planning: not done
    - possible problem: automatic environment reset
- training:
    - 1,000,000 steps from scratch: poor performance
    - next steps: pretrained checkpoint (from multitask training), more training steps (on PC)

### General approach:
- too hard to score goals? Make the problem easier by making goals/ball bigger?
- other reward terms (such as ball velocity)

### Multi-agent training:
- opponent actions chosen from copy of the agent
- possibly better approach: update agent copy only every n iterations and keep opponent constant inbetween

### Other models:

### Rewards:
- distance to opponent goal
- touch ball with timeout after initial touch
- curriculum: decay of reward weights, after specific steps switch

### Error Hunting:
- look at action space of agent vs env action space (does it match?)
Why does the player move to the right??