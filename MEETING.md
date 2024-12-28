### SAC with HER: 
- implemented terminal observation return, still have to test

### TD-MPC2:
- implementation only supports single environment
- vectorized environment adaptation:
    - rollout collection, replay buffer for vectorized data: done
    - vectorized planning: not done
    - possible problem: automatic environment reset
- training:
    - 1,000,000 steps finetuning (using basketball as initialization task embedding): poor performance
        - bad task initialization?
    - next steps: vectorized implementation?

### Reward shaping:
- rewards adding up to 1
- additional reward terms:
    - distance player ball if ball is in player's half
    - collision between player and ball weighted by players velocity
    - ball stationary (negative reward)
- implementation of curriculum wrapper: modification of reward weights over time
- decay of shaped rewards over time: to be done

### General approach:
- too hard to score goals? Make the problem easier by making goals/ball bigger?
- other reward terms (such as ball velocity)

### Multi-agent training:
- opponent actions chosen from copy of the agent
- possibly better approach: update agent copy only every n iterations and keep opponent constant inbetween

### Other models:
- DreamerV3? Used in air hockey paper but high latency

### Error Hunting:
- look at action space of agent vs env action space (does it match?)
Why does the player move to the right??