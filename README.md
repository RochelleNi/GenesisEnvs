# Genesis Environment
This repository contains example RL environment for Genesis general-purpose physics platform.

## Requirements
All necessary dependencies have been listed in `requirements.txt`.
You can create a conda environment by:

```bash
conda create --name genesis_env --file requirements.txt
```

## Command-line Arguments

- `-v` or `--vis` enables visualization.
- `-l` or `--load` loads the model from a checkpoint.
- `-n` or `--num_envs` specifies the number of parallel environments.
- `-b` or `--batch_size` defines the batch size used for training.
- `-hd` or `--hidden_dim` sets the hidden dimension for the network.
- `-t` or `--task` specifies the task to train on. Available tasks include:
  - `GraspFixedBlock`: Environment for grasping a fixed block.
  - `GraspFixedRod`: Environment for grasping a fixed rod.
  - `GraspRandomBlock`: Environment for grasping a randomly placed block.
  - `GraspRandomRod`: Environment for grasping a randomly placed rod.



## Usage

- Training

You can run different learning algorithms with the following command structure. Here is an example of running training with 10 envs:
```bash
python run_{algo}.py -n 10
```
where `algo` can be `dqn`, `ppo` or `heuristic`.

<img  src="figs/train.gif" width="300">

- Evaluation

To test the trained policy, you can load a pretrained model from the checkpoint and visualize the rollout, by executing the script with the following command-line arguments:
```bash
python run_{algo}.py -l -v -n 1
```
Similarly, you can specify `algo` as you like.

<img  src="figs/eval.gif" width="300">

## Saving and Loading Checkpoints

The agent periodically saves the model's weights and the target network state for later resumption. 

```python
def save_checkpoint(self, file_path):
    checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'target_model_state_dict': self.target_model.state_dict()
    }
    torch.save(checkpoint, file_path)
```
You can load a checkpoint by setting the `--load` flag. We've provided a successfully trained checkpoint `dqn_checkpoint.pth` for a Franka robot to grasp a block, which you can use for evaluation.
```python
    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.model.eval()
        self.target_model.eval()
```


