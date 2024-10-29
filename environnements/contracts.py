import tensorflow as tf
import numpy as np

class Forward:
    def forward(self, input_tensor: tf.Tensor) -> tf.Tensor:
        raise "Unimplemented"

class DeepDiscreteActionsEnv:
    def state_description(self) -> np.ndarray:
        raise "Unimplemented"

    def available_actions(self) -> np.ndarray:
        raise "Unimplemented"

    def action_mask(self) -> np.ndarray:
        raise "Unimplemented"

    def step(self, action: int):
        raise "Unimplemented"

    def reset(self):
        raise "Unimplemented"