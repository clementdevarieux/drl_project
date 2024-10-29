use burn::prelude::{Backend, Tensor};

pub trait Forward {
    type B: Backend;
    fn forward<const DIM: usize>(&self, input: Tensor<Self::B, DIM>) -> Tensor<Self::B, DIM>;
}

pub trait DeepDiscreteActionsEnv<const NUM_STATES_FEATURES: usize, const NUM_ACTIONS: usize>: Default + Clone {
    fn state_description(&self) -> [f32; NUM_STATES_FEATURES];
    fn available_actions_ids(&self) -> impl Iterator<Item=usize>;
    fn action_mask(&self) -> [f32; NUM_ACTIONS];
    fn step(&mut self, action: usize);
    fn is_game_over(&self) -> bool;
    fn score(&self) -> f32;
    fn reset(&mut self);
}