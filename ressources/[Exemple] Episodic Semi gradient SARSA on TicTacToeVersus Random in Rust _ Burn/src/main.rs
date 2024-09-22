mod contracts;
mod algorithms;
mod environments;

use crate::algorithms::{episodic_semi_gradient_sarsa, epsilon_greedy_action};
use crate::contracts::{DeepDiscreteActionsEnv, Forward};
use crate::environments::tic_tac_toe::{TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES};
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::nn;
use burn::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Use your favourite backend here
// type Backend = burn::backend::NdArray;
type MyBackend = burn::backend::LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

type GameEnv = TicTacToeVersusRandom;

#[derive(Module, Debug)]
struct MyQMLP<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
    output_layer: nn::Linear<B>,
}

impl<B: burn::prelude::Backend> MyQMLP<B> {
    fn new(device: &B::Device, input_state_features: usize, output_actions: usize) -> Self {
        let linear1 = nn::LinearConfig::new(input_state_features, 64)
            .with_bias(true)
            .init(device);
        let linear2 = nn::LinearConfig::new(64, 32)
            .with_bias(true)
            .init(device);
        let linear3 = nn::LinearConfig::new(32, 16)
            .with_bias(true)
            .init(device);
        let output_layer = nn::LinearConfig::new(16, output_actions)
            .with_bias(true)
            .init(device);
        MyQMLP {
            linear1,
            linear2,
            linear3,
            output_layer,
        }
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear1.forward(x);
        let x = x.tanh();
        let x = self.linear2.forward(x);
        let x = x.tanh();
        let x = self.linear3.forward(x);
        let x = x.tanh();
        self.output_layer.forward(x)
    }
}

impl<B: Backend> Forward for MyQMLP<B> {
    type B = B;

    fn forward<const DIM: usize>(&self, input: Tensor<Self::B, DIM>) -> Tensor<Self::B, DIM> {
        self.forward(input)
    }
}

fn main() {
    let device = &Default::default();

    // Create the model
    let model = MyQMLP::<MyAutodiffBackend>::new(device,
                                                 NUM_STATE_FEATURES,
                                                 NUM_ACTIONS);

    // Train the model
    let model =
        episodic_semi_gradient_sarsa::<
            NUM_STATE_FEATURES,
            NUM_ACTIONS,
            _,
            MyAutodiffBackend,
            GameEnv,
        >(
            model,
            50_000,
            0.999f32,
            3e-3,
            1.0f32,
            1e-5f32,
            device,
        );

    // Let's play some games (press enter to show the next game)
    let device = &Default::default();
    let mut env = GameEnv::default();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    loop {
        env.reset();
        while !env.is_game_over() {
            println!("{}", env);
            let s = env.state_description();
            let s_tensor: Tensor<MyBackend, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<MyBackend, 1> = Tensor::from(mask).to_device(device);
            let q_s = model.valid().forward(s_tensor);

            let a = epsilon_greedy_action::<MyBackend, NUM_STATE_FEATURES, NUM_ACTIONS>(&q_s, &mask_tensor, env.available_actions_ids(), 1e-5f32, &mut rng);
            env.step(a);
        }
        println!("{}", env);
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).unwrap();
    }
}