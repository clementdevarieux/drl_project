use kdam::tqdm;
use rand::prelude::IteratorRandom;
use crate::contracts::DeepDiscreteActionsEnv;
use crate::environments::tic_tac_toe::TicTacToeVersusRandom;

mod environments;
mod contracts;
type GameEnv = TicTacToeVersusRandom;

fn mcts<const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize>(env: &impl DeepDiscreteActionsEnv<NUM_STATES_FEATURES, NUM_ACTIONS>, num_iterations: usize, c: f32) -> usize {
    let mut best_a = 0;
    let mut best_q_s_z = f32::NEG_INFINITY;

    todo!("Implement the MCTS algorithm here");

    best_a
}


pub fn main() {
    let mut env = GameEnv::default();
    const NUM_GAMES: usize = 10000;

    for num_rollouts_per_action in [1, 10, 100, 1000, 10000].iter() {
        let mut mean_score = 0.0;
        for _ in tqdm!(0..NUM_GAMES) {
            env.reset();
            while !env.is_game_over() {
                let a = mcts(&env, *num_rollouts_per_action, 2.0f32.sqrt());
                env.step(a);
            }
            mean_score += env.score();
        }
        mean_score /= NUM_GAMES as f32;
        println!("Mean score for {} iterations: {}", num_rollouts_per_action, mean_score);
    }
}