use kdam::tqdm;
use rand::prelude::IteratorRandom;
use crate::contracts::DeepDiscreteActionsEnv;
use crate::environments::tic_tac_toe::TicTacToeVersusRandom;

mod environments;
mod contracts;
type GameEnv = TicTacToeVersusRandom;

fn mcrr<const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize>(env: &impl DeepDiscreteActionsEnv<NUM_STATES_FEATURES, NUM_ACTIONS>, num_rollouts_per_action: usize) -> usize {
    let mut best_a = 0;
    let mut best_q_s_z = f32::NEG_INFINITY;

    for a in env.available_actions_ids() {
        let mut q_s_a = 0.0;
        for _ in 0..num_rollouts_per_action {
            let mut env_copy = env.clone();
            env_copy.step(a);
            while !env_copy.is_game_over() {
                let a = env_copy.available_actions_ids().choose(&mut rand::thread_rng()).unwrap();
                env_copy.step(a);
            }
            q_s_a += env_copy.score();
        }
        q_s_a /= num_rollouts_per_action as f32;

        if q_s_a > best_q_s_z {
            best_q_s_z = q_s_a;
            best_a = a;
        }
    }
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
                let a = mcrr(&env, *num_rollouts_per_action);
                env.step(a);
            }
            mean_score += env.score();
        }
        mean_score /= NUM_GAMES as f32;
        println!("Mean score for {} rollouts per action: {}", num_rollouts_per_action, mean_score);
    }
}