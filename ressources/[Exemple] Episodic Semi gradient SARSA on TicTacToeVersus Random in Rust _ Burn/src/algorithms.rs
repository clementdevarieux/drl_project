use crate::contracts::{DeepDiscreteActionsEnv, Forward};
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn epsilon_greedy_action<B: Backend<FloatElem=f32, IntElem=i64>, const NUM_STATES_FEATURES: usize, const NUM_ACTIONS: usize>(
    q_s: &Tensor<B, 1>,
    mask_tensor: &Tensor<B, 1>,
    available_actions: impl Iterator<Item=usize>,
    epsilon: f32,
    rng: &mut impl Rng,
) -> usize {
    if rng.gen_range(0f32..=1f32) < epsilon {
        available_actions.choose(rng).unwrap()
    } else {
        let inverted_mask = mask_tensor.clone().mul([-1f32; NUM_ACTIONS].into()).add([1f32; NUM_ACTIONS].into());
        let masked_q_s = (q_s.clone() * mask_tensor.clone()).add(inverted_mask.mul([f32::MIN; NUM_ACTIONS].into()));
        masked_q_s.clone().argmax(0).into_scalar() as usize
    }
}

pub fn episodic_semi_gradient_sarsa<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B=B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem=f32, IntElem=i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS>
>(
    mut model: M,
    num_episodes: usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    device: &B::Device) -> M
where
    M::InnerModule: Forward<B=B::InnerBackend>,
{
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();

    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;

    let mut env = Env::default();
    for ep_id in tqdm!(0..num_episodes) {
        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        if ep_id % 1000 == 0 {
            println!("Mean Score: {}", total_score / 1000.0);
            total_score = 0.0;
        }
        env.reset();

        if env.is_game_over() {
            continue;
        }

        let s = env.state_description();
        let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

        let mask = env.action_mask();
        let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
        let mut q_s = model.forward(s_tensor);

        let mut a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
            &q_s,
            &mask_tensor,
            env.available_actions_ids(),
            decayed_epsilon,
            &mut rng,
        );

        while !env.is_game_over() {
            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;

            let s_p = env.state_description();
            let s_p_tensor: Tensor<B, 1> = Tensor::from_floats(s_p.as_slice(), device);

            let mask_p = env.action_mask();
            let mask_p_tensor: Tensor<B, 1> = Tensor::from(mask_p).to_device(device);
            let q_s_p = Tensor::from_inner(model.valid().forward(s_p_tensor.clone().inner()));

            let (a_p, q_s_p_a_p) = if env.is_game_over() {
                (0, Tensor::from([0f32]).to_device(device))
            } else {
                let a_p = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                    &q_s_p,
                    &mask_p_tensor,
                    env.available_actions_ids(),
                    decayed_epsilon,
                    &mut rng,
                );
                #[allow(clippy::single_range_in_vec_init)]
                let q_s_p_a_p = q_s_p.clone().slice([a_p..(a_p + 1)]);
                (a_p, q_s_p_a_p)
            };
            let q_s_p_a_p = q_s_p_a_p.detach();
            #[allow(clippy::single_range_in_vec_init)]
            let q_s_a = q_s.clone().slice([a..(a + 1)]);

            let loss = (q_s_a - q_s_p_a_p.mul_scalar(gamma).add_scalar(r)).powf_scalar(2f32);
            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &model);

            model = optimizer.step(alpha.into(), model, grads);

            q_s = model.forward(s_p_tensor);
            a = a_p;
        }
        total_score += env.score();
    }

    model
}
