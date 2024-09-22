use std::fmt;
use rand::prelude::*;
use std::iter::IntoIterator;
use rand::seq::SliceRandom;
use std::cmp;
use std::f32;
// use tqdm::tqdm;

pub fn policy_iteration(S: Vec<i32>,
                        A:Vec<i32>,
                        R:Vec<i32>,
                        T:Vec<i32>,
                        p:Vec<Vec<Vec<Vec<f32>>>>,
                        theta: f32,
                        gamma: f32) -> Vec<i32> {

    let len_S= S.clone().len();
    let mut rng = rand::thread_rng();
    let mut V: Vec<f32> = Vec::with_capacity(len_S);

    for _ in 0..len_S {
        V.push(rng.gen_range(0f32..1f32));
    }

    let mut Pi= Vec::with_capacity(len_S);

    for _ in 0..len_S {
        let random_index = rng.gen_range(0..A.len());
        Pi.push(A[random_index]); // mettre des valeurs aléatoires de A
    }

    loop {
        // policy evaluation
        loop {
            let mut delta: f32 = 0.0;
            for s in 0..S.len() {
                let mut v = V[s];
                let mut total: f32 = 0f32;
                for s_p in 0..S.len() {
                    for r in 0..R.len() {
                        total = total + p[s][Pi[s] as usize][s_p][r] * (R[r] as f32 + gamma * V[s_p]);
                    }
                }
                V[s] = total;
                delta = delta.max((v - V[s]).abs());
            }
            if delta < theta {
                break;
            }
        }

        let mut policy_stable = true;

        for s in 0..S.len() {
            if T.contains(&(s as i32)) {
                continue;
            }

            let mut old_action = Pi[s];

            let mut argmax_a: i32 = -9999999;
            let mut max_a: f32 = -9999999.0;

            for a in 0..A.len() {
                let mut total: f32 = 0.0;

                for s_p in 0..S.len() {
                    for r_index in 0..R.len() {
                        total += p[s][a][s_p][r_index] * (R[r_index] as f32 + gamma * V[s_p])
                    }
                }

                if argmax_a == -9999999 || total >= max_a {
                    argmax_a = a as i32;
                    max_a = total;
                }
            }

            Pi[s] = argmax_a;

            if old_action != Pi[s] {
                policy_stable = false;
            }
        }

        if policy_stable {
            break
        }
    }
    return Pi
}

pub fn value_iteration(S: Vec<i32>,
                       A:Vec<i32>,
                       R:Vec<i32>,
                       T:Vec<i32>,
                       p:Vec<Vec<Vec<Vec<f32>>>>,
                       theta: f32,
                       gamma: f32) -> Vec<i32> {

    let len_S= S.clone().len();
    let mut rng = rand::thread_rng();
    let mut V: Vec<f32> = Vec::with_capacity(len_S);

    for i in 0..len_S {
        if T.contains(&(i as i32)) {
            V.push(0f32);
        } else {
            V.push(rng.gen_range(0f32..1f32));
        }
    }


    loop {
        let mut delta = 0f32;
        for s in 0..len_S {
            if T.contains(&(s as i32)) {
                continue;
            }

            let v = V[s];
            let mut max_value: f32 = -9999f32;
            for a in 0..A.len() {
                let mut total: f32 = 0.0;
                for s_p in 0..S.len() {
                    for r in 0..R.len() {
                        total += p[s][a][s_p][r] * (R[r] as f32 + gamma * V[s_p]);
                    }
                }
                if total > max_value {
                    max_value = total;
                }
            }

            V[s] = max_value;
            delta = delta.max((v - V[s]).abs());
        }
        if delta < theta {
            break;
        }
    }

    let mut Pi: Vec<i32> = vec![-1; len_S];
    for s in 0..S.len() {
        if T.contains(&(s as i32)) {
            continue;
        }

        let mut argmax_a: i32 = -1;
        let mut max_value: f32 = -99999f32;

        for a in 0..A.len() {
            let mut total: f32 = 0.0;
            for s_p in 0..S.len() {
                for r in 0..R.len() {
                    total += p[s][a][s_p][r] * (R[r] as f32 + gamma * V[s_p]);
                }
            }

            if total > max_value {
                max_value = total;
                argmax_a = a as i32;
            }
        }

        Pi[s] = argmax_a;
    }

    Pi
}

/*
// TODO
pub fn monte_carlo_exploring_starts(S: Vec<i32>,
                      A:Vec<i32>,
                      R:Vec<i32>,
                      T:Vec<i32>,
                      p:Vec<Vec<Vec<Vec<f32>>>>,
                      theta: f32,
                      gamma: f32,
                      nb_iter: i32,
                      max_steps: i32) -> Vec<i32> {

    let len_S= S.clone().len();
    println!("len de S");
    println!("{:?}", len_S);
    let len_A = A.clone().len();
    println!("len de A");
    println!("{:?}",len_A);

    let len_R = R.clone().len();

    let mut rng = rand::thread_rng();

    let mut Pi= Vec::with_capacity(len_S);

    let mut q_s_a: Vec<Vec<f32>>= vec![vec![0.0;len_A]; len_S];

    for s in 0..len_S {
        for a in 0..len_A {
            q_s_a[s][a] = rng.gen_range(-10.0..10.0);
        }
    }

    let mut returns_s_a: Vec<Vec<Vec<usize>>>= vec![vec![vec![];len_A]; len_S];

    println!("{:?}",returns_s_a);

    for _ in tqdm(0..nb_iter) {
        let mut is_first_action: bool = true;
        let mut trajectory: Vec<usize> = vec![];
        let mut steps_count: i32 = 0;
        let mut prev_score: i32 = 0;
        while steps_count < max_steps {
            let mut s:i32 = rng.gen_range(0..len_S) as i32;

            let aa = A.clone();

            if !Pi.contains(&s) {
                let random_index = rng.gen_range(0..A.len());
                Pi[s] = A[random_index];
            }

            if is_first_action {
                let mut a:i32 = rng.gen_range(0..len_A) as i32;
                is_first_action = false;
            } else {
                let mut a:i32 = Pi[s];
            }

            let mut score:i32 = 0;
            for s_p in 0..len_S {
                for r in 0..len_R {
                    score = score + p[s][a][s_p][r] * R[r];
                }
            }

            let mut reward = score - prev_score;
            prev_score = score;

            //trajectory = trajectory.push((s, a, reward, ))


            if T.contains(&s) {
                break;
            }
        }
    }

    Pi
}


// TODO
pub fn monte_carlo_on_policy_first_visit(S: Vec<i32>,
                                         A:Vec<i32>,
                                         R:Vec<i32>,
                                         T:Vec<i32>,
                                         p:Vec<Vec<Vec<Vec<f32>>>>,
                                         theta: f32,
                                         gamma: f32) -> Vec<i32> {

    pass
}

// TODO
pub fn monte_carlo_off_policy_control(S: Vec<i32>,
                                         A:Vec<i32>,
                                         R:Vec<i32>,
                                         T:Vec<i32>,
                                         p:Vec<Vec<Vec<Vec<f32>>>>,
                                         theta: f32,
                                         gamma: f32) -> Vec<i32> {

    pass
}

// TODO
pub fn sarsa(S: Vec<i32>,
                                      A:Vec<i32>,
                                      R:Vec<i32>,
                                      T:Vec<i32>,
                                      p:Vec<Vec<Vec<Vec<f32>>>>,
                                      theta: f32,
                                      gamma: f32) -> Vec<i32> {

    pass
}

// TODO
pub fn q_learning(S: Vec<i32>,
             A:Vec<i32>,
             R:Vec<i32>,
             T:Vec<i32>,
             p:Vec<Vec<Vec<Vec<f32>>>>,
             theta: f32,
             gamma: f32) -> Vec<i32> {

    pass
}

// TODO
pub fn dyna_q(S: Vec<i32>,
             A:Vec<i32>,
             R:Vec<i32>,
             T:Vec<i32>,
             p:Vec<Vec<Vec<Vec<f32>>>>,
             theta: f32,
             gamma: f32) -> Vec<i32> {

    pass
}
*/
