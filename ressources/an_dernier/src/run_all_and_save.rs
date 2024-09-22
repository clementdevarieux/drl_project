use crate::Env;

use rand::prelude::*;
use std::iter::IntoIterator;
use rand::seq::SliceRandom;
use colored::*;
use std::time;
use chrono::Local;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::PathBuf;
use csv::WriterBuilder;
use uuid::Uuid;

pub(crate) fn run_all_and_save() {


    let file_path_res = PathBuf::from("./results").join("results.csv");

    let file_res = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path_res)
        .unwrap();

    let mut wtr = WriterBuilder::new().has_headers(false).from_writer(file_res);

    let metadata = std::fs::metadata(&file_path_res);
    if metadata.unwrap().len() == 0 {
        wtr.write_record(&["run_number", "run_date", "env", "model", "gamma", "theta", "epsilon", "alpha", "number_iteration", "max_step", "actual_run", "time", "score"]);
    }

    let run_number = Uuid::new_v4();
    let date_now = Local::now();
    let run_date = date_now.format("%Y-%m-%d_%H-%M-%S").to_string();
    let number_of_tests = 5;
    let gamma = 0.9999f32;
    let epsilon = 0.20;
    let alpha = 0.10;
    let nb_iter = 5000;
    let max_steps = 50000;
    let theta = 0.20f32;

    //////// LineWorld

    let model_name = "policy_iteration";
    let env_name = "LineWorld";

    let file_path = PathBuf::from("./2024-07-23").join("LineWorld.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "LineWorld\n");

    let mut durations_lineworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut lineworld = Env::LineWorld::LineWorld::init();
        let now = time::Instant::now();
        let res = lineworld.policy_iteration(theta, gamma);
        durations_lineworld.push(now.elapsed().as_millis());
        lineworld.run_game_vec(res.clone());
        final_score.push(lineworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            theta.to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_lineworld.get(i as usize).unwrap().to_string(),
            lineworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} policy_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_lineworld.iter().sum::<u128>() / durations_lineworld.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    // ///////////////////
    let model_name = "value_iteration";

    let mut durations_lineworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut lineworld = Env::LineWorld::LineWorld::init();
        let now = time::Instant::now();
        let res = lineworld.value_iteration(theta, gamma);
        durations_lineworld.push(now.elapsed().as_millis());
        lineworld.run_game_vec(res.clone());
        final_score.push(lineworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            theta.to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_lineworld.get(i as usize).unwrap().to_string(),
            lineworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} value_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_lineworld.iter().sum::<u128>() / durations_lineworld.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    // ////////////////

    let model_name = "monte_carlo_exploring_starts";

    let mut durations_lineworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut lineworld = Env::LineWorld::LineWorld::init();
        let now = time::Instant::now();
        let res = lineworld.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_lineworld.push(now.elapsed().as_millis());
        lineworld.run_game_random_state_hashmap(res.clone());
        final_score.push(lineworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_lineworld.get(i as usize).unwrap().to_string(),
            lineworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_lineworld.iter().sum::<u128>() / durations_lineworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);
    ////////////////////
    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_lineworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut lineworld = Env::LineWorld::LineWorld::init();
        let now = time::Instant::now();
        let res = lineworld.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_lineworld.push(now.elapsed().as_millis());
        lineworld.run_game_random_hashmap(res.clone());
        final_score.push(lineworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_lineworld.get(i as usize).unwrap().to_string(),
            lineworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_lineworld.iter().sum::<u128>() / durations_lineworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ///////////////
    let model_name = "monte_carlo_off_policy";

    let mut durations_lineworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut lineworld = Env::LineWorld::LineWorld::init();
        let now = time::Instant::now();
        let res = lineworld.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_lineworld.push(now.elapsed().as_millis());
        lineworld.run_game_hashmap(res.clone());
        final_score.push(lineworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_lineworld.get(i as usize).unwrap().to_string(),
            lineworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_lineworld.iter().sum::<u128>() / durations_lineworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ///////////
    let model_name = "Q_learning_off_policy";

    let mut durations_lineworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut lineworld = Env::LineWorld::LineWorld::init();
        let now = time::Instant::now();
        let res = lineworld.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_lineworld.push(now.elapsed().as_millis());
        lineworld.run_game_hashmap(res.clone());
        final_score.push(lineworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_lineworld.get(i as usize).unwrap().to_string(),
            lineworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_lineworld.iter().sum::<u128>() / durations_lineworld.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);
    ///////////
    let model_name = "sarsa";

    let mut durations_lineworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut lineworld = Env::LineWorld::LineWorld::init();
        let now = time::Instant::now();
        let res = lineworld.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_lineworld.push(now.elapsed().as_millis());
        lineworld.run_game_hashmap(res.clone());
        final_score.push(lineworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_lineworld.get(i as usize).unwrap().to_string(),
            lineworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_lineworld.iter().sum::<u128>() / durations_lineworld.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);





    //////// GridWorld
    let model_name = "policy_iteration";
    let env_name = "GridWorld";

    let file_path = PathBuf::from("./2024-07-23").join("GridWorld.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "GridWorld\n");

    let mut durations_gridworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut gridworld = Env::GridWorld::GridWorld::init();
        let now = time::Instant::now();
        let res = gridworld.policy_iteration(theta, gamma);
        durations_gridworld.push(now.elapsed().as_millis());
        gridworld.run_game_vec(res.clone());
        final_score.push(gridworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            theta.to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_gridworld.get(i as usize).unwrap().to_string(),
            gridworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} policy_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_gridworld.iter().sum::<u128>() / durations_gridworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //////
    let model_name = "value_iteration";

    let mut durations_gridworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut gridworld = Env::GridWorld::GridWorld::init();
        let now = time::Instant::now();
        let res = gridworld.value_iteration(0.0001, gamma);
        durations_gridworld.push(now.elapsed().as_millis());
        gridworld.run_game_vec(res.clone());
        final_score.push(gridworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "0.01".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_gridworld.get(i as usize).unwrap().to_string(),
            gridworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} value_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_gridworld.iter().sum::<u128>() / durations_gridworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //////
    let model_name = "monte_carlo_exploring_starts";

    let mut durations_gridworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut gridworld = Env::GridWorld::GridWorld::init();
        let now = time::Instant::now();
        let res = gridworld.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_gridworld.push(now.elapsed().as_millis());
        gridworld.run_game_random_state_hashmap(res.clone());
        final_score.push(gridworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_gridworld.get(i as usize).unwrap().to_string(),
            gridworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_gridworld.iter().sum::<u128>() / durations_gridworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    /////
    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_gridworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut gridworld = Env::GridWorld::GridWorld::init();
        let now = time::Instant::now();
        let res = gridworld.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_gridworld.push(now.elapsed().as_millis());
        gridworld.run_game_random_hashmap(res.clone());
        final_score.push(gridworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_gridworld.get(i as usize).unwrap().to_string(),
            gridworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_gridworld.iter().sum::<u128>() / durations_gridworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////
    let model_name = "monte_carlo_off_policy";

    let mut durations_gridworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut gridworld = Env::GridWorld::GridWorld::init();
        let now = time::Instant::now();
        let res = gridworld.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_gridworld.push(now.elapsed().as_millis());
        gridworld.run_game_hashmap(res.clone());
        final_score.push(gridworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_gridworld.get(i as usize).unwrap().to_string(),
            gridworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_gridworld.iter().sum::<u128>() / durations_gridworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);
    ///////
    let model_name = "Q_learning_off_policy";

    let mut durations_gridworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let now = time::Instant::now();
        let mut gridworld = Env::GridWorld::GridWorld::init();
        let res = gridworld.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_gridworld.push(now.elapsed().as_millis());
        gridworld.run_game_hashmap(res.clone());
        final_score.push(gridworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_gridworld.get(i as usize).unwrap().to_string(),
            gridworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_gridworld.iter().sum::<u128>() / durations_gridworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ///////
    let model_name = "sarsa";

    let mut durations_gridworld: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let now = time::Instant::now();
        let mut gridworld = Env::GridWorld::GridWorld::init();
        let res = gridworld.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_gridworld.push(now.elapsed().as_millis());
        gridworld.run_game_hashmap(res.clone());
        final_score.push(gridworld.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_gridworld.get(i as usize).unwrap().to_string(),
            gridworld.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_gridworld.iter().sum::<u128>() / durations_gridworld.len() as u128);

    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);



    //////// Shifumi
    let model_name = "policy_iteration";
    let env_name = "Shifumi";

    let file_path = PathBuf::from("./2024-07-23").join("Shifumi.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "Shifumi\n");

    let mut durations_shifumi: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut shifumi = Env::Shifumi::Shifumi::init();
        let now = time::Instant::now();
        let res = shifumi.policy_iteration(theta, gamma);
        durations_shifumi.push(now.elapsed().as_millis());
        shifumi.run_game_vec(res.clone());
        final_score.push(shifumi.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            theta.to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_shifumi.get(i as usize).unwrap().to_string(),
            shifumi.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} policy_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_shifumi.iter().sum::<u128>() / durations_shifumi.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////
    let model_name = "value_iteration";

    let mut durations_shifumi: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut shifumi = Env::Shifumi::Shifumi::init();
        let now = time::Instant::now();
        let res = shifumi.value_iteration(theta, gamma);
        durations_shifumi.push(now.elapsed().as_millis());
        shifumi.run_game_vec(res.clone());
        final_score.push(shifumi.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            theta.to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_shifumi.get(i as usize).unwrap().to_string(),
            shifumi.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} value_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_shifumi.iter().sum::<u128>() / durations_shifumi.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////
    let model_name = "monte_carlo_exploring_starts";

    let mut durations_shifumi: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut shifumi = Env::Shifumi::Shifumi::init();
        let now = time::Instant::now();
        let res = shifumi.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_shifumi.push(now.elapsed().as_millis());
        shifumi.run_game_random_state_hashmap(res.clone());
        final_score.push(shifumi.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_shifumi.get(i as usize).unwrap().to_string(),
            shifumi.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_shifumi.iter().sum::<u128>() / durations_shifumi.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);
    /////////
    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_shifumi: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut shifumi = Env::Shifumi::Shifumi::init();
        let now = time::Instant::now();
        let res = shifumi.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_shifumi.push(now.elapsed().as_millis());
        shifumi.run_game_random_hashmap(res.clone());
        final_score.push(shifumi.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_shifumi.get(i as usize).unwrap().to_string(),
            shifumi.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_shifumi.iter().sum::<u128>() / durations_shifumi.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //////////
    let model_name = "monte_carlo_off_policy";

    let mut durations_shifumi: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut shifumi = Env::Shifumi::Shifumi::init();
        let now = time::Instant::now();
        let res = shifumi.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_shifumi.push(now.elapsed().as_millis());
        shifumi.run_game_hashmap(res.clone());
        final_score.push(shifumi.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_shifumi.get(i as usize).unwrap().to_string(),
            shifumi.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_shifumi.iter().sum::<u128>() / durations_shifumi.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);
    //////
    let model_name = "Q_learning_off_policy";

    let mut durations_shifumi: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut shifumi = Env::Shifumi::Shifumi::init();
        let now = time::Instant::now();
        let res = shifumi.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_shifumi.push(now.elapsed().as_millis());
        shifumi.run_game_hashmap(res.clone());
        final_score.push(shifumi.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_shifumi.get(i as usize).unwrap().to_string(),
            shifumi.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_shifumi.iter().sum::<u128>() / durations_shifumi.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //////
    let model_name = "sarsa";

    let mut durations_shifumi: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut shifumi = Env::Shifumi::Shifumi::init();
        let now = time::Instant::now();
        let res = shifumi.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_shifumi.push(now.elapsed().as_millis());
        shifumi.run_game_hashmap(res.clone());
        final_score.push(shifumi.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_shifumi.get(i as usize).unwrap().to_string(),
            shifumi.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_shifumi.iter().sum::<u128>() / durations_shifumi.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);



    //////// MontyHall

    let model_name = "policy_iteration";
    let env_name = "MontyHall";

    let file_path = PathBuf::from("./2024-07-23").join("MontyHall.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "MontyHall\n");

    let mut durations_montyhall: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut montyhall = Env::MontyHall::MontyHall::init();
        let now = time::Instant::now();
        let res = montyhall.policy_iteration(theta, gamma);
        durations_montyhall.push(now.elapsed().as_millis());
        montyhall.run_game_vec(res.clone());
        final_score.push(montyhall.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            theta.to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_montyhall.get(i as usize).unwrap().to_string(),
            montyhall.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} policy_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_montyhall.iter().sum::<u128>() / durations_montyhall.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    /////
    let model_name = "value_iteration";

    let mut durations_montyhall: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut montyhall = Env::MontyHall::MontyHall::init();
        let now = time::Instant::now();
        let res = montyhall.value_iteration(theta, gamma);
        durations_montyhall.push(now.elapsed().as_millis());
        montyhall.run_game_vec(res.clone());
        final_score.push(montyhall.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            theta.to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_montyhall.get(i as usize).unwrap().to_string(),
            montyhall.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} value_iteration (theta = {}, gammma = {}): {}ms", number_of_tests, theta, gamma, durations_montyhall.iter().sum::<u128>() / durations_montyhall.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    /////
    let model_name = "monte_carlo_exploring_starts";

    let mut durations_montyhall: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut montyhall = Env::MontyHall::MontyHall::init();
        let now = time::Instant::now();
        let res = montyhall.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_montyhall.push(now.elapsed().as_millis());
        montyhall.run_game_random_state_hashmap(res.clone());
        final_score.push(montyhall.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_montyhall.get(i as usize).unwrap().to_string(),
            montyhall.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_montyhall.iter().sum::<u128>() / durations_montyhall.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //
    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_montyhall: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut montyhall = Env::MontyHall::MontyHall::init();
        let now = time::Instant::now();
        let res = montyhall.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_montyhall.push(now.elapsed().as_millis());
        montyhall.run_game_random_hashmap(res.clone());
        final_score.push(montyhall.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_montyhall.get(i as usize).unwrap().to_string(),
            montyhall.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_montyhall.iter().sum::<u128>() / durations_montyhall.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //
    let model_name = "monte_carlo_off_policy";

    let mut durations_montyhall: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut montyhall = Env::MontyHall::MontyHall::init();
        let now = time::Instant::now();
        let res = montyhall.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_montyhall.push(now.elapsed().as_millis());
        montyhall.run_game_hashmap(res.clone());
        final_score.push(montyhall.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_montyhall.get(i as usize).unwrap().to_string(),
            montyhall.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_montyhall.iter().sum::<u128>() / durations_montyhall.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //
    let model_name = "Q_learning_off_policy";

    let mut durations_montyhall: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut montyhall = Env::MontyHall::MontyHall::init();
        let now = time::Instant::now();
        let res = montyhall.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_montyhall.push(now.elapsed().as_millis());
        montyhall.run_game_hashmap(res.clone());
        final_score.push(montyhall.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_montyhall.get(i as usize).unwrap().to_string(),
            montyhall.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_montyhall.iter().sum::<u128>() / durations_montyhall.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    /////
    let model_name = "sarsa";

    let mut durations_montyhall: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut montyhall = Env::MontyHall::MontyHall::init();
        let now = time::Instant::now();
        let res = montyhall.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_montyhall.push(now.elapsed().as_millis());
        montyhall.run_game_hashmap(res.clone());
        final_score.push(montyhall.score() as f32);
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_montyhall.get(i as usize).unwrap().to_string(),
            montyhall.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_montyhall.iter().sum::<u128>() / durations_montyhall.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ///// SecretEnv0
    let model_name = "monte_carlo_exploring_starts";
    let env_name = "SecretEnv0";

    let file_path = PathBuf::from("./2024-07-23").join("SecretEnv0.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "SecretEnv0\n");

    let mut durations_secretenv0: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut secretenv0 = Env::SecretEnv0::SecretEnv0::new();
        let now = time::Instant::now();
        let res = secretenv0.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_secretenv0.push(now.elapsed().as_millis());
        secretenv0.run_game_random_state_hashmap(res.clone());
        final_score.push(secretenv0.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv0.get(i as usize).unwrap().to_string(),
            secretenv0.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_secretenv0.iter().sum::<u128>() / durations_secretenv0.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////
    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_secretenv0: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut secretenv0 = Env::SecretEnv0::SecretEnv0::new();
        let now = time::Instant::now();
        let res = secretenv0.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv0.push(now.elapsed().as_millis());
        secretenv0.run_game_random_hashmap(res.clone());
        final_score.push(secretenv0.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv0.get(i as usize).unwrap().to_string(),
            secretenv0.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv0.iter().sum::<u128>() / durations_secretenv0.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //////
    let model_name = "monte_carlo_off_policy";

    let mut durations_secretenv0: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut secretenv0 = Env::SecretEnv0::SecretEnv0::new();
        let now = time::Instant::now();
        let res = secretenv0.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv0.push(now.elapsed().as_millis());
        secretenv0.run_game_hashmap(res.clone());
        final_score.push(secretenv0.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv0.get(i as usize).unwrap().to_string(),
            secretenv0.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv0.iter().sum::<u128>() / durations_secretenv0.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ///////
    let model_name = "Q_learning_off_policy";

    let mut durations_secretenv0: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut secretenv0 = Env::SecretEnv0::SecretEnv0::new();
        let now = time::Instant::now();
        let res = secretenv0.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv0.push(now.elapsed().as_millis());
        secretenv0.run_game_hashmap(res.clone());
        final_score.push(secretenv0.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv0.get(i as usize).unwrap().to_string(),
            secretenv0.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv0.iter().sum::<u128>() / durations_secretenv0.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

///////
    let model_name = "sarsa";

    let mut durations_secretenv0: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut secretenv0 = Env::SecretEnv0::SecretEnv0::new();
        let now = time::Instant::now();
        let res = secretenv0.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv0.push(now.elapsed().as_millis());
        secretenv0.run_game_hashmap(res.clone());
        final_score.push(secretenv0.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv0.get(i as usize).unwrap().to_string(),
            secretenv0.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv0.iter().sum::<u128>() / durations_secretenv0.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    //////// SecretEnv1
    let model_name = "monte_carlo_exploring_starts";
    let env_name = "SecretEnv1";

    let file_path = PathBuf::from("./2024-07-23").join("SecretEnv1.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "SecretEnv1\n");


    let mut durations_secretenv1: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();

    for i in 0..number_of_tests {
        let mut secretenv1 = Env::SecretEnv1::SecretEnv1::new();
        let now = time::Instant::now();
        let res = secretenv1.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_secretenv1.push(now.elapsed().as_millis());
        secretenv1.run_game_random_state_hashmap(res.clone());
        final_score.push(secretenv1.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv1.get(i as usize).unwrap().to_string(),
            secretenv1.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_secretenv1.iter().sum::<u128>() / durations_secretenv1.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////

    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_secretenv1: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut secretenv1 = Env::SecretEnv1::SecretEnv1::new();
        let now = time::Instant::now();
        let res = secretenv1.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv1.push(now.elapsed().as_millis());
        secretenv1.run_game_random_hashmap(res.clone());
        final_score.push(secretenv1.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv1.get(i as usize).unwrap().to_string(),
            secretenv1.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv1.iter().sum::<u128>() / durations_secretenv1.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////

    let model_name = "monte_carlo_off_policy";

    let mut durations_secretenv1: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut secretenv1 = Env::SecretEnv1::SecretEnv1::new();
        let now = time::Instant::now();
        let res = secretenv1.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv1.push(now.elapsed().as_millis());
        secretenv1.run_game_hashmap(res.clone());
        final_score.push(secretenv1.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv1.get(i as usize).unwrap().to_string(),
            secretenv1.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv1.iter().sum::<u128>() / durations_secretenv1.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////

    let model_name = "Q_learning_off_policy";

    let mut durations_secretenv1: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut secretenv1 = Env::SecretEnv1::SecretEnv1::new();
        let now = time::Instant::now();
        let res = secretenv1.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv1.push(now.elapsed().as_millis());
        secretenv1.run_game_hashmap(res.clone());
        final_score.push(secretenv1.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv1.get(i as usize).unwrap().to_string(),
            secretenv1.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv1.iter().sum::<u128>() / durations_secretenv1.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////

    let model_name = "sarsa";

    let mut durations_secretenv1: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut secretenv1 = Env::SecretEnv1::SecretEnv1::new();
        let now = time::Instant::now();
        let res = secretenv1.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv1.push(now.elapsed().as_millis());
        secretenv1.run_game_hashmap(res.clone());
        final_score.push(secretenv1.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv1.get(i as usize).unwrap().to_string(),
            secretenv1.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv1.iter().sum::<u128>() / durations_secretenv1.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

//
// //////// SeceretEnv2
//
    let model_name = "monte_carlo_exploring_starts";
    let env_name = "SecretEnv2";

    let file_path = PathBuf::from("./2024-07-23").join("SecretEnv2.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "SecretEnv2\n");

    let mut durations_secretenv2: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut secretenv2 = Env::SecretEnv2::SecretEnv2::new();
        let now = time::Instant::now();
        let res = secretenv2.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_secretenv2.push(now.elapsed().as_millis());
        secretenv2.run_game_random_state_hashmap(res.clone());
        final_score.push(secretenv2.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv2.get(i as usize).unwrap().to_string(),
            secretenv2.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_secretenv2.iter().sum::<u128>() / durations_secretenv2.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////


    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_secretenv2: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut secretenv2 = Env::SecretEnv2::SecretEnv2::new();
        let now = time::Instant::now();
        let res = secretenv2.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv2.push(now.elapsed().as_millis());
        secretenv2.run_game_random_hashmap(res.clone());
        final_score.push(secretenv2.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv2.get(i as usize).unwrap().to_string(),
            secretenv2.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv2.iter().sum::<u128>() / durations_secretenv2.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////

    let model_name = "monte_carlo_off_policy";

    let mut durations_secretenv2: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut secretenv2 = Env::SecretEnv2::SecretEnv2::new();
        let now = time::Instant::now();
        let res = secretenv2.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv2.push(now.elapsed().as_millis());
        secretenv2.run_game_hashmap(res.clone());
        final_score.push(secretenv2.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv2.get(i as usize).unwrap().to_string(),
            secretenv2.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv2.iter().sum::<u128>() / durations_secretenv2.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);


    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    /////

    let model_name = "Q_learning_off_policy";

    let mut durations_secretenv2: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut secretenv2 = Env::SecretEnv2::SecretEnv2::new();
        let now = time::Instant::now();
        let res = secretenv2.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv2.push(now.elapsed().as_millis());
        secretenv2.run_game_hashmap(res.clone());
        final_score.push(secretenv2.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv2.get(i as usize).unwrap().to_string(),
            secretenv2.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv2.iter().sum::<u128>() / durations_secretenv2.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////
    let model_name = "sarsa";

    let mut durations_secretenv2: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut secretenv2 = Env::SecretEnv2::SecretEnv2::new();
        let now = time::Instant::now();
        let res = secretenv2.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv2.push(now.elapsed().as_millis());
        secretenv2.run_game_hashmap(res.clone());
        final_score.push(secretenv2.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv2.get(i as usize).unwrap().to_string(),
            secretenv2.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv2.iter().sum::<u128>() / durations_secretenv2.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

//////// SecretEnv3
    let model_name = "monte_carlo_exploring_starts";
    let env_name = "SecretEnv3";

    let file_path = PathBuf::from("./2024-07-23").join("SecretEnv3.txt");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "SecretEnv3\n");

    let mut durations_secretenv3: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();



    for i in 0..number_of_tests {
        let mut secretenv3 = Env::SecretEnv3::SecretEnv3::new();
        let now = time::Instant::now();
        let res = secretenv3.monte_carlo_exploring_starts(gamma, nb_iter, max_steps);
        durations_secretenv3.push(now.elapsed().as_millis());
        secretenv3.run_game_random_state_hashmap(res.clone());
        final_score.push(secretenv3.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(),
            "None".to_string(),
            "None".to_string(),
            "None".to_string(),
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv3.get(i as usize).unwrap().to_string(),
            secretenv3.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_exploring_starts (gammma = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, nb_iter, max_steps, durations_secretenv3.iter().sum::<u128>() / durations_secretenv3.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);


    ////

    let model_name = "monte_carlo_fv_on_policy";

    let mut durations_secretenv3: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut secretenv3 = Env::SecretEnv3::SecretEnv3::new();
        let now = time::Instant::now();
        let res = secretenv3.monte_carlo_fv_on_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv3.push(now.elapsed().as_millis());
        secretenv3.run_game_random_hashmap(res.clone());
        final_score.push(secretenv3.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            theta.to_string(), //theta
            "None".to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv3.get(i as usize).unwrap().to_string(),
            secretenv3.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_fv_on_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv3.iter().sum::<u128>() / durations_secretenv3.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);


    ////

    let model_name = "monte_carlo_off_policy";

    let mut durations_secretenv3: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();




    for i in 0..number_of_tests {
        let mut secretenv3 = Env::SecretEnv3::SecretEnv3::new();
        let now = time::Instant::now();
        let res = secretenv3.monte_carlo_off_policy(gamma, epsilon, nb_iter, max_steps);
        durations_secretenv3.push(now.elapsed().as_millis());
        secretenv3.run_game_hashmap(res.clone());
        final_score.push(secretenv3.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            "None".to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv3.get(i as usize).unwrap().to_string(),
            secretenv3.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} monte_carlo_off_policy (gammma = {}, epsilon = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, nb_iter, max_steps, durations_secretenv3.iter().sum::<u128>() / durations_secretenv3.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    ////

    let model_name = "Q_learning_off_policy";

    let mut durations_secretenv3: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();





    for i in 0..number_of_tests {
        let mut secretenv3 = Env::SecretEnv3::SecretEnv3::new();
        let now = time::Instant::now();
        let res = secretenv3.Q_learning_off_policy(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv3.push(now.elapsed().as_millis());
        secretenv3.run_game_hashmap(res.clone());
        final_score.push(secretenv3.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv3.get(i as usize).unwrap().to_string(),
            secretenv3.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} Q_learning_off_policy (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv3.iter().sum::<u128>() / durations_secretenv3.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);

    /////

    let model_name = "sarsa";

    let mut durations_secretenv3: Vec<u128> = Vec::new();
    let mut final_score: Vec<f32> = Vec::new();


    for i in 0..number_of_tests {
        let mut secretenv3 = Env::SecretEnv3::SecretEnv3::new();
        let now = time::Instant::now();
        let res = secretenv3.sarsa(gamma, epsilon, alpha, nb_iter, max_steps);
        durations_secretenv3.push(now.elapsed().as_millis());
        secretenv3.run_game_hashmap(res.clone());
        final_score.push(secretenv3.score());
        wtr.write_record(&[
            run_number.clone().to_string(),
            run_date.clone(),
            env_name.to_string(),
            model_name.to_string(),
            gamma.to_string(), //gamma
            "None".to_string(), //theta
            epsilon.to_string(), // epsilon
            alpha.to_string(), // alpha
            number_of_tests.to_string(),
            max_steps.to_string(),
            i.to_string(),
            durations_secretenv3.get(i as usize).unwrap().to_string(),
            secretenv3.score().to_string(),
        ]).expect("Error in writing csv");

        let mut file_Pi = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./results/Pi_values/".to_owned() + &*run_number.to_string() + "_" + &*run_date.clone() + "_" + env_name + "_" + model_name + "_" + &*i.to_string())
            .unwrap();
        writeln!(file_Pi, "{:?}", res.clone());
    }
    let output = format!("Average elapsed time over {} sarsa (gammma = {}, epsilon = {}, alpha = {}, nb_iter = {}, max_steps = {}): {}ms", number_of_tests, gamma, epsilon, alpha, nb_iter, max_steps, durations_secretenv3.iter().sum::<u128>() / durations_secretenv3.len() as u128);
    let average_score: f32 = final_score.iter().sum::<f32>() / final_score.len() as f32;
    let output_score = format!("Score moyen= {}", average_score);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .unwrap();
    writeln!(file, "{}", output);
    writeln!(file, "{}", output_score);


}
