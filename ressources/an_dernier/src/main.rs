mod Env;
mod secret_env;
mod run_all_and_save;
use std::fs::File;
use std::io::{self, Read};

use std::collections::HashMap;
use crate::Env::GridWorld::GridWorld;


fn get_vector(chemin: &str) -> Vec<i32> {
    // Ouvrir le fichier
    let mut file = File::open(chemin).expect("Erreur lors de l'ouverture du fichier");

    // Lire le contenu du fichier dans une chaîne de caractères
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Erreur lors de la lecture du fichier");

    // Nettoyer et parser le contenu
    let contents = contents.trim(); // Supprime les espaces blancs autour
    let contents = &contents[1..contents.len()-1]; // Supprime les crochets '[' et ']'

    // Convertir la chaîne en vecteur de f64
    let pi_vector: Vec<i32> = contents
        .split(',')
        .map(|s| s.trim().parse().expect("Erreur lors du parsing d'un nombre"))
        .collect();

    pi_vector
}

fn get_hashmap(chemin: &str) -> HashMap<i32, i32> {
    // Ouvrir le fichier
    let mut file = File::open(chemin).expect("Erreur lors de l'ouverture du fichier");

    // Lire le contenu du fichier dans une chaîne de caractères
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Erreur lors de la lecture du fichier");

    // Nettoyer et parser le contenu
    let contents = contents.trim(); // Supprime les espaces blancs autour
    let contents = &contents[1..contents.len()-1]; // Supprime les crochets '{' et '}'

    // Convertir la chaîne en HashMap<String, f64>
    let mut hashmap = HashMap::new();

    for pair in contents.split(',') {
        let pair = pair.trim();
        let parts: Vec<&str> = pair.split(':').collect();
        if parts.len() == 2 {
            let key: i32 = parts[0].trim().parse().expect("Erreur lors du parsing d'une valeur en f64");
            let value: i32 = parts[1].trim().parse().expect("Erreur lors du parsing d'une valeur en f64");
            hashmap.insert(key, value);
        }
    }

    hashmap
}

fn get_nested_hashmap(chemin: &str) -> HashMap<i32, HashMap<i32, f32>> {
    use std::fs::File;
    use std::io::{self, Read};

    let mut file = File::open(chemin).expect("Erreur lors de l'ouverture du fichier");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Erreur lors de la lecture du fichier");

    // Nettoyer et afficher le contenu brut pour le débogage

    // Nettoyer la chaîne pour enlever les accolades extérieures
    let cleaned_contents = contents
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .replace("}, ", "}||")
        .replace("{ ", "{")
        .replace(" }", "}");


    let mut outer_map = HashMap::new();

    // Séparer les paires externes
    for outer_pair in cleaned_contents.split("||") {
        let outer_pair = outer_pair.trim();
        let parts: Vec<&str> = outer_pair.splitn(2, ':').collect();


        if parts.len() == 2 {
            let outer_key: i32 = parts[0].trim().parse().expect("Erreur lors du parsing d'une clé externe en i32");
            let inner_map_str = parts[1].trim();

            let inner_map_str = inner_map_str
                .trim_start_matches('{')
                .trim_end_matches('}')
                .replace(", ", "||");

            let mut inner_map = HashMap::new();

            // Séparer les paires internes
            for inner_pair in inner_map_str.split("||") {
                let inner_pair = inner_pair.trim();
                let inner_parts: Vec<&str> = inner_pair.splitn(2, ':').collect();


                if inner_parts.len() == 2 {
                    let inner_key: i32 = inner_parts[0].trim().parse().expect("Erreur lors du parsing d'une clé interne en i32");
                    let inner_value: f32 = inner_parts[1].trim().parse().expect("Erreur lors du parsing d'une valeur en f32");
                    inner_map.insert(inner_key, inner_value);
                }
            }

            outer_map.insert(outer_key, inner_map);

        }
    }

    outer_map
}

fn main() {
    //permet de tout lancer d'un coup et save les resultats intermédiaires dans un fichier results/results.csv, dans le dossier "2024-07-23" et le dossier results/Pi_values
    //run_all_and_save::run_all_and_save();

    //// Pour Init et lancer un algo sur un environnement spécifique
    // let mut secretenv1 = Env::SecretEnv1::SecretEnv1::new();
    // let res = secretenv1.monte_carlo_exploring_starts(0.9999, 1000, 50000);
    // secretenv1.run_game_random_state_hashmap(res);


    let vector = get_vector("./results/Pi_values/6d9acd28-70a5-4bd0-a130-3fffbcac9173_2024-07-24_20-09-23_GridWorld_policy_iteration_2");
    let hashmap = get_hashmap("./results/Pi_values/6d9acd28-70a5-4bd0-a130-3fffbcac9173_2024-07-24_20-09-23_GridWorld_monte_carlo_off_policy_3");
    let random_hashmap = get_nested_hashmap("./results/Pi_values/6d9acd28-70a5-4bd0-a130-3fffbcac9173_2024-07-24_20-09-23_LineWorld_monte_carlo_fv_on_policy_3");

    let mut grid = Env::LineWorld::LineWorld::init();
    grid.run_game_random_hashmap(random_hashmap);

}