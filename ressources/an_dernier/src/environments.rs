pub fn line_world() -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<Vec<Vec<Vec<f32>>>>) {
    // CREATION DE L'ENVIRONNEMENT LINE WORLD
    // ensemble des états possibles
    let mut S: Vec<i32>  = vec![0, 1, 2, 3, 4];
    // ensemble des actions possibles, O gauche, 1 droite
    let mut A: Vec<i32>= vec![0, 1];
    // ensemble des rewards possibles
    let mut R: Vec<i32> = vec![-1, 0, 1];
    // ensemble des états terminaux
    let mut T: Vec<i32> = vec![0, 4];
    // définition de p
    let mut p= vec![
        vec![
            vec![vec![0f32; R.len()]; S.len()];
            A.len()
        ];
        S.len()
    ];

    // mise à jour de p
    for s in 0..S.len() {
        for a in 0..A.len() {
            for s_p in 0..S.len() {
                for r in 0..R.len() {
                    if s_p == (s + 1) && a == 1 && R[r] == 0 && [1, 2].contains(&S[s]) {
                        p[s][a][s_p][r] = 1f32;
                    }
                    if s > 0 && s_p == (s - 1) && a == 0 && R[r] == 0 && [2, 3].contains(&S[s]) {
                        p[s][a][s_p][r] = 1f32;
                    }
                }
            }
        }
    }

    p[3][1][4][2] = 1f32;
    p[1][0][0][0] = 1f32;

    return (S, A, R, T, p);
}

pub fn grid_world() -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<Vec<Vec<Vec<f32>>>>){
    // ensemble des états possibles
    let mut S: Vec<i32>  = (0..49).collect();
    // ensemble des actions possibles, O gauche, 1 droite
    let mut A: Vec<i32>= vec![0, 1, 2, 3];  // left right up down
    // ensemble des rewards possibles
    let mut R: Vec<i32> = vec![-3, -1, 0, 1];
    // ensemble des états terminaux
    let mut T: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 20, 21, 27, 28, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48];
    // définition de p
    let mut p= vec![
        vec![
            vec![vec![0f32; R.len()]; S.len()];
            A.len()
        ];
        S.len()
    ];

    // mise à jour de p
    for s in 0..S.len() {
        for a in 0..A.len() {
            for s_p in 0..S.len() {
                for r in 0..R.len() {
                    // actions terminales :
                    // si on monte depuis la premiere ligne :
                    if 7 < s && s < 12 && a == 2 && s_p == s - 7 && R[r] == -1 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on descend depuis la derniere ligne :
                    if 35 < s && s < 40 && a == 3 && s_p == s + 7 && R[r] == -1 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à gauche depuis la première colonne :
                    if s % 7 == 1 && 7 < s && s < 37 && a == 0 && s_p == s - 1 && R[r] == -1 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à droite depuis la dernière colonne :
                    if s % 7 == 5 && 7 < s && s < 37 && a == 1 && s_p == s + 1 && R[r] == -1 {
                        p[s][a][s_p][r] = 1f32;
                    }

                    // actions banales :
                    // si on est sur la premiere ligne:
                    // si on descend
                    if 7 < s && s < 12 && a == 3 && s_p == s + 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à gauche
                    if 8 < s && s < 12 && a == 0 && s_p == s - 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à droite
                    if 7 < s && s < 11 && a == 1 && s_p == s + 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }

                    // si on est sur la deuxième ligne:
                    // si on monte
                    if 14 < s && s < 19 && a == 2 && s_p == s - 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on descend
                    if 14 < s && s < 20 && a == 3 && s_p == s + 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à droite
                    if 14 < s && s < 19 && a == 1 && s_p == s + 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à gauche
                    if 15 < s && s < 20 && a == 0 && s_p == s - 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }

                    // si on est sur la troisième ligne:
                    // si on monte
                    if 21 < s && s < 27 && a == 2 && s_p == s - 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on descend
                    if 21 < s && s < 27 && a == 3 && s_p == s + 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à gauche
                    if 22 < s && s < 27 && a == 0 && s_p == s - 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à droite
                    if 21 < s && s < 26 && a == 1 && s_p == s + 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }

                    // si on est sur la quatrième ligne:
                    // si on monte
                    if 28 < s && s < 34 && a == 2 && s_p == s - 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on descend
                    if 28 < s && s < 33 && a == 3 && s_p == s + 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à gauche
                    if 29 < s && s < 34 && a == 0 && s_p == s - 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à droite
                    if 28 < s && s < 33 && a == 1 && s_p == s + 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }

                    // si on est sur la dernière ligne
                    // si on monte
                    if 35 < s && s < 40 && a == 2 && s_p == s - 7 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à gauche
                    if 36 < s && s < 40 && a == 0 && s_p == s - 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                    // si on va à droite
                    if 35 < s && s < 39 && a == 0 && s_p == s - 1 && R[r] == 0 {
                        p[s][a][s_p][r] = 1f32;
                    }
                }
            }
        }
    }

    p[11][1][12][0] = 1f32;
    p[19][2][12][0] = 1f32;
    p[39][1][40][3] = 1f32;
    p[33][3][40][3] = 1f32;

    return (S, A, R, T, p);
}

pub fn shifumi() -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<Vec<Vec<Vec<f32>>>>){
    // ensemble des états possibles
    let mut S: Vec<i32>  = (0..37).collect();
    // ensemble des actions possibles, O gauche, 1 droite
    let mut A: Vec<i32>= vec![0, 1, 2]; // P F C
    // ensemble des rewards possibles
    let mut R: Vec<i32> = vec![-1, 0, 1];
    // ensemble des états terminaux
    let mut T: Vec<i32> = (10..37).collect();
    // définition de p
    let mut p= vec![
        vec![
            vec![vec![0f32; R.len()]; S.len()];
            A.len()
        ];
        S.len()
    ];

    p[0][0][1][1] = 1f32/3f32; //# PP  0
    p[0][0][2][0] = 1f32/3f32; //# PF -1
    p[0][0][3][2] = 1f32/3f32; //# PC  1

    p[0][1][4][2] = 1f32/3f32; //# FP  1
    p[0][1][5][1] = 1f32/3f32; // # FF 0
    p[0][1][6][0] = 1f32/3f32; // # FC -1

    p[0][2][7][0] = 1f32/3f32; // # CP -1
    p[0][2][8][2] = 1f32/3f32; // # CF  1
    p[0][2][9][1] = 1f32/3f32; // # CC  0
    // ###############
    p[1][0][10][1] = 1f32; // # PP PP  0
    p[1][1][11][2] = 1f32; // # PP FP  1
    p[1][2][12][0] = 1f32; // # PP CP  -1

    p[2][0][13][1] = 1f32; // # PF PP  0
    p[2][1][14][2] = 1f32; // # PF FP  1
    p[2][2][15][0] = 1f32; // # PF CP  -1

    p[3][0][16][1] = 1f32; // # PC PP  0
    p[3][1][17][2] = 1f32; // # PC FP  1
    p[3][2][18][0] = 1f32; // # PC CP -1
    //###############
    p[4][0][19][0] = 1f32; // # FP PF  -1
    p[4][1][20][1] = 1f32; // FP FF     0
    p[4][2][21][2] = 1f32; //# FP CF    1

    p[5][0][22][0] = 1f32; // FF PF   -1
    p[5][1][23][1] = 1f32; // FF FF    0
    p[5][2][24][2] = 1f32; // FF CF    1

    p[6][0][25][0] = 1f32; // FC PF   -1
    p[6][1][26][1] = 1f32; // FC FF    0
    p[6][2][27][2] = 1f32; // FC CF    1
    //###############
    p[7][0][28][2] = 1f32; // CP PC    1
    p[7][1][29][0] = 1f32; // CP FC   -1
    p[7][2][30][1] = 1f32; // CP CC    0

    p[8][0][31][2] = 1f32; // CF PC    1
    p[8][1][32][0] = 1f32; // CF FC   -1
    p[8][2][33][1] = 1f32; // CF CC    0

    p[9][0][34][2] = 1f32; // CC PC    1
    p[9][1][35][0] = 1f32; // CC FC   -1
    p[9][2][36][1] = 1f32; // CC CC    0

    return (S, A, R, T, p);
}

pub fn montyhall_standard() -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<Vec<Vec<Vec<f32>>>>) {
    // ensemble des états possibles
    let mut S: Vec<i32> = vec![0, 1, 2, 3, 4, 5]; // état initial / on a choisi la porte A / B / C / même porte / on change
    // ensemble des actions possibles, O gauche, 1 droite
    let mut A: Vec<i32> = vec![0, 1, 2, 3, 4]; // porte A / B / C / on reste / on change
    // ensemble des rewards possibles
    let mut R: Vec<i32> = vec![0, 1];
    // ensemble des états terminaux
    let mut T: Vec<i32> = vec![4, 5];
    // définition de p
    let mut p = vec![
        vec![
            vec![vec![0f32; R.len()]; S.len()];
            A.len()
        ];
        S.len()
    ];

    // état initiaux
    p[0][0][1][0] = 1f32;
    p[0][1][2][0] = 1f32;
    p[0][2][3][0] = 1f32;

    // on reste
    // on perd
    p[1][3][4][0] = 2f32/3f32;
    p[2][3][4][0] = 2f32/3f32;
    p[3][3][4][0] = 2f32/3f32;

    // on gagne
    p[1][3][4][1] = 1f32/3f32;
    p[2][3][4][1] = 1f32/3f32;
    p[3][3][4][1] = 1f32/3f32;

    // on change
    // on gagne
    p[1][4][5][1] = 2f32/3f32;
    p[2][4][5][1] = 2f32/3f32;
    p[3][4][5][1] = 2f32/3f32;

    // on perd
    p[1][4][5][0] = 1f32/3f32;
    p[2][4][5][0] = 1f32/3f32;
    p[3][4][5][0] = 1f32/3f32;

    return (S, A, R, T, p);
}

// TODO
// tester les secrets envs aussi