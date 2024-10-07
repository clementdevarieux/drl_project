pub mod tic_tac_toe {
    use crate::contracts::DeepDiscreteActionsEnv;
    use rand::prelude::IteratorRandom;
    use std::fmt::Display;

    pub const NUM_STATE_FEATURES: usize = 27;
    pub const NUM_ACTIONS: usize = 9;

    #[derive(Clone)]
    pub struct TicTacToeVersusRandom {
        pub board: [f32; NUM_ACTIONS],
        pub player: u8,
        pub score: f32,
        pub is_game_over: bool,
    }

    impl Default for TicTacToeVersusRandom {
        fn default() -> Self {
            Self {
                board: [0f32; NUM_ACTIONS],
                player: 0,
                score: 0.0,
                is_game_over: false,
            }
        }
    }

    impl DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> for TicTacToeVersusRandom {
        fn state_description(&self) -> [f32; NUM_STATE_FEATURES] {
            std::array::from_fn(|idx| {
                let cell = idx / 3;
                let feature = idx % 3;
                if self.board[cell] == feature as f32 {
                    1.0
                } else {
                    0.0
                }
            })
        }

        fn available_actions_ids(&self) -> impl Iterator<Item=usize> {
            self.board.iter().enumerate().filter_map(|(idx, &val)| {
                if val == 0.0 {
                    Some(idx)
                } else {
                    None
                }
            })
        }

        fn action_mask(&self) -> [f32; 9] {
            std::array::from_fn(|idx| {
                if self.board[idx] == 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
        }

        fn step(&mut self, action: usize) {
            if self.is_game_over {
                panic!("Trying to play while Game is Over");
            }

            if action >= NUM_ACTIONS {
                panic!("Invalid action : {}", action);
            }

            if self.board[action] != 0.0 {
                panic!("Cell {} already occupied : {}", action, self.board[action]);
            }

            self.board[action] = self.player as f32 + 1.0;

            let row = action / 3;
            let col = action % 3;

            // check line, column and diagonals
            if self.board[row * 3] == self.board[row * 3 + 1] && self.board[row * 3 + 1] == self.board[row * 3 + 2] ||
                self.board[col] == self.board[col + 3] && self.board[col + 3] == self.board[col + 6] ||
                self.board[0] == self.board[4] && self.board[4] == self.board[8] && self.board[0] == self.board[action] ||
                self.board[2] == self.board[4] && self.board[4] == self.board[6] && self.board[2] == self.board[action]
            {
                self.is_game_over = true;
                self.score = if self.player == 0 { 1.0 } else { -1.0 };
                return;
            }

            // check if board is full
            if self.board.iter().all(|&val| val != 0.0) {
                self.is_game_over = true;
                self.score = 0.0;
                return;
            }

            if self.player == 0 {
                self.player = 1;

                // random move
                let mut rng = rand::thread_rng();
                let random_action = self.available_actions_ids().choose(&mut rng).unwrap();
                self.step(random_action);
            } else {
                self.player = 0;
            }
        }

        fn is_game_over(&self) -> bool {
            self.is_game_over
        }

        fn score(&self) -> f32 {
            self.score
        }

        fn reset(&mut self) {
            self.board = [0f32; NUM_ACTIONS];
            self.player = 0;
            self.score = 0.0;
            self.is_game_over = false;
        }
    }

    impl Display for TicTacToeVersusRandom {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for i in 0..3 {
                for j in 0..3 {
                    let idx = i * 3 + j;
                    f.write_str(match self.board[idx] as u8 {
                        0 => "_",
                        1 => "X",
                        2 => "O",
                        _ => panic!("Invalid value in board : {}", self.board[idx]),
                    })?;
                }
                f.write_str("\n")?;
            }
            writeln!(f, "Score: {}", self.score)?;
            writeln!(f, "Player {} to play", self.player)?;
            writeln!(f, "Game Over: {}", self.is_game_over)?;
            Ok(())
        }
    }
}