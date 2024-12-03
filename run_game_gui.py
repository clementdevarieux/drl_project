from environnements.Farkle_GUI_v4 import Farkle_GUI_v4

def run_GUI(isWithModel: bool, model = None):
    env = Farkle_GUI_v4()
    if isWithModel:
        env.run_game_GUI_vs_model(model)
    else:
        env.run_game_GUI()