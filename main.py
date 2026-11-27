import os
from comm.snake_comm import SnakeComm
# from player.keyboard_player import KeyboardPlayer
from player.dqn.dqn_player import DqnPlayer
from player.dqn.ex_dqn_player import ExDqnPlayer
from player.dqn.hc_dqn_player import HcDqnPlayer
from player.dqn.alpha_player import AlphaPlayer

from player.bfs_player import BFSPlayer
from player.dqn.Im_player import ImPlayer
from tools import Recorder
from player.ppo.mlp_ppo_player import MLPPPOPlayer
from player.ppo.cnn_ppo_player import CNNPPOPlayer
from player.neuro.fly_player_fixed_brain import FlyFixedBrainPlayer
from player.neuro.fly_player_plastic_brain import FlyPlasticBrainPlayer
from player.neuro.fly_player_train_encoder import FlyTrainEncoderPlayer
from player.neuro.fly_player_snake import FlySnakePlayer


class MdlInfo:
    def __init__(self, mdl, model_file):
        self.mdl = mdl
        self.model_file = model_file

def Mdl():
    mdl_tbl={
        "dqn": MdlInfo(DqnPlayer, SnakeComm.dqn_model_file),
        "exdqn": MdlInfo(ExDqnPlayer, SnakeComm.exdqn_model_file),
        "hcdqn": MdlInfo(HcDqnPlayer, SnakeComm.hcdqn_model_file),
        "alpha": MdlInfo(AlphaPlayer, SnakeComm.alpha_model_file),
        "bfs": MdlInfo(BFSPlayer, None),
        "im": MdlInfo(ImPlayer, SnakeComm.im_model_file),
        "mlpppo": MdlInfo(MLPPPOPlayer, SnakeComm.mlp_ppo_model_file),
        "cnnppo": MdlInfo(CNNPPOPlayer, SnakeComm.cnn_ppo_model_file),
        "flyfixed": MdlInfo(FlyFixedBrainPlayer, SnakeComm.fly_fixed_model_file),
        "flyplastic": MdlInfo(FlyPlasticBrainPlayer, SnakeComm.fly_plastic_model_file),
        "flyenc": MdlInfo(FlyTrainEncoderPlayer, SnakeComm.fly_enc_model_file),
        "flysnake": MdlInfo(FlySnakePlayer, SnakeComm.fly_snake_model_file),
        "rec": MdlInfo(Recorder, None)
    }
    return mdl_tbl

def ShowHelp():
    print("Usage: python main.py <model name> <action> :"
        "\n   <model name>:"
        "\n       dqn             1 hidden layer"
        "\n       exdqn           extended dqn"
        "\n       hcdqn          hamilton cycle dqn"
        "\n       alpha           alphaplayer"
        "\n       bfs             bfs"
        "\n       im              imitation learning (BFS expert)"
        "\n       mlpppo          MLP PPO (SB3/sb3_contrib)"
        "\n       cnnppo          CNN PPO (SB3/sb3_contrib)"
        "\n       flyfixed        fly brain fixed encoder+brain, train decoder"
        "\n       flyplastic      fly brain Hebbian plastic + train decoder"
        "\n       flyenc          fly brain fixed, train encoder+decoder"
        "\n       flysnake        fly-brain-snake aligned computeStim+decide, BC+PPO"
        "\n       rec             for showing recorder playing the game"
        "\n   <action>:"
        "\n       re|retrain      for retraining new model"
        "\n       tr|train        for continuing train the existing model"
        "\n       p|play          for showing nn playing the game"
        "\n       c|collection    for collecting expert data, only for recorder"
        "\n       rp|recplay      for showing recorder playing the game, only for recorder")
        

def exec_action(mdl_info, action, extra_args=None):
    extra_args = extra_args or []
    print("action: %s, model file: %s， extra args: %s." % (action, mdl_info.model_file, extra_args))
    if action in ["tr", "train", "re", "retrain"] and mdl_info.model_file:
        model_dir = os.path.dirname(mdl_info.model_file)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
    if action in ["tr", "train"]:
        mdl_info.mdl.train(model_file=mdl_info.model_file, new_model=False)# 继续训练已有模型
    elif action in ["re", "retrain"]:
        mdl_info.mdl.train(model_file=mdl_info.model_file, new_model=True) # 训练新模型
    elif action in ["p", "play"]:
        mdl_file = extra_args[0] if len(extra_args) > 0 else mdl_info.model_file
        print("play model file: %s." % mdl_file)
        mdl_info.mdl.play(mdl_file)
    elif action in ["c", "collection"]:
        mdl_file = extra_args[0] if len(extra_args) > 0 else mdl_info.model_file
        print("collection model file: %s." % mdl_file)
        mdl_info.mdl.collection(mdl_file)
    elif action in ["rp", "recplay"]:
        mdl_info.mdl.recplay(file_list=extra_args)
    else:
        ShowHelp()
        print("Invalid action: %s or model file: %s." % (action, mdl_info.model_file))

def main(mdl_tbl):
    import sys
    if len(sys.argv) < 3:
        ShowHelp()
        return

    if not sys.argv[1].lower() in mdl_tbl.keys():
        ShowHelp()
        print("Invalid model name: %s." % sys.argv[1])
        return

    exec_action(mdl_tbl[sys.argv[1].lower()], sys.argv[2].lower(), sys.argv[3:])

if __name__ == "__main__":
    main(Mdl())
