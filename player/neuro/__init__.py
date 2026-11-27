"""果蝇脑 Snake players：fixed / plastic / train-encoder / snake-aligned。"""

__all__ = [
    "FlyFixedBrainPlayer",
    "FlyPlasticBrainPlayer",
    "FlyTrainEncoderPlayer",
    "FlySnakePlayer",
]


def __getattr__(name):
    if name == "FlyFixedBrainPlayer":
        from player.neuro.fly_player_fixed_brain import FlyFixedBrainPlayer
        return FlyFixedBrainPlayer
    if name == "FlyPlasticBrainPlayer":
        from player.neuro.fly_player_plastic_brain import FlyPlasticBrainPlayer
        return FlyPlasticBrainPlayer
    if name == "FlyTrainEncoderPlayer":
        from player.neuro.fly_player_train_encoder import FlyTrainEncoderPlayer
        return FlyTrainEncoderPlayer
    if name == "FlySnakePlayer":
        from player.neuro.fly_player_snake import FlySnakePlayer
        return FlySnakePlayer
    raise AttributeError(name)
