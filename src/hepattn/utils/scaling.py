import torch


class RegressionTargetScaler:
    def scale(self, target):
        raise NotImplementedError

    def inverse(self, target):
        raise NotImplementedError


class TrackScaler(RegressionTargetScaler):
    def __init__(self):
        self.REG_SCALING = {
            "sintheta": lambda x: x / 1,
            "costheta": lambda x: x / 1,
            "sinphi": lambda x: x / 1,
            "cosphi": lambda x: x / 1,
            "phi": lambda x: x / 1,
            "px": lambda x: x / 1,
            "py": lambda x: x / 1,
            "pz": lambda x: x / 2,
            "p": lambda x: x / 2,
            "vz": lambda x: x / 50,
            "qOp": lambda x: x * 20,
            "eta": lambda x: x / 2,
        }

        self.REG_INVERSE = {
            "sintheta": lambda x: x * 1,
            "costheta": lambda x: x * 1,
            "sinphi": lambda x: x * 1,
            "cosphi": lambda x: x * 1,
            "phi": lambda x: x * 1,
            "px": lambda x: x * 1,
            "py": lambda x: x * 1,
            "pz": lambda x: x * 2,
            "p": lambda x: x * 2,
            "vz": lambda x: x * 50,
            "qOp": lambda x: x / 20,
            "eta": lambda x: x * 2,
        }

    def scale(self, target):
        return self.REG_SCALING[target]

    def inverse(self, target):
        return self.REG_INVERSE[target]
