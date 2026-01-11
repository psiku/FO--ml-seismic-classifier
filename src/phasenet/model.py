import seisbench.models as sbm
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pretrained_model_list():
    return sbm.PhaseNet.list_pretrained()


def get_phasenet_model(pretrained_version: str = None):
    device = get_device()

    if not pretrained_version:
        return sbm.PhaseNet().to(device)
    else:
        return sbm.PhaseNet.from_pretrained(pretrained_version).to(device)


def load_model_from_path(model_path: str, pretrained_version: str = None):
    device = get_device()
    model = get_phasenet_model(pretrained_version)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    return model
