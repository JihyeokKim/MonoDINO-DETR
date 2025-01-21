from lib.models.monodinodetr import build_monodinodetr


def build_model(cfg):
    return build_monodinodetr(cfg)
