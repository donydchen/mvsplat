from ...dataset import DatasetCfg
from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
