import sys
import os
import argparse

import torch

from plonk.models.networks.mlp import GeoAdaLNMLP
from huggingface_hub import PyTorchModelHubMixin

models_overrides = {
    "YFCC100M_geoadalnmlp_r3_small_sigmoid_flow_riemann_10M_10M": "YFCC100M_geoadalnmlp_r3_small_sigmoid_flow_riemann",
    "iNaturalist_geoadalnmlp_r3_small_sigmoid_flow_riemann_-7_3": "iNaturalist_geoadalnmlp_r3_small_sigmoid_flow_riemann",
    "osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann_-7_3": "osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann",
    "plonk/checkpoints/multidisaster20_osm":  "multidisaster20_osm.yaml" ,
    "plonk/checkpoints/multidisaster30_osm" : "multidisaster30_osm.yaml" ,
    "plonk/checkpoints/multidisaster20_yfcc" :"multidisaster20_yfcc.yaml" ,
    "plonk/checkpoints/multidisaster30_yfcc" :"multidisaster30_yfcc.yaml",
    "plonk/checkpoints/iandisaster20_osm":  "iandisaster20_osm.yaml" ,
    "plonk/checkpoints/iandisaster30_osm" : "iandisaster30_osm.yaml" ,
    "plonk/checkpoints/iandisaster20_yfcc" :"iandisaster20_yfcc.yaml" ,
    "plonk/checkpoints/iandisaster30_yfcc" :"iandisaster30_yfcc.yaml",


}

class Plonk(
    GeoAdaLNMLP,
    PyTorchModelHubMixin,
    repo_url="https://github.com/nicolas-dufour/plonk",
    tags=["plonk", "geolocalization", "diffusion"],
    license="mit",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def save_model_locally(checkpoint_dir, output_dir):
    import hydra
    from omegaconf import OmegaConf

    # initialize hydra config
    hydra.initialize(version_base=None, config_path=f"configs")
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            f"exp={models_overrides[checkpoint_dir]}",
        ],
    )

    # get network config
    network_config = cfg.model.network
    serialized_network_config = OmegaConf.to_container(network_config, resolve=True)
    print(f"✅ Network config:\n{serialized_network_config}")

    # remove target key if present
    serialized_network_config.pop("_target_", None)

    # instantiate model
    model = Plonk(**serialized_network_config)

    # load checkpoint
    ckpt_path = f"{checkpoint_dir}/last.ckpt"
    print(f"📄 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)

    # extract only ema_network weights and strip prefix
    ckpt_state_dict = ckpt["state_dict"]
    ema_state_dict = {
        k.replace("ema_network.", ""): v
        for k, v in ckpt_state_dict.items()
        if k.startswith("ema_network")
    }

    # load weights
    missing, unexpected = model.load_state_dict(ema_state_dict, strict=False)
    print(f"⚠️ Missing keys: {missing}")
    print(f"⚠️ Unexpected keys: {unexpected}")

    # save locally
    output_dir = os.path.abspath(output_dir)
    print(f"💾 Saving model repository locally at: {output_dir}")
    model.save_pretrained(output_dir)

    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Name of checkpoint dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save local repository")
    args = parser.parse_args()

    save_model_locally(args.checkpoint_dir, args.output_dir)

