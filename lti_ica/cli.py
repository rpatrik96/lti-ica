from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

from lti_ica.datamodule import NonstationaryLTIDatamodule
from lti_ica.runner import LTILightning


class LTILightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--notes",
            type=str,
            default=None,
            help="Notes for the run on Weights and Biases",
        )
        # todo: process notes based on args in before_instantiate_classes
        parser.add_argument(
            "--tags",
            type=str,
            nargs="*",  # 0 or more values expected => creates a list
            default=None,
            help="Tags for the run on Weights and Biases",
        )

        parser.link_arguments("data.num_comp", "model.num_comp")
        parser.link_arguments("data.num_segment", "model.num_segment")
        parser.link_arguments("data.dt", "model.dt")
        parser.link_arguments("data.max_variability", "model.max_variability")
        parser.link_arguments("data.ar_order", "model.ar_order")
        parser.link_arguments("data.zero_means", "model.zero_means")
        parser.link_arguments("data.use_B", "model.use_B")
        parser.link_arguments("data.use_C", "model.use_C")


if __name__ == "__main__":
    cli = LTILightningCLI(
        LTILightning,
        NonstationaryLTIDatamodule,
        save_config_callback=None,
        run=True,
        parser_kwargs={"parse_as_dict": False, "parser_mode": "omegaconf"},
    )
