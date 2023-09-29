from lti_ica.cli import LTILightningCLI
from lti_ica.runner import LTILightning
from lti_ica.datamodule import NonstationaryLTIDatamodule
from os.path import abspath, dirname, join


def test_cli_fast_dev_run():
    config_path = join(dirname(dirname(abspath(__file__))), "configs", "config.yaml")

    args = [
        "fit",
        "--config",
        config_path,
        "--trainer.fast_dev_run",
        "true",
    ]
    cli = LTILightningCLI(
        LTILightning,
        NonstationaryLTIDatamodule,
        save_config_callback=None,
        run=True,
        args=args,
        parser_kwargs={"parse_as_dict": False},
    )
