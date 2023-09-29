from lti_ica.cli import LTILightningCLI
from lti_ica.runner import LTILightning
from lti_ica.datamodule import NonstationaryLTIDatamodule


def test_cli_fast_dev_run():
    args = [
        "fit",
        "--config",
        "../configs/config.yaml",
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
