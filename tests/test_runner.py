from pytorch_lightning.trainer import Trainer


def test_runner_fast_dev_mode(datamodule, runner):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(runner, datamodule=datamodule)
