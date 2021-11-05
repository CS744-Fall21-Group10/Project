import marius as m
from marius.tools import preprocess

def fb15k_example():

    preprocess.fb15k(output_dir="output_dir/")
    
    config_path = "mariusConfig.ini"
    config = m.parseConfig(config_path)

    train_set, eval_set = m.initializeDatasets(config)

    model = m.initializeModel(config.model.encoder_model, config.model.decoder_model)

    trainer = m.SynchronousTrainer(train_set, model)
    evaluator = m.SynchronousEvaluator(eval_set, model)

    trainer.train(1)
    evaluator.evaluate(True)


if __name__ == "__main__":
    fb15k_example()
