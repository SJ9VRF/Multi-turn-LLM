from together import Together

class FineTuningManager:
    def __init__(self, api_key):
        self.client = Together(api_key=api_key)

    def upload_dataset(self, file_path):
        return self.client.files.upload(file_path, check=True)

    def create_fine_tuning_job(self, training_file_id, model, epochs, checkpoints, wandb_api_key, lora, warmup_ratio, learning_rate, suffix):
        return self.client.fine_tuning.create(
            training_file=training_file_id,
            model=model,
            train_on_inputs="auto",
            n_epochs=epochs,
            n_checkpoints=checkpoints,
            wandb_api_key=wandb_api_key,
            lora=lora,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            suffix=suffix
        )

