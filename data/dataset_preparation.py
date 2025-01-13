from datasets import load_dataset

class DatasetPreparation:
    def __init__(self, dataset_name, system_prompt):
        self.dataset = load_dataset(dataset_name)
        self.system_prompt = system_prompt

    def map_fields(self, row):
        messages = [
            {"role": "system", "content": self.system_prompt.format(row["story"])}
        ]
        for question, answer in zip(row["questions"], row["answers"]["input_text"]):
            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
        return {"messages": messages}

    def prepare_dataset(self):
        train_messages = self.dataset["train"].map(self.map_fields, remove_columns=self.dataset["train"].column_names)
        train_messages.to_json("coqa_prepared_train.jsonl")

