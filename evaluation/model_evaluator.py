import transformers.data.metrics.squad_metrics as squad_metrics
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool

class ModelEvaluator:
    def __init__(self, client, system_prompt, coqa_dataset):
        self.client = client
        self.system_prompt = system_prompt
        self.coqa_dataset = coqa_dataset

    def get_model_answers(self, model_name):
        model_answers = []

        def get_answers(data):
            answers = []
            messages = [{"role": "system", "content": self.system_prompt.format(data["story"])}]
            for question, true_answer in zip(data["questions"], data["answers"]["input_text"]):
                messages.append({"role": "user", "content": question})
                chat_completion = self.client.chat.completions.create(messages=messages, model=model_name, max_tokens=64)
                answer = chat_completion.choices[0].message.content
                answers.append(answer)
            return answers

        with ThreadPool(8) as pool:
            for answers in tqdm(pool.imap(get_answers, self.coqa_dataset["validation"]), total=len(self.coqa_dataset["validation"])):
                model_answers.append(answers)

        return model_answers

    def get_metrics(self, pred_answers):
        em_metrics, f1_metrics = [], []
        for pred, data in tqdm(zip(pred_answers, self.coqa_dataset["validation"]), total=len(pred_answers)):
            for pred_answer, true_answer in zip(pred, data["answers"]["input_text"]):
                em_metrics.append(squad_metrics.compute_exact(true_answer, pred_answer))
                f1_metrics.append(squad_metrics.compute_f1(true_answer, pred_answer))

        return sum(em_metrics) / len(em_metrics), sum(f1_metrics) / len(f1_metrics)

