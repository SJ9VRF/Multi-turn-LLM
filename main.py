from data.dataset_preparation import DatasetPreparation
from models.fine_tuning_manager import FineTuningManager
from evaluation.model_evaluator import ModelEvaluator
from utils.config import TOGETHER_API_KEY, WANDB_API_KEY, SYSTEM_PROMPT

if __name__ == "__main__":
    dataset_prep = DatasetPreparation("stanfordnlp/coqa", SYSTEM_PROMPT)
    dataset_prep.prepare_dataset()
    ft_manager = FineTuningManager(TOGETHER_API_KEY)
    training_file_response = ft_manager.upload_dataset("coqa_prepared_train.jsonl")
    ft_job_response = ft_manager.create_fine_tuning_job(training_file_response.id, 'meta-llama/Meta-Llama-3.1-8B-Instruct-Reference', 3, 1, WANDB_API_KEY, True, 0, 1e-5, 'my-demo-finetune')
    print(ft_job_response.id)

    evaluator = ModelEvaluator(ft_manager.client, SYSTEM_PROMPT, dataset_prep.dataset)
    answers = evaluator.get_model_answers("meta-llama/Meta-Llama-3.1-8B-Instruct-Reference")
    em_score, f1_score = evaluator.get_metrics(answers)
    print(f"EM: {em_score}, F1: {f1_score}")

