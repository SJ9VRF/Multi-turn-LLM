import os

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
SYSTEM_PROMPT = "Read the story and extract answers for the questions.\nStory: {}"

