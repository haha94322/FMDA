import torch
print(torch.__version__)
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor

model_name = "/home/zhouhao/lmy/paper6/继续投/VideoLLaMA3-2B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
video_path = "/home/zhouhao/lmy/paper6/继续投/039.mp4"
question = "who are you."

# Video conversation
conversation = [
    {
        "role": "system",
        "content":
            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer."
    },
    {
        "role": "user",
                "content":
            "Please evaluate the following video-based question-answer pair:\n\n"
            "Question: what is lifting large pieces of rocks?\n"
            "Correct Answer: girl\n"
            "Predicted Answer: lady\n\n"
            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."
    },
]

inputs = processor(conversation=conversation, return_tensors="pt")
inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
output_ids = model.generate(**inputs, max_new_tokens=128)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(response)