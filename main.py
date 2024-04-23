import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig

# Настройки модели
MODEL_NAME = "IlyaGusev/saiga_7b_lora"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

class Conversation:
    def __init__(self, message_template=DEFAULT_MESSAGE_TEMPLATE, system_prompt=DEFAULT_SYSTEM_PROMPT, start_token_id=1, bot_token_id=9225):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output_ids = model.generate(**inputs)[0]
    output_ids = output_ids[len(inputs["input_ids"][0]):]  # remove input ids from the output
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
config = PeftConfig.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, MODEL_NAME, torch_dtype=torch.float16)
model.eval()

# Streamlit интерфейс
st.title('Диалог с моделью Saiga 7B LoRA')
user_input = st.text_input("Введите ваш вопрос или комментарий:")
if st.button('Отправить'):
    if user_input:
        conversation = Conversation()
        conversation.add_user_message(user_input)
        prompt = conversation.get_prompt(tokenizer)
        response = generate(model, tokenizer, prompt)
        st.write(response)
    else:
        st.write("Пожалуйста, введите текст для отправки.")
