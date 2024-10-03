import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def chatbot_response(user_input, chat_history_ids):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    

    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    
    chat_history_ids = model.generate(bot_input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    response = response.replace(":D", "")  
    response = response.strip()  
    

    if not response:
        response = "I'm here to help! What would you like to talk about?"

    return response, chat_history_ids

def chatbot():
    print("Chatbot: Hello! I'm an AI chatbot. Type 'bye' to exit the conversation.")
    
    chat_history_ids = None

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
