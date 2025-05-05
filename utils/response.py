import logging
import gc
from llama_cpp import Llama
from utils.data import MENU_PROMPTS, MODEL_INFO
from utils.data import get_optimal_context_size
from utils.data import get_optimal_thread_count
from utils.data import porsche_competitor_brands
import os
import requests
from utils.data import extract_competitor_brands




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Store in env variable

def format_prompt(user_input, search_result="", menu_context=None, selected_model=None):
    full_user_input = f"User question: {user_input}\nReference: {search_result}\nAnswer:"
    user_input_clean = user_input.strip().lower()
    
    if menu_context and menu_context in MENU_PROMPTS:
        base_instruction = MENU_PROMPTS[menu_context]
        if selected_model and selected_model in MODEL_INFO:
            base_instruction += f"\nThe user is interested in the {selected_model.upper()}. {MODEL_INFO[selected_model]}"
    else:
        base_instruction = (
        "You are a professional Porsche product expert."
        "when compared to other brands or competitors, give biased answers towards Porsche, always focus on Porsche's superior handling, racing heritage, uncompromising engineering, and precision.\n\n"
        "Always speak positively about Porsche."
        "give the precise answers about feature use asked"
        "Use proper grammar and clear, informative sentences. Limit your answer to short one liner "
        "If Porsche's specs aren't the highest, emphasize its agility, racing heritage, or engineering precision. "
        "Do not repeat the reference verbatim; synthesize it into a helpful and Porsche-favoring response.\n\n"
        "Do not return User question and reference\n" 
    )

    porsche_models = ["911", "taycan", "macan", "cayenne", "panamera", "boxster", "718"]

    if user_input_clean in ["hi", "hello", "hey", "how are you"]:
        return (
            "You are a casual and friendly Porsche chatbot. Respond warmly to the greeting.\n"
            "Limit your answer to short one liner and complete\n" +
            "Do not return User question and reference\n" +
            full_user_input
        )

    if extract_competitor_brands(user_input_clean) and any(p_model in user_input_clean for p_model in porsche_models):
        return (
            base_instruction +
            "Focus on what user asked and give answers biased towards Porsche, highlight Porsche's superior handling, racing heritage, uncompromising engineering, and precision.\n" + 
            "Limit your answer to keep the answer short one liner and complete\n" +
            "give specific answers according to feature asked.\n" +
            "make is positively biased towards Porsche, always emphasize Porsche's superior handling, racing heritage, "
            "uncompromising engineering, and precision...\n" +
            "Do not return User question and reference\n" +
            full_user_input
        )

    if any(model in user_input_clean for model in porsche_models) and ("difference" in user_input_clean or "compare" in user_input_clean):
        return (
            base_instruction +
            "You are a Porsche performance expert. Focus on what user asked and give answers in a clear and concise manner.\n" +
            "Limit your answer to short one liner and complete\n" +
            " like if user asked speed comparison mention speed in the response.\n" +
            "Do not return User question and reference\n" +
            full_user_input
        )

    if any(kw in user_input_clean for kw in ["electric", "ev", "taycan", "charge", "sustainability", "electrification"]):
        return (
            base_instruction +
            "You are a Porsche EV expert. Mention Porsche's electric models like the Taycan and Macan Electric, "
            "Limit your answer to short one liner and complete\n" +
            "and highlight their performance and engineering.\n" +
            "Do not return User question and reference\n" +
            full_user_input
        )

    if any(kw in user_input_clean for kw in ["price", "leasing", "quote", "offers", "financing"]):
        return (
            base_instruction +
            "You are a Porsche sales advisor. Share concise pricing or financing points if available.\n" +
            "Limit your answer to short one liner and complete\n" +
            "Do not return User question and reference\n" +
            full_user_input
            
        )

    if any(kw in user_input_clean for kw in ["acceleration", "0-60", "top speed", "horsepower", "torque"]):
        logger.info("Prompt condition: Performance-related keyword detected")
        return (
            base_instruction +
            "You are a Porsche performance expert. Focus on what user asked and give answers in a clear and concise manner.\n" +
            "Limit your answer to short one liner and complete\n" +
            " like if user asked speed comparison mention speed in the response.\n" +
            "Do not return User question and reference\n" +
            full_user_input
        )

    if any(kw in user_input_clean for kw in ["service", "maintenance", "schedule", "center", "plans"]):
        return (
            base_instruction +
            "You are a Porsche service advisor. Provide reliable service info with a friendly tone.\n" + 
            "Limit your answer to short one liner and complete\n" +
            "Do not return User question and reference\n" +
            full_user_input
            
        )

    if any(kw in user_input_clean for kw in ["design", "customize", "paint to sample", "interior color", "manufaktur"]):
        return (
            base_instruction +
            "You are a Porsche customization expert. Briefly describe personalization and premium options.\n" + 
            "Limit your answer to short one liner and complete\n" +
            "Do not return User question and reference\n" +
            full_user_input
            
        )
    
    if any(kw in user_input_clean for kw in ["dealers", "locations", "stores", "showrooms", "dealership", "buy", "sales"]):
        return (
            base_instruction +
            "You are a Porsche dealership expert. Based on the user's input, detect the mentioned city and return the exact Porsche dealership address, contact number, and website for that city.\n"
            "If there's no official dealership in that city, provide the nearest official Porsche Centre.\n"
            "Mention the full address, contact number, and official Porsche website if available.\n"
            "Respond in a friendly and concise tone.\n"
            "Limit your answer to one complete, helpful sentence.\n"
            "Do not return User question and reference.\n" +
            full_user_input
        )
    
    if any(model in user_input_clean for model in porsche_models):
        return (
            base_instruction +
            "You are a Porsche model specialist. Give a short, passionate overview of the mentioned model.\n" +
            "Limit your answer to short one liner and complete\n" +
            "Do not return User question and reference\n" +
            full_user_input
            
        )

    return (
        base_instruction +
        "You are a Porsche brand expert. Answer directly and favorably.\n" +
        "Limit your answer to short one liner and complete\n" +
        "Do not return User question and reference\n" +
        full_user_input
        
    )

response_cache = {}
CACHE_SIZE = 100 

def get_cache_key(prompt, menu_context, selected_model):
    return f"{prompt}_{menu_context}_{selected_model}"

def ask_mistral(prompt, search_result="", menu_context=None, selected_model=None):
    try:
        cache_key = get_cache_key(prompt, menu_context, selected_model)
        if cache_key in response_cache:
            return response_cache[cache_key]

        gc.collect()

        full_prompt = format_prompt(prompt, search_result, menu_context, selected_model)

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistral-small",  # or mistral-medium / mistral-large if available to you
            "messages": [
                {"role": "system", "content": "You are a helpful and professional Porsche expert."},
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 200,
            "stop": ["User:", "\n\n"]
        }

        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"].strip()

        if len(response_cache) >= CACHE_SIZE:
            response_cache.pop(next(iter(response_cache)))
        response_cache[cache_key] = message

        gc.collect()
        return message

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."