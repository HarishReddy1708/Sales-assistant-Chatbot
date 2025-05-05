import random
from utils.data import SUGGESTED_QUESTIONS
from utils.data import extract_models


def get_suggested_questions(menu_context, user_input=None):

    """ if selected_model:
        model = selected_model.lower() """
    if user_input:
        models_mentioned = extract_models(user_input)
        model = models_mentioned[0] if models_mentioned else None
    else:
        model = None

    if menu_context == 'models' and model:
        questions = SUGGESTED_QUESTIONS['models'].get(model, [])
        return random.sample(questions, min(2, len(questions))) if questions else []
    
    elif menu_context in SUGGESTED_QUESTIONS:
        questions = SUGGESTED_QUESTIONS[menu_context]
        if isinstance(questions, list):
            return random.sample(questions, min(3, len(questions)))
    
    return []
