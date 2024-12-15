from transformers import pipeline

text_qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2") # or other suitable model

def process_text(problem_text, context=None):
    if context:
        result = text_qa_pipeline(question=problem_text, context=context)
        return result
    else:
        # Basic problem parsing (can be improved)
        return problem_text.split() # Example: split into words

# ... other text processing functions (extract variables, constraints)