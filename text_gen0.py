# !pip install transformers

from transformers.utils import logging
logging.set_verbosity_error() # supresses warning messages

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = 'facebook/blenderbot-400M-distill'
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

user_message = """
What are some fun activities I can do in the winter?
"""

inputs = tokenizer(user_message, return_tensors="pt")
result = model.generate(**inputs)

tokenizer.decode(result[0])

'''Conversation.add_message(
    {"role": "user",
     "content": """
What else do you recommend?
"""
    })'''
