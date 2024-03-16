from langchain.prompts import PromptTemplate

#------------------------------------------------------------------------------#

# SQL
with_instruction_sql_template = "I want you to act as a SQL terminal in front of an example database, \
you need only to return the sql command to me. Below is an instruction that describes a task, \
Write a response that appropriately completes the request.\n"  \
"##PHISHCODE Context:\n{context}\n###Input:\n{question}\n\n###SQL Query:\n{sqlquery}"

no_instruction_sql_template = "I want you to act as a SQL terminal in front of an example database, \
you need only to return the sql command to me. \
Write a response that appropriately completes the request.\n"  \
"###Input:\n{question}\n\n###SQL Query:\n{sqlquery}"

inference_sql_template = "I want you to act as a SQL terminal in front of an example database, \
you need only to return the sql command to me. Below is an instruction that describes a task, \
Write a response that appropriately completes the request.\n"  \
"###Input:\n{question}.\n\n###SQL Query:"

# Open Ended
open_ended_tempelate = "###Question: {question}\n\n###Answer: {answer}"

inference_open_ended_template = "###Question: {question}\n\n###Answer:"

#------------------------------------------------------------------------------#

# SQL
sql_prompt_with_instruction = PromptTemplate(template=with_instruction_sql_template, input_variables=['context', 'question', 'sqlquery'])

sql_prompt_without_instruction = PromptTemplate(template=no_instruction_sql_template, input_variables=['question', 'sqlquery'])

sql_inference_prompt = PromptTemplate(template=inference_sql_template, input_variables=['question'])

# Open Ended
open_ended_prompt = PromptTemplate(template=open_ended_tempelate, input_variables=['question', 'answer'])

open_ended_inference_prompt = PromptTemplate(template=inference_open_ended_template, input_variables=['question'])

#------------------------------------------------------------------------------#

# SQL
def format_sql_text_to_prompt(example):
    if example['Context'] != "":
      ans = sql_prompt_with_instruction.format(context=example['Context'],
                           question=example['Questions'],
                           sqlquery=example['SQL Queries'])
    else:
      ans = sql_prompt_without_instruction.format(question=example['Questions'],
                           sqlquery=example['SQL Queries'])

    return ans

def format_sql_inference_prompt(example):
    ans = sql_inference_prompt.format(question=example)
    return ans

# Open Ended
def format_open_ended_text_to_prompt(example):
    ans = open_ended_prompt.format(question=example['Questions'],
                           answer=example['Answers'])

    return ans

def format_open_ended_inference_prompt(example):
    ans = open_ended_inference_prompt.format(question=example)
    return ans
