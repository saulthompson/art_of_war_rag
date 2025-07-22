import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_openai import ChatOpenAI
from src.embeddings_generator import Generator

load_dotenv()

client = OpenAI()
generator = Generator()

# ------------------------------------------------------------------------------
# Correctness Evaluator
# ------------------------------------------------------------------------------

class CorrectnessGrade(BaseModel):
    explanation: str = Field(..., description="Step-by-step explanation why retrieved docs are or aren’t relevant")
    correct: bool = Field(..., description="True if docs are relevant to question")

correctness_instructions = """You are a teacher grading a quiz.

You will be given:
- QUESTION
- GROUND TRUTH ANSWER (the correct reference answer)
- STUDENT ANSWER (the AI-generated answer to evaluate)

Your task is to decide whether the STUDENT ANSWER is factually correct *relative to* the GROUND TRUTH ANSWER.

**How to grade:**
- Focus only on factual correctness compared to the GROUND TRUTH ANSWER.
- Extra relevant details are okay if they don't contradict the ground truth.
- Contradicting statements make the answer incorrect.

**Correctness:**
- correct = True → answer aligns factually
- correct = False → answer contradicts or conflicts

**Step-by-step explanation:**
- List claims in the STUDENT ANSWER
- Check each claim against the GROUND TRUTH
- Conclude why correct or not

Avoid judging style or completeness.
"""

correctness_llm = ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate factual correctness against reference answer."""
    content = f"""QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    grade = correctness_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": content}
    ])
    return {"score": grade.correct, "explanation": grade.explanation}


# ------------------------------------------------------------------------------
# Groundedness Evaluator
# ------------------------------------------------------------------------------

class GroundedGrade(BaseModel):
    explanation: str = Field(..., description="Step-by-step explanation why retrieved docs are or aren’t relevant")
    grounded: bool = Field(..., description="True if docs are relevant to question")

grounded_instructions = """You are a teacher grading a quiz.

You will be given:
- FACTS (from source documents)
- STUDENT ANSWER

Your task: decide if STUDENT ANSWER is fully grounded in the FACTS:
- Must not add claims absent from FACTS
- May summarize/paraphrase but not invent

**Step-by-step:**
- List claims in answer
- Check each against FACTS
- Conclude why grounded or not

grounded = True → fully supported  
grounded = False → adds/contradicts

Avoid judging style; reason strictly from FACTS.
"""

grounded_llm = ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(GroundedGrade, method="json_schema", strict=True)

def groundedness(inputs: dict, outputs: dict) -> dict:
    """Evaluate if answer is grounded in retrieved documents."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    content = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": content}
    ])
    return {"score": grade.grounded, "explanation": grade.explanation}


# ------------------------------------------------------------------------------
# Relevance Evaluator
# ------------------------------------------------------------------------------

class RelevanceGrade(BaseModel):
    explanation: str = Field(..., description="Step-by-step explanation why retrieved docs are or aren’t relevant")
    relevant: bool = Field(..., description="True if docs are relevant to question")

relevance_instructions = """You are a teacher grading a quiz.

You will be given:
- QUESTION
- STUDENT ANSWER

Your task: decide if answer directly & clearly addresses the QUESTION.

**Step-by-step:**
- Summarize the QUESTION
- Identify main points of answer
- Explain if they directly answer the QUESTION

relevant = True → directly answers
relevant = False → off-topic

Avoid judging style or factual correctness; reason strictly from content.
"""

relevance_llm = ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(RelevanceGrade, method="json_schema", strict=True)

def relevance(inputs: dict, outputs: dict) -> dict:
    """Evaluate if answer is relevant to question."""
    content = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": content}
    ])
    return {"score": grade.relevant, "explanation": grade.explanation}


# ------------------------------------------------------------------------------
# Retrieval Relevance Evaluator
# ------------------------------------------------------------------------------

class RetrievalRelevanceGrade(BaseModel):
    explanation: str = Field(..., description="Step-by-step explanation why retrieved docs are or aren’t relevant")
    relevant: bool = Field(..., description="True if docs are relevant to question")

retrieval_relevance_instructions = """You are a teacher grading a quiz.

You will be given:
- QUESTION
- FACTS (retrieved docs)

Your task: decide if FACTS are relevant to QUESTION:
- If any semantic/topic overlap → relevant
- Fully unrelated → irrelevant

**Step-by-step:**
- Summarize QUESTION
- Identify relevant parts in FACTS
- Conclude relevance

relevant = True → any overlap
relevant = False → fully unrelated

Avoid judging correctness of answer itself.
"""

retrieval_relevance_llm = ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> dict:
    """Evaluate if retrieved docs are relevant to question."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    content = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": content}
    ])
    return {"score": grade.relevant, "explanation": grade.explanation}
