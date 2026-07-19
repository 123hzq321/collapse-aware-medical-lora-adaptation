PUBMEDQA_TEMPLATES = [
    """Answer the biomedical research question using only the provided context.

Label meanings:
- yes: the context supports an affirmative answer.
- no: the context supports a negative answer.
- maybe: the context is explicitly inconclusive, mixed, or insufficient.

Do not choose maybe just because the topic is medical. Choose the best label supported by the abstract.

Question:
{question}

Context:
{context}

Return exactly one label: yes, no, or maybe.
Answer:""",
    """You are evaluating a PubMedQA-style biomedical question.

Use the abstract context as evidence. Choose one of: yes, no, maybe.

Context:
{context}

Question:
{question}

Label:""",
    """Given the medical literature excerpt below, decide whether the answer to the question is yes, no, or maybe.

Excerpt:
{context}

Question:
{question}

One-word answer:""",
]


VALID_LABELS = {"yes", "no", "maybe"}
CHOICE_LABELS = ("A", "B", "C", "D")


def parse_pubmedqa_label(text: str) -> str | None:
    normalized = text.strip().lower()
    for token in ("yes", "no", "maybe"):
        if normalized.startswith(token):
            return token
    return None


def build_pubmedqa_user_text(question: str, context: str, template_id: int = 0) -> str:
    return PUBMEDQA_TEMPLATES[template_id].format(
        question=question,
        context=context,
    )


def build_pubmedqa_prompt(tokenizer, question: str, context: str, template_id: int = 0) -> str:
    user_prompt = build_pubmedqa_user_text(question, context, template_id)
    messages = [
        {
            "role": "system",
            "content": "You are a biomedical QA classifier. Return only yes, no, or maybe.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_prompt


def build_medmcqa_user_text(question: str, choices: dict[str, str]) -> str:
    rendered_choices = "\n".join(
        f"{label}. {choices[label]}"
        for label in CHOICE_LABELS
    )
    return f"""Answer the medical multiple-choice question.

Choose exactly one option: A, B, C, or D.

Question:
{question}

Options:
{rendered_choices}

Return exactly one letter: A, B, C, or D.
Answer:"""


def build_medmcqa_prompt(tokenizer, question: str, choices: dict[str, str]) -> str:
    user_prompt = build_medmcqa_user_text(question, choices)
    messages = [
        {
            "role": "system",
            "content": "You are a medical multiple-choice QA classifier. Return only A, B, C, or D.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_prompt
