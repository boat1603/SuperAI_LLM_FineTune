IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "[INST] <<SYS>>\n"
        "You are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด\n"  # noqa: E501
        "<</SYS>>\n"
        "{instruction}###{input} [/INST]"
    ),
    "prompt_no_input": (
        "[INST] <<SYS>>\n"
        "You are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด\n"  # noqa: E501
        "<</SYS>>\n"
        "{instruction} [/INST]"
    ),
}
