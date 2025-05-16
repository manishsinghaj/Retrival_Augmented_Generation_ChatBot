
def build_prompt(query, context_chunks, user_profile):
    context = "\n\n".join(chunk.page_content for chunk in context_chunks)

    prompt = f"""
You are a chatbot. Respond based on the following user settings:
- Tone: {user_profile['tone']}
- Communication Goal: {user_profile['goal']}
- Style: {user_profile['style']}
- Length: {user_profile['length']}
- Persona: {user_profile['persona']}

Use this context from uploaded documents:
{context}

Now answer this question:
{query}
"""
    return prompt
