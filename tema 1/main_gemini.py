from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

pregunta = "¿En qué año llegó el ser humano a la luna por primera vez?"
print("Pregunta:", pregunta)

respuesta = llm.invoke(pregunta)
print("Respuesta:", respuesta.content)
