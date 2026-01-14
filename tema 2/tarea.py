from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import json


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
)

def preprocess_text(text):
    return text.strip()[:500]

preprocessor = RunnableLambda(preprocess_text)

def generate_summary(text):
    prompt = (
        "Resume el siguiente texto en **una sola oración clara y precisa**, "
        "manteniendo la idea principal y evitando detalles secundarios:\n\n"
        f"{text}"
    )
    response = llm.invoke(prompt)
    return response.content

def analyze_sentiment(text):
    prompt = f"""Analiza el sentimiento del siguiente texto.
                Devuelve SOLO un JSON válido.
                No markdown.
                No texto adicional.
                Formato exacto:
                {{"sentimiento":"positivo|negativo|neutro","razon":"justificación breve"}}

                Texto: {text}"""

    response = llm.invoke(prompt)

    raw = response.content

    cleaned = raw.strip()

    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception as e:
        print("ERROR JSON:", repr(raw))
        return {"sentimiento": "neutro", "razon": "Error en análisis"}


def merge_results(data):
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }

def process_one(t):
    resumen = generate_summary(t)              
    sentimiento_data = analyze_sentiment(t) 
    return merge_results({
        "resumen": resumen,
        "sentimiento_data": sentimiento_data
    })
    
process = RunnableLambda(process_one)
    
chain = preprocessor | process

textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
]
 
for texto in textos_prueba:
    resultado = chain.invoke(texto)
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-" * 50)