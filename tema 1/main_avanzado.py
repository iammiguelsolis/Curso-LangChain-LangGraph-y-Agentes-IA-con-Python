from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0,
)

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template=(
        "Saluda al usuario con su nombre.\n"
        "Nombre del usuario: {nombre}\n"
        "Asistente:"
    )
)

chain = plantilla | chat

resultado = chain.invoke({"nombre": "Miguel"})

print(resultado.content)
