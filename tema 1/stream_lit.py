import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# CONFIGURACIÓN STREAMLIT
# -------------------------
st.set_page_config(page_title="ChatBot Gemini", page_icon="🤖")
st.title("ChatBot con LangChain + Gemini")
st.markdown("ChatBot con **historial real**, streaming y personalidad configurable")

# -------------------------
# MODELO (CACHEADO)
# -------------------------
@st.cache_resource
def crear_chat_model(model_name: str, temperature: float):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("Configuración")

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    model_name = st.selectbox(
        "Modelo",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    )

    personalidad = st.selectbox(
        "Personalidad del asistente",
        [
            "Útil y amigable",
            "Profesional y formal",
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido"
        ]
    )

# -------------------------
# MENSAJES DEL SISTEMA
# -------------------------
system_messages = {
    "Útil y amigable": "Eres un asistente útil y amigable. Responde de forma clara y concisa.",
    "Profesional y formal": "Eres un asistente profesional y formal. Proporciona respuestas precisas y bien estructuradas.",
    "Casual y relajado": "Eres un asistente casual y relajado. Habla de forma natural y cercana.",
    "Experto técnico": "Eres un asistente experto técnico. Proporciona respuestas detalladas y precisas.",
    "Creativo y divertido": "Eres un asistente creativo y divertido. Usa analogías y ejemplos originales."
}

# -------------------------
# PROMPT (BEST PRACTICE)
# -------------------------
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_messages[personalidad]),
    ("placeholder", "{messages}")
])

# -------------------------
# CADENA
# -------------------------
chat_model = crear_chat_model(model_name, temperature)
cadena = chat_prompt | chat_model

# -------------------------
# ESTADO DE SESIÓN
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# MOSTRAR HISTORIAL
# -------------------------
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# -------------------------
# INPUT DEL USUARIO
# -------------------------
user_input = st.chat_input("Escribe tu mensaje")

if user_input:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    # Generar respuesta con streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in cadena.stream({
                "messages": st.session_state.messages
            }):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            st.session_state.messages.append(
                AIMessage(content=full_response)
            )

        except Exception as e:
            st.error(f"Error al generar respuesta: {e}")
            st.info("Verifica tu API Key y el modelo seleccionado")
