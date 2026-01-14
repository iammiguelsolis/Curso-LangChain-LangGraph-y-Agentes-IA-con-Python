from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st

propmt_template = PromptTemplate(
    input_variables=['mensaje', 'historial'],
    template='''
    Eres un asistente útil y amigable llamado KevinBot.
    
    Historial de la conversación:
    
    {historial}
    
    Responde de manera clara y concisa a la siguiente pregunta:
    Usuario: {mensaje}
    Asistente:
    '''
)

def construir_historial(mensajes):
    historial = ''
    for msg in mensajes:
        if isinstance(msg, HumanMessage):
            historial += f'Usuario: {msg.content}\n'
        elif isinstance(msg, AIMessage):
            historial += f'Asistente: {msg.content}\n'
    return historial

st.set_page_config(page_title='ChatBot Básico', page_icon='🤖')
st.title('ChatBot Básico con LangChain')
st.markdown('Este es un **ChatBot de ejemplo** construido por mí')

@st.cache_resource
def crear_chat_model(model_name, temperature):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )
    
with st.sidebar:
    st.header('Configuración')
    temperature = st.slider('Temperatura', 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox(
        'Modelo',
        ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    )
    
    if st.button('Nueva Conversación'):
        st.session_state.mensajes = []
        st.rerun()

chat = crear_chat_model(model_name, temperature)

cadena = propmt_template | chat

if 'mensajes' not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue
    
    role = 'assistant' if isinstance(msg, AIMessage) else 'user'
    
    with st.chat_message(role):
        st.markdown(msg.content)
        
pregunta = st.chat_input('Escribe tu mensaje')

if pregunta:
    with st.chat_message('user'):
        st.markdown(pregunta)
    
    try:
        
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        
        historial_texto = construir_historial(st.session_state.mensajes)
        
        with st.chat_message('assistant'):
            response_placeholder = st.empty()
            full_response = ''
            
            for chunk in cadena.stream({
                'mensaje': pregunta,
                'historial': historial_texto
            }) :
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
        st.session_state.mensajes.append(
            AIMessage(content=full_response)
        )
        
    except Exception as e:
        st.error(f'Error al generar respuesta: {str(e)}')
        st.info('Verifica tu API Key y el modelo seleccionado')