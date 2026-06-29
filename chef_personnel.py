
# TP_Soumia KHAY

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from tavily import TavilyClient

# Initialiser le modèle Ollama
model = ChatOllama(model="llama3.2:3b", temperature=0)

# Définir le rôle du chef cuisinier
system_prompt = "Vous êtes un chef cuisinier personnel. Proposez des plats adaptés aux ingrédients et aux préférences."

# Initialiser Tavily avec ta clé API
tavily_client = TavilyClient(api_key="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")

# Définir le Tool de recherche web (docstring obligatoire)
@tool("web_search")
def web_search(query: str):
    """Recherche des recettes ou informations culinaires sur le web via Tavily"""
    return tavily_client.search(query)

# Activer la mémoire
memory = InMemorySaver()

# Créer l’agent
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=memory,
)

# Configuration pour la mémoire
config = {"configurable": {"thread_id": "1"}}

# --- Tests ---

# 1. Liste d’ingrédients du réfrigérateur
frigo = ["poulet", "tomates", "riz"]
question1 = HumanMessage(content=f"Ingrédients disponibles dans le réfrigérateur : {', '.join(frigo)}")
response1 = agent.invoke({"messages": [question1]}, config)
print("Test 1 (ingrédients du frigo) :", response1['messages'][-1].content)

# 2. Préférence utilisateur (mémoire)
pref = HumanMessage(content="Je suis végétarien, évite la viande.")
agent.invoke({"messages": [pref]}, config)

# 3. Nouvelle demande avec préférence mémorisée
question2 = HumanMessage(content="J’ai des lentilles et des carottes")
response2 = agent.invoke({"messages": [question2]}, config)
print("Test 2 (préférence mémorisée) :", response2['messages'][-1].content)

# 4. Plat épicé avec ingrédients
question3 = HumanMessage(content="J’ai des lentilles et des carottes, propose un plat épicé.")
response3 = agent.invoke({"messages": [question3]}, config)
print("Test 3 (plat épicé) :", response3['messages'][-1].content)
