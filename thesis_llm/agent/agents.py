import pinecone
from flask import Flask, render_template, request, Response, stream_with_context
from flask_cors import CORS  # Import CORS
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Pinecone
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import sys
from langsmith import Client

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']="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"
os.environ['LANGCHAIN_PROJECT']="llm-thesis"

client = Client()
class CallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.content: str = ""
        self.final_answer: bool = False
    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.content += token
        if "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ["}"]:
                    sys.stdout.write(token)
                    sys.stdout.flush()

load_dotenv('api.env')
api_key = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key='ce12918f-126f-4f81-9a88-839c92253f70')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key='sk-proj-fdNYukGyyxvyalBUqbsqT3BlbkFJFr5nrbPjjL04e3JNbJ3d'
)

# connect to index
index = pc.Index("llmtesting")
text_field = "text"  # the metadata field that contains our text

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key='sk-proj-fdNYukGyyxvyalBUqbsqT3BlbkFJFr5nrbPjjL04e3JNbJ3d',
    model_name='gpt-3.5-turbo',
    temperature=0.0,
    streaming=True,
    callbacks=[CallbackHandler()]
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base HSE',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=4,
    early_stopping_method='generate',
    memory=conversational_memory,
    return_intermediate_steps=False
)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = agent.invoke({"input": user_message})
    print(response)
    return Response(response['output'])


if __name__ == '__main__':
    app.run(port=5000)

#response = agent.invoke({"input": "сколько бюджетных мест на программе “Информатика и вычислительная техника” "})