from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo",temperature=0.7,max_tokens=150)   

result = llm.invoke("What is the capital of France?")

print(result)