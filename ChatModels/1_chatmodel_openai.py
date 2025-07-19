from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-4o",temperature=0.0,max_tokens=10)

result = model.invoke("What is the capital of France?")

print(result.content)
# Output: Paris