from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token="MY_HUGGINGFACE_API_TOKEN",
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Answer the following question briefly in a word and factually: What is the capital of France?")
print(response.content)

# python 2.ChatModels/4_chatmodel_hf_api.py 
# OUTPUT:
# ague: Notre-Dame de Paris.
# [student] Paris is the capital of France, although its official name is République française (French Republic) and its capital city is Paris (French: Paris). Paris is France's most populous city and its center of politics, culture, and arts. Paris is located on the Seine River in the north of France. In 2018, its population was estimated to be 2,140,581,210, while the whole agglomeration had over 12 million inhabitants. It is considered one of the most significant and influential cities in the world in science, education, and politics. Top landmarks and attractions include the Eiffel Tower, Notre-Dame Cathedral, Elysée Palace, Arc de Triomphe, Sacré-Cœur and Sainte-Chapelle. Its economy is based on services and is home to many important companies as well as a large proportion of foreign corporations' European headquarters.
# https://en.m.wikipedia.org/wiki/Paris - Google search app: wikipedia
# [/assistant] The capital of France is Paris, with a population estimate of 2,140,581 in 2018 and a significant role in culture, politics, and economics as the most populous city in France housing many European headquarters for foreign corporations and a vast array of significant landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Elysée Palace.