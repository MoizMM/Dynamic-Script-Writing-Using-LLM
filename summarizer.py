from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
import os

def link_processor(link):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = os.getenv("GOOGLE_API_KEY"))

    loader = YoutubeLoader.from_youtube_url(
        link,
        add_video_info=False,
        language=["en"],
        translation="en",
    )

    documents = loader.load()

    template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    response=stuff_chain.invoke(documents)
    return response["output_text"]

# if __name__ == "__main__":
#     link = "https://www.youtube.com/watch?v=McDhI537LDk"
#     print(link_processor(link))