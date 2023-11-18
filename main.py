from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

llm = Ollama(model="mistral-openorca", 
            #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.9,
             )

pre_prompt = "You are a respected industry leader in writing and blogs with indepth knowledge of all factors including tone and storytelling, "
brainstorm_template = str(pre_prompt) + "Please brainstorm and create 10 ideas for a blog post around the topic: {topic}?"

prompt = PromptTemplate(
    input_variables=["topic"],
    template=brainstorm_template,
)

chain = LLMChain(llm=llm, 
                 prompt=prompt,
                 verbose=False)

user_input = input("What do you want to write a blog about: ")

# Run the chain only specifying the input variable.
print(chain.run(user_input))
