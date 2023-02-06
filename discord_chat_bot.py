"""
This will be a simple gpt based chatbot with discord integrations
The idea here is to allow a chatbot to participate in a discord and remember what each participant has said.

based on a conversation with chatgpt here is the code it gave me the following code.

I can use that to create a simple chatbot that can be used in discord. I'll need to figure out the memory thing later.
I kinda stole the idea from TheGuy on Twitter (@Guygies). he has a bot called Geepers on discord.
"""

import discord
import json
import openai
from langchain.llms import OpenAI
import os
from langchain import PromptTemplate, ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.agents import load_tools
from langchain.agents import initialize_agent


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    key = f.read().strip()
    openai.api_key = key
    os.environ["OPENAI_API_KEY"] = key

with open("/Users/alexthe5th/Documents/API Keys/GoogleSearchAPI_key.txt", "r") as f:
    key = f.read().strip()
    os.environ["GOOGLE_API_KEY"] = key

with open("/Users/alexthe5th/Documents/API Keys/GoogleSearch_ID.txt", "r") as f:
    key = f.read().strip()
    os.environ["GOOGLE_CSE_ID"] = key

with open("prompt_template.txt", "r") as f:
    template = f.read()
auth = load_json('/Users/alexthe5th/Documents/API Keys/Discord_auth.json')
TOKEN = auth['token'].strip()
CHAN_ID = int(auth['chan_id'].strip())
intents = discord.Intents.all()
client = discord.Client(intents=intents)
llm = OpenAI(temperature=0.9,
             model_name='text-davinci-003',
             max_tokens=1024,
)
memory = ConversationalBufferWindowMemory()
prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)

bort = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=2048),
)


@client.event
async def on_message(message):
    if not message.author.bot:
        discord_text = message.content
        reply = bort.predict(human_input=discord_text, history=bort.memory)
        await message.channel.send(reply)


@client.event
async def on_ready():
    channel = client.get_channel(CHAN_ID)
    if channel is not None:
        await channel.send("Hello World!")
        await channel.send(f"{client.user} has connected to Discord!")
    else:
        print("Channel not found")


if __name__ == "__main__":
    client.run(TOKEN)
