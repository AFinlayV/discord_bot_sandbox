"""
Starting from scratch with the discord bot template using the Bot class instead of the Client class.
"""
import asyncio
import discord
from discord.ext import commands
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

TEMPLATE = """
the following is a message from a user on discord:
{message}
the following is a response from a bot:
"""

intents = discord.Intents.all()
CHAN = int(os.environ.get('SANDBOX_DISCORD_CHAN_ID'))
TOKEN = os.environ.get('SANDBOX_DISCORD_TOKEN')
bot = commands.Bot(
    intents=intents,
    command_prefix='!',
    channel_id=CHAN,
)
LLM = OpenAI(temperature=0.0)
MEMORY = ConversationBufferMemory(memory_key="chat_history")
TOOL_NAMES = ['llm-math', 'google-search', 'wolfram-alpha']
TOOLS = load_tools(TOOL_NAMES, llm=LLM)
AGENT = initialize_agent(llm=LLM, memory=MEMORY, tools=TOOLS, agent='conversational-react-description', verbose=True)


def process_message(message):
    try:
        prompt = PromptTemplate(input_variables=['message'], template=TEMPLATE)
        text = prompt.format(message=message.content)
        response = AGENT.run(text)
        return response
    except Exception as e:
        return f'<Error>: {e}'


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    channel = bot.get_channel(CHAN)
    await channel.send(f'{bot.user} has connected to Discord!')


@bot.event
async def on_message(message):
    if message.channel.id != CHAN or message.author == bot.user or message.content.startswith('!') or message.content == "":
        return
    else:
        # use asyncio to run the process_message function in the background
        response = await asyncio.get_event_loop().run_in_executor(None, process_message, message.content)
        channel = bot.get_channel(int(os.environ.get('SANDBOX_DISCORD_CHAN_ID')))
        await channel.send(response)


bot.run(TOKEN)
