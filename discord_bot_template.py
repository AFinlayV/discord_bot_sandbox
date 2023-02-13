"""
A template for building discord bots that use OpenAI's GPT-3 API.
"""

import asyncio
import discord
from discord.ext import
import os
import openai
import json
from langchain.llms import OpenAI


def init_discord_client():
    intents = discord.Intents.all()
    bot = discord.Client(intents=intents,
                         shard_id=0,
                         shard_count=1,
                         reconnect=True)
    return bot


def init_keys():
    with open(OPENAI_KEY_FILENAME) as f:
        key = f.read().strip()
        openai.api_key = key
        os.environ["OPENAI_API_KEY"] = key
    with open(DISCORD_AUTH_FILENAME, 'r', encoding='utf-8') as f:
        auth = json.load(f)
        os.environ['DISCORD_TOKEN'] = auth['token'].strip()
        os.environ['DISCORD_CHAN_ID'] = auth['chan_id'].strip()


def process_message(message, gpt_client):
    text = message.content
    response = gpt_client(text)
    return response


if __name__ == "__main__":
    init_keys()
    discord_client = init_discord_client()
    gpt = OpenAI(
        temperature=0.0,
        max_tokens=128,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )


    @discord_client.event
    async def on_ready():
        channel = discord_client.get_channel(int(os.environ['DISCORD_CHAN_ID']))
        if channel is not None:
            await channel.send(f"{discord_client.user} has connected to Discord!")
        else:
            print("Channel not found")


    @discord_client.event
    async def on_message(message):
        if message.author == discord_client.user:
            return
        elif message.content.startswith('!') or message.content == "":
            return
        else:
            # use asyncio to run the process_message function in the background
            output = await asyncio.get_event_loop().run_in_executor(None, process_message, message, gpt)
            await message.channel.send(output)


    discord_client.run(os.environ['DISCORD_TOKEN'])
