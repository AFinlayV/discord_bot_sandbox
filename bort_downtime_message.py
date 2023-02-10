"""
Bort is down for maintenance, Run this to respond with down message
"""

import asyncio
import datetime
import discord
import os
import openai
import json
import re
from time import time, sleep
from uuid import uuid4
import numpy as np
from numpy.linalg import norm
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.agents import agent
from langchain.chains import ConversationChain


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def init_keys():
    # load openai api key
    with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
        key = f.read().strip()
        openai.api_key = key
        os.environ["OPENAI_API_KEY"] = key

    # load discord auth token
    auth = load_json('/Users/alexthe5th/Documents/API Keys/Discord_auth.json')
    os.environ['DISCORD_TOKEN'] = auth['token'].strip()
    os.environ['DISCORD_CHAN_ID'] = auth['chan_id'].strip()


def init_discord_client():
    intents = discord.Intents.all()
    bort = discord.Client(intents=intents,
                          shard_id=0,
                          shard_count=1,
                          reconnect=True)
    return bort


if __name__ == "__main__":
    init_keys()
    bort = init_discord_client()


    @bort.event
    async def on_ready():
        channel = bort.get_channel(int(os.environ['DISCORD_CHAN_ID']))
        if channel is not None:
            await channel.send(f"{bort.user} has connected to Discord!")
        else:
            print("Channel not found")


    @bort.event
    async def on_message(message):
        if message.author == bort.user:
            return
        await message.channel.send("Bort is down for a rewrite of his code... please check back later!")


    bort.run(os.environ['DISCORD_TOKEN'])
