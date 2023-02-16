import requests
import os
APP_ID = os.environ.get('RULES_LAWYER_APP_ID')
TOKEN = os.environ.get('RULES_LAWYER_DISCORD_TOKEN')

application_id = APP_ID  # Replace with your application ID
bot_token = TOKEN  # Replace with your bot's token

headers = {
    "Authorization": f"Bot {bot_token}"
}

data = {
    "name": "rules",
    "description": "Returns the texto of rules that are semantically similar to the query."
}

response = requests.post(f"https://discord.com/api/v8/applications/{application_id}/commands", headers=headers, json=data)

print(response.status_code)
print(response.json())
