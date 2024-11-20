import os
from telethon import TelegramClient, events
from dotenv import load_dotenv
import csv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Telegram API credentials
api_id = os.getenv('TELEGRAM_API_ID')
api_hash = os.getenv('TELEGRAM_API_HASH')
phone_number = os.getenv('TELEGRAM_PHONE_NUMBER')

# Channels to scrape (replace with actual channel usernames)
channels = ['@StockMarketChat', '@WallStreetBets']

# Initialize the Telegram client
client = TelegramClient('session', api_id, api_hash)

async def scrape_channel(channel, days_to_scrape=30):
    messages = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_scrape)
    
    async for message in client.iter_messages(channel, limit=None, offset_date=start_date):
        if message.date > end_date:
            break
        messages.append({
            'channel': channel,
            'date': message.date.strftime('%Y-%m-%d %H:%M:%S'),
            'text': message.text
        })
    return messages

async def main():
    await client.start(phone=phone_number)
    print("Client Created")
    
    all_messages = []
    for channel in channels:
        channel_messages = await scrape_channel(channel)
        all_messages.extend(channel_messages)
        print(f"Scraped {len(channel_messages)} messages from {channel}")
    
    # Save messages to CSV
    with open('telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['channel', 'date', 'text'])
        writer.writeheader()
        writer.writerows(all_messages)
    
    print(f"Total messages scraped: {len(all_messages)}")
    print("Data saved to telegram_data.csv")

if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())
