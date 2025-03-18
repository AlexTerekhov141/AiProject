import os
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram import F
from aiogram.types import ContentType


API_TOKEN = os.environ.get("TG_TOKEN")
API_URL = os.environ.get("MODEL_API_URL")  # Ensure this is the correct URL for your FastAPI

bot = Bot(token=API_TOKEN)
router = Dispatcher()


@router.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.answer("Hey there! Send me an audio file, and I'll predict it for you! 🎧")


@router.message()
async def handle_audio(message: types.Message):
    # Get the audio file
    if message.audio is None or message.audio.file_id is None:
        await message.answer("Send me audio message!")
        return

    audio_file = await bot.get_file(message.audio.file_id)
    audio_file_path = os.path.join('downloads', f"{audio_file.file_id}.ogg")
    print("get audio request")
    # Download the audio file
    await bot.download_file(audio_file.file_path, audio_file_path)

    # Send to FastAPI for prediction
    async with aiohttp.ClientSession() as session:
        with open(audio_file_path, 'rb') as f:

            files = {'file': f.read()}
            async with session.post(API_URL, data=files) as response:
                if response.status == 200:
                    result = await response.json()
                    await message.answer(f"Prediction: {result['predicted_label']} 🎤")
                else:
                    await message.answer("Oops! Something went wrong while predicting! 😢")

    # Clean up the downloaded file
    os.remove(audio_file_path)


async def main():
    await bot.delete_my_commands()  # Clean up previous commands if any
    await router.start_polling(bot)


if __name__ == "__main__":
    print("starting")
    os.makedirs('downloads', exist_ok=True)  # Ensure the downloads folder exists
    import asyncio

    asyncio.run(main())
