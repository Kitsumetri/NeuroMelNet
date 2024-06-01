from torchvision import transforms

import constants as consts
from model import NeuroMelNet
from processor import AudioProcessor
from utils import load_checkpoint

from typing import Tuple
import torch


preprocessor = transforms.Compose(
    [
        AudioProcessor(sample_rate=consts.SAMPLE_RATE, 
                       size=consts.AUDIO_SIZE)
    ]
)


net = load_checkpoint(load_path='Detection task/pretrained/NeuroMelNet.pth', 
                      model=NeuroMelNet().to(consts.DEVICE), 
                      pretrained_eval=True)


def predict_fake_input_audio(file_path: str) -> Tuple[str, torch.Tensor]:
    classes = {0: 'Нейросеть 🤖', 1: "Человек 👤"}
    net.eval()

    try:
        mel_tensor = preprocessor(file_path).to(consts.DEVICE)
    except FileNotFoundError:
        return None

    with torch.no_grad():
        outputs = net(mel_tensor.squeeze())
        predicted = torch.argmax(outputs, dim=1).cpu().numpy().item()
        return classes[predicted], torch.round(outputs, decimals=4)


import asyncio
import logging
import sys

import os

from dotenv import load_dotenv
from os import getenv

from aiogram import __version__ as aiogram_version
print(aiogram_version, end='\n\n')

from aiogram import Bot, Dispatcher, html, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command

from aiogram.types import (Message)


load_dotenv()

dp = Dispatcher()
router = Router()
TOKEN: str = getenv("NN_BOT_TOKEN")
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

info_message = """
\nЭтот бот создан для проверки аудиозаписи речи на DeepFake 🦻 

Для пользования нужно всего лишь загрузить в бота звуковой файл формата .mp3 или .wav, либо же 
записать голосовое сообщение.

Удачи!
"""

INPUT_FOLDER = "Detection task/API/input_files"

@dp.message(Command('info'))  # /info
async def command_info_handler(message: Message) -> None:
    await message.reply(info_message)

@dp.message(CommandStart())  # /start
async def command_start_handler(message: Message) -> None:
    start_message = f"Привет, {html.bold(message.from_user.full_name)} 👋"
    start_message += info_message
    await message.answer(start_message)

@dp.message(F.audio | F.document | F.voice)
async def audio_input(message: Message):
    try:
        file_name = message.audio.file_name
        file_size = message.audio.file_size // (1024 * 1024)
        file_id = message.audio.file_id
    except AttributeError:
        try:
            file_name = message.document.file_name
            file_size = message.document.file_size // (1024 * 1024)
            file_id = message.document.file_id
        except AttributeError:
            try:
                import datetime
                file_name = message.chat.first_name + str(datetime.datetime.now()) + '.wav'
                file_size = message.voice.file_size // (1024 * 1024)
                file_id = message.voice.file_id
            except AttributeError as e:
                print(e)
                await message.reply('Неподдерживаемый формат.\nПоддерживаемые форматы: wav, mp3 и голосовые сообщения')
                return

    if file_name.split('.')[-1] not in ('wav', 'mp3'):
        await message.reply('Неподдерживаемый формат.\nПоддерживаемые форматы: ogg, wav, mp3')
        return

    if file_size > 5:
        await message.reply('Аудиозапись не должна весить более 5 mb')
        return
    
    process_message = await message.reply('Обрабатываем ваш запрос...')
    
    try:
        file = await bot.get_file(file_id)
        file_path = file.file_path
        new_file_name = f"{INPUT_FOLDER}/{file_name}"
        await bot.download_file(file_path, new_file_name)

    except Exception as e:
        await bot.delete_message(process_message.chat.id, process_message.message_id)
        await message.reply('Что-то пошло не так... Повторите попытку позже :(')
        return
    
    await bot.delete_message(process_message.chat.id, process_message.message_id)

    class_name, _ = predict_fake_input_audio(new_file_name)
    await message.reply(f'Предсказанный класс для файла {new_file_name.split("/")[-1]}:\n{class_name}')
    # os.remove(new_file_name)


@dp.message()
async def echo(message: Message):
    await message.reply("Я не такой умный, чтобы обрабатывать это 🤡")


async def main() -> None:
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
