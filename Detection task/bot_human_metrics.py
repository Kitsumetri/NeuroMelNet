import asyncio
import logging
import sys

from dotenv import load_dotenv
from os import getenv
from random import randint

import sqlite3
import pandas as pd

conn = sqlite3.connect("Detection task/testing.sqlite")
cursor = conn.cursor()

sql_create_table ='''CREATE TABLE IF NOT EXISTS HumanTest(  
   time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
   audio_path TEXT NOT NULL,
   human_answer INT NOT NULL,
   true_answer INT NOT NULL
)'''

from aiogram import __version__ as aiogram_version

print(aiogram_version, end='\n\n')

load_dotenv()

from aiogram import Bot, Dispatcher, html, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.filters.callback_data import CallbackData, CallbackQuery


from aiogram.types import (
    InlineKeyboardButton,
    Message,
    InlineKeyboardMarkup,
    FSInputFile)


class UserChoiceData(CallbackData, prefix="uc"):
    user_answer: str
    audio_name: str
    true_answer: int


def preprocess_metadata_wild(df):
    df = df.drop(['speaker'], axis=1)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'spoof' else 1)
    return df

WILD_DATASET_DIR = f'Datasets/In_The_wild/release_in_the_wild'
metadata_wild: pd.DataFrame = preprocess_metadata_wild(pd.read_csv(f"{WILD_DATASET_DIR}/meta.csv"))

dp = Dispatcher()
router = Router()
TOKEN: str = getenv("BOT_TOKEN")
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


info_message = """
\nЭтот бот создан для проверки вашего слуха 🦻 

Каждый раз будет присылаться аудиозапись, a также 2 выбора ответа: 'Нейросеть' и 'Человек'. 

Аудиозапись представляет собой небольшой отрывок озвученной речи. Необходимо распознать, настоящий человек это говорил или нейросеть.

Если возникнут какие-либо технические проблемы, то пишите @kr1stt

Для старта введите команду /test

И, пожалуйста, не отвечайте на рандом, это только ухудшит статистику.

Удачи!
"""

@dp.message(Command('info'))  # /info
async def command_info_handler(message: Message) -> None:
    await message.reply(info_message)

@dp.message(CommandStart())  # /start
async def command_start_handler(message: Message) -> None:
    start_message = f"Привет, {html.bold(message.from_user.full_name)} 👋"
    start_message += info_message
    await message.answer(start_message)

async def get_random_audio() -> tuple[str, int]:
    idx = randint(0, len(metadata_wild) - 1)
    file_path = f"{WILD_DATASET_DIR}/{metadata_wild.iloc[idx]['file']}"
    label = metadata_wild.iloc[idx]['label']
    
    return file_path, label

@dp.message(Command('test'))  # /test
async def command_test_handler(message: Message) -> None:

    audio_path, true_label = await get_random_audio()

    audio_path_name: str = audio_path.split('/')[-1]

    keyboard_test = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text='Нейросеть 🤖', 
                                     callback_data=UserChoiceData(user_answer='AI',
                                                                  audio_name=audio_path_name,
                                                                  true_answer=true_label).pack()),
                InlineKeyboardButton(text='Человек 👤', 
                                     callback_data=UserChoiceData(user_answer='Human',
                                                                  audio_name=audio_path_name,
                                                                  true_answer=true_label).pack()),
            ]
        ],
        resize_keyboard=True,
)
    try:
        audio_file = FSInputFile(audio_path)
        await message.answer_voice(voice=audio_file, reply_markup=keyboard_test)
    except Exception as e:
        print(e)
        await message.reply('Что-то пошло не так, попробуйте ещё раз команду /test')

@router.callback_query(UserChoiceData.filter((F.user_answer == "AI") | (F.user_answer == 'Human')))
async def user_choose_callback(query: CallbackQuery, callback_data: UserChoiceData) -> None:
    audio_path, true_label = await get_random_audio()

    audio_file = FSInputFile(audio_path)

    sql_insert = 'INSERT INTO HumanTest (audio_path, human_answer, true_answer) VALUES (?, ?, ?)'

    cursor.execute(sql_insert, 
                         (
                          f"{WILD_DATASET_DIR}/{callback_data.audio_name}",
                          0 if callback_data.user_answer == 'AI' else 1,
                          callback_data.true_answer
                          )
    )
    conn.commit()

    await query.answer('Следующее аудио ➡️')
    await query.message.delete()
    await command_test_handler(query.message)

@dp.message()
async def echo(message: Message):
    await message.reply("Я не такой умный, чтобы обрабатывать это 🤡")

async def main() -> None:
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    cursor.execute(sql_create_table)

    asyncio.run(main())