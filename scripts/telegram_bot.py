import asyncio
import telegram

class NotificationBot:
    def __init__(self) -> None:
        self.token = None
        self.chat_id = None

        self.token, self.chat_id = self.read_credentials()
        
        self.bot = telegram.Bot(self.token)
        
    @staticmethod
    def read_credentials():
        with open("telegram_credentials.txt", "r") as fp:
            lines = [_.strip() for _ in fp.readlines()]
            _, token = lines[0].split(",")
            _, chat_id = lines[1].split(",")
        return token, chat_id

    async def send_message(self, message):
        async with self.bot:
            await self.bot.send_message(text=message, chat_id=self.chat_id)

        # asyncio.run(send_message(self.token, self.chat_id, message))    


if __name__ == '__main__':
    bot = NotificationBot()
    
    asyncio.run(bot.send_message("b"))
