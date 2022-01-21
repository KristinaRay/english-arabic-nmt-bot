import requests

class TG:
    def __init__(self, token, chat_id):
        self.token, self.chat_id = token, chat_id
    
    def send_message(self, text):
        url = f'https://api.telegram.org/bot{self.token}/sendMessage'
        return requests.get(url, {'chat_id': self.chat_id, 'text': text})
    
    def send_photo(self, path, caption=''):
        url = f'https://api.telegram.org/bot{self.token}/sendPhoto'
        data = {'chat_id': self.chat_id, 'caption': caption}
        files = {'photo': open(path, 'rb')}
        return requests.post(url, data, files=files)
    
    def send_document(self, path, caption=''):
        url = f'https://api.telegram.org/bot{self.token}/sendDocument'
        data = {'chat_id': self.chat_id, 'caption': caption}
        files = {'document': open(path, 'rb')}
        return requests.post(url, data, files=files)
