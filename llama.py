# Llama3 class to interact with the Llama model API
from typing import Dict, Any, List
import json
import requests


class Llama3:
    def __init__(self, llama_url: str, model: str, stream: bool, output: str, messages: List[Dict[str, Any]]):
        self.llama_url = llama_url
        self.model = model
        self.stream = stream
        self.output = output
        self.messages = messages

    def add_message(self, role: str, content: str):
        """Add a message to the list of messages to be sent to the Llama model."""
        if role not in ['system', 'user', 'assistant']:
            raise ValueError("Invalid role")
        print(content)
        self.messages.append({"role": role, "content": content})

    def send_query(self) -> Dict[str, Any]:
        """Send the query to the Llama model and return the response."""
        request = {
            "model": self.model,
            "messages": self.messages[:],
            "stream": self.stream
        }
        # request["messages"][-1]['content'] =  + request["messages"][-1]['content']
        response = requests.post(self.llama_url, json=request)
        response_json = response.json()
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, ensure_ascii=False, indent=4)
        
        self.add_message("assistant", response_json['message']['content'])
        return response_json
