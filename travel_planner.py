import importlib
import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any
from pandas import DataFrame
from tqdm import tqdm
from datasets import load_dataset
import requests

from ReAct import ReActFramework
from prompts import zeroshot_react_agent_prompt

# Update system path to include necessary directories
sys.path.extend([
    os.path.abspath(os.path.join(os.getcwd(), "..")),
    os.path.abspath(os.path.join(os.getcwd(), "tools/planner")),
    os.path.abspath(os.path.join(os.getcwd(), "../tools/planner"))
])

# Set environment variables
os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

# Action mapping dictionary to map action types to tool names
action_mapping = {
    "FlightSearch": "flights",
    "AttractionSearch": "attractions",
    "GoogleDistanceMatrix": "googleDistanceMatrix",
    "AccommodationSearch": "accommodation",
    "RestaurantSearch": "restaurants",
    "Planner": "planner",
    "NotebookWrite": "notebook",
    "CitySearch": "cities"
}



# Custom exceptions for handling specific errors
class CityError(Exception):
    pass


class DateError(Exception):
    pass

class ActionHandler:
    def handle_flightsearch(self, args: str):
        """Handle the FlightSearch action."""
        from_city, to_city, date = args.split(', ')
        if not validate_date_format(date):
            raise DateError(f"Invalid date format: {date}")
        if from_city not in self.city_set or to_city not in self.city_set:
            raise CityError(f"Invalid cities: {from_city}, {to_city}")
        self.current_data = self.tools['flights'].run(from_city, to_city, date)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_attractionsearch(self, args: str):
        """Handle the AttractionSearch action."""
        city = args.strip()
        if city not in self.city_set:
            raise CityError(f"Invalid city: {city}")
        self.current_data = self.tools['attractions'].run(city)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_accommodationsearch(self, args: str):
        """Handle the AccommodationSearch action."""
        city = args.strip()
        if city not in self.city_set:
            raise CityError(f"Invalid city: {city}")
        self.current_data = self.tools['accommodations'].run(city)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_restaurantsearch(self, args: str):
        """Handle the RestaurantSearch action."""
        city = args.strip()
        if city not in self.city_set:
            raise CityError(f"Invalid city: {city}")
        self.current_data = self.tools['restaurants'].run(city)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_citysearch(self, args: str):
        """Handle the CitySearch action."""
        state = args.strip()
        self.current_data = self.tools['cities'].run(state)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_googledistancematrix(self, args: str):
        """Handle the GoogleDistanceMatrix action."""
        origin, destination, mode = args.split(', ')
        self.current_data = self.tools['googleDistanceMatrix'].run(origin, destination, mode)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_notebookwrite(self, args: str):
        """Handle the NotebookWrite action."""
        print("writing to notebook")
        self.current_observation = str(self.tools['notebook'].write(self.current_data, args))
        self.json_log[-1]['state'] = 'Successful'

    def handle_planner(self, args: str):
        """Handle the Planner action."""
        self.current_observation = str(self.tools['planner'].run(str(self.tools['notebook'].list_all()), args))
        self.answer = self.current_observation
        self.json_log[-1]['state'] = 'Successful'


def validate_date_format(date_str: str) -> bool:
    """Validate the date format to ensure it matches YYYY-MM-DD."""
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))


def to_string(data) -> str:
    """Convert data to a string format, handling different data types."""
    if data is not None:
        if isinstance(data, DataFrame):
            return data.to_string(index=False)
        return str(data)
    return "None"


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_url", default="http://localhost:11434/api/chat", type=str)
    parser.add_argument("--model_name", default="llama3:8b-instruct-fp16", type=str)  # llama3:70b-instruct-q5_1
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--stream", default=False, action='store_true')
    parser.add_argument("--set_type", default="validation", type=str)
    args = parser.parse_args()

    # Load the dataset based on the set type
    dataset = load_dataset('osunlp/TravelPlanner', args.set_type)[args.set_type]
    tools_list = ["notebook", "flights", "attractions", "accommodations", "restaurants", "googleDistanceMatrix",
                  "planner", "cities"]

    # Initialize the ReactAgent
    agent = ReActFramework(args, mode='zero_shot', tools=tools_list, max_steps=20, max_retries=3,
                       illegal_early_stop_patience=3,
                       react_llm_name=args.model_name, planner_llm_name=args.model_name,
                       city_file_path='../database/background/citySet.txt',
                       agent_prompt=zeroshot_react_agent_prompt,
                       action_mapping=action_mapping,
                       action_handler=ActionHandler)

    # Create output directory if it doesn't exist
    output_path = os.path.join(args.output_dir, args.set_type)
    os.makedirs(output_path, exist_ok=True)

    # Process each query in the dataset
    for number, data in enumerate(tqdm(dataset), start=1):
        if number > 1: continue
        query = data['query']
        output_file = os.path.join(output_path, f'generated_plan_{number}.json')

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result = json.load(f)
        else:
            result = [{}]

        # Run the agent to get the results
        planner_results, scratchpad, action_log = agent.run(query)
        if planner_results == 'Max Token Length Exceeded.':
            result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
            result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
            action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
        else:
            result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
            result[-1][f'{args.model_name}_two-stage_results'] = planner_results
            result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

        # Save the results to a file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
