import os
import re

from agent import OllamaAgent


# ReactFramework class to manage the interaction process
class ReActFramework:
    def __init__(self, args, mode: str, max_steps: int, max_retries: int,
                 illegal_early_stop_patience: int,
                 react_llm_name: str, planner_llm_name: str, agent_prompt: str,
                 action_mapping: dict, action_handler):
        self.max_steps = max_steps
        self.mode = mode
        self.react_name = react_llm_name
        self.planner_name = planner_llm_name
        self.agent_prompt = agent_prompt
        self.illegal_early_stop_patience = illegal_early_stop_patience
        self.max_retries = max_retries
        self.llm = OllamaAgent(llama_url=args.llama_url, model=react_llm_name, stream=args.stream, output=os.path.join(args.output_dir, "output.json"),
                          messages=[])
        self.action_mapping = action_mapping
        self.action_handler = action_handler
        self.__reset_agent()

    def run(self, query: str, reset: bool = True):
        """Run the agent with a given query."""
        self.query = query
        if reset:
            self.__reset_agent()
        self.llm.add_message("system", self.agent_prompt.format(query=self.query, scratchpad=self.scratchpad))

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self):
        """Perform a single step in the agent's reasoning process."""
        self.json_log.append({"step": self.step_n, "thought": "", "action": "", "observation": "", "state": ""})
        thought = self.prompt_agent(f"Give me thought number {self.step_n} and that only (without extra dialogue) in the following example format:\n\nThought {self.step_n}: [reasoning inserted here]\n")
        self.json_log[-1]['thought'] = thought
        action = self.prompt_agent(f"Give me action number {self.step_n} and that only (without extra dialogue) in the following example format:\n\nAction {self.step_n}: \nActionName[Required Information]\n")
        self.json_log[-1]['action'] = action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()
        self.last_actions.append(action)

        if len(self.last_actions) == 3:
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return

        self.scratchpad = f'Observation {self.step_n}: '
        action_type, action_arg = self.parse_action(action)
        if action_type:
            self.handle_action(action_type, action_arg)
        self.json_log[-1]['observation'] = self.current_observation
        self.step_n += 1

        if action_type == 'Planner' and self.retry_record['planner'] == 0:
            self.finished = True
            self.answer = self.current_observation

        self.llm.add_message('user', self.current_observation)

        # print(self.llm.messages[-5:])

    def prompt_agent(self, message: str) -> str:
        """Prompt the agent with a message and return the response."""
        response = self.llm.send_query()
        return response['message']['content']

    def __reset_agent(self):
        """Reset the agent's state."""
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad = ''
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []
        self.llm.messages = []
        self.retry_record = {key: 0 for key in self.action_mapping.values()}
        self.retry_record['invalidAction'] = 0

    def parse_action(self, action_str: str):
        """Parse the action string to extract the action type and arguments."""
        pattern = r'(\w+)\[(.+)]'
        match = re.search(pattern, action_str, re.M)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def handle_action(self, action_type: str, action_arg: str):
        """Handle the action based on its type."""
        if action_type in self.action_mapping:
            try:
                action_func = getattr(self.action_handler, f'handle_{action_type.lower()}')
                action_func(action_arg)
            except Exception as e:
                self.current_observation = f'Error in {action_type}: {str(e)}'
                self.json_log[-1]['state'] = 'Error'

    def is_finished(self) -> bool:
        """Check if the agent has finished its process."""
        return self.finished

    def is_halted(self) -> bool:
        """Check if the agent has halted due to reaching maximum steps or token length."""
        return (self.step_n > self.max_steps) and not self.finished