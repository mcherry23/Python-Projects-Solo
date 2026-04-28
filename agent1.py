import json
import time
from typing import List, Dict, Any, Callable


# =========================
# MEMORY SYSTEM
# =========================
class Memory:
    def __init__(self):
        self.store: List[Dict[str, Any]] = []

    def add(self, role: str, content: str):
        self.store.append({"role": role, "content": content})

    def recent(self, n=10):
        return self.store[-n:]

    def dump(self):
        return json.dumps(self.store, indent=2)


# =========================
# TOOL SYSTEM
# =========================
class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def run(self, name: str, *args, **kwargs):
        if name not in self.tools:
            raise Exception(f"Tool '{name}' not found")
        return self.tools[name].func(*args, **kwargs)

    def list_tools(self):
        return {name: tool.description for name, tool in self.tools.items()}


# =========================
# PLANNER (SIMPLE RULE-BASED)
# =========================
class Planner:
    def decide(self, goal: str, memory: Memory, tools: ToolRegistry):
        """
        Very simple planner:
        - If keyword matches tool, use tool
        - Else respond normally
        """
        goal_lower = goal.lower()

        for tool_name in tools.tools:
            if tool_name in goal_lower:
                return {
                    "action": "tool",
                    "tool": tool_name,
                    "input": goal
                }

        return {
            "action": "respond",
            "input": goal
        }


# =========================
# AGENT CORE
# =========================
class Agent:
    def __init__(self, name: str):
        self.name = name
        self.memory = Memory()
        self.tools = ToolRegistry()
        self.planner = Planner()
        self.running = False

    def add_tool(self, tool: Tool):
        self.tools.register(tool)

    def think(self, goal: str):
        return self.planner.decide(goal, self.memory, self.tools)

    def act(self, decision: Dict[str, Any]):
        if decision["action"] == "tool":
            result = self.tools.run(decision["tool"], decision["input"])
            self.memory.add("tool_result", str(result))
            return result

        # default response behavior
        response = f"I processed: {decision['input']}"
        self.memory.add("assistant", response)
        return response

    def run(self, goal: str, steps: int = 5):
        self.running = True
        current_goal = goal

        for i in range(steps):
            decision = self.think(current_goal)
            result = self.act(decision)

            print(f"[Step {i+1}] {result}")

            # simple loop evolution
            current_goal = f"Continue: {current_goal}"

            time.sleep(0.2)

        self.running = False


# =========================
# EXAMPLE TOOLS
# =========================
def calculator_tool(input_text: str):
    try:
        expr = input_text.replace("calculator", "").strip()
        return eval(expr)
    except:
        return "Invalid expression"


def web_search_tool(input_text: str):
    return f"Simulated search results for: {input_text}"


def file_writer_tool(input_text: str):
    filename = "output.txt"
    with open(filename, "a") as f:
        f.write(input_text + "\n")
    return f"Wrote to {filename}"


# =========================
# RUN EXAMPLE
# =========================
if __name__ == "__main__":
    agent = Agent("NeoAgent")

    # register tools
    agent.add_tool(Tool("calculator", calculator_tool, "Evaluates math expressions"))
    agent.add_tool(Tool("search", web_search_tool, "Simulates web search"))
    agent.add_tool(Tool("write", file_writer_tool, "Writes text to file"))

    # run agent
    agent.run("calculator 5 + 7 * 2", steps=3)
