import re
from smolagents import CodeAgent, PromptTemplates

_RAW_SYS_PROMPT = """You are a Python expert.
TASK: {{ task }}
RULES:
1. Write ONLY valid Python code. No markdown.
2. Imports: {{ authorized_imports }}
3. Tools:
{{code_block_opening_tag}}
{%- for tool in tools.values() %}
{{ tool.to_code_prompt() }}
{% endfor %}
{{code_block_closing_tag}}
4. Use print() to inspect variables.
5. You MUST call final_answer(result) to finish.
"""

_RAW_TEMPLATES = PromptTemplates(
    system_prompt=_RAW_SYS_PROMPT,
    planning={"initial_plan": "", "update_plan_pre_messages": "", "update_plan_post_messages": ""},
    managed_agent={"task": "{{ task }}", "report": "{{ final_answer }}"},
    final_answer={"pre_messages": "", "post_messages": ""}
)

class _PythonInjector:
    def __init__(self, model):
        self.model = model
        self.trigger = re.compile(r'(print|final_answer)\s*\(')

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, messages, stop_sequences=None, **kwargs):
        resp = self.model.generate(messages, stop_sequences, **kwargs)
        content = resp.content or ""
        if m := self.trigger.search(content):
            end_idx = content.find('\n', m.end())
            content = content[:end_idx] if end_idx != -1 else content
        resp.content = f"```python\n{content.strip()}\n```"
        return resp

    def generate_stream(self, messages, stop_sequences=None, **kwargs):
        yield "```python\n"
        buf, closed = "", False
        
        for delta in self.model.generate_stream(messages, stop_sequences, **kwargs):
            token = delta.content if hasattr(delta, 'content') else str(delta)
            if not token: continue
            
            buf += token
            yield delta
            
            # Stop early if trigger detected and parentheses balanced
            if self.trigger.search(buf) and (token == ')' or token == '\n'):
                if buf.count('(') == buf.count(')'):
                    yield "\n```"
                    closed = True
                    break
        
        # Ensure block is closed if model stops naturally (EOS)
        if not closed:
            yield "\n```"

class PythonAgent(CodeAgent):
    def __init__(self, model, **kwargs):
        kwargs['model'] = _PythonInjector(model)
        kwargs['prompt_templates'] = _RAW_TEMPLATES
        kwargs['code_block_tags'] = ("```python", "```")
        super().__init__(**kwargs)

if __name__ == "__main__":
    from smolagents import OpenAIModel
    
    model = OpenAIModel(
        model_id="qwen3:30b-a3b-instruct-16k", 
        api_base="http://localhost:11434/v1",
        api_key="na"
    )
    
    agent = PythonAgent(model=model)
    agent.run("Start by defining x=10. Then stop. Then in next step print x.")