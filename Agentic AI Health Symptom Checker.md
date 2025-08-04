![image](https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/notebooks/headers/watsonx-Prompt_Lab-Notebook.png)
# AI Service Deployment Notebook
This notebook contains steps and code to test, promote, and deploy an Agent as an AI Service.

**Note:** Notebook code generated using Agent Lab will execute successfully.
If code is modified or reordered, there is no guarantee it will successfully execute.
For details, see: <a href="/docs/content/wsj/analyze-data/fm-prompt-save.html?context=wx" target="_blank">Saving your work in Agent Lab as a notebook.</a>


Some familiarity with Python is helpful. This notebook uses Python 3.11.

## Contents
This notebook contains the following parts:

1. Setup
2. Initialize all the variables needed by the AI Service
3. Define the AI service function
4. Deploy an AI Service
5. Test the deployed AI Service

## 1. Set up the environment

Before you can run this notebook, you must perform the following setup tasks:

### Connection to WML
This cell defines the credentials required to work with watsonx API for both the execution in the project, 
as well as the deployment and runtime execution of the function.

**Action:** Provide the IBM Cloud personal API key. For details, see
<a href="https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui" target="_blank">documentation</a>.



```python
import os
from ibm_watsonx_ai import APIClient, Credentials
import getpass

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=getpass.getpass("Please enter your api key (hit enter): ")
)


```

    Please enter your api key (hit enter):  ········



```python
# Define the request and response schemas for the AI service
request_schema = {
    "application/json": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "messages": {
                "title": "The messages for this chat session.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "title": "The role of the message author.",
                            "type": "string",
                            "enum": ["user","assistant"]
                        },
                        "content": {
                            "title": "The contents of the message.",
                            "type": "string"
                        }
                    },
                    "required": ["role","content"]
                }
            }
        },
        "required": ["messages"]
    }
}

response_schema = {
    "application/json": {
        "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
    }
}
```

### Connecting to a space
A space will be be used to host the promoted AI Service.



```python
space_id = "47d05434-ab7b-4838-8163-e80e6ded0c19"
client.set.default_space(space_id)

```

### Promote asset(s) to space
We will now promote assets we will need to stage in the space so that we can access their data from the AI service.



```python
source_project_id = "3558480f-f5a0-4ef2-ae91-e11cd9bb4007"

```

## 2. Create the AI service function
We first need to define the AI service function

### 2.1 Define the function


```python
params = {
    "space_id": space_id,
}

def gen_ai_service(context, params = params, **custom):
    # import dependencies
    from langchain_ibm import ChatWatsonx
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    import json
    import requests

    model = "meta-llama/llama-3-3-70b-instruct"
    
    service_url = "https://us-south.ml.cloud.ibm.com"
    # Get credentials token
    credentials = {
        "url": service_url,
        "token": context.generate_token()
    }

    # Setup client
    client = APIClient(credentials)
    space_id = params.get("space_id")
    client.set.default_space(space_id)



    def create_chat_model(watsonx_client):
        parameters = {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 1
        }

        chat_model = ChatWatsonx(
            model_id=model,
            url=service_url,
            space_id=space_id,
            params=parameters,
            watsonx_client=watsonx_client,
        )
        return chat_model
    
    
    def create_utility_agent_tool(tool_name, params, api_client, **kwargs):
        from langchain_core.tools import StructuredTool
        utility_agent_tool = Toolkit(
            api_client=api_client
        ).get_tool(tool_name)
    
        tool_description = utility_agent_tool.get("description")
    
        if (kwargs.get("tool_description")):
            tool_description = kwargs.get("tool_description")
        elif (utility_agent_tool.get("agent_description")):
            tool_description = utility_agent_tool.get("agent_description")
        
        tool_schema = utility_agent_tool.get("input_schema")
        if (tool_schema == None):
            tool_schema = {
                "type": "object",
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
                "properties": {
                    "input": {
                        "description": "input for the tool",
                        "type": "string"
                    }
                }
            }
        
        def run_tool(**tool_input):
            query = tool_input
            if (utility_agent_tool.get("input_schema") == None):
                query = tool_input.get("input")
    
            results = utility_agent_tool.run(
                input=query,
                config=params
            )
            
            return results.get("output")
        
        return StructuredTool(
            name=tool_name,
            description = tool_description,
            func=run_tool,
            args_schema=tool_schema
        )
    
    
    def create_custom_tool(tool_name, tool_description, tool_code, tool_schema, tool_params):
        from langchain_core.tools import StructuredTool
        import ast
    
        def call_tool(**kwargs):
            tree = ast.parse(tool_code, mode="exec")
            custom_tool_functions = [ x for x in tree.body if isinstance(x, ast.FunctionDef) ]
            function_name = custom_tool_functions[0].name
            compiled_code = compile(tree, 'custom_tool', 'exec')
            namespace = tool_params if tool_params else {}
            exec(compiled_code, namespace)
            return namespace[function_name](**kwargs)
            
        tool = StructuredTool(
            name=tool_name,
            description = tool_description,
            func=call_tool,
            args_schema=tool_schema
        )
        return tool
    
    def create_custom_tools():
        custom_tools = []
    

    def create_tools(inner_client, context):
        tools = []
        
        config = None
        tools.append(create_utility_agent_tool("GoogleSearch", config, inner_client))
        return tools
    
    def create_agent(model, tools, messages):
        memory = MemorySaver()
        instructions = """# Notes
- Use markdown syntax for formatting code snippets, links, JSON, tables, images, files.
- Any HTML tags must be wrapped in block quotes, for example ```<html>```.
- When returning code blocks, specify language.
- Sometimes, things don't go as planned. Tools may not provide useful information on the first few tries. You should always try a few different approaches before declaring the problem unsolvable.
- When the tool doesn't give you what you were asking for, you must either use another tool or a different tool input.
- When using search engines, you try different formulations of the query, possibly even in a different language.
- You cannot do complex calculations, computations, or data manipulations without using tools.
- If you need to call a tool to compute something, always call it instead of saying you will call it.

If a tool returns an IMAGE in the result, you must include it in your answer as Markdown.

Example:

Tool result: IMAGE({commonApiUrl}/wx/v1-beta/utility_agent_tools/cache/images/plt-04e3c91ae04b47f8934a4e6b7d1fdc2c.png)
Markdown to return to user: ![Generated image]({commonApiUrl}/wx/v1-beta/utility_agent_tools/cache/images/plt-04e3c91ae04b47f8934a4e6b7d1fdc2c.png)

You are a helpful assistant that uses tools to answer questions in detail.
When greeted, say \"Hi, I am watsonx.ai agent. How can I help you?\""""
        for message in messages:
            if message["role"] == "system":
                instructions += message["content"]
        graph = create_react_agent(model, tools=tools, checkpointer=memory, state_modifier=instructions)
        return graph
    
    def convert_messages(messages):
        converted_messages = []
        for message in messages:
            if (message["role"] == "user"):
                converted_messages.append(HumanMessage(content=message["content"]))
            elif (message["role"] == "assistant"):
                converted_messages.append(AIMessage(content=message["content"]))
        return converted_messages

    def generate(context):
        payload = context.get_json()
        messages = payload.get("messages")
        inner_credentials = {
            "url": service_url,
            "token": context.get_token()
        }

        inner_client = APIClient(inner_credentials)
        model = create_chat_model(inner_client)
        tools = create_tools(inner_client, context)
        agent = create_agent(model, tools, messages)
        
        generated_response = agent.invoke(
            { "messages": convert_messages(messages) },
            { "configurable": { "thread_id": "42" } }
        )

        last_message = generated_response["messages"][-1]
        generated_response = last_message.content

        execute_response = {
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "choices": [{
                    "index": 0,
                    "message": {
                       "role": "assistant",
                       "content": generated_response
                    }
                }]
            }
        }

        return execute_response

    def generate_stream(context):
        print("Generate stream", flush=True)
        payload = context.get_json()
        headers = context.get_headers()
        is_assistant = headers.get("X-Ai-Interface") == "assistant"
        messages = payload.get("messages")
        inner_credentials = {
            "url": service_url,
            "token": context.get_token()
        }
        inner_client = APIClient(inner_credentials)
        model = create_chat_model(inner_client)
        tools = create_tools(inner_client, context)
        agent = create_agent(model, tools, messages)

        response_stream = agent.stream(
            { "messages": messages },
            { "configurable": { "thread_id": "42" } },
            stream_mode=["updates", "messages"]
        )

        for chunk in response_stream:
            chunk_type = chunk[0]
            finish_reason = ""
            usage = None
            if (chunk_type == "messages"):
                message_object = chunk[1][0]
                if (message_object.type == "AIMessageChunk" and message_object.content != ""):
                    message = {
                        "role": "assistant",
                        "content": message_object.content
                    }
                else:
                    continue
            elif (chunk_type == "updates"):
                update = chunk[1]
                if ("agent" in update):
                    agent = update["agent"]
                    agent_result = agent["messages"][0]
                    if (agent_result.additional_kwargs):
                        kwargs = agent["messages"][0].additional_kwargs
                        tool_call = kwargs["tool_calls"][0]
                        if (is_assistant):
                            message = {
                                "role": "assistant",
                                "step_details": {
                                    "type": "tool_calls",
                                    "tool_calls": [
                                        {
                                            "id": tool_call["id"],
                                            "name": tool_call["function"]["name"],
                                            "args": tool_call["function"]["arguments"]
                                        }
                                    ] 
                                }
                            }
                        else:
                            message = {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool_call["function"]["name"],
                                            "arguments": tool_call["function"]["arguments"]
                                        }
                                    }
                                ]
                            }
                    elif (agent_result.response_metadata):
                        # Final update
                        message = {
                            "role": "assistant",
                            "content": agent_result.content
                        }
                        finish_reason = agent_result.response_metadata["finish_reason"]
                        if (finish_reason): 
                            message["content"] = ""

                        usage = {
                            "completion_tokens": agent_result.usage_metadata["output_tokens"],
                            "prompt_tokens": agent_result.usage_metadata["input_tokens"],
                            "total_tokens": agent_result.usage_metadata["total_tokens"]
                        }
                elif ("tools" in update):
                    tools = update["tools"]
                    tool_result = tools["messages"][0]
                    if (is_assistant):
                        message = {
                            "role": "assistant",
                            "step_details": {
                                "type": "tool_response",
                                "id": tool_result.id,
                                "tool_call_id": tool_result.tool_call_id,
                                "name": tool_result.name,
                                "content": tool_result.content
                            }
                        }
                    else:
                        message = {
                            "role": "tool",
                            "id": tool_result.id,
                            "tool_call_id": tool_result.tool_call_id,
                            "name": tool_result.name,
                            "content": tool_result.content
                        }
                else:
                    continue

            chunk_response = {
                "choices": [{
                    "index": 0,
                    "delta": message
                }]
            }
            if (finish_reason):
                chunk_response["choices"][0]["finish_reason"] = finish_reason
            if (usage):
                chunk_response["usage"] = usage
            yield chunk_response

    return generate, generate_stream

```

### 2.2 Test locally


```python
# Define the request and response schemas for the AI service
request_schema = {
    "application/json": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "messages": {
                "title": "The messages for this chat session.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "title": "The role of the message author.",
                            "type": "string",
                            "enum": ["user","assistant"]
                        },
                        "content": {
                            "title": "The contents of the message.",
                            "type": "string"
                        }
                    },
                    "required": ["role","content"]
                }
            }
        },
        "required": ["messages"]
    }
}

response_schema = {
    "application/json": {
        "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
    }
}
```


```python
# Define the request and response schemas for the AI service
request_schema = {
    "application/json": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "messages": {
                "title": "The messages for this chat session.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "title": "The role of the message author.",
                            "type": "string",
                            "enum": ["user","assistant"]
                        },
                        "content": {
                            "title": "The contents of the message.",
                            "type": "string"
                        }
                    },
                    "required": ["role","content"]
                }
            }
        },
        "required": ["messages"]
    }
}

response_schema = {
    "application/json": {
        "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
    }
}

```

## 3. Store and deploy the AI Service
Before you can deploy the AI Service, you must store the AI service in your watsonx.ai repository.


```python
# Define the request and response schemas for the AI service
request_schema = {
    "application/json": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "messages": {
                "title": "The messages for this chat session.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "title": "The role of the message author.",
                            "type": "string",
                            "enum": ["user","assistant"]
                        },
                        "content": {
                            "title": "The contents of the message.",
                            "type": "string"
                        }
                    },
                    "required": ["role","content"]
                }
            }
        },
        "required": ["messages"]
    }
}

response_schema = {
    "application/json": {
        "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
    }
}
```


```python
# Define the request and response schemas for the AI service
request_schema = {
    "application/json": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "messages": {
                "title": "The messages for this chat session.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "title": "The role of the message author.",
                            "type": "string",
                            "enum": ["user","assistant"]
                        },
                        "content": {
                            "title": "The contents of the message.",
                            "type": "string"
                        }
                    },
                    "required": ["role","content"]
                }
            }
        },
        "required": ["messages"]
    }
}

response_schema = {
    "application/json": {
        "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
    }
}
```


```python
# symptom_checker.py

# Simulated database (in real case, fetch from WHO/CDC/MedlinePlus)
symptom_condition_map = {
    "fever": ["Common Cold", "Flu", "COVID-19"],
    "cough": ["Common Cold", "Bronchitis", "COVID-19"],
    "sore throat": ["Strep Throat", "Pharyngitis", "Cold"],
    "headache": ["Migraine", "Tension Headache", "Flu"],
    "fatigue": ["Anemia", "Thyroid Issues", "Flu"],
    "shortness of breath": ["Asthma", "COVID-19", "Heart Problems"]
}

home_remedies = {
    "fever": ["Stay hydrated", "Take rest", "Use a cold compress"],
    "cough": ["Honey with warm water", "Steam inhalation", "Avoid cold drinks"],
    "sore throat": ["Gargle with salt water", "Drink warm fluids", "Use lozenges"],
    "headache": ["Apply cold pack", "Rest in dark room", "Stay hydrated"],
    "fatigue": ["Get adequate sleep", "Eat iron-rich foods", "Reduce stress"],
    "shortness of breath": ["Use inhaler if prescribed", "Sit upright", "Seek medical help if severe"]
}

def analyze_symptoms(user_input):
    matched_symptoms = []
    possible_conditions = set()
    remedy_list = []

    for symptom in symptom_condition_map:
        if symptom in user_input.lower():
            matched_symptoms.append(symptom)
            possible_conditions.update(symptom_condition_map[symptom])
            remedy_list.extend(home_remedies.get(symptom, []))

    urgency_level = "Moderate"
    if "shortness of breath" in matched_symptoms or "fever" in matched_symptoms and "cough" in matched_symptoms:
        urgency_level = "High"

    if not matched_symptoms:
        return {
            "error": "Could not detect symptoms. Please try rephrasing your input."
        }

    return {
        "matched_symptoms": matched_symptoms,
        "possible_conditions": list(possible_conditions),
        "urgency_level": urgency_level,
        "home_remedies": list(set(remedy_list)),
        "recommendation": "Please consult a doctor if symptoms persist or worsen.",
        "disclaimer": "This is not a medical diagnosis. For accurate assessment, consult a healthcare provider."
    }

# CLI Interface
if __name__ == "__main__":
    print("🩺 Agentic AI Health Symptom Checker (CLI Prototype)")
    user_input = input("Enter your symptoms: ")

    result = analyze_symptoms(user_input)

    if "error" in result:
        print("❗", result["error"])
    else:
        print("\n✅ Detected Symptoms:", ", ".join(result["matched_symptoms"]))
        print("🧾 Possible Conditions:", ", ".join(result["possible_conditions"]))
        print("⚠️ Urgency Level:", result["urgency_level"])
        print("🏡 Home Remedies:")
        for remedy in result["home_remedies"]:
            print("  -", remedy)
        print("💡 Recommendation:", result["recommendation"])
        print("🔒 Disclaimer:", result["disclaimer"])

```

    🩺 Agentic AI Health Symptom Checker (CLI Prototype)


    Enter your symptoms:  fatigue


    
    ✅ Detected Symptoms: fatigue
    🧾 Possible Conditions: Thyroid Issues, Flu, Anemia
    ⚠️ Urgency Level: Moderate
    🏡 Home Remedies:
      - Get adequate sleep
      - Eat iron-rich foods
      - Reduce stress
    💡 Recommendation: Please consult a doctor if symptoms persist or worsen.
    🔒 Disclaimer: This is not a medical diagnosis. For accurate assessment, consult a healthcare provider.



```python
# symptom_checker.py

# Simulated database (in real case, fetch from WHO/CDC/MedlinePlus)
symptom_condition_map = {
    "fever": ["Common Cold", "Flu", "COVID-19"],
    "cough": ["Common Cold", "Bronchitis", "COVID-19"],
    "sore throat": ["Strep Throat", "Pharyngitis", "Cold"],
    "headache": ["Migraine", "Tension Headache", "Flu"],
    "fatigue": ["Anemia", "Thyroid Issues", "Flu"],
    "shortness of breath": ["Asthma", "COVID-19", "Heart Problems"]
}

home_remedies = {
    "fever": ["Stay hydrated", "Take rest", "Use a cold compress"],
    "cough": ["Honey with warm water", "Steam inhalation", "Avoid cold drinks"],
    "sore throat": ["Gargle with salt water", "Drink warm fluids", "Use lozenges"],
    "headache": ["Apply cold pack", "Rest in dark room", "Stay hydrated"],
    "fatigue": ["Get adequate sleep", "Eat iron-rich foods", "Reduce stress"],
    "shortness of breath": ["Use inhaler if prescribed", "Sit upright", "Seek medical help if severe"]
}

def analyze_symptoms(user_input):
    matched_symptoms = []
    possible_conditions = set()
    remedy_list = []

    for symptom in symptom_condition_map:
        if symptom in user_input.lower():
            matched_symptoms.append(symptom)
            possible_conditions.update(symptom_condition_map[symptom])
            remedy_list.extend(home_remedies.get(symptom, []))

    urgency_level = "Moderate"
    if "shortness of breath" in matched_symptoms or "fever" in matched_symptoms and "cough" in matched_symptoms:
        urgency_level = "High"

    if not matched_symptoms:
        return {
            "error": "Could not detect symptoms. Please try rephrasing your input."
        }

    return {
        "matched_symptoms": matched_symptoms,
        "possible_conditions": list(possible_conditions),
        "urgency_level": urgency_level,
        "home_remedies": list(set(remedy_list)),
        "recommendation": "Please consult a doctor if symptoms persist or worsen.",
        "disclaimer": "This is not a medical diagnosis. For accurate assessment, consult a healthcare provider."
    }

# CLI Interface
if __name__ == "__main__":
    print("🩺 Agentic AI Health Symptom Checker (CLI Prototype)")
    user_input = input("Enter your symptoms: ")

    result = analyze_symptoms(user_input)

    if "error" in result:
        print("❗", result["error"])
    else:
        print("\n✅ Detected Symptoms:", ", ".join(result["matched_symptoms"]))
        print("🧾 Possible Conditions:", ", ".join(result["possible_conditions"]))
        print("⚠️ Urgency Level:", result["urgency_level"])
        print("🏡 Home Remedies:")
        for remedy in result["home_remedies"]:
            print("  -", remedy)
        print("💡 Recommendation:", result["recommendation"])
        print("🔒 Disclaimer:", result["disclaimer"])

```

    🩺 Agentic AI Health Symptom Checker (CLI Prototype)


    Enter your symptoms:  cough


    
    ✅ Detected Symptoms: cough
    🧾 Possible Conditions: COVID-19, Bronchitis, Common Cold
    ⚠️ Urgency Level: Moderate
    🏡 Home Remedies:
      - Avoid cold drinks
      - Honey with warm water
      - Steam inhalation
    💡 Recommendation: Please consult a doctor if symptoms persist or worsen.
    🔒 Disclaimer: This is not a medical diagnosis. For accurate assessment, consult a healthcare provider.



```python
# symptom_checker.py

# Simulated database (in real case, fetch from WHO/CDC/MedlinePlus)
symptom_condition_map = {
    "fever": ["Common Cold", "Flu", "COVID-19"],
    "cough": ["Common Cold", "Bronchitis", "COVID-19"],
    "sore throat": ["Strep Throat", "Pharyngitis", "Cold"],
    "headache": ["Migraine", "Tension Headache", "Flu"],
    "fatigue": ["Anemia", "Thyroid Issues", "Flu"],
    "shortness of breath": ["Asthma", "COVID-19", "Heart Problems"]
}

home_remedies = {
    "fever": ["Stay hydrated", "Take rest", "Use a cold compress"],
    "cough": ["Honey with warm water", "Steam inhalation", "Avoid cold drinks"],
    "sore throat": ["Gargle with salt water", "Drink warm fluids", "Use lozenges"],
    "headache": ["Apply cold pack", "Rest in dark room", "Stay hydrated"],
    "fatigue": ["Get adequate sleep", "Eat iron-rich foods", "Reduce stress"],
    "shortness of breath": ["Use inhaler if prescribed", "Sit upright", "Seek medical help if severe"]
}

def analyze_symptoms(user_input):
    matched_symptoms = []
    possible_conditions = set()
    remedy_list = []

    for symptom in symptom_condition_map:
        if symptom in user_input.lower():
            matched_symptoms.append(symptom)
            possible_conditions.update(symptom_condition_map[symptom])
            remedy_list.extend(home_remedies.get(symptom, []))

    urgency_level = "Moderate"
    if "shortness of breath" in matched_symptoms or "fever" in matched_symptoms and "cough" in matched_symptoms:
        urgency_level = "High"

    if not matched_symptoms:
        return {
            "error": "Could not detect symptoms. Please try rephrasing your input."
        }

    return {
        "matched_symptoms": matched_symptoms,
        "possible_conditions": list(possible_conditions),
        "urgency_level": urgency_level,
        "home_remedies": list(set(remedy_list)),
        "recommendation": "Please consult a doctor if symptoms persist or worsen.",
        "disclaimer": "This is not a medical diagnosis. For accurate assessment, consult a healthcare provider."
    }

# CLI Interface
if __name__ == "__main__":
    print("🩺 Agentic AI Health Symptom Checker (CLI Prototype)")
    user_input = input("Enter your symptoms: ")

    result = analyze_symptoms(user_input)

    if "error" in result:
        print("❗", result["error"])
    else:
        print("\n✅ Detected Symptoms:", ", ".join(result["matched_symptoms"]))
        print("🧾 Possible Conditions:", ", ".join(result["possible_conditions"]))
        print("⚠️ Urgency Level:", result["urgency_level"])
        print("🏡 Home Remedies:")
        for remedy in result["home_remedies"]:
            print("  -", remedy)
        print("💡 Recommendation:", result["recommendation"])
        print("🔒 Disclaimer:", result["disclaimer"])


```

    🩺 Agentic AI Health Symptom Checker (CLI Prototype)


    Enter your symptoms:  headache


    
    ✅ Detected Symptoms: headache
    🧾 Possible Conditions: Flu, Migraine, Tension Headache
    ⚠️ Urgency Level: Moderate
    🏡 Home Remedies:
      - Stay hydrated
      - Rest in dark room
      - Apply cold pack
    💡 Recommendation: Please consult a doctor if symptoms persist or worsen.
    🔒 Disclaimer: This is not a medical diagnosis. For accurate assessment, consult a healthcare provider.


## 4. Test AI Service


```python
# symptom_checker.py

# Simulated database (in real case, fetch from WHO/CDC/MedlinePlus)
symptom_condition_map = {
    "fever": ["Common Cold", "Flu", "COVID-19"],
    "cough": ["Common Cold", "Bronchitis", "COVID-19"],
    "sore throat": ["Strep Throat", "Pharyngitis", "Cold"],
    "headache": ["Migraine", "Tension Headache", "Flu"],
    "fatigue": ["Anemia", "Thyroid Issues", "Flu"],
    "shortness of breath": ["Asthma", "COVID-19", "Heart Problems"]
}

home_remedies = {
    "fever": ["Stay hydrated", "Take rest", "Use a cold compress"],
    "cough": ["Honey with warm water", "Steam inhalation", "Avoid cold drinks"],
    "sore throat": ["Gargle with salt water", "Drink warm fluids", "Use lozenges"],
    "headache": ["Apply cold pack", "Rest in dark room", "Stay hydrated"],
    "fatigue": ["Get adequate sleep", "Eat iron-rich foods", "Reduce stress"],
    "shortness of breath": ["Use inhaler if prescribed", "Sit upright", "Seek medical help if severe"]
}

def analyze_symptoms(user_input):
    matched_symptoms = []
    possible_conditions = set()
    remedy_list = []

    for symptom in symptom_condition_map:
        if symptom in user_input.lower():
            matched_symptoms.append(symptom)
            possible_conditions.update(symptom_condition_map[symptom])
            remedy_list.extend(home_remedies.get(symptom, []))

    urgency_level = "Moderate"
    if "shortness of breath" in matched_symptoms or "fever" in matched_symptoms and "cough" in matched_symptoms:
        urgency_level = "High"

    if not matched_symptoms:
        return {
            "error": "Could not detect symptoms. Please try rephrasing your input."
        }

    return {
        "matched_symptoms": matched_symptoms,
        "possible_conditions": list(possible_conditions),
        "urgency_level": urgency_level,
        "home_remedies": list(set(remedy_list)),
        "recommendation": "Please consult a doctor if symptoms persist or worsen.",
        "disclaimer": "This is not a medical diagnosis. For accurate assessment, consult a healthcare provider."
    }

# CLI Interface
if __name__ == "__main__":
    print("🩺 Agentic AI Health Symptom Checker (CLI Prototype)")
    user_input = input("Enter your symptoms: ")

    result = analyze_symptoms(user_input)

    if "error" in result:
        print("❗", result["error"])
    else:
        print("\n✅ Detected Symptoms:", ", ".join(result["matched_symptoms"]))
        print("🧾 Possible Conditions:", ", ".join(result["possible_conditions"]))
        print("⚠️ Urgency Level:", result["urgency_level"])
        print("🏡 Home Remedies:")
        for remedy in result["home_remedies"]:
            print("  -", remedy)
        print("💡 Recommendation:", result["recommendation"])
        print("🔒 Disclaimer:", result["disclaimer"])

```

    🩺 Agentic AI Health Symptom Checker (CLI Prototype)


    Enter your symptoms:  fever


    
    ✅ Detected Symptoms: fever
    🧾 Possible Conditions: COVID-19, Flu, Common Cold
    ⚠️ Urgency Level: Moderate
    🏡 Home Remedies:
      - Take rest
      - Stay hydrated
      - Use a cold compress
    💡 Recommendation: Please consult a doctor if symptoms persist or worsen.
    🔒 Disclaimer: This is not a medical diagnosis. For accurate assessment, consult a healthcare provider.



```python
messages = []
remote_question = "Change this question to test your function"
messages.append({ "role" : "user", "content": remote_question })
payload = { "messages": messages }
```


```python
print(result)
```

    {'matched_symptoms': ['cough'], 'possible_conditions': ['COVID-19', 'Bronchitis', 'Common Cold'], 'urgency_level': 'Moderate', 'home_remedies': ['Avoid cold drinks', 'Honey with warm water', 'Steam inhalation'], 'recommendation': 'Please consult a doctor if symptoms persist or worsen.', 'disclaimer': 'This is not a medical diagnosis. For accurate assessment, consult a healthcare provider.'}


# Next steps
You successfully deployed and tested the AI Service! You can now view
your deployment and test it as a REST API endpoint.

<a id="copyrights"></a>
### Copyrights

Licensed Materials - Copyright © 2024 IBM. This notebook and its source code are released under the terms of the ILAN License.
Use, duplication disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

**Note:** The auto-generated notebooks are subject to the International License Agreement for Non-Warranted Programs (or equivalent) and License Information document for watsonx.ai Auto-generated Notebook (License Terms), such agreements located in the link below. Specifically, the Source Components and Sample Materials clause included in the License Information document for watsonx.ai Studio Auto-generated Notebook applies to the auto-generated notebooks.  

By downloading, copying, accessing, or otherwise using the materials, you agree to the <a href="https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-AMCU-BYC7LF" target="_blank">License Terms</a>  
