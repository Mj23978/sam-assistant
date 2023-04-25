# import os
# import cohere
# # from langchain import Cohere

# # model = "command-xlarge-nightly" 
# cohere_api = os.environ.get("COHERE_API_KEY")

# # llm = Cohere(
# #     cohere_api_key=cohere_api,
# #     verbose=True,
# # )

# # result = llm.generate(["Name 10 Footbal Players :"])

# # print(result)

# co = cohere.Client(f'{cohere_api}')  # This is your trial API key
# response = co.generate(
#     model='command-xlarge-nightly',
#     prompt='You are an AI assistant for writing documentation about code.\nYou are given the following extracted parts of a long code. Provide a documentation.\nProvide inputs and outputs and an example for it and short Step by Step explanation In Markdown. \ncode: \nfunction sum(a, b) {\n    return a + b;\n}\ndocumentation: \n### Function `sum`\n\n**Inputs:**\n- `a`: a number\n- `b`: a number\n\n**Outputs:**\n- The sum of `a` and `b`\n\n**Example:**\n\n```\nsum(3, 4) // returns 7\n```\n\n**Step by Step Explanation:**\n1. The `sum` function takes two numbers, `a` and `b`, as inputs. \n2. It adds `a` and `b` together.\n3. The sum of `a` and `b` is returned.\ncode: \ndef find_files(directory):\n    files_list = []\n    for root, dirs, files in os.walk(directory):\n        for file in files:\n            if file.endswith(\'.js\'):\n                files_list.append(os.path.join(root, file))\n    return files_list\ndocumentation : ',
#     max_tokens=876,
#     temperature=0.5,
#     k=0,
#     stop_sequences=[],
#     return_likelihoods='NONE')
# print('Prediction: {}'.format(response.generations[0].text))
