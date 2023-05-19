# import forefront


# # create an account
# account_data = forefront.Account.signin(logging=False)

# # get a response
# response = forefront.Completion.create(
#     account_data=account_data,
#     prompt='tell me 10 footbal players',
#     model='gpt-3.5'
# )
# print(response)

# import you

# response = you.Completion.create(
#     prompt="tell me 10 footbal players",
#     detailed=True,
#     include_links=True, )

# print(response.dict())

import usesless

message_id = ""
while True:
    prompt = input("Question: ")
    if prompt == "!stop":
        break

    req = usesless.Completion.create(prompt=prompt, parentMessageId=message_id)

    print(f"Answer: {req['text']}")
    message_id = req["id"]

# import theb

# for token in theb.Completion.create('hello world'):
#     print(token, end='', flush=True)
# print("")


# from quora import Poe

# poe = Poe(model='ChatGPT', driver='firefox', cookie_path='cookie.json')
# poe.chat('who won the football world cup most?')


# import cocalc

# res = cocalc.Completion.create(prompt="How are you!", cookieInput="cookieinput") ## Tutorial 
# print(res)