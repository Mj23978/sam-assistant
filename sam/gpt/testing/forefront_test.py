import forefront

# create an account
token = forefront.Account.create(logging=True)
print(token)

# get a response
for response in forefront.StreamingCompletion.create(token=token,
                                                     prompt='hello world', model='gpt-4'):
    print(response.completion.choices[0].text, end='')
