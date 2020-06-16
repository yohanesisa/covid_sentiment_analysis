# import tweepy

# auth = tweepy.AppAuthHandler('brYXGvHTSesmDSzRtLYR2wMvT', '125Kr7B7gQJtyvlBfOTGAk1HWDp0UUKL9OQFYKoSdJmW3Nq9lx')

# api = tweepy.API(auth)

# status = api.get_status(1220944131444203520, tweet_mode="extended")

# try:
#     print(status.retweeted_status.full_text)
# except AttributeError:  # Not a Retweet
#     print(status.full_text)


import os

kernel = 'linear'

os.system('say "model '+kernel+' generated"')