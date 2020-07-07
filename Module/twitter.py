import tweepy

twitterAuth = tweepy.OAuthHandler('brYXGvHTSesmDSzRtLYR2wMvT', '125Kr7B7gQJtyvlBfOTGAk1HWDp0UUKL9OQFYKoSdJmW3Nq9lx')
twitterAuth.set_access_token('170599223-NC3EvlFxt5bvR7LqY8cQLlIz98qzuuGxKVtsKPIb', 'xwO5ZLya6ZumgxxMjOkXit8gHJeAjRg65OjzQboRZjcPI')
twitterAPI = tweepy.API(twitterAuth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def retrieveTweets(raw):
    id_dict = raw.keys()
    result_full = []

    try:
        for i in range((len(id_dict) / 100) + 1):
            end_loc = min((i + 1) * 100, len(id_dict))
            result = twitterAPI.statuses_lookup(id_=id_dict[i * 100:end_loc], tweet_mode="extended")
            result_full.extend(result)
        
        for tweet in result_full:
            raw[str(tweet.id)].setSentence(getText(tweet))

    except (tweepy.TweepError, tweepy.RateLimitError) as error:
        print error

def getText(tweet):
    try:
        return tweet.retweeted_status.full_text 
    except AttributeError:  # Not a Retweet
        return tweet.full_text 


        