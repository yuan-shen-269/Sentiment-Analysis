def get_usernames():
    ''' 
    Returns list of twitter handles from selected source website
    Scraping usernames from: https://rareaction.org/state-by-state-senator-governor-information/
    Website date last modified: 2017-11-15T21:06:35-05:00
    '''

    from bs4 import BeautifulSoup

    # downloaded html from website directly, as requests was returning different html page source
    with open('twitter_handles.html') as source:
        soup = BeautifulSoup(source, 'lxml')

        # decided to scrape by style declaration, as only twitter usernames were styled this way
        elems = soup.find_all(
            'td', {"style": "height: 47px; width: 148px;"})

        usernames = [elem.text for elem in elems if '@' in elem.text]

    return usernames


def scrape_tweets(username, count=5):
    '''
    Returns list of Tweet objects, scraped from input username. count variable sets maximum
    number of tweets to scrape. 
    GetOldTweets3 Documentation: https://pypi.org/project/GetOldTweets3/
    '''

    import GetOldTweets3 as got

    tweetCriteria = got.manager.TweetCriteria().setUsername(
        username).setMaxTweets(count)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    return tweets


def write_txt(username, texts):
    '''
    Creates output .txt files in separate directory. 
    Naming convention is twitter handle, minus the @ character. Distinct tweets are newline separated.
    '''

    # create directory to store output files
    import os
    dirname = 'output_files/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # naming text files by twitter handle
    filename = dirname + username[1:] + '.txt'
    with open(filename, 'w') as file:
        for text in texts:
            file.write(text)
            file.write('\n\n')


def main():
    '''
    TODO: add command line functionality to directly scrape based on CLI usernames, or --all option
    '''

    # append venv libraries to path before importing
    import os
    import sys
    sys.path.append(os.getcwd() + ('/venv/bin/'))

    # define maximum number of tweets to scrape per username
    max_count = 10

    # test scrape with given username
    username = '@GavinNewsom'
    tweets = scrape_tweets(username, max_count)
    texts = [tweet.text for tweet in tweets]
    write_txt(username, texts)

    # uncomment this to scrape all 147 governors/senators from website
    # usernames = get_usernames()
    # for username in usernames:
    #     tweets = scrape_tweets(username, max_count)
    #     texts = [tweet.text for tweet in tweets]
    #     write_txt(username, texts)


if __name__ == "__main__":
    main()
