import praw
import datetime
import random, json
from tqdm import tqdm
from collections import defaultdict

# Replace these values with your own Reddit API application credentials
client_id = '' # 'YOUR_CLIENT_ID'
client_secret = '' # 'YOUR_CLIENT_SECRET'
user_agent = '' # 'YOUR_USER_AGENT'
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

# Define the time range
start_time = datetime.datetime(2022, 1, 1)
end_time = datetime.datetime(2023, 3, 31)

# Define the domains and number of posts to scrape
domains = ['biology', 'physics', 'chemistry', 'economics', 'law', 'technology']
num_posts = 500

# Function to filter posts within the time range
def in_time_range(post, start_time, end_time):
    post_time = datetime.datetime.fromtimestamp(post.created_utc)
    return start_time <= post_time <= end_time

# Scrape questions and their longest answers
questions_and_answers = defaultdict(list)

for domain in domains:
    print(f"Scraping '{domain}' domain...")
    domain_questions = []
    count = 0
    for submission in reddit.subreddit('explainlikeimfive').search(f"flair:{domain}", sort='top', time_filter='year', limit=None):
        count += 1
        if count % 100 == 0:    
            print(f"Scraped {count} posts...")

        if in_time_range(submission, start_time, end_time):
            domain_questions.append(submission)

    random.shuffle(domain_questions)
    domain_questions = domain_questions[:num_posts]

    count2 = 0
    for question in tqdm(domain_questions):
        question.comment_sort = 'best'
        question.comments.replace_more(limit=None)

        longest_answer = None
        longest_length = 0

        for comment in question.comments.list():
            if not comment.author or comment.author == "AutoModerator" or comment.author == "[deleted]":
                continue

            comment_length = len(comment.body)
            if comment_length > 1200 and comment_length < 5000 and comment.body[:100]!='**Please read this entire message**\n\n---\n\nYour comment has been removed for the following reason(s):' and comment.body[:100] !=  '**Please read this entire message**\r\n\r\n---\r\n\r\nYour comment has been removed for the following reason':
                
                longest_answer = comment.body
                questions_and_answers[domain].append((question.title, longest_answer))
                break

print("Scraping completed.")
# Save results to a JSON file
output_filename = "reddit_ELI5.json"

with open(output_filename, "w", encoding="utf-8") as output_file:
    json.dump(questions_and_answers, output_file, ensure_ascii=False, indent=4)

print(f"Results saved to {output_filename}.")