import requests, json
from bs4 import BeautifulSoup
from datetime import datetime

titles = []
abstracts = []
for subject in ['earth-and-environmental', 'health', 'biological', 'physical']:
    url = 'https://www.nature.com/subjects/' + subject + '-sciences/ncomms' # biological, physical
    # url = 'https://www.nature.com/subjects/biological-sciences/ncomms?searchType=journalSearch&sort=PubDate&page=2'
    # Send an HTTP GET request to the URL and get the content
    response = requests.get(url)
    html_content = response.text
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    # Find all article links by searching for 'a' tags with itemprop='url'
    article_links = soup.find_all('a', itemprop='url')
    # Extract the href attribute (URL) of each link and store in a list
    paper_links = [link['href'] for link in article_links]
    print('number of links:', len(paper_links))
    for paper_id in paper_links:
        url = 'https://www.nature.com' + paper_id
        # Send a request to the URL and get the content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract the title
        title = soup.find('h1', {'class': 'c-article-title'}).text.strip()
        # Extract the abstract (update the class name if needed after inspecting the HTML structure)
        abstract_div = soup.find('div', {'class': 'c-article-section__content'})
        abstract = abstract_div.text.strip() if abstract_div else ''
        if len( abstract ) > 1000:
            titles.append(title)
            abstracts.append(abstract)
        if len(titles) % 50 == 0:
            break

with open( 'natural_abstract.json', 'w') as f:
    json.dump({'title': titles, 'abstract': abstracts}, f)