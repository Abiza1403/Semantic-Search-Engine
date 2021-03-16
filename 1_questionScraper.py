import json
import scrapy
from scrapy.crawler import CrawlerProcess
import csv

class Questionscraper(scrapy.Spider):
    name = 'questionscraper'
    base_url = 'https://ask.learncbse.in/latest.json?no_definitions=true&page='    
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'}
    
    def __init__(self):
        with open('question.csv', 'w') as csv_file:
            csv_file.write('id,title,question,slug,category,tags,views\n')
      
    def start_requests(self):
        # scrape data from infinite scroll
        for page in range(0, 1643):    # specify page range you would like to scrape data for
            next_page = self.base_url + str(page)
            yield scrapy.Request(url=next_page, headers=self.headers, callback=self.parse)
    
    def parse(self, res):
        data = ''
        with open('res.json', 'w') as json_file:
            json_file.write(res.text)
        with open('res.json', 'r') as json_file:
            for line in json_file.read():
                data += line
        data = json.loads(data)
        
        # data extraction logic
        for topics in data['topic_list']['topics']:
            items = {
                'id': topics['id'],
                'title': topics['title'],
                'question': topics['fancy_title'],
                'slug': topics['slug'],
                'category': topics['category_id'],
                'tags': topics['tags'],
                'views': topics['views']
            }

            # append results to CSV
            with open('question.csv', 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=items.keys())
                writer.writerow(items)
            
# run scraper
process = CrawlerProcess()
process.crawl(Questionscraper)
process.start()