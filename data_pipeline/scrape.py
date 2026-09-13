import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "http://books.toscrape.com/"
CATALOGUE_URL = "http://books.toscrape.com/catalogue/"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_category_links(min_categories=3):
    """Get links to at least `min_categories` book category pages."""
    resp = requests.get(BASE_URL)
    soup = BeautifulSoup(resp.text, "html.parser")
    category_tags = soup.select("div.side_categories ul li ul li a")
    links = []
    for tag in category_tags:
        href = tag["href"]
        full_url = BASE_URL + href
        name = tag.text.strip()
        links.append((name, full_url))
    return links[:min_categories] if min_categories else links


def scrape_category(category_name, category_url):
    """Scrape all books from a single category, following pagination."""
    books = []
    url = category_url
    while url:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("article.product_pod")

        for art in articles:
            title = art.h3.a["title"].strip()
            price_text = art.select_one("p.price_color").text.strip()
            rating_class = art.select_one("p.star-rating")["class"]
            star_rating = [c for c in rating_class if c != "star-rating"][0]
            availability = art.select_one("p.instock.availability").text.strip()

            books.append({
                "title": title,
                "price": price_text,          
                "star_rating": star_rating,   
                "availability": availability, 
                "category": category_name,
            })

       
        next_tag = soup.select_one("li.next a")
        if next_tag:
            next_href = next_tag["href"]
            url = url.rsplit("/", 1)[0] + "/" + next_href
        else:
            url = None

        time.sleep(0.2)  

    return books


def scrape_all(min_categories=3):
    all_books = []
    categories = get_category_links(min_categories=min_categories)
    print(f"Scraping {len(categories)} categories: {[c[0] for c in categories]}")

    for name, link in categories:
        print(f"  -> scraping category: {name}")
        books = scrape_category(name, link)
        print(f"     found {len(books)} books")
        all_books.extend(books)

    return pd.DataFrame(all_books)


if __name__ == "__main__":
    df = scrape_all(min_categories=3)
    print(f"\nTotal books scraped: {len(df)}")

    
    if len(df) < 60:
        print("Fewer than 60 books, scraping additional categories...")
        all_categories = get_category_links(min_categories=None)
        extra_needed = 3
        for name, link in all_categories[3:3 + extra_needed]:
            books = scrape_category(name, link)
            df = pd.concat([df, pd.DataFrame(books)], ignore_index=True)
        print(f"Total books after topping up: {len(df)}")

    df.to_csv("books_raw.csv", index=False)
    print("Saved raw scraped data to books_raw.csv")