from pathlib import Path
import requests
import time
import os
folder = Path("shiina")
folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist

total = 0
limit = 100
total_limit = 1000
supported_types = (".png", ".jpg", ".jpeg")

def get_json(url):
    print(url)
    r = requests.get(url)
    return r.json()

def filter_images(data):
    return [(post["file_url"], post["tag_string"]) for post in data if "file_url" in post]
import random
def download_images(url):
    data = get_json(url)
    image_urls = set(filter_images(data))

    # Pagination loop
    page = 0
    while len(data) < total_limit and len(data) > 0:
        time.sleep(0.1)  # Small delay to avoid rate-limiting
        data = get_json(f"{url}&page={page}")
        image_urls.update(filter_images(data))
        page += 1

    if not image_urls:
        print("📷 No results found")
        return

    print(f"🎯 Found {len(image_urls)} results")
    print("📩 Grabbing image list...")

    # Save URLs to a file
    scrape_file = folder / "scrape.txt"
    with open(scrape_file, "w") as f:
        f.write("\n".join([img[0] for img in image_urls]))

    print(f"🌐 Saved links to {scrape_file}\n\n🔁 Downloading images...")

    # Download images with aria2c
    os.system(f"aria2c --console-log-level=warn -c -x 16 -k 1M -s 16 -i {scrape_file}")

    # Save tags
    for img_url, tags in image_urls:
        img_name = img_url.split("/")[-1].rsplit(".", 1)[0]
        tag_file = folder / f"{img_name}.txt"
        with open(tag_file, "w") as f:
            tags = tags.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
            tags_list = tags.split(" ")
            print(tags_list)
            random.shuffle(tags_list)
            # Preserve special tags
            specialtags = [tag for tag in tags_list if tag in urls]
            tags_list = [tag for tag in tags_list if tag not in specialtags]

            final_tags = ", ".join(specialtags + tags_list).replace("_", " ")
            final_tags = final_tags.replace("(", "\\(").replace(")", "\\)")
            print(final_tags.strip())
            f.write(final_tags.strip())
        dat = requests.get(img_url)
        img_file = folder / img_url.split("/")[-1]
        img_file.write_bytes(dat.content)
urls = ["shiina_excel"]

for url in urls:
    download_images(f"https://danbooru.donmai.us/posts.json?tags={url}")
