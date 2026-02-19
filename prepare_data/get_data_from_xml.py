from bs4 import BeautifulSoup
import os

def extract_text(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    text_parts = []

    # Title
    title = soup.find("article-title")
    if title:
        text_parts.append(title.get_text())

    # Abstract
    abstract = soup.find("abstract")
    if abstract:
        text_parts.append(abstract.get_text())

    # Body paragraphs
    for p in soup.find_all("p"):
        text_parts.append(p.get_text())

    return " ".join(text_parts)


dataset = []

folder = "/home/jarvis/inerg_demo/oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23"

for file in os.listdir(folder):
    if file.endswith(".xml"):
        path = os.path.join(folder, file)
        text = extract_text(path)

        if text.strip():
            dataset.append(text)

print("Documents extracted:", len(dataset))
