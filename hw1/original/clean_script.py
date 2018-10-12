import csv
import nltk
from nltk.tokenize import word_tokenize
import re, string; pattern = re.compile('[^a-zA-Z0-9_]+')

def write_fake():
    titles = set()
    try:
        for line in csv.DictReader(open("data/fake.csv")):
            if line['thread_title']:
                otitle = line['thread_title'].lower()
                if "trump" not in otitle:
                    continue
                title = otitle.replace("(video)","") \
                              .replace("[video]","") \
                              .replace("re:","") \
                              .replace("?","") \
                              .replace("100percentfedup.com","")

                title = pattern.sub(' ', title)
                twords = word_tokenize(title)
                twords = [w for w in twords if w != 's']
                ntitle = ' '.join(twords)

                # "don t" -> "dont"; "wasn t" -> "wasnt"; etc
                ntitle = ntitle.replace("n t ", "nt ") 
                titles.add(ntitle)
    except:
        pass

    outfile = open("data/clean_fake.txt", "w")
    for ntitle in titles:
        outfile.write(ntitle + "\n")

def write_real():
    titles = set()
    for line in csv.reader(open("data/abcnews-date-text.csv")):
        date = line[0]
        if date[:5] >= "20161":
            title = line[1].lower()
            if "trump" not in title:
                continue
            title = pattern.sub(' ', title)
            twords = word_tokenize(title)
            twords = [w for w in twords if w != 's']
            ntitle = ' '.join(twords)
            titles.add(ntitle)

    outfile = open("data/clean_real.txt", "w")
    for ntitle in titles:
        outfile.write(ntitle + "\n")

if __name__ == "__main__":
    write_fake()
    write_real()
