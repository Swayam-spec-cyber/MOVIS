import wikipedia

def fetch_wikipedia_summary(name):
    try:
        cleaned_name = name.replace(".", "").strip()

        try_names = [cleaned_name, name.strip(), name.replace("_", " ")]

        for candidate in try_names:
            try:
                summary = wikipedia.summary(candidate, sentences=2, auto_suggest=False)
                page = wikipedia.page(candidate, auto_suggest=False)
                return {"summary": summary, "url": page.url}
            except:
                continue

        
        results = wikipedia.search(name)
        if results:
            best = results[0]
            summary = wikipedia.summary(best, sentences=2)
            page = wikipedia.page(best)
            return {"summary": summary, "url": page.url}

        return {"summary": "", "url": ""}

    except Exception as e:
        return {"summary": "", "url": ""}
