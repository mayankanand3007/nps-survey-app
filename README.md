# NPS-Survey-App

## Installation of Required Libraries
```
Install dependencies
pip install -r requirements.txt

For nltk librabry (in python interpreter)
>> import nltk
>> nltk.download('punkt')
```

## Run example
To run the examples simply run:
```
python main.py
```

## Library Requirements
```
fastapi==0.88.0
pydantic==1.10.2
uvicorn==0.20.0
jinja2==3.1.2
python-multipart==0.0.5
fuzzywuzzy==0.18.0
nltk==3.7
matplotlib==3.6.2
wordcloud==1.8.2.2
pandas==1.5.2
openpyxl==3.0.10
python-Levenshtein==0.20.8
```

## Additional Usage
### Run FASTAPI without stopping
```
uvicorn main:app --reload
```