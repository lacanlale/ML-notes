FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CMD [ "python", "./sklearn_ver.py" ]
CMD ["python", "./self_made_ver.py"]
