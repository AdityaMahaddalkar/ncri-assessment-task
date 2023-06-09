FROM python:3.6.5

WORKDIR .

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "gunicorn", "-w 4", "app:app" ]
