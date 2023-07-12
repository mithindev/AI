import requests
import json

city = input("Enter your city\n")

url = 'https://api.tomorrow.io/v4/timelines?location=40.75872069597532,-73.98529171943665&fields=temperature&timesteps=1h&units=metric&apikey=xUNSGwTb29aTJddaFdyH048VKXkQQrny'
