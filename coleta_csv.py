import requests
from bs4 import BeautifulSoup
import os

url = "https://www.football-data.co.uk/englandm.php"

save_folder = "football_data"
os.makedirs(save_folder, exist_ok=True)

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

base_url = "https://www.football-data.co.uk/"

# mapa código → liga
league_map = {
    "E0": "PremierLeague",
    "E1": "Championship",
    "E2": "LeagueOne",
    "E3": "LeagueTwo",
    "EC": "Conference"
}

for link in soup.find_all("a"):
    href = link.get("href")

    if href and href.endswith(".csv"):

        full_url = base_url + href

        filename = href.split("/")[-1]
        season = href.split("/")[-2]

        league_code = filename.split(".")[0]

        league_name = league_map.get(league_code, league_code)

        new_name = f"{league_name}_{season}.csv"

        file_path = os.path.join(save_folder, new_name)

        print(f"baixando {new_name}")

        r = requests.get(full_url)

        with open(file_path, "wb") as f:
            f.write(r.content)

print("Download completo!")