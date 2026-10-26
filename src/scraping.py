import requests
from bs4 import BeautifulSoup
import pandas as pd
url = "https://www.bari91.com/competition/Tunisia%2C_Tunisian_Ligue_Professionnelle_1/2024-2025"

session = requests.Session()

response = session.get(url)

timezone_url = "https://www.bari91.com/tz.php"

params = {
    "zone": "Africa/Tunis",
    "r": "/competition/Tunisia%2C_Tunisian_Ligue_Professionnelle_1/2024-2025"
}

response2 = session.get(timezone_url, params=params)

soup = BeautifulSoup(response2.text, "html.parser")

print("Page title:", soup.title.get_text(strip=True))

match_days = soup.select(".match_day")
matches = soup.select(".results_line")

all_results_url = "https://www.bari91.com/competition/get_scores/15584/99"

response3 = session.get(all_results_url)

soup_results = BeautifulSoup(response3.text, "html.parser")

dates = soup_results.select(".results_date")
matches = soup_results.select(".results_line")

print("Number of dates:", len(dates))
print("Number of matches:", len(matches))
elements = soup_results.select(".results_date, .results_line")
all_matches=[]
current_date=None
for element in elements:
    if "results_date" in element.get("class", []):
        current_date=element.get_text(strip=True)
        current_date=current_date.split("(")[0].strip()
        current_date = pd.to_datetime(current_date, format="%d.%m.%Y")

    else:
        home_team=element.select_one(".col-xs-5.text-right").get_text(strip=True)
        away_team=element.select_one(".col-xs-5.text-left").get_text(strip=True)
        score=element.select_one(".col-xs-2").get_text(strip=True)
        home_goals, away_goals = map(int,score.split(":"))
        match_data={
            "date":current_date,
            "home_team":home_team,
            "home_goals":home_goals,
            "away_goals":away_goals,
            "away_team":away_team,
        }
        all_matches.append(match_data)

df = pd.DataFrame(all_matches)

df.to_csv("tunisia_ligue1_2024_2025.csv",index=False)
print("Dataset saved successfully!")