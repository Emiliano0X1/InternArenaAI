from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from models.playerItem import Player
import httpx
import asyncio


class Settings(BaseSettings):
    LEETCODE_API: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()


# 1. Helper function for a single player
async def fetch_single_player(client: httpx.AsyncClient, username: str) -> Optional[Player]:
    """
    Fetches profile stats and recent submissions for a single LeetCode user.
    """
    try:
        # Launch both requests concurrently for this specific user
        user_req = client.get(f"{settings.LEETCODE_API}/user/{username}")
        sub_req = client.get(f"{settings.LEETCODE_API}/user/{username}/submissions?limit={20}")
        
        user_res, sub_res = await asyncio.gather(user_req, sub_req, return_exceptions=True)
        
        # Handle network errors or non-200 HTTP responses
        if isinstance(user_res, Exception) or user_res.status_code != 200:
            print(f"Failed to fetch profile for user: {username}")
            return None
            
        user_data = user_res.json()
        sub_data = sub_res.json()

        # Instantiate Player model using dict key access
        # Pasamos los valores directamente en la instanciación

        ac_submissions = user_data["submitStats"]["acSubmissionNum"]
        new_player = Player(
            name=username,
            cantEasy=ac_submissions[1]["count"] if len(ac_submissions) > 1 else 0,
            cantMed=ac_submissions[2]["count"] if len(ac_submissions) > 2 else 0,
            cantHard=ac_submissions[3]["count"] if len(ac_submissions) > 3 else 0,
            last_active=sub_data[0]["timestamp"] if len(sub_data) >= 1 else "", # o la fecha/timestamp correspondiente
            recent_submissions=(
                [s.get("statusDisplay") for s in sub_data[:20]] 
                if isinstance(sub_data, list) else []
            )
        )
        
        return new_player

    except Exception as e:
        print(f"Unexpected error processing {username}: {e}")
        return None


# 2. Main function coordinating all players
async def getStatsFromAPI(listUsernames: List[str]) -> List[Player]:
    """
    Fetches statistics for a list of LeetCode usernames concurrently.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Create an async task for each player
        tasks = [fetch_single_player(client, username) for username in listUsernames]
        
        # Execute all player tasks in parallel
        results = await asyncio.gather(*tasks)
        
        # Filter out None values caused by errors or missing users
        players_data = [player for player in results if player is not None]

    return players_data