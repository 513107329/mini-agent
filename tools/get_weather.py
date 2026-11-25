import json
import urllib.request
import os
from dotenv import load_dotenv

load_dotenv()


def get_weather_from_api(city: str, key: str) -> str:
    """
    Get the weather of a city from the API.
    """
    encoded_city = urllib.request.quote(city)
    request = urllib.request.Request(
        f"https://restapi.amap.com/v3/weather/weatherInfo?city={encoded_city}&key={key}"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw_data = response.read().decode("utf-8")
        result = json.loads(raw_data)
        if result.get("status") != "1":
            raise ValueError(result.get("info"))
        print(result)
        live = result.get("lives")[0]
        province = live.get("province", "未知")
        city_name = live.get("city", city)
        weather = live.get("weather", "未知")
        temperature = live.get("temperature", "未知")
        wind_direction = live.get("winddirection", "未知")
        wind_power = live.get("windpower", "未知")
        humidity = live.get("humidity", "未知")
        report_time = live.get("reporttime", "未知")
        return f"{report_time} [{province} {city_name}] 天气是：{weather}，温度是：{temperature}℃，风向是{wind_direction}，风力是{wind_power}级，湿度是{humidity}%"


def get_weather(city: str) -> str:
    """
    Get the weather of a city.
    """
    if not city or not city.strip():
        raise ValueError("city is not set")
    amap_key = os.getenv("AMAP_KEY")
    if not amap_key:
        raise ValueError("AMAP_KEY is not set")
    weather_info = get_weather_from_api(city=city, key=amap_key)
    return weather_info


GET_WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather of a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather of.",
                },
            },
            "required": ["city"],
        },
    },
}


if __name__ == "__main__":
    print(get_weather("北京"))
