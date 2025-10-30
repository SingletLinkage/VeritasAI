from dotenv import load_dotenv
load_dotenv()
import os
import requests
import base64


def upload_image(file_path) -> str:
    # upload file to image host, return public URL
    return imgbb_upload(file_path)

def imgbb_upload(file_path: str) -> str:
    api_key = os.environ.get("IMGB_API_KEY")
    if not api_key:
        raise RuntimeError("IMGB_API_KEY environment variable is not set")

    with open(file_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("ascii")

    url = "https://api.imgbb.com/1/upload"
    payload = {"key": api_key, "image": b64_image}

    response = requests.post(url, data=payload)
    response.raise_for_status()
    resp_json = response.json()

    if not resp_json.get("success"):
        raise RuntimeError(f"imgbb upload failed: {resp_json}")

    return resp_json["data"]["url"]


def reverse_image_search(image_url: str) -> list[dict]:
    return serpapi_reverse_image_search(image_url)
    

def serpapi_reverse_image_search(image_url: str) -> list[dict]:
    from serpapi import GoogleSearch

    params = {
    "engine": "google_reverse_image",
    "image_url": image_url,
    "api_key": os.environ.get("SERPAPI_API_KEY"),
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    # save as json
    # import json
    # with open("serpapi_rev_search_results.json", "w") as f:
    #     json.dump(results, f, indent=4)
    return results['image_results']



def process_file(file_path: str) -> dict:
    url = upload_image(file_path)
    return reverse_image_search(url)

if __name__ == "__main__":
    # test_file = "./reports/media/backbone.png"
    # result = upload_image(test_file)
    # print(result)

    test_url = "https://www.webmd.com/food-recipes/benefits-apples" # "https://i.ibb.co/JwfqbSzr/6dfd413ac61e.png"
    result = reverse_image_search(test_url)
    print(result)