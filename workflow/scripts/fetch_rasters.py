"""
Methods to fetch raster data from copernicus archive.
Requires sentinelHub API. 

EXAMPLE SCRIPT
"""

import json
import jwt
import time
import os
import requests
import logging

logger = logging.getLogger(__name__)
# Load saved key from filesystem
TOKEN_PATH = os.path.expanduser("~/documents/token_land_cover_copernicus.json")


def authenticate() -> dict:
    """authenticae with the copernicus API

    Returns:
        dict: the access token dict
    """
    service_key = json.load(open(TOKEN_PATH, "rb"))

    private_key = service_key["private_key"].encode("utf-8")

    claim_set = {
        "iss": service_key["client_id"],
        "sub": service_key["user_id"],
        "aud": service_key["token_uri"],
        "iat": int(time.time()),
        "exp": int(time.time() + (60 * 60)),
    }
    grant = jwt.encode(claim_set, private_key, algorithm="RS256")
    base_url = claim_set["aud"]
    token_request = requests.post(
        base_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=f"grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion={grant}",
    )
    if token_request.status_code != 200:
        print(token_request.text)
        exit(1)
    token = token_request.json()
    return token


def search_items(json_items: dict, target_name="global-dynamic-land-cover") -> list:
    """filter items for target name

    Args:
        json_items (dict): the json response to the seach query
        target_name (str, optional): The items to find. Defaults to "global-dynamic-land-cover".

    Returns:
        list: the items list
    """

    return [itm["@id"] for itm in json_items["items"] if itm["@id"].find(target_name) != -1]


def find_dataset_ids(name="global-dynamic-land-cover") -> tuple[str]:
    """find the catalogue ID of the dataset

    Args:
        name (str, optional): dataset product name. Defaults to "global-dynamic-land-cover".

    Returns:
        tuple: the results (download_info_id, download_id)
    """
    bid = 0
    # batches of 25
    while True:
        batch = "" if bid == 0 else f"b_start={bid}&"
        search_req = requests.get(
            f"https://land.copernicus.eu/api/@search?{batch}portal_type=DataSet&metadata_fields=UID&metadata_fields=dataset_full_format&&metadata_fields=dataset_download_information",
            headers={"Accept": "application/json"},
        )
        if search_req.status_code != 200:
            logger.error(f"failed request: {search_req.text}")
            exit(1)
        search_results = search_req.json()
        res = search_items(search_results, target_name=name)
        if res == []:
            bid += 25
        else:
            break

    res_list = [r for r in search_results["items"] if r["@id"].find("2019") != -1]
    download_info_id = res_list[0]["dataset_download_information"]["items"][0]["@id"]
    download_id = res_list[0]["UID"]
    # dataset_url = res_list[0]["@id"]
    return download_info_id, download_id


def post_data_request(token: dict, download_info_id: str, download_id: str) -> tuple[dict, str]:
    """post the request to the copernicus API

    Args:
        token (dict): the access token
        download_info_id (str): output of find_dataset_ids
        download_id (str): output of find_dataset_ids

    Returns:
        tuple: the data request (json dict) and task id (string?)
    """
    base_url = "https://land.copernicus.eu"
    data_req = requests.post(
        f"{base_url}/api/@datarequest_post",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token['access_token']}",
        },
        json={
            "Datasets": [
                {
                    "DatasetID": download_id,
                    "DatasetDownloadInformationID": download_info_id,
                    "Layer": "Cover Fraction: Built-up ",
                    "NUTS": "CN",
                    "OutputFormat": "Geotiff",
                    "OutputGCS": "EPSG:4326",
                }
            ]
        },
    )

    return data_req.json(), data_req.json()["TaskIds"][0]["TaskID"]


if __name__ == "__main__":
    token = authenticate()
    download_info_id, download_id = find_dataset_ids()

    data_req, task_id = post_data_request(token, download_info_id, download_id)
    logger.info(f"data request post returned: {data_req}, {task_id}")

    # once ready run this
    status = requests.get(
        "https://land.copernicus.eu/api/@datarequest_search?status=Finished_ok",
        headers={
            "Accept": "application/json",
            "Authorization": f'Bearer {token["access_token"]}',
        },
    )
    download = status.json()[str(task_id)]["DownloadURL"]
    dl = requests.get(download)
    with open("CLMS_download.zip", "wb") as f:
        f.write(dl.content)
