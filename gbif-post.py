import requests
from requests.auth import HTTPBasicAuth
import json
import time

# Define the API endpoint for occurrence downloads
api_endpoint = "https://api.gbif.org/v1/occurrence/download/request"

# Define the payload for the POST request
payload = {
    "creator": "gmougeot", 
    "notificationAddresses": [
        "gmo@ecos.au.dk"
    ],
    "sendNotification": True,
    "format": "SIMPLE_CSV",
    "predicate": {
        "type": "and",
        "predicates": [
            {
                "type": "equals",
                "key": "TAXON_KEY",
                "value": "54"  # TaxonKey for Arthropoda
            },
            {
                "type": "equals",
                "key": "MEDIA_TYPE",
                "value": "StillImage"
            },
            {
                "type": "in",
                "key": "COUNTRY",
                "values": [
                    # List of European country codes (ISO 3166-1 alpha-2)
                    "AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CY",
                    "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS",
                    "IE", "IT", "LV", "LI", "LT", "LU", "MT", "MD", "MC",
                    "ME", "NL", "MK", "NO", "PL", "PT", "RO", "RU", "SM",
                    "RS", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "GB",
                    "VA"
                ]
            },
            {
                "type": "equals",
                "key": "YEAR",
                "value": "2017"
            },
            {
                "type": "equals",
                "key": "MONTH",
                "value": "12"
            }
        ]
    },
    # "options": {
    #     "limit": 10000  # Set to 0 to indicate no limit; adjust as needed
    # }
}

# If you have an API key, include it in the headers
headers = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer YOUR_API_KEY"  # Uncomment and replace if you have an API key
}

# Make the POST request to initiate the download
response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), auth=HTTPBasicAuth("gmougeot", "Ezaqwxcds78544521!"))

# Handle the response based on the 201 status code
if response.status_code == 201:  # The correct response for a successful download request
    # download_key = response.json().get("key")
    download_key = response.text
    print(f"Download initiated successfully. Key: {download_key}")

    # Polling to check the status of the download
    status_endpoint = f"https://api.gbif.org/v1/occurrence/download/{download_key}"
    print(f"Polling status from: {status_endpoint}")

    while True:
        status_response = requests.get(status_endpoint)

        if status_response.status_code == 200:
            status = status_response.json()
            download_status = status.get("status")
            print(f"Current status: {download_status}")

            if download_status == "SUCCEEDED":
                download_url = f"https://api.gbif.org/occurrence/download/request/{download_key}.zip"
                print(f"Download succeeded! You can download the file from: {download_url}")
                break
            elif download_status in ["RUNNING", "PENDING", "PREPARING"]:
                print("Download is still processing. Checking again in 60 seconds...")
                time.sleep(60)  # Wait for 60 seconds before polling again
            else:
                print(f"Download failed with status: {download_status}")
                break
        else:
            print(f"Failed to get download status. HTTP Status Code: {status_response.status_code}")
            print(f"Response Text: {status_response.text}")
            break
else:
    print(f"Failed to initiate download. HTTP Status Code: {response.status_code}")
    print(f"Response: {response.text}")

# TODO:
# - Parse python input arguments for username and password, for data directory
# - Download the file to data directory
