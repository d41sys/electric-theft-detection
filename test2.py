import requests

# URL of the CSV file
url = "http://localhost:6006/data/plugin/custom_scalars/download_data?tag=AUC&run=transformer_2024-09-24+15%3A44%3A41&format=csv"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
file_name = "AUC.csv"
if response.status_code == 200:
    # Save the content to a local file
    with open(file_name, "wb") as file:
        file.write(response.content)
    print(f"CSV file downloaded and saved as {file_name}")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")