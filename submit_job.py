
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AzureBlobDatastore, Data
from azure.ai.ml.constants import AssetTypes
from azure.storage.blob import BlobServiceClient

import json

config = json.load(open("infra_config.json"))  # Load the Terraform output JSON file

# 1. Configuration Constants (Match your Azure/Terraform setup)
SUBSCRIPTION_ID = "YOUR_SUBSCRIPTION_ID" # run: az account show --query id --output tsv
RESOURCE_GROUP = config["resource_group"]
WORKSPACE_NAME = config["workspace_name"]
COMPUTE_NAME   = config["compute_name"]

STORAGE_ACCOUNT_NAME = config["storage_account_name"]  # Matches your Terraform storage account name
STORAGE_CONTAINER_NAME = config["container_name"]  # Matches your Terraform container name

STORAGE_ACCOUNT_URI = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

LOCAL_DATA_PATH = "./data"  # Local folder to upload (ensure it exists)
DESTINATION_BLOB_PATH = "rossmann/"  # Destination path inside the container

DATASTORE_NAME = "data_container_datastore"  # Name for the Datastore that points to the custom container
DATASTORE_URI = f"azureml://datastores/{DATASTORE_NAME}/paths/{DESTINATION_BLOB_PATH}"  # Path to the uploaded data in the datastore

def main():

    # 1. Authenticate to Azure ML (Moved up so it exists before we try to use it)
    print("Connecting to Azure ML Workspace...")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # 2. Register your custom Terraform container as a new Datastore
    print("Registering custom datastore...")
    custom_datastore = AzureBlobDatastore(
        name=DATASTORE_NAME,
        description="Datastore pointing to our custom data_container container",
        account_name=STORAGE_ACCOUNT_NAME,
        container_name=STORAGE_CONTAINER_NAME # Matches your Terraform container name!
    )
    ml_client.datastores.create_or_update(custom_datastore)

    # 3. Upload local data and register it as a Data asset on the custom datastore.
    #    Step 1: Upload the local folder to the custom container via the Azure Storage SDK.
    #    (Assumes the local ./data folder exists and azure-storage-blob is installed)
    
    data_asset_name = "my_uploaded_dataset"

    print(f"Uploading local directory '{LOCAL_DATA_PATH}' to custom datastore container...")

    storage_account_key = ml_client.datastores.get(DATASTORE_NAME).credentials.account_key
    blob_service = BlobServiceClient(
        account_url=STORAGE_ACCOUNT_URI,
        credential=storage_account_key
    )
    import os
    container_client = blob_service.get_container_client(STORAGE_CONTAINER_NAME)
    for root, _, files in os.walk(LOCAL_DATA_PATH):
        for file in files:
            local_file_path = os.path.join(root, file)
            blob_name = DESTINATION_BLOB_PATH + os.path.relpath(local_file_path, LOCAL_DATA_PATH).replace("\\", "/")
            with open(local_file_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    print("Upload complete.")

    #    Step 2: Register the cloud path as a Data asset using the azureml:// URI.
    my_data_asset = Data(
        name=data_asset_name,
        version="1.0.0",
        description="Dataset uploaded to the custom data_container_datastore",
        path=DATASTORE_URI,
        type=AssetTypes.URI_FOLDER
    )
    registered_data_asset = ml_client.data.create_or_update(my_data_asset)
    print(f"Data asset registered as: {registered_data_asset.name}")
    

    # 4. Define the Command Job configuration
    job = command(
        display_name="Generic ML Pipeline Run",
        experiment_name="generic-jobs",
        description="A generic template for running code on Azure Spot ML Clusters",
        
        # Point to local directories and define execution script
        code="./src",  # Packs everything inside ./src (including main.py and config.yaml)
        command="python main.py --config debug_config.yaml --data_dir ${{inputs.raw_data}}",
        compute=COMPUTE_NAME,
        
        # The software environment to run inside the container
        # List of curated environments: https://ml.azure.com/registries/azureml/environments
        environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
        
        # Reference the freshly registered cloud dataset directly!
        inputs={
            "raw_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml:{data_asset_name}:1.0.0", 
                mode="ro_mount"
            ),
        },
        
        # Guardrail timeout (e.g., 3600 seconds = 1 hour limit)
        limits={"timeout": 3600} 
    )

    # 5. Submit to Azure ML
    print("Submitting command job...")
    returned_job = ml_client.jobs.create_or_update(job)
    
    print("Job Submitted successfully!")
    print(f"Job Name:   {returned_job.name}")
    print(f"Status:     {returned_job.status}")
    print(f"Studio Link: {returned_job.studio_url}")

if __name__ == "__main__":
    main()