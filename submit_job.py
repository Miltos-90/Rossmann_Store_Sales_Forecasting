
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AzureBlobDatastore, Data  # Added Data import
from azure.ai.ml.constants import AssetTypes               # Added AssetTypes import

# 1. Configuration Constants (Match your Azure/Terraform setup)
SUBSCRIPTION_ID = "YOUR_SUBSCRIPTION_ID" # run: az account show --query id --output tsv
RESOURCE_GROUP = "rg-ml-workspace"
WORKSPACE_NAME = "mlw-workspace"
COMPUTE_NAME   = "ml-training-cluster"

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
        name="data_container_datastore",
        description="Datastore pointing to our custom data_container container",
        account_name="stmlws123",
        container_name="data_container" # Matches your Terraform container name!
    )
    ml_client.datastores.create_or_update(custom_datastore)

    # 3. Define, upload, and register the local folder
    local_data_path = "./data"  # Path to your local dataset directory
    data_asset_name = "my_uploaded_dataset"

    my_data_asset = Data(
        name=data_asset_name,
        version="1.0.0",
        description="Dataset uploaded dynamically via Python SDK",
        path=local_data_path,
        type=AssetTypes.URI_FOLDER # Specifies that we are uploading an entire folder
    )

    # Trigger the upload and registration in one go
    print(f"Uploading local directory '{local_data_path}' to cloud blob storage...")
    registered_data_asset = ml_client.data.create_or_update(my_data_asset)
    print(f"Upload complete! Data registered as: {registered_data_asset.name}")


    # 4. Define the Command Job configuration
    job = command(
        display_name="Generic ML Pipeline Run",
        experiment_name="generic-jobs",
        description="A generic template for running code on Azure Spot ML Clusters",
        
        # Point to local directories and define execution script
        code="./src",  # Packs everything inside ./src (including main.py and config.yaml)
        command="python main.py --config config.yaml --data_dir ${{inputs.raw_data}}",
        compute=COMPUTE_NAME,
        
        # The software environment to run inside the container
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