""" Utility functions for Azure ML and Azure Blob Storage operations. """

import os

from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import HttpResponseError


def connect_to_workspace(subscription_id: str,
                         resource_group: str,
                         workspace_name: str) -> MLClient:
    """
    Connects to the Azure ML workspace. Returns an instance of MLClient if successful, 
    otherwise raises an exception.

    Args:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        workspace_name (str): Azure ML workspace name.

    Returns:
        MLClient: An instance of MLClient connected to the specified workspace.
    """
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )

        print(f"Successfully connected to Azure ML Workspace: {workspace_name}")
        return ml_client

    except Exception as e:
        print(f"Failed to connect to Azure ML Workspace: {e}")
        raise


def get_storage_uri(storage_account_name: str) -> str:
    """
    Constructs the Azure Blob Storage URI for the specified storage account.

    Args:
        storage_account_name (str): The name of the Azure Storage account.
    Returns:
        str: The Azure Blob Storage URI for the specified storage account.
    """
    return f"https://{storage_account_name}.blob.core.windows.net"


def get_blob_client(
        storage_account_name: str,
        chunk_size: int
        ) -> BlobServiceClient:
    """
    Creates and returns a BlobServiceClient for the specified Azure Storage account.

    Args:
        storage_account_name (str): The name of the Azure Storage account.
        chunk_size (int): The maximum size of a single block to upload in bytes.
    Returns:
        BlobServiceClient: The BlobServiceClient instance for the specified storage account.
    """

    # ==============================================================================
    # WE CHUNK AND LIMIT SINGLE PUT SIZE:
    # 
    # 1. THE 64MB DEFAULT BOTTLENECK:
    #    By default, the Azure Python SDK uploads any file under 64MB as a single,
    #    unbroken HTTP PUT request. 
    # 
    # 2. THE TIMEOUT ISSUE:
    #    For a large file on a slower or firewalled/VPN connection, a single-stream
    #    upload takes too long. If there is even a micro-hiccup in the connection, 
    #    the stream stalls, triggers a network timeout (usually at 2 minutes), 
    #    and the entire upload fails.
    # 
    # 3. CHUNKING FIXES IT:
    #    By setting `max_single_put_size` to 4MB, we force the SDK to slice any 
    #    file larger than 4MB into smaller blocks (e.g., nine 4MB blocks for 37MB).
    #    
    #    - Reliability: If one 4MB chunk fails due to a network hiccup, the SDK 
    #      only has to retry that specific 4MB chunk, not start the whole 37MB over.
    #
    #    - Speed: Using `max_concurrency=4` allows the SDK to upload up to 4 of 
    #      these chunks simultaneously, significantly speeding up the transfer.
    # ==============================================================================

    storage_uri = get_storage_uri(storage_account_name)

    client = BlobServiceClient(account_url=storage_uri,
                               credential=DefaultAzureCredential(),
                               max_single_put_size=chunk_size,
                               max_block_size=chunk_size)

    return client


def _make_blob_name(
        local_file_path: str,
        local_data_path: str,
        destination_blob_path: str) -> str:
    """
    Constructs the blob name for the given local file path.

    Args:
        local_file_path (str): The full path of the local file.
        local_data_path (str): The base path of the local data directory.
        destination_blob_path (str): The destination path inside the blob container.
    Returns:
        str: The constructed blob name.
    """
    local_relpath = os.path.relpath(local_file_path, local_data_path)
    blob_path     = local_relpath.replace("\\", "/")
    blob_name     = destination_blob_path + blob_path
    return blob_name


def upload_to_blob(
        client: ContainerClient, 
        source: str, 
        destination: str,
        **kwargs) -> list:
    """
    Uploads all files from the local data directory to the specified Azure Blob Storage container.

    Args:
        client (ContainerClient): The Azure Blob Storage container client.
        source (str): The local directory path containing files to upload.
        destination (str): The destination path inside the blob container.
        kwargs: Additional keyword arguments to pass to the upload_blob method 
        (e.g., overwrite, max_concurrency, timeout).
    Returns:
        list: A list of dictionaries containing information about any files that failed to upload.

    """
    # Convert the path string to a Path object
    base_path = Path(source)
    failed_files = []

    for file_path in base_path.rglob('*'):
        if file_path.is_file():

            try:
                print(f"Uploading {file_path.name}...")
                blob_name   = _make_blob_name(str(file_path), source, destination)
                blob_client = client.get_blob_client(blob_name)

                with file_path.open("rb") as data:
                    blob_client.upload_blob(data, **kwargs)

            except (HttpResponseError, OSError, Exception) as e:

                error_msg = f"{type(e).__name__}: {str(e)}"
                failed_files.append({"file": str(file_path), "error": error_msg})
                print(f"Failed to upload {file_path.name}: {error_msg}")

    # Summary Report
    if failed_files:
        print(f"Upload complete with {len(failed_files)} errors")
        for entry in failed_files:
            print(f"File: {entry['file']}\nReason: {entry['error']}\n")
    else:
        print("All files uploaded successfully!")

    return failed_files


def get_datastore_uri(datastore_name: str, destination_blob_path: str) -> str:
    """
    Constructs the Azure ML datastore URI for the specified datastore and blob path.

    Args:
        datastore_name (str): The name of the Azure ML datastore.
        destination_blob_path (str): The destination path inside the blob container.
    Returns:
        str: The Azure ML datastore URI for the specified datastore and blob path.
    """
    return f"azureml://datastores/{datastore_name}/paths/{destination_blob_path}"
