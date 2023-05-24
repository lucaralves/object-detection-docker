from google.cloud import aiplatform
from google.cloud import storage
import os
from main import MODEL_FILE_DIR

def export_model_sample(
    project: str,
    model_id: str,
    gcs_destination_output_uri_prefix: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    timeout: int = 300,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
    output_config = {
        "artifact_destination": {
            "output_uri_prefix": gcs_destination_output_uri_prefix
        },
        # For information about export formats: https://cloud.google.com/ai-platform-unified/docs/export/export-edge-model#aiplatform_export_model_sample-drest
        "export_format_id": "tf-saved-model"
    }
    name = client.model_path(project=project, location=location, model=model_id)
    response = client.export_model(name=name, output_config=output_config)
    print("Long running operation:", response.operation.name)
    print("output_info:", response.metadata.output_info)
    export_model_response = response.result(timeout=timeout)
    print("export_model_response:", export_model_response)

def download_model_from_gcs(bucket_name: str, gcs_file_path: str, local_file_path: str):
    """Downloads a file from Google Cloud Storage to the local machine."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)
    blob.download_to_filename(local_file_path)
    print(f"Model downloaded from GCS to {local_file_path}")

# Autenticação através da google service account.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "teak-node-386212-87b132b61be4.json"

# Export the model to GCS.
export_model_sample("teak-node-386212", "1423042924249088000", "gs://pepsi-cocacola-bucket-model-edge/")

# Call the function to download the model from GCS.
download_model_from_gcs("pepsi-cocacola-bucket-model-edge",
                        "model-1423042924249088000/tf-saved-model/2023-05-23T11:13:54.160040Z/saved_model.pb",
                        str(MODEL_FILE_DIR))