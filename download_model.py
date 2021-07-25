from google.cloud import storage 
from zipfile import ZipFile
import os

def download(bucket_name,source_blob_name,destination):


    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination
        )
    )


def extract(file):
    
    # opening the zip file in READ mode
    with ZipFile(file, 'r') as zip:

        # printing all the contents of the zip file
        zip.printdir()
  
        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall()
        os.remove(file)
        print('Done!')

