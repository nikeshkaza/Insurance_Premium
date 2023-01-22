import boto3
import botocore
from constant.environment.variable_key import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME
__access_key_id=AWS_ACCESS_KEY_ID_ENV_KEY
__secret_access_key=AWS_SECRET_ACCESS_KEY_ENV_KEY
region_name=REGION_NAME


# Create an S3 client
s3 = boto3.client('s3',aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key,
                    region_name=region_name
                )

# Call S3 to list current buckets
try:
    response = s3.list_buckets()
    print(response)
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "InvalidAccessKeyId":
        print("Invalid Access Key Id")
        print(e.response)
    elif e.response['Error']['Code'] == "SignatureDoesNotMatch":
        print("Invalid Secret Access Key")
    else:
        print(e.response)
        raise
