import os
import base64
import requests
import boto3
import json
import MySQLdb
from botocore.exceptions import ClientError
_connection = None


def get_secret(secret_id):
    """
    Use AWS Secret Manager to get the credentials for the annotations
    NER database
    :param secret_id: str
    :param region_name: str
    :return: secret: dict()
    """

    # Get region name
    r = requests.get("http://169.254.169.254/latest/dynamic/instance-identity/document")
    region_name = r.json().get('region')

    # Create a Secrets Manager client
    # Make sure you set the region in your instance using "aws configure" command
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_id
        )
    except ClientError as e:
        raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            secret = base64.b64decode(get_secret_value_response['SecretBinary'])

        return json.loads(secret)

    return None


def get_connection(secret='health-rds'):
    """
    Connect to the RDS
    :return: connection
    """

    # If secret is a string, get the secret for that key
    # Otherwise assume it is secret dictionary
    if isinstance(secret, str):
        secret = get_secret(secret)

    global _connection
    if not _connection:
        try:
            _connection = MySQLdb.connect(
                    host=secret['host'],
                    user=secret['username'],
                    passwd=secret['password'],
                    db=secret['dbClusterIdentifier'],
                    use_unicode=True,
                    charset="utf8"
            )
        except:
            raise

    return _connection


def get_query(prefix):
    query_filename = os.path.join('..', 'sql', prefix + '.sql')
    with open(query_filename, 'r') as fh:
        return fh.read()