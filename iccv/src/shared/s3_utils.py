from contextlib import contextmanager
import boto3
import tempfile


def is_s3_path(path):
    return path.startswith('s3://')


def split_s3_path(path):
    path = path.replace('s3://', '', 1)
    parts = path.split('/')
    return parts[0], '/'.join(parts[1:])


@contextmanager
def S3FileContext(path):
    if is_s3_path(path):
        s3_client = boto3.client('s3')
        s3_bucket, s3_key = split_s3_path(path)
        with tempfile.NamedTemporaryFile() as fp:
            s3_client.download_fileobj(s3_bucket, s3_key, fp)
            yield fp.name        
    else:
        yield path
