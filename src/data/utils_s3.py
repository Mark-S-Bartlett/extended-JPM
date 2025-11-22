from databricks.sdk.runtime import *

def mount_aws_bucket(aws_bucket_name, mount_name):
    """
    Mounts an AWS S3 bucket to a Databricks mount point if it's not already mounted.
    
    Parameters:
    aws_bucket_name (str): Name of the AWS S3 bucket to mount.
    mount_name (str): Name of the mount point in Databricks.
    
    Returns:
    None. Prints mount status and displays contents of the mount point.
    """
    # Refresh table of mounted file systems
    dbutils.fs.refreshMounts()

    # Check if mount_name already exists in mount table
    if not any(m.mountPoint == f"/mnt/{mount_name}" for m in dbutils.fs.mounts()):
        # Mount the directory if mount point doesn't exist
        dbutils.fs.mount(f"s3a://{aws_bucket_name}", f"/mnt/{mount_name}")
        print(f"Mounted bucket '{aws_bucket_name}' to mount point '/mnt/{mount_name}'")
    else:
        print(f"Bucket '{aws_bucket_name}' already mounted to mount point '/mnt/{mount_name}'")

    # Display contents of mount point
    dbutils.fs.ls(f"/mnt/{mount_name}")
