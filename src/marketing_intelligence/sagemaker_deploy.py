from __future__ import annotations

"""Lightweight SageMaker deployment stub.

Fill in the IAM role, S3 model artifact path, and container image before use.
This file exists so you can discuss how the project would be pushed to SageMaker.
"""

import boto3


def deploy_model(
    model_name: str,
    image_uri: str,
    model_data_url: str,
    execution_role_arn: str,
    endpoint_config_name: str,
    endpoint_name: str,
    instance_type: str = "ml.m5.large",
) -> None:
    sm = boto3.client("sagemaker")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri, "ModelDataUrl": model_data_url},
        ExecutionRoleArn=execution_role_arn,
    )
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
            }
        ],
    )
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
