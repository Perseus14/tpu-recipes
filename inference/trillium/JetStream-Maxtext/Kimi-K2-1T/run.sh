#!/bin/bash
# Required variables to be set
export PROJECT_ID=<project-id>
export REGION=<region>
export CLUSTER_NAME=<cluster-name>
export CLUSTER_ZONE=<cluster-zone>
export GCS_BUCKET=<gcs-bucket>
export TPU_RESERVATION=<tpu-reservation>

# Required variables with default values
export TPU_TYPE=v6e-128
export NUM_SLICES=1
export CLUSTER_CPU_MACHINE_TYPE=n2d-standard-32
export CLUSTER_CKPT_NODEPOOL_NAME=ckpt-conversion-node-pool-0
export CLUSTER_CKPT_NODE_MACHINE_TYPE=m4-ultramem-224
export CLUSTER_CKPT_NODE_REGION=us-east4
export CLUSTER_CKPT_NODE_DISK_SIZE=8000
export CLUSTER_CKPT_NUM_NODES=1
export ARTIFACT_REGISTRY_REPO_NAME=jetstream-maxtext-ar-kimi-k2
export ARTIFACT_REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_NAME}
export JETSTREAM_MAXTEXT_IMAGE=jetstream-maxtext
export JETSTREAM_MAXTEXT_VERSION=latest
export HF_MODEL_NAME="moonshotai/Kimi-K2-Thinking"
export HF_MODEL_VARIANT="thinking"
export MODEL_NAME=kimi-k2-1t
export LOCAL_CKPT_BASE_PATH=/mnt/disks/persist
export GCS_CKPT_PATH_UNSCANNED=gs://${GCS_BUCKET}/ninja_poc/zentiq/models/${MODEL_NAME}/${HF_MODEL_VARIANT}/unscanned

export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/inference/trillium/JetStream-Maxtext/Kimi-K2-1T