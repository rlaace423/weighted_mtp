name: stable-diffusion
description: This is the inference example of stable diffusion.
tags:
- "best"
- "A100-80g"
- "20epochs"
resources:
  cluster: vessl-oci-sanjose
  preset: gpu-l4-small   
  node_names:
    - "n01"
    - "n03"
    - "n04"
import:
  /import/code: git://github.com/{accountName}/{repoName}
  /import/code-verbose:
    git:
      url: https://github.com/{accountName}/{repoName}
      ref: c0ffee
      credential_name: my-git-cred-name
  /import/dataset: vessl-dataset://{organizationName}/{datasetName}
  /import/dataset-verbose:
    dataset:
      organization_name: {organizationName}
      dataset_name: {datasetName}
  /import/model: vessl-model://{organizationName}/{modelRepositoryName}/{modelNumber}
  /import/model-verbose:
    model:
      organization_name: {organizationName}
      model_repository_name: {modelRepositoryName}
      model_number: {modelNumber}
  /import/artifact: vessl-artifact://{organiztionName}/{projectName}/{artifactName}
  /import/artifact-verbose:
    artifact:
      organization_name: {organizationName}
      project_name: {projectName}
      name: {artifactName}
  /import/artifact-verbose-same-project:
    artifact:
      name: {artifactName}
  /import/s3: s3://{bucketName}/{path}
  /import/s3-verbose:
    s3:
      bucket: {bucketName}
      prefix: {prefix}
      credential_name: my-s3-cred-name
  /import/gs: gs://{buckeName}/{path}
  /import/gs-verbose:
    gs:
      bucket: {bucketName}
      prefix: {prefix}
      credential_name: my-gs-cred-name
mount: 
  /mount/dataset: vessl-dataset://{organizationName}/{datasetName}
  /mount/dataset-verbose: 
    dataset:
      organization_name: {organizationName}
      dataset_name: {datasetName}
  /mount/hostpath: hostpath://{path}
  /mount/hostpath-verbose:
    hostpath:
      path: {path}
    readonly: true
  /mount/nfs: nfs://{server}/{path}
  /mount/nfs-verbose:
    nfs:
      server: {server}
      path: {path}
    readonly: false
export:
  /export/output-artifact: vessl-artifact:// 
  /export/output-artifact-verbose:
    artifact: 
  /export/backup-artifact: vessl-artifact://{organizationName}/{projectName}/{artifactName}
  /export/backup-artifact-verbose:
    artifact:
      organization_name: {organizationName}
      project_name: {projectName}
      artifact_name: {artifactName}
  /export/dataset: vessl-dataset://{organizationName}/{datasetName}
  /export/dataset-verbose:
    dataset:
      organization_name: {organizationName}
      dataset_name: {datasetName}
  /export/model: vessl-model://{organizationName}/{modelRepositoryName}
  /export/model-verbose:
    model:
      organization_name: {organizationName}
      model_repository_name: {modelRepositoryName}
  /export/s3: s3://{buckeName}/{prefix}
  /export/s3-verbose:
    s3:
      bucket: {bucketName}
      prefix: {prefix}
      endpoint: in-house.endpoint.co.kr
      credential_name: my-s3-cred-name
  /export/gs: gs://{bucketName}/{prefix}
  /export/gs-verbose:
    gs:
      bucket: {bucketName}
      prefix: {prefix}
run:
  - workdir: /input/data1
    command: | 
      python data_preprocessing.py
  - wait: 10s
  - workdir: /root/git-examples
    command: |
      python train.py --learning_rate=$learning_rate --batch_size=$batch_size
interactive:
  max_runtime: 24h      # required if interactive
  jupyter:              # required if interactive
    idle_timeout: 120m  # required if interactive
ports:
  - 3000
  - name: streamlit
    type: http
    port: 8501
env:
  learning_rate: 0.001
  postgres_password:
    value: OUR_DB_PW
    secret: true