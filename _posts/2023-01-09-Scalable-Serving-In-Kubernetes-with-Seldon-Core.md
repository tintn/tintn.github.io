---
title:  "Scalable Serving In Kubernetes with Seldon Core: A tutorial"
date:   2023-01-09 23:00:00 +0700
tags:
    - pytorch
    - triton
    - seldon-core
    - kubernetes
    - autoscaling
---

## Overview

<figure>
	<img src="/assets/posts/Scalable-Serving-In-Kubernetes-with-Seldon-Core/ml-k8s.png">
</figure>

Deploying trained models to production is an essential step in most of ML applications. It’s where the models actually show their values by providing their predictions for customers or other systems.

Deploying a model can be as straight-forward as implementing a Flask server then export its endpoints for users to call. However, having a system that can robustly and reliably serve large amount of requests with strict requirements in response time or throughput is not trivial.

For medium and large companies, it’s very likely they require the systems to be able to scale to process heavier workload without significantly changing the codebase. Perhaps the company is growing over time and they need a system that can handle growing number of requests (this characteristic is scalability). Or the company has unstable traffic, the traffic is fluctuating over the course of a week and the company want the system to be able to adapt with the traffic fluctuation (this characteristic is elasticity). These characteristics can be achieved if the systems have the ability to scale up and down based on the traffic volume.

In this tutorial, we’re going to learn how to deploy ML models in Kubernetes clusters with Seldon Core. We’ll also learn to implement autoscaling for our deployment with HPA and KEDA. The code for this tutorial can be found from this [repo](https://github.com/tintn/ml-model-deployment-tutorials).



##  Train a PyTorch model

We need to have a model to work through the deployment process. We use the model from a [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) from the official PyTorch website. It’s a simple image classification model that can run with CPU easily, so we can test the whole deployment process on local machines like your laptop.

Assume you’re in the `toy-model` folder of this [repo](https://github.com/tintn/ml-model-deployment-tutorials). You can train the model on the CIFAR10 dataset with:

```bash
python train.py
# Output: model.pth -> the trained weights of the model
```

Seldon Core uses [Triton Inference Server](https://github.com/triton-inference-server/server) to serve PyTorch models, so we need to prepare the model in a format that it can be served with Triton. First, we need to export the model to TorchScript (it’s also possible to serve PyTorch models with Triton’s [python backend](https://github.com/triton-inference-server/python_backend), but it’s generally less efficient and more complicated to deploy).

There’re two methods to export a model to TorchScript: tracing and scripting. Both of them are supported by PyTorch. Which method is better is still controversial, this [article](https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/) discusses pros and cons of each method. We’ll use the tracing method to export the model:

```bash
python export_to_ts.py -c model.pth -o model.ts
# Output: model.ts 
# -> serialized model containing both trained weights and the model's architecture
```

Triton loads models from a model repository. It must contain information that the server needs to serve a model such as the model’s input/output information, backend to use… A model repository must follow the following structure:

```bash
<model-repository-path>/
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```

In our case, we only have one model. Let’s call the model `cifar10-pytorch`, our model repo should have the following structure:

```bash
cifar10-model
└── cifar10-pytorch
    ├── 1
    │   └── model.ts
    └── config.pbtxt
```

`cifar10-model` is the repository’s name, `cifar10-pytorch` is the model name and `model.ts` is the TorchScript model we just exported. `config.pdtxt` defines how the model should be served with Triton:

```bash
platform: "pytorch_libtorch"
default_model_filename: "model.ts"
max_batch_size: 0
input [
  {
    name: "image__0"
    data_type: TYPE_UINT8
    dims: [-1, 3, -1, -1]
  }
]
output [
  {
    name: "probs__0"
    data_type: TYPE_FP32
    dims: [-1, 10]
  }
]
```

You can find the final repository [here](https://github.com/tintn/ml-model-deployment-tutorials/tree/main/toy-model/cifar10-model). Triton supports many features that you can possibly use to tune the model’s performance. You can also group multiple steps or multiple models into an inference pipeline to implement your business logic. However, I deliberately keep the model config simple to illustrate the whole process of model deployment instead of focusing on tuning performance. 

If you want to see a more realistic example of how to export and serve a PyTorch model with Triton, take a look at this [post](https://tintn.github.io/deploy-detectron2-with-triton/). It shows how to serve the MaskRCNN model from Detectron2 with Triton, which is a very popular model for instance segmentation and is used in many real-world computer vision systems.

Triton can access models from local filesystem or cloud storage services (e.g. S3, Google Storage, or Azure Storage). As we’re going to deploy the model in Kubernetes, using a cloud storage service is more convenient because all nodes in the Kubernetes cluster can access the same models. We’ll use AWS S3 as the model repository in this tutorial. Assume that you already have an AWS account, let’s create an S3 bucket and upload the folder we have prepared:

```bash
aws s3 cp --recursive cifar10-model s3://<YOUR_BUCKET>/cifar10-model
```

Replace `<YOUR_BUCKET>` to the name of your bucket. We now have the model repository on AWS S3, we can start to deploy the model.

## Deploy models with Seldon Core

We’ll deploy the model to a Kubernetes cluster with Seldon Core, a framework specializing in ML model deployment and monitoring. Let’s create a local Kubernetes cluster, so we can test the deployment process using our local machine.

[Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) can be used to create local clusters. At the time of writing, Seldon Core has an [issue](https://github.com/SeldonIO/seldon-core/issues/4339) with k8s ≥ 1.25, so we have to use version 1.24 or older. To specify the k8s version with Kind, just choose the image with the corresponding version to start a cluster. The following command creates a local cluster named `kind-seldon` with `k8s==1.24.7`:

```bash
kind create cluster --name seldon --image kindest/node:v1.24.7
```

Also make sure you have `docker`, `kubectl`, `helm` installed on your local machine. Switching the context of `kubectl` to `kind-seldon` instructs `kubectl` to connect to the newly created cluster by default:

```bash
kubectl cluster-info --context kind-seldon
```

### Install Seldon Core

We’ll use [Istio](https://istio.io/) as the cluster’s Ingress and Seldon Core as the serving platform. You can find the installation instruction [here](https://docs.seldon.io/projects/seldon-core/en/latest/install/kind.html). After installing Istio and Seldon Core, run these commands to check if they are all correctly installed:

```bash
kubectl get svc -n istio-system
# NAME                   TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)                                                                      AGE
# istio-egressgateway    ClusterIP      10.96.90.103   <none>        80/TCP,443/TCP                                                               3m29s
# istio-ingressgateway   LoadBalancer   10.96.229.8    <pending>     15021:30181/TCP,80:32431/TCP,443:30839/TCP,31400:32513/TCP,15443:32218/TCP   3m28s
# istiod                 ClusterIP      10.96.195.7    <none>        15010/TCP,15012/TCP,443/TCP,15014/TCP                                        8m48s
```

Check if the Istio gateway is running:

```bash
kubectl get gateway -n istio-system
# NAME             AGE
# seldon-gateway   5m17s
```

Check if the Seldon controller is running:

```bash
kubectl get pods -n seldon-system
# NAME                                        READY   STATUS    RESTARTS   AGE
# seldon-controller-manager-b74d66684-qndf6   1/1     Running   0          4m18Create an Istio gateway to manage the cluster’s traffic:
```

If you haven’t done it, make sure the label `istio-injection` is enabled:

```bash
kubectl label namespace default istio-injection=enabled
```

The Istio gateway is running on port 80 in the cluster, we need to forward a port from your local machine to that port so we can access it externally:

```bash
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```

### Serving with Seldon Core

If your model repository is stored in a private bucket, you need to grant permission to access your bucket from within the cluster. It can be done by creating a secret and then referring to it when creating a deployment. This is a template to create secrets for S3 buckets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: seldon-rclone-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: aws
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: "<AWS_ACCESS_KEY_ID>"
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: "<AWS_SECRET_ACCESS_KEY>"
```

Replace  `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` with your actual AWS access key ID and secret access key. Create the secret:

```bash
kubectl apply -f secret.yaml
```

We can deploy the model with this [manifest](https://github.com/tintn/ml-model-deployment-tutorials/blob/main/scalable-serving/cifar10-deploy.yaml), notice the created secret is refered in the manifest with the `envSecretRefName` key. Also make sure that `spec.predictors[].graph.name` matches the model name you uploaded to your model repository. Apply the manifest to create a deployment:

```bash
kubectl apply -f cifar10-deploy.yaml
```

If this is your first deployment in this cluster, it’ll take a while to download the necessary docker images. Check if the model is successfully deployed:

```bash
kubectl get deploy
# NAME                                READY   UP-TO-DATE   AVAILABLE   AGE
# cifar10-default-0-cifar10-pytorch   1/1     1            1           41m
```

I created a [script](https://github.com/tintn/ml-model-deployment-tutorials/blob/main/testing/test.py) using Locust to test the deployed models. You need to install the [requirements](https://github.com/tintn/ml-model-deployment-tutorials/blob/main/testing/requirements.txt) needed for the script to run first:

```bash
pip install -r requirements.txt
```

Given that `localhost:8080` has been port-forwarded to the cluster’s gateway, run the following command to send requests to models deployed with Seldon:

```bash
locust -f test.py --headless -u 100 -r 10 --run-time 180 -H http://localhost:8080
```

If your deployment name or model names are different, you can adjust the deployment’s URL accordingly in the script. The URL for a deployed model follows Seldon Core’s [inference protocol](https://docs.seldon.io/projects/seldon-core/en/latest/reference/apis/v2-protocol.html):

```
/seldon/{namespace}/{model_repo}/v2/models/{model_name}/versions/{model_version}/infer
```

We’ve deployed our custom model with Seldon Core and tested it by sending inference requests to the model. In the next section, we’ll explore how to scale the deployment to handle more users or higher traffic.  

## Pod Autoscaling with HPA

HPA (Horizontal Pod Autoscaling) is an important feature of Kubernetes when it comes to scalability. It can add more pods to a deployment when certain metrics exceed their thresholds (e.g. CPU or memory), so we have more resources to process the incoming traffic.

### Install Metrics Server

HPA needs to fetch metrics from an aggregated API, which is usually provided through a [Metrics Server](https://github.com/kubernetes-sigs/metrics-server). You can install a metrics server for your cluster with:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

If your cluster is local, you also need to disable certificate validation by passing the argument `-kubelet-insecure-tls` to the server:

```bash
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

### Deploy models with HPA

We can enable HPA in the deployment manifest by adding `hpaSpec` for the corresponding component:

```bash
hpaSpec:
  maxReplicas: 2
  metrics:
  - resource:
      name: cpu  # This can be either "cpu" or "memory"
      targetAverageUtilization: 50
    type: Resource
  minReplicas: 1
```

The HPA spec tells the deployment to scale up when the current metric value (CPU usage in this case) is higher than 50% of the desired value, and the maximum replicas that the deployment can possibly have are 2.

Apply this [manifest](https://github.com/tintn/ml-model-deployment-tutorials/blob/main/scalable-serving/cifar10-deploy-hpa.yaml) to create a deployment with HPA, make sure you replace `<YOUR_BUCKET>` with your bucket name and the secret for accessing the bucket (as mentioned in the previous section) has been created: 

```bash
kubectl apply -f cifar10-deploy-hpa.yaml
```

You can see the current metric value with:

```bash
kubectl get hpa
# NAME                                REFERENCE                                      TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
# cifar10-default-0-cifar10-pytorch   Deployment/cifar10-default-0-cifar10-pytorch   0%/50%    1         2         1          98m
```

Let’s check the running pods. You should see a running pod for the deployed model:

```bash
kubectl get pods
# NAME                                                 READY   STATUS    RESTARTS   AGE
# cifar10-default-0-cifar10-pytorch-7744fdc4dd-vzx4x   3/3     Running   0          101m
```

Now we can test the deployed model with our [test script](https://github.com/tintn/ml-model-deployment-tutorials/blob/main/testing/test.py) (that was mentioned in the previous section):

```bash
locust -f test.py --headless -u 100 -r 10 --run-time 180 -H http://localhost:8080
```

Monitoring the current metric value with `kubectl get hpa -w`, you can see after a while the metrics value exceeds the threshold, and HPA will trigger the creation of a new pod:

```bash
kubectl get pods
# NAME                                                 READY   STATUS    RESTARTS   AGE
# cifar10-default-0-cifar10-pytorch-7744fdc4dd-pgpxm   3/3     Running   0          10s
# cifar10-default-0-cifar10-pytorch-7744fdc4dd-vzx4x   3/3     Running   0          108m
```

If the current metric value is lower than the threshold for a certain period (it’s 5 minutes by default), HPA will scale down the deployment. The period can be configured with the argument `--horizontal-pod-autoscaler-downscale-stabilization` flag to `kube-controller-manager`:

```bash
kubectl get pods
# NAME                                                 READY   STATUS        RESTARTS   AGE
# cifar10-default-0-cifar10-pytorch-7744fdc4dd-pgpxm   3/3     Terminating   0          7m3s
# cifar10-default-0-cifar10-pytorch-7744fdc4dd-vzx4x   3/3     Running       0          114m
```

In this section, we’ve learned how to scale the number of pods up and down based on CPU usage. In the next section, we’ll use KEDA to scale our deployment more flexibly based on custom metrics.

## Pod Autoscaling with KEDA

KEDA can fetch metrics from many sources (they are called scalers), see the list of supported scalers [here](https://keda.sh/docs/2.9/scalers/). We’ll set up KEDA to fetch metrics from a Prometheus server. The Prometheus server collects metrics from Seldon deployments in the cluster. The collected metrics are later used to trigger pod scaling.

### Install Seldon Monitoring and KEDA

Follow this [instruction](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/analytics.html) to install Seldon’s stack for monitoring, which includes a Prometheus server. The following pods should now be present in the `seldon-monitoring` namespace:

```bash
kubectl get pods -n seldon-monitoring
# NAME                                                    READY   STATUS    RESTARTS     AGE
# alertmanager-seldon-monitoring-alertmanager-0           2/2     Running   3 (8h ago)   26h
# prometheus-seldon-monitoring-prometheus-0               2/2     Running   2 (8h ago)   26h
# seldon-monitoring-blackbox-exporter-dbbcd845d-qszj8     1/1     Running   1 (8h ago)   26h
# seldon-monitoring-kube-state-metrics-7588b77796-nrd9g   1/1     Running   1 (8h ago)   26h
# seldon-monitoring-node-exporter-fmlh6                   1/1     Running   1 (8h ago)   26h
# seldon-monitoring-operator-6dc8898f89-fkwx8             1/1     Running   1 (8h ago)   26h
```

Also check if the pod monitor for Seldon Core has been created:

```bash
kubectl get PodMonitor -n seldon-monitoring
# NAME                AGE
# seldon-podmonitor   26h
```

Run the following command to enable KEDA in Seldon Core:

```bash
helm upgrade seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set keda.enabled=true \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system
```

Install KEDA to the cluster, and make sure that the previously installed KEDA (if any) is completely uninstalled:

```bash
kubectl delete -f https://github.com/kedacore/keda/releases/download/v2.9.1/keda-2.9.1.yaml
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.9.1/keda-2.9.1.yaml
```

### Deploy models with KEDA

We’ve had everything set up. Let’s create a Seldon deployment with KEDA. Similar to HPA, to enable KEDA in a deployment, we only need to include `kedaSpec` in the deployment’s manifest. Consider the following spec:

```yaml
kedaSpec:
  pollingInterval: 15
  minReplicaCount: 1
  maxReplicaCount: 2
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://seldon-monitoring-prometheus.seldon-monitoring.svc.cluster.local:9090
      metricName: access_frequency
      threshold: '20'
      query: avg(rate(seldon_api_executor_client_requests_seconds_count{deployment_name=~"cifar10"}[1m]))
```

`serverAddress` is the address of the Prometheus server within the cluster, it should be the URL of the Prometheus service, we can check the service with `kubectl get svc -n seldon-monitoring`. The scaling will be triggered when the metric value exceeds `threshold`. `query` is the query to fetch the metrics we want to monitor, in this case, it’s the average number of requests per second across running replicas.

Apply this [manifest](https://github.com/tintn/ml-model-deployment-tutorials/blob/main/scalable-serving/cifar10-deploy-keda.yaml) to deploy the model:

```bash
kubectl apply -f cifar10-deploy-keda.yaml
```

Let’s trigger the autoscaling by sending requests to the deployment:

```bash
locust -f test.py --headless -u 100 -r 10 --run-time 180 -H http://localhost:8080
```

After a few seconds, you can see a new pod created:

```bash
kubectl get pods
# NAME                                                 READY   STATUS     RESTARTS      AGE
# cifar10-default-0-cifar10-pytorch-5dc484599c-2zrv8   3/3     Running    3 (18m ago)   35h
# cifar10-default-0-cifar10-pytorch-5dc484599c-ljk74   0/3     Init:0/2   0             9s
```

Similar to HPA, downscaling will be triggered after a certain period (5 minutes by default) of low traffic.

```bash
kubectl get pods -w
# NAME                                                 READY   STATUS    RESTARTS      AGE
# cifar10-default-0-cifar10-pytorch-5dc484599c-2zrv8   3/3     Running   3 (22m ago)   35h
# cifar10-default-0-cifar10-pytorch-5dc484599c-ljk74   3/3     Running   0             3m55s
# cifar10-default-0-cifar10-pytorch-5dc484599c-ljk74   3/3     Terminating   0             5m49s
# cifar10-default-0-cifar10-pytorch-5dc484599c-ljk74   2/3     Terminating   0             6m
# cifar10-default-0-cifar10-pytorch-5dc484599c-ljk74   1/3     Terminating   0             6m1s
# cifar10-default-0-cifar10-pytorch-5dc484599c-ljk74   0/3     Terminating   0             6m3s
```

## Conclusion

We’ve learned how to deploy machine learning models to Kubernetes clusters with Seldon Core. Although we focused mostly on deploying PyTorch models in this tutorial, the same steps can be applied to deploying models from other frameworks with minor changes.

We’ve also made the deployments scalable using HPA and KEDA. Compared to HPA, KEDA provides more flexible ways to scale the system based on Prometheus metrics (or other supported scalers from KEDA). Technically, we can implement any scaling rules from metrics that can be fetched from the Prometheus server.