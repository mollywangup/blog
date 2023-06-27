- [安装Docker](#安装docker)
- [下载并启动Superset Docker镜像](#下载并启动superset-docker镜像)
  - [Start a superset instance on port 8080](#start-a-superset-instance-on-port-8080)
  - [Initialize a local Superset Instance](#initialize-a-local-superset-instance)
  - [启动 Apache Drill](#启动-apache-drill)


# 安装Docker
```shell
brew install docker

docker --version
```
或者[Install Docker Desktop on Mac](https://docs.docker.com/desktop/install/mac-install/)


# 下载并启动Superset Docker镜像
注意：不挂VPN也能拉
```shell
docker pull apache/superset

docker run -d -p 8080:8088 --name superset apache/superset

openssl rand -hex 32

```


## Start a superset instance on port 8080
```
docker run -d -p 8080:8088 --name superset apache/superset
```

## Initialize a local Superset Instance
```
docker exec -it superset superset fab create-admin \
              --username admin \
              --firstname Superset \
              --lastname Admin \
              --email admin@superset.com \
              --password admin

docker exec -it superset superset db upgrade

docker exec -it superset superset load_examples

docker exec -it superset superset init


http://localhost:8080/login/ -- u/p: [admin/admin]
```


```shell
cd /Users/liwang/GoogleDrive/python

git clone https://github.com/apache/superset.git


cd superset


```


MongoBI
mongobi://root?source=admin:root@poc_ssm_mongo_bi:3307/samples



## 启动 Apache Drill
```shell
docker run -p 8047:8047 -p 31010:31010 -it apache/drill /bin/bash
```

[MongoDB to Apache Drill to Apache Superset](https://www.shubhamdipt.com/blog/mongodb-to-apache-drill-to-apache-superset/)

Apache Drill
drill+sadrill://localhost:8047/mongo?use_ssl=False


mongodb://127.0.0.1:27017/



```shell
cd /Users/liwang/GoogleDrive/python

git clone https://github.com/apache/drill.git

cd drill
```