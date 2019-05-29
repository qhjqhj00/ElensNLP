# Lensnlp Docker 快速开始

​	为方便使用，和快速部署，lensnlp提供了docker版本。docker中内置了CUDA 9.0，Pytorch 1.0，lensnlp以及其他依赖环境。

​	欢迎大家使用，如果遇到任何问题，请联系Hongjin。

#### Step 1 

下载lensnlp docker 的 .tar包，下载地址为：

http://101.254.159.164:4501/source/docker/lensnlp_docker.tar

如果没有权限访问，请联系Hongjin开通新账号。

#### Step 2

配置docker环境（需要root权限）：

a. 安装docker ce

详情请参考：https://docs.docker.com/install/

b. 如果是GPU环境，请安装nvidia-docker

详情请参考：https://github.com/NVIDIA/nvidia-docker

如果您的环境是ubuntu，有网络链接，可以执行脚本安装：

脚本地址：http://101.254.159.164:4501/source/docker/docker.sh

```shell
sudo sh docker.sh
```

#### Step 3 

加载docker file成镜像：

```
docker load<lensnlp_docker.tar
```

然后：

```shell
docker images
```

会看到加载好的镜像：

![镜像](./images/image.png)

其中REPOSITORY:TAG构成镜像名称，例如：lensnlp:0522

#### Step 4

接下来，需要运行一个基于镜像的容器，可以使用以下命令：

```shell
docker run -itd --restart=always -v /home/lensAI:/home/lensAI  -p 4000-4100:4000-4100 --name  lensnlp_20 lensnlp:0522
```

其中：

-i： 以交互模式运行容器，通常与 -t 同时使用；

-t：为容器重新分配一个伪输入终端，通常与 -i 同时使用；

-d：后台运行容器，并返回容器ID；

-v：定义文件夹映射；

-p：定义端口映射；

--name：容器名字；

--restart：定义重启机制，always为不管容器的退出状态是什么，都可以重启。

运行容器的各种标签可以参考：

https://docs.docker.com/engine/reference/commandline/run/

http://www.runoob.com/docker/docker-run-command.html

##### Tips: 非root账户执行docker命令需要sudo，如果想要直接执行docker命令，可以执行以下命令。

```shell
sudo usermod -aG docker your_username
```

执行完之后记得重新登录哦。

#### Step 5

现在就可以进入容器来使用lensnlp包了！

查看容器列表：

```shell
docker ps
```

可以看到已经运行的容器：

![容器](./images/container.png)

进入容器：

```shell
docker exec -it lensnlp bash
```

##### 更多docker信息，请参考：https://docs.docker.com

