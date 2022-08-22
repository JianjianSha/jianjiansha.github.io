---
title: Slurm
date: 2022-04-29 17:23:52
tags:
---

本文各服务最终目的是为了配合 Slurm，某些服务的配置或执行的操作需要配合 Slurm，所以建议阅读全文之后再安装这些服务。

# 1. NFS

网络文件系统 NFS ： 网络文件系统，最早由SUN公司研发的一种在网络中主机之间进行数据传输的技术，提供的网络服务让使用者访问网络上的文件就像访问自己本地的文件一样。

以 ubuntu 系统为例进行说明。

__服务端__，`ip:192.168.1.1`

1. 安装 NFS 服务
    ```sh
    sudo apt install nfs-kernel-server
    ```

2. 创建服务器共享文件目录，例如 `/work/share`
    ```sh
    sudo mkdir -p /work/share
    ```
3. 创建配置文件 `/etc/exports`
    ```sh
    sudo vim /etc/exports
    ```
4. 配置文件中添加配置
    ```sh
    # /etc/exports
    # add content
    /work/share *(rw,sync,no_root_squash)
    ```
5. 开启服务
    ```sh
    sudo systemctl restart nfs-kernel-server
    # or
    # sudo service nfs-kernel-server restart
    # sudo /etc/init.d/nfs-kernel-server restart
    ```
6. 展示共享文件路径位置和属性
    ```sh
    sudo showmount -e
    ```

__客户端__， `ip: 192.168.1.2`
1. 安装 NFS 工具
    ```sh
    sudo apt-get install nfs-common
    ```
2. 创建挂载目录，例如 `share`
    ```sh
    mkdir -p /work/share
    # 这里的目录与服务器上的目录可以不同
    ```
3. 再次显示服务器上的共享目录
    ```sh
    showmount -e 192.168.1.1
    ```
4. 挂载
    ```sh
    mount -t nfs 192.168.1.1:/work/share /work/share
    ```
5. 取消挂载
    ```sh
    umount /work/share

    # 强制解挂，例如有进程正在使用这个挂载目录
    # umount -f /work/share
    # or 找出这个进程 kill -9 <pid>，然后在正常解挂
    ```

# 2. munge
解决高性能计算（HPC）控制节点和计算节点的通信认证，常见的认证方式有：

- authd
- munge：允许进程对一组具有相同 普通用户(UID)和组(GID)的主机对另一个本地或远程的进程进行身份验证。这些主机组成了 共享密码密钥 的 安全领域(security realm)。此安全领域中的客户端可以在不使用 root特权、保留端口 或 特定于平台 的创建和验证凭据。

简单的说是通过对数据（payload）的加解密（或者称编码），实现数据的安全传输。

## 2.1 安装
```sh
# 安装，会自动创建munge用户和组，默认key在/etc/munge/munge.key,这个key要归属到munge用户上，权限是400
sudo apt install munge
```

也可以手动创建密钥
```sh
# -u munge 确保密钥文件属于 munge 用户和 munge 组
sudo -u munge /usr/sbin/mungekey -v
# 查看密钥文件信息
ls -lh /etc/munge/munge.key
```

## 2.2 服务操作
```sh
sudo systemctl enable munge
sudo systemctl start munge
sudo systemctl stop munge
sudo systemctl status munge
```
## 2.3测试

### 2.3.1 本地-编码解码
```sh
# 编码测试, -n 为 --no-input，即空字符串
munge -n
# 对字符串编码
munge -s "some string"
# 编码加解码测试，没有输入字符串
munge -n | unmunge
# 编码解码测试，有输入字符串
munge -s "some string" | unmunge
```

将本地munge.key发到远端服务器上，覆盖远端服务器的key，重启远端服务器的munge就可以联机测试

### 2.3.2 联机-编码解码

```sh
// 本地编码，远端解码
munge -n -t 10 | ssh 192.168.1.2 unmunge
munge -s "some string" -t 10 | ssh 192.168.1.2 unmunge
// 远端编码，本地解码
ssh 192.168.1.2 munge -n -t 10 | unmunge
ssh 192.168.1.2 munge -s "some string" -t 10 | unmunge
```
## 2.4 查看
查看 munged 服务打开的文件，
```sh
lsof -p $(pgrep -f munged)
```
可以看到 munge 日志文件位于 `/var/log/munge/munged.log

## 2.4 应用
一般和 Slurm 配合使用，进行如下配置：
```sh
# /etc/slurm/slurm.conf
AuthType=auth/munge
```

# 3. pdsh

- pdsh是一个多线程远程shell客户机，它在多个远程主机上并行执行命令
- pdsh可以使用几种不同的远程shell服务，包括标准的 rsh、Kerberos IV 和 ssh
- 在使用pdsh之前，必须保证本地主机和要管理远程主机之间的单向信任
- pdsh还附带了pdcp命令，该命令可以将本地文件批量复制到远程的多台主机上，这在大规模的文件分发环境下非常有用

> 使用pdcp命令要求本地主机和远程主机必须安装pdsh工具

## 3.1 安装
源码安装
```sh
su  # 切换到 root 用户
wget https://github.com/grondo/pdsh/archive/pdsh-2.31.tar.gz
tar xf pdsh-2.31.tar.gz -C /usr/local/src/
cd /usr/local/src/pdsh-pdsh-2.31/
./configure \
--prefix=/usr/local/pdsh \
--with-ssh \
--with-machines=/usr/local/pdsh/machines \
--with-dshgroups=/usr/local/pdsh/group \
--with-rcmd-rank-list=ssh \
--with-exec && \
make && \
make install
```
- --with-ssh ssh模块（支持ssh）
- --with-rcmd-rank-list=ssh 指定默认模式为ssh
- --with-dshgroups= 指定默认主机组路径
- --with-machines= 指定默认主机列表
    - 在该文件中写入主机地址（或主机名，需要在hosts中写好主机解析），每行一个
    - 存在machines文件，使用pdsh执行时若不指定主机，则默认对machines文件中所有主机执行该命令
- --with-exec exec模块
- 其他模块参数可以在pdsh-pdsh-2.31目录下使用 ./configure --help 命令查看

## 3.2 使用
- 语法：pdsh <参数> <需要并行执行的命令>
    - 如果只输入前面两部分，回车后可进入pdsh交互式命令行（若是编译安装需要启用--with-readline），再输入并行执行的命令部分
- 常用参数：
    - -w 指定主机 -x 排除指定的主机
        - 目标主机可以使用Ip地址或主机名（确保该主机名已经在/etc/hosts中存在解析）
        - 多个主机之间可以使用逗号分隔，可重复使用该参数指定多个主机；可以使用简单的正则
    - -g 指定主机组 -G 排除指定主机组
    - -l 目标主机的用户名
        - 如果不指定用户名，默认以当前用户名作为在目标主机上执行命令的用户名
    - -N 用来关闭目标主机所返回值前的主机名显示

### 3.2.1 使用示例

__指定主机__
```sh
pdsh -w ssh:192.168.72.12,192.168.72.13,192.168.72.14 date
# 或者使用
pdsh -w ssh:192.168.72.[12-14] date
```

__指定用户__
```sh
# 指定名为 'linux' 的用户
pdsh -w ssh:192.168.72.[12-14] -l linux date
```

__指定主机组__
```sh
# 创建组文件所在目录
mkdir /usr/local/pdsh/group
# 将主机列表写入 test1 文件中
cat > /usr/local/pdsh/group/test1 <<EOF
192.168.1.1
192.168.1.2
192.168.1.3
EOF
# 在组文件所列的主机列表上执行命令
pdsh -g test1 'uname -r'
```

__主机列表__

```sh
# 查看主机列表
cat > /usr/local/pdsh/machines <<EOF
# 执行命令
pdsh -a uptime
```

__交互式界面__
```sh
# 有exec模块即可，或者readline模块
pdsh -a
```

进入交互式界面
```sh
pdsh> date
...
pdsh> whoami
...
pdsh> exit
```

__时间同步__
```sh
# 发现各节点时间不同
pdsh -a date
# 执行同步命令
pdsh -a ntpdate 192.168.1.1
```
使用 NTP 服务通过定时任务的方式同步各节点的时间。

__文件拷贝__

```sh
# 创建一个临时文件
touch test2
# 使用 'pdcp' 命令进行拷贝
pdcp -a test2 /tmp/
# 查看文件是否存在
pdsh -a ls -lrt /tmp/ | grep test2
```

# 4. Slurm
参考 [Slurm资源管理与作业调度系统安装配置](http://hmli.ustc.edu.cn/doc/linux/slurm-install/index.html)