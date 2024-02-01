---
title: 安装 gitlab
date: 2022-08-22 18:50:17
tags:
---

访问网站 [gitlab.cn/install](https://gitlab.cn/install/)，根据其中的安装说明进行安装。

注意，不配置 DNS 时，需要修改访问 URL 为，

```sh
sudo EXTERNAL_URL="http://192.168.xx.xx:18888" apt-get install gitlab-jh
```

当然也可以在安装之后到配置文件 `/etc/gitlab/gitlab.rb` 中修改 `external_url`。由于我这里使用的是 WSL2 部署 gitlab 服务，而 WSL2 每次开机 ip 均会变，所以这里配置的 ip 为 `http://0.0.0.0:18888`

修改好配置文件后，运行

```sh
sudo /opt/gitlab/embedded/bin/runsvdir-start &  # 下一行配置执行报错时需要先执行这行
sudo gitlab-ctl reconfigure
```

刷新配置后，重启服务

```sh
sudo gitlab-ctl restart
```

这里我是在本机的 WSL2 中安装的 gitlab，导致其他局域网机器无法访问 gitlab，处理方法参考 [如何在局域网的其他主机上访问本机的WSL2](https://zhuanlan.zhihu.com/p/652237989)，

在 WSL2 中输入 `ip address` 查看 ip，然后配置到 `/etc/gitlab/gitlab.rb`，刷新配置并启动服务后，根据上面的网址配置本机（windows）上的端口转发规则，

```sh
netsh interface portproxy add v4tov4 listenport=18889 listenaddress=0.0.0.0 connectport=18888 connectaddress=localhost
```

其中，`18888` 是 WSL2 中 gitlab 的服务端口，`18889` 是 windows 系统中用于转发的端口，这样在其他机器上就可以访问 gitlab 了，

```sh
http://<windows-ip>:18889
```

初次登录需要修改密码，初始密码查看方法，

```sh
sudo vim /etc/gitlab/initial_root_password
```

登录账号为 `root`。

WSL2 中，没有启动 `systemctl`，而是使用 `init` 代替，使用如下命令查看，

```sh
ps -p 1 -o comm=
```

如果 gitlab 停止，从而需要重启的时候，需要执行

```sh
sudo /opt/gitlab/embedded/bin/runsvdir-start &  # 下一行配置执行报错时需要先执行这行
sudo gitlab-ctl restart # 这一句可不执行，直接上一条命令即可
```

手动停止服务，

```sh
sudo gitlab-ctl stop
```