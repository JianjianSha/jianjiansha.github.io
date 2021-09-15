---
title: ubuntu zsh 使用
date: 2021-01-23 18:27:51
tags:
---
安装 zsh
```
sudo apt-get install zsh
```
查看版本，验证是否安装成功
```
zsh --version
```
更改默认 shell 为 zsh
```
chsh -s /bin/zsh
```
查看 shell
```
echo $SHELL
```
如果失败，那么重启 shell 即可。


安装 oh-my-zsh，这是一个 zsh 配置框架。
```
wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh
```
修改文件权限，
```
chmod +x install.sh
```
 执行安装，
 ```
 ./install.sh
 ```

安装 powerlevel10k 这个主题，
```
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

设置主题，将以下内容添加到 `~/.zshrc`，
```
ZSH_THEME="powerlevel10k/powerlevel10k"
```

# redirect printing message to file
## stop output to console but file
```sh
ls -al >> output.txt
ls -al >> output.txt 2>&1 # redirect the stderr to stdout
```

## output to console and file 
```sh
ls -al 2>&1 | tee output.txt
```